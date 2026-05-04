from __future__ import annotations

import base64
import time
from typing import Optional

import cv2
import numpy as np
from ollama import Client

from config import (
    CAMERA_EMOTION_BACKEND,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    CAMERA_SAMPLE_EVERY_SECONDS,
    CAMERA_WARMUP_SECONDS,
    DEEPFACE_ACTIONS,
    DEEPFACE_ALIGN,
    DEEPFACE_DETECTOR_BACKEND,
    OLLAMA_HOST,
    VLM_EMOTION_IMAGE_JPEG_QUALITY,
    VLM_EMOTION_MODEL,
    VLM_EMOTION_TEMPERATURE,
)
from json_utils import safe_json_extract
from models import FaceEmotionCapture
from time_utils import now_ts, print_ts

SUPPORTED_CAMERA_EMOTION_BACKENDS = {"deepface", "vlm"}
CANONICAL_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def list_camera_devices(max_indices: int = 4) -> None:
    print("\nAvailable camera devices (best-effort probe):")
    found = False
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            found = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [cam {idx}] available | resolution={width}x{height}")
        finally:
            cap.release()
    if not found:
        print("  No camera devices found.")
    print()


def choose_camera_device(current_camera_device: Optional[int]) -> Optional[int]:
    list_camera_devices()
    selection = input("Enter camera device index (blank to keep current/default): ").strip()
    if not selection:
        return current_camera_device
    try:
        camera_index = int(selection)
        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                print("That camera could not be opened.\n")
                return current_camera_device
            ok, frame = cap.read()
            if not ok or frame is None:
                print("That camera opened but did not return a frame.\n")
                return current_camera_device
            print_ts(f"Using camera device [{camera_index}]")
            return camera_index
        finally:
            cap.release()
    except Exception as exc:
        print(f"Invalid camera device selection: {exc}\n")
        return current_camera_device


def open_camera(camera_device: Optional[int]) -> cv2.VideoCapture:
    camera_index = 0 if camera_device is None else camera_device
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    return cap


def normalize_camera_emotion_backend(raw_backend: str) -> str:
    backend = raw_backend.strip().lower()
    if backend not in SUPPORTED_CAMERA_EMOTION_BACKENDS:
        raise ValueError(
            f"Unsupported CAMERA_EMOTION_BACKEND={raw_backend!r}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_CAMERA_EMOTION_BACKENDS))}."
        )
    return backend


def analyze_frame_emotion_scores(frame_bgr: np.ndarray) -> Optional[dict[str, float]]:
    return analyze_frame_emotion_scores_with_backend(frame_bgr, CAMERA_EMOTION_BACKEND)


def analyze_frame_emotion_scores_with_backend(
    frame_bgr: np.ndarray,
    backend: str,
    ollama_client: Optional[Client] = None,
) -> Optional[dict[str, float]]:
    backend = normalize_camera_emotion_backend(backend)
    if backend == "deepface":
        return analyze_frame_emotion_scores_with_deepface(frame_bgr)
    return analyze_frame_emotion_scores_with_vlm(frame_bgr, ollama_client=ollama_client)


def analyze_frame_emotion_scores_with_deepface(frame_bgr: np.ndarray) -> Optional[dict[str, float]]:
    try:
        from deepface import DeepFace

        result = DeepFace.analyze(
            img_path=frame_bgr,
            actions=DEEPFACE_ACTIONS,
            enforce_detection=False,
            detector_backend=DEEPFACE_DETECTOR_BACKEND,
            align=DEEPFACE_ALIGN,
            silent=True,
        )
        face_result = result[0] if isinstance(result, list) and result else result
        scores = face_result.get("emotion") if isinstance(face_result, dict) else None
        return sanitize_emotion_scores(scores) if isinstance(scores, dict) else None
    except Exception:
        return None


def encode_frame_as_base64_jpeg(frame_bgr: np.ndarray, jpeg_quality: int = VLM_EMOTION_IMAGE_JPEG_QUALITY) -> Optional[str]:
    quality = int(max(1, min(jpeg_quality, 100)))
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def build_vlm_emotion_prompt() -> str:
    return """
You are analyzing a single webcam frame for facial expression only.
Return JSON only. Do not include markdown.
Use these exact emotion keys: angry, disgust, fear, happy, sad, surprise, neutral.
Rules:
- Estimate visible facial expression, not the person's true internal emotion.
- If no clear face is visible, set face_visible to false and use low confidence.
- Scores must be percentages from 0 to 100 and sum to approximately 100.
- Be conservative. Prefer neutral when the expression is unclear.
Return exactly this structure:
{
  "face_visible": true,
  "scores": {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 100
  },
  "reason": "brief visual explanation"
}
""".strip()


def analyze_frame_emotion_scores_with_vlm(
    frame_bgr: np.ndarray,
    ollama_client: Optional[Client] = None,
) -> Optional[dict[str, float]]:
    image_b64 = encode_frame_as_base64_jpeg(frame_bgr)
    if not image_b64:
        return None

    client = ollama_client or Client(host=OLLAMA_HOST)
    try:
        response = client.chat(
            model=VLM_EMOTION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": build_vlm_emotion_prompt(),
                    "images": [image_b64],
                }
            ],
            options={"temperature": VLM_EMOTION_TEMPERATURE},
            stream=False,
        )
        raw = response.get("message", {}).get("content", "")
        data = safe_json_extract(raw)
        if not isinstance(data, dict) or data.get("face_visible") is False:
            return None
        scores = data.get("scores")
        return sanitize_emotion_scores(scores) if isinstance(scores, dict) else None
    except Exception:
        return None


def sanitize_emotion_scores(raw_scores: dict) -> Optional[dict[str, float]]:
    cleaned: dict[str, float] = {}
    for emotion in CANONICAL_EMOTIONS:
        value = raw_scores.get(emotion, 0.0)
        try:
            cleaned[emotion] = max(0.0, min(float(value), 100.0))
        except (TypeError, ValueError):
            cleaned[emotion] = 0.0
    total = sum(cleaned.values())
    if total <= 0:
        return None
    return {emotion: (score / total) * 100.0 for emotion, score in cleaned.items()}


def capture_face_emotion_during_recording(
    duration_seconds: int,
    camera_device: Optional[int],
    sample_every_seconds: float = CAMERA_SAMPLE_EVERY_SECONDS,
) -> FaceEmotionCapture:
    started_at = now_ts()
    emotion_score_samples: list[dict[str, float]] = []
    frame_count = 0
    sampled_frame_count = 0

    try:
        backend = normalize_camera_emotion_backend(CAMERA_EMOTION_BACKEND)
    except ValueError as exc:
        return FaceEmotionCapture([], 0, 0, started_at, now_ts(), str(exc))

    cap = open_camera(camera_device)
    if not cap.isOpened():
        return FaceEmotionCapture([], 0, 0, started_at, now_ts(), "Could not open camera.")

    ollama_client = Client(host=OLLAMA_HOST) if backend == "vlm" else None

    try:
        end_time = time.time() + duration_seconds
        capture_start_time = time.time()
        last_sample_time = 0.0
        print_ts(f"Camera emotion capture is active during voice recording. backend={backend}")

        while time.time() < end_time:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame_count += 1
            current_time = time.time()
            if current_time - capture_start_time < CAMERA_WARMUP_SECONDS:
                continue
            if current_time - last_sample_time < sample_every_seconds:
                continue
            last_sample_time = current_time
            sampled_frame_count += 1
            scores = analyze_frame_emotion_scores_with_backend(frame, backend, ollama_client=ollama_client)
            if scores:
                emotion_score_samples.append(scores)
            time.sleep(0.02)
    finally:
        cap.release()

    return FaceEmotionCapture(emotion_score_samples, frame_count, sampled_frame_count, started_at, now_ts())
