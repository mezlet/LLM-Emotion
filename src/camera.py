"""
camera.py
---------
Webcam management and DeepFace facial-emotion analysis.

The facial signal produced here is intentionally treated as a *weak hint*
and is never the primary determinant of emotional interpretation.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace

from config import (
    CAMERA_DEVICE,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    CAMERA_SAMPLE_EVERY_SECONDS,
    CAMERA_WARMUP_SECONDS,
    DEEPFACE_ACTIONS,
    DEEPFACE_ALIGN,
    DEEPFACE_DETECTOR_BACKEND,
)
from models import FaceEmotionCapture
from utils import now_ts, print_ts


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_camera_devices(max_indices: int = 4) -> None:
    """Probe camera indices 0 … *max_indices*-1 and print the available ones."""
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
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [cam {idx}] available | resolution={width}x{height}")
        finally:
            cap.release()

    if not found:
        print("  No camera devices found.")
    print()


def choose_camera_device(current_device: Optional[int]) -> Optional[int]:
    """
    Prompt the user to pick a camera by index.

    Returns the new device index, or *current_device* if the selection
    is blank or invalid.
    """
    list_camera_devices()
    selection = input(
        "Enter camera device index (blank to keep current/default): "
    ).strip()

    if not selection:
        return current_device

    try:
        camera_index = int(selection)
        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                print("That camera could not be opened.\n")
                return current_device
            ok, _ = cap.read()
            if not ok:
                print("That camera opened but did not return a frame.\n")
                return current_device
            print_ts(f"Using camera device [{camera_index}]")
            return camera_index
        finally:
            cap.release()

    except Exception as exc:
        print(f"Invalid camera device selection: {exc}\n")
        return current_device


# ---------------------------------------------------------------------------
# Camera open helper
# ---------------------------------------------------------------------------

def open_camera(camera_device: Optional[int]) -> cv2.VideoCapture:
    """Open the camera at *camera_device* (defaults to index 0) and set resolution."""
    index = 0 if camera_device is None else camera_device
    cap   = cv2.VideoCapture(index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    return cap


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def analyze_frame_emotion_scores(frame_bgr: np.ndarray) -> Optional[dict[str, float]]:
    """
    Run DeepFace on a single BGR frame and return the full emotion score dict.

    Using per-emotion scores (rather than a single dominant label per frame)
    gives a much more stable averaged signal across multiple frames.

    Returns ``None`` if no face is detected or DeepFace raises an exception.
    """
    try:
        result = DeepFace.analyze(
            img_path=frame_bgr,
            actions=DEEPFACE_ACTIONS,
            enforce_detection=False,
            detector_backend=DEEPFACE_DETECTOR_BACKEND,
            align=DEEPFACE_ALIGN,
            silent=True,
        )

        face_result = result[0] if isinstance(result, list) else result
        scores = face_result.get("emotion")

        if not isinstance(scores, dict):
            return None

        return {emotion: float(score) for emotion, score in scores.items()}

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def capture_face_emotion_during_recording(
    duration_seconds: int,
    camera_device: Optional[int] = CAMERA_DEVICE,
    sample_every_seconds: float = CAMERA_SAMPLE_EVERY_SECONDS,
) -> FaceEmotionCapture:
    """
    Capture webcam frames for *duration_seconds* and accumulate DeepFace
    emotion scores.

    The result is a :class:`FaceEmotionCapture` that averages scores across
    all sampled frames.  Designed to run in a background thread while audio
    is being recorded.
    """
    started_at = now_ts()
    emotion_score_samples: list[dict[str, float]] = []
    frame_count = 0
    sampled_frame_count = 0

    cap = open_camera(camera_device)
    if not cap.isOpened():
        return FaceEmotionCapture(
            emotion_score_samples=[],
            frame_count=0,
            sampled_frame_count=0,
            started_at=started_at,
            ended_at=now_ts(),
            error="Could not open camera.",
        )

    try:
        end_time          = time.time() + duration_seconds
        capture_start     = time.time()
        last_sample_time  = 0.0

        print_ts("Camera emotion capture is active during voice recording.")

        while time.time() < end_time:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()

            # Skip the warm-up window so auto-exposure can settle.
            if now - capture_start < CAMERA_WARMUP_SECONDS:
                continue

            if now - last_sample_time < sample_every_seconds:
                continue

            last_sample_time = now
            sampled_frame_count += 1

            scores = analyze_frame_emotion_scores(frame)
            if scores:
                emotion_score_samples.append(scores)

            time.sleep(0.02)

    finally:
        cap.release()

    return FaceEmotionCapture(
        emotion_score_samples=emotion_score_samples,
        frame_count=frame_count,
        sampled_frame_count=sampled_frame_count,
        started_at=started_at,
        ended_at=now_ts(),
        error=None,
    )