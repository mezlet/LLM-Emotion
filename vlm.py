import base64
import json
import re
import sys
import time
from io import BytesIO

import cv2
import requests
import torch
from PIL import Image
from facenet_pytorch import MTCNN


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen2.5vl:7b"
#MODEL_NAME = "llama3.2-vision:11b"

CAMERA_DEVICE = 0  # try 1 or 2 if needed

ZED_STEREO_WIDTH = 2560
ZED_STEREO_HEIGHT = 720

PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

ANALYZE_EVERY_N_FRAMES = 60
FACE_MARGIN = 0.25


device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    keep_all=True,
    device=device,
    post_process=False
)


def get_camera_backend():
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_zed_camera(camera_device):
    cap = cv2.VideoCapture(camera_device, get_camera_backend())

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {camera_device}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ZED_STEREO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ZED_STEREO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    time.sleep(0.5)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Camera opened, but no frame was received.")

    print(
        f"Camera opened: /dev/video{camera_device}, "
        f"actual size={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    return cap


def split_zed_stereo_frame(stereo_frame):
    height, width = stereo_frame.shape[:2]
    mid = width // 2
    return stereo_frame[:, :mid], stereo_frame[:, mid:]

def combine_faces_side_by_side(left_face, right_face):
    faces = []

    if left_face is not None:
        faces.append(left_face)

    if right_face is not None:
        faces.append(right_face)

    if not faces:
        return None

    if len(faces) == 1:
        return faces[0]

    target_height = min(f.shape[0] for f in faces)

    resized_faces = []
    for face in faces:
        h, w = face.shape[:2]
        new_width = int(w * target_height / h)
        resized = cv2.resize(face, (new_width, target_height))
        resized_faces.append(resized)

    return cv2.hconcat(resized_faces)


def crop_best_face(frame_bgr):
    """
    Detect faces with MTCNN and return:
    - cropped face image
    - bounding box
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or probs is None:
        return None, None

    best_index = int(probs.argmax())
    if probs[best_index] < 0.80:
        return None, None

    x1, y1, x2, y2 = boxes[best_index]

    h, w = frame_bgr.shape[:2]

    face_w = x2 - x1
    face_h = y2 - y1

    margin_x = face_w * FACE_MARGIN
    margin_y = face_h * FACE_MARGIN

    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(w, int(x2 + margin_x))
    y2 = min(h, int(y2 + margin_y))

    cropped = frame_bgr[y1:y2, x1:x2]

    if cropped.size == 0:
        return None, None

    return cropped, (x1, y1, x2, y2)


def frame_to_base64_jpeg(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def ask_qwen_emotion_from_faces(face_images_base64):
    prompt = """
You are analyzing cropped face images from a ZED stereo camera.

The images may include:
- left camera cropped face
- right camera cropped face

Use all provided face crops together.
Classify the most likely facial emotion.

Return ONLY valid JSON:

{
  "face_detected": true,
  "emotion": "happy",
  "confidence": 0.0,
  "reason": "short explanation"
}

Allowed emotions:
happy, sad, angry, surprised, neutral, fearful, disgusted, unknown

Rules:
- confidence must be between 0.0 and 1.0
- if no clear human face is visible, set face_detected to false
- if unsure, use emotion "unknown"
- do not include markdown
- do not include extra text outside JSON
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": face_images_base64,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 160
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    if response.status_code != 200:
        print("Ollama error body:")
        print(response.text)
        response.raise_for_status()

    raw = response.json().get("response", "").strip()
    data = extract_json(raw)

    if data is None:
        return {
            "face_detected": False,
            "emotion": "unknown",
            "confidence": 0.0,
            "reason": f"Could not parse model response: {raw}"
        }

    return {
        "face_detected": bool(data.get("face_detected", False)),
        "emotion": str(data.get("emotion", "unknown")).lower(),
        "confidence": float(data.get("confidence", 0.0)),
        "reason": str(data.get("reason", ""))
    }


def draw_status(frame, label, reason, box=None):
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    if reason:
        cv2.putText(
            frame,
            reason[:90],
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


def main():
    print(f"Using MTCNN on: {device}")

    cap = open_zed_camera(CAMERA_DEVICE)

    frame_count = 0
    latest_label = "Waiting for analysis..."
    latest_reason = ""
    latest_left_box = None

    print("Running. Press q to quit.")

    try:
        while True:
            ok, stereo_frame = cap.read()

            if not ok or stereo_frame is None:
                print("Could not read frame.")
                continue

            left_frame, right_frame = split_zed_stereo_frame(stereo_frame)

            preview_frame = left_frame.copy()

            frame_count += 1

            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                try:
                    left_face, left_box = crop_best_face(left_frame)
                    right_face, right_box = crop_best_face(right_frame)

                    latest_left_box = left_box

                    combined_face = combine_faces_side_by_side(left_face, right_face)

                    if combined_face is None:
                        result = {
                            "face_detected": False,
                            "emotion": "unknown",
                            "confidence": 0.0,
                            "reason": "MTCNN did not detect a clear face."
                        }
                    else:
                        face_images_base64 = [frame_to_base64_jpeg(combined_face)]
                        result = ask_qwen_emotion_from_faces(face_images_base64)

                    print(json.dumps(result, indent=2))

                    if result["face_detected"]:
                        latest_label = (
                            f"{result['emotion']} "
                            f"({result['confidence']:.2f})"
                        )
                    else:
                        latest_label = "No face detected"

                    latest_reason = result.get("reason", "")

                except Exception as exc:
                    latest_label = "Analysis error"
                    latest_reason = str(exc)
                    print(f"Analysis error: {exc}")

            draw_status(
                preview_frame,
                latest_label,
                latest_reason,
                latest_left_box
            )

            preview = cv2.resize(preview_frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            cv2.imshow("ZED preview + cropped stereo face emotion", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()