from __future__ import annotations

import threading
from typing import Optional, Tuple

from audio_service import record_audio_to_wav
from camera_service import capture_face_emotion_during_recording
from config import CAMERA_DEVICE, CAMERA_SAMPLE_EVERY_SECONDS, INPUT_DEVICE, MAX_RECORD_SECONDS
from models import FaceEmotionCapture


def record_audio_and_capture_face_emotion(
    max_seconds: int = MAX_RECORD_SECONDS,
    input_device: Optional[int] = INPUT_DEVICE,
    camera_device: Optional[int] = CAMERA_DEVICE,
) -> Tuple[Optional[str], Optional[FaceEmotionCapture]]:
    face_result: dict[str, Optional[FaceEmotionCapture]] = {"data": None}

    def face_worker() -> None:
        face_result["data"] = capture_face_emotion_during_recording(
            duration_seconds=max_seconds,
            camera_device=camera_device,
            sample_every_seconds=CAMERA_SAMPLE_EVERY_SECONDS,
        )

    thread = threading.Thread(target=face_worker, daemon=True)
    thread.start()
    wav_path = record_audio_to_wav(max_seconds=max_seconds, input_device=input_device)
    thread.join(timeout=max_seconds + 5)
    return wav_path, face_result.get("data")
