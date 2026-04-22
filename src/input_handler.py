"""
input_handler.py
----------------
Reads user input from the keyboard and dispatches slash-commands.

The /speak command runs microphone recording and camera emotion capture
in parallel using a background thread.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

from config import CAMERA_DEVICE, INPUT_DEVICE, MAX_RECORD_SECONDS
from audio import (
    choose_input_device,
    list_input_devices,
    record_audio_to_wav,
    transcribe_with_whisperx,
)
from camera import (
    capture_face_emotion_during_recording,
    choose_camera_device,
    list_camera_devices,
)
from models import FaceEmotionCapture
from utils import normalize_command, print_ts


# Type alias for the 4-tuple returned by get_user_input().
UserInputResult = Tuple[
    Optional[str],          # user text (None → quit)
    Optional[int],          # (possibly updated) mic device
    Optional[int],          # (possibly updated) camera device
    Optional[FaceEmotionCapture],  # facial emotion data (speak only)
]


# ---------------------------------------------------------------------------
# Parallel recording helper
# ---------------------------------------------------------------------------

def record_audio_and_capture_face_emotion(
    max_seconds: int = MAX_RECORD_SECONDS,
    input_device: Optional[int] = INPUT_DEVICE,
    camera_device: Optional[int] = CAMERA_DEVICE,
) -> Tuple[Optional[str], Optional[FaceEmotionCapture]]:
    """
    Run microphone recording and camera emotion capture concurrently.

    The camera capture runs in a daemon thread so it is automatically
    terminated if the main thread exits unexpectedly.

    Returns:
        wav_path   – path to the recorded WAV file (None on failure / silence)
        face_data  – :class:`FaceEmotionCapture` result (None on thread error)
    """
    face_result: dict[str, Optional[FaceEmotionCapture]] = {"data": None}

    def _face_worker() -> None:
        face_result["data"] = capture_face_emotion_during_recording(
            duration_seconds=max_seconds,
            camera_device=camera_device,
        )

    thread = threading.Thread(target=_face_worker, daemon=True)
    thread.start()

    wav_path = record_audio_to_wav(
        max_seconds=max_seconds,
        input_device=input_device,
    )

    thread.join(timeout=max_seconds + 5)

    return wav_path, face_result.get("data")


# ---------------------------------------------------------------------------
# Main input function
# ---------------------------------------------------------------------------

def get_user_input(
    whisper_model,
    current_input_device: Optional[int],
    current_camera_device: Optional[int],
) -> UserInputResult:
    """
    Read one line of input from the user and handle slash-commands.

    Returns a 4-tuple:
      - ``None`` as the first element signals the caller to exit.
      - An empty string signals "no message this turn" (command handled).
      - Any other string is the user's message to process.
    """
    try:
        raw = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None, current_input_device, current_camera_device, None

    if not raw:
        return "", current_input_device, current_camera_device, None

    command = normalize_command(raw)

    # ---- device listing / selection commands --------------------------------

    if command == "devices":
        list_input_devices()
        return "", current_input_device, current_camera_device, None

    if command == "mic":
        new_device = choose_input_device(current_input_device)
        return "", new_device, current_camera_device, None

    if command == "cams":
        list_camera_devices()
        return "", current_input_device, current_camera_device, None

    if command == "cam":
        new_camera = choose_camera_device(current_camera_device)
        return "", current_input_device, new_camera, None

    # ---- voice input --------------------------------------------------------

    if command == "speak":
        if whisper_model is None:
            print("Voice input is unavailable because WhisperX failed to load.\n")
            return "", current_input_device, current_camera_device, None

        wav_path, face_capture = record_audio_and_capture_face_emotion(
            max_seconds=MAX_RECORD_SECONDS,
            input_device=current_input_device,
            camera_device=current_camera_device,
        )

        if not wav_path:
            return "", current_input_device, current_camera_device, face_capture

        try:
            transcript = transcribe_with_whisperx(wav_path, whisper_model)
        except Exception as exc:
            print(f"WhisperX transcription error: {exc}\n")
            transcript = ""
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if transcript:
            print_ts(f"Transcribed: {transcript}")
        else:
            print("No speech detected.\n")

        if face_capture:
            if face_capture.error:
                print_ts(f"Face emotion capture error: {face_capture.error}")
            else:
                print_ts(f"Face emotion summary: {face_capture.summary_text}")

        print()
        return transcript, current_input_device, current_camera_device, face_capture

    # ---- plain text ---------------------------------------------------------

    return raw, current_input_device, current_camera_device, None