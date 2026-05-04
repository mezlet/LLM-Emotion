from __future__ import annotations

from typing import Optional, Tuple

from audio_service import choose_input_device, delete_file_safely, list_input_devices
from camera_service import choose_camera_device, list_camera_devices
from commands import normalize_command
from config import MAX_RECORD_SECONDS
from face_emotion_service import classify_face_capture_to_plutchik, unavailable_face
from models import FaceEmotionCapture, ModalityEmotion
from prosody_service import classify_prosody_from_audio, unavailable_prosody
from time_utils import print_ts
from transcription_service import Transcriber
from voice_face_service import record_audio_and_capture_face_emotion

UserInputResult = Tuple[
    Optional[str],          # user_text
    Optional[int],          # current_input_device
    Optional[int],          # current_camera_device
    Optional[FaceEmotionCapture],
    ModalityEmotion,        # prosody emotion
    ModalityEmotion,        # face emotion
    str,                    # input_mode: typed | speech
]


def get_user_message_from_keyboard_or_voice(
    transcriber: Optional[Transcriber],
    current_input_device: Optional[int],
    current_camera_device: Optional[int],
) -> UserInputResult:
    empty_prosody = unavailable_prosody()
    empty_face = unavailable_face()

    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None, current_input_device, current_camera_device, None, empty_prosody, empty_face, "typed"

    if not user_text:
        return "", current_input_device, current_camera_device, None, empty_prosody, empty_face, "typed"

    normalized = normalize_command(user_text)

    if normalized == "devices":
        list_input_devices()
        return "", current_input_device, current_camera_device, None, empty_prosody, empty_face, "typed"
    if normalized == "mic":
        return "", choose_input_device(current_input_device), current_camera_device, None, empty_prosody, empty_face, "typed"
    if normalized == "cams":
        list_camera_devices()
        return "", current_input_device, current_camera_device, None, empty_prosody, empty_face, "typed"
    if normalized == "cam":
        return "", current_input_device, choose_camera_device(current_camera_device), None, empty_prosody, empty_face, "typed"
    if normalized == "speak":
        return _handle_voice_input(transcriber, current_input_device, current_camera_device)

    return user_text, current_input_device, current_camera_device, None, empty_prosody, empty_face, "typed"


def _handle_voice_input(
    transcriber: Optional[Transcriber],
    current_input_device: Optional[int],
    current_camera_device: Optional[int],
) -> UserInputResult:
    if transcriber is None:
        print("Voice input is unavailable because the selected transcription backend failed to load.\n")
        return "", current_input_device, current_camera_device, None, unavailable_prosody(), unavailable_face(), "speech"

    wav_path, face_capture = record_audio_and_capture_face_emotion(
        max_seconds=MAX_RECORD_SECONDS,
        input_device=current_input_device,
        camera_device=current_camera_device,
    )
    if not wav_path:
        return "", current_input_device, current_camera_device, face_capture, unavailable_prosody("No valid WAV was recorded."), classify_face_capture_to_plutchik(face_capture), "speech"

    try:
        # Extract prosody before deleting the temporary WAV.
        prosody_emotion = classify_prosody_from_audio(wav_path)

        try:
            transcript = transcriber.transcribe(wav_path)
        except Exception as exc:
            print(f"{transcriber.backend} transcription error: {exc}\n")
            transcript = ""
    finally:
        delete_file_safely(wav_path)

    face_emotion = classify_face_capture_to_plutchik(face_capture)

    if transcript:
        print_ts(f"Transcribed: {transcript}")
    else:
        print("No speech detected.\n")

    print_ts(f"Prosody emotion: {prosody_emotion.emotion.primary} ({prosody_emotion.summary})")

    if face_capture:
        if face_capture.error:
            print_ts(f"Face emotion capture error: {face_capture.error}")
        else:
            print_ts(f"Face emotion summary: {face_capture.summary_text}")
            print_ts(f"Face Plutchik emotion: {face_emotion.emotion.primary}")
    print()
    return transcript, current_input_device, current_camera_device, face_capture, prosody_emotion, face_emotion, "speech"
