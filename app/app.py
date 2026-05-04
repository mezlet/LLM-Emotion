from __future__ import annotations

import sys

from ollama import Client

from commands import normalize_command
from config import (
    CAMERA_DEVICE,
    INPUT_DEVICE,
    MAX_HISTORY_MESSAGES,
    MODEL_NAME,
    OLLAMA_HOST,
)
from face_emotion_service import classify_face_capture_to_plutchik, unavailable_face
from input_service import get_user_message_from_keyboard_or_voice
from llm_service import (
    analyze_user_message_with_llm,
    build_multimodal_context,
    generate_assistant_reply,
    trim_history,
)
from fusion import fuse_emotions
from prosody_service import unavailable_prosody
from time_utils import build_system_time_reply, looks_like_time_question, now_ts, print_ts
from transcription_service import load_transcriber


def print_startup_banner() -> None:
    print_ts(f"Python version: {sys.version.split()[0]}")
    print_ts(f"Platform: {sys.platform}")
    print()
    print_ts(f"Local chat with {MODEL_NAME} started.")
    print("Commands:")
    print("  /exit    - quit")
    print("  /quit    - quit")
    print("  /clear   - clear conversation history")
    print("  /speak   - record from microphone and capture camera emotion in parallel")
    print("  /devices - list microphone input devices")
    print("  /mic     - choose microphone input device")
    print("  /cams    - list camera devices")
    print("  /cam     - choose camera device")
    print()
    print("Typing works at all times. The camera only activates during /speak.")
    print()


def _emotion_dict(modality_emotion) -> dict:
    emotion = modality_emotion.emotion
    return {
        "available": modality_emotion.available,
        "source": modality_emotion.source,
        "primary": emotion.primary,
        "secondary": emotion.secondary,
        "blended_emotion": emotion.blended_emotion,
        "intensity_level": emotion.intensity_level,
        "intensity_score": emotion.intensity_score,
        "confidence": emotion.confidence,
        "summary": modality_emotion.summary,
    }


def append_exchange(
    history: list[dict],
    user_text: str,
    assistant_text: str,
    facial_hint: str | None,
    text_emotion=None,
    prosody_emotion=None,
    face_emotion=None,
    source: str | None = None,
) -> list[dict]:
    user_message = {
        "role": "user",
        "content": user_text,
        "timestamp": now_ts(),
        "facial_emotion_hint": facial_hint,
    }
    if text_emotion is not None:
        user_message["text_emotion"] = _emotion_dict(text_emotion)
    if prosody_emotion is not None:
        user_message["prosody_emotion"] = _emotion_dict(prosody_emotion)
    if face_emotion is not None:
        user_message["face_emotion"] = _emotion_dict(face_emotion)

    history.append(user_message)

    assistant_message = {"role": "assistant", "content": assistant_text, "timestamp": now_ts()}
    if source:
        assistant_message["source"] = source
    history.append(assistant_message)
    return trim_history(history, MAX_HISTORY_MESSAGES)


def main() -> None:
    client = Client(host=OLLAMA_HOST)
    print_startup_banner()
    transcriber = load_transcriber()

    current_input_device = INPUT_DEVICE
    current_camera_device = CAMERA_DEVICE
    history: list[dict] = []

    while True:
        (
            user_text,
            current_input_device,
            current_camera_device,
            face_capture,
            prosody_emotion,
            face_emotion,
            input_mode,
        ) = get_user_message_from_keyboard_or_voice(
            transcriber=transcriber,
            current_input_device=current_input_device,
            current_camera_device=current_camera_device,
        )

        if user_text is None:
            break
        if not user_text:
            continue

        normalized = normalize_command(user_text)
        if normalized in {"exit", "quit"}:
            print_ts("Goodbye.")
            break
        if normalized == "clear":
            history = []
            print_ts("Conversation cleared.")
            print()
            continue

        print_ts(f"You: {user_text}")
        facial_hint = face_capture.summary_text if face_capture else None

        try:
            text_analysis = analyze_user_message_with_llm(
                client=client,
                user_text=user_text,
                facial_emotion_hint=None,
            )
            # print(": "f"text_analysis={text_analysis}")
            print("\n")
            text_emotion = build_multimodal_context(
                text_analysis=text_analysis,
                prosody_emotion=unavailable_prosody(),
                face_emotion=unavailable_face(),
            ).text

            # print(":"f"text_emotion={text_emotion}")
            print("\n")

            if input_mode == "typed":
                prosody_emotion = unavailable_prosody()
                face_emotion = classify_face_capture_to_plutchik(face_capture)

            multimodal_context = build_multimodal_context(
                text_analysis=text_analysis,
                prosody_emotion=prosody_emotion,
                face_emotion=face_emotion,
            )

            fused_emotion = fuse_emotions(
                text=text_emotion.emotion.primary if text_emotion else None,
                prosody=prosody_emotion.emotion.primary if prosody_emotion.available else None,
                face=face_emotion.emotion.primary if face_emotion.available else None,
            )

            # print(":"f"multimodal_context={multimodal_context}")
            print("\n")

            print_ts(
                "Emotion context: "
                f"text={multimodal_context.text.emotion.primary}, "
                f"prosody={multimodal_context.prosody.emotion.primary if multimodal_context.prosody.available else 'unavailable'}, "
                f"face={multimodal_context.face.emotion.primary if multimodal_context.face.available else 'unavailable'}"
                f"fused={fused_emotion}"
            )

            if looks_like_time_question(user_text):
                final_reply = build_system_time_reply(user_text)
                print_ts(f"Assistant: {final_reply}")
                print()
                history = append_exchange(
                    history,
                    user_text,
                    final_reply,
                    facial_hint,
                    text_emotion=text_emotion,
                    prosody_emotion=prosody_emotion,
                    face_emotion=face_emotion,
                    source="system_time",
                )
                continue

            final_reply = generate_assistant_reply(
                client=client,
                conversation_history=history,
                user_text=user_text,
                text_analysis=text_analysis,
                multimodal_context=multimodal_context,
            )
            print_ts(f"Assistant: {final_reply}")
            print()
            history = append_exchange(
                history,
                user_text,
                final_reply,
                facial_hint,
                text_emotion=text_emotion,
                prosody_emotion=prosody_emotion,
                face_emotion=face_emotion,
            )
        except Exception as exc:
            print(f"\nError talking to Ollama server: {exc}\n")


if __name__ == "__main__":
    main()
