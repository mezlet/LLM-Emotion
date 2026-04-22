"""
main.py
-------
Entry point for the HRI chat application.

Run with:
    python main.py
"""

from __future__ import annotations

import sys

from ollama import Client

from config import MAX_HISTORY_MESSAGES, MODEL_NAME, OLLAMA_HOST
from audio import load_whisper_model
from input_handler import get_user_input
from llm import analyze_user_message, generate_assistant_reply, trim_history
from utils import (
    build_system_time_reply,
    looks_like_time_question,
    normalize_command,
    now_ts,
    print_ts,
)


# ---------------------------------------------------------------------------
# Application loop
# ---------------------------------------------------------------------------

def main() -> None:
    client = Client(host=OLLAMA_HOST)

    print_ts(f"Python version: {sys.version.split()[0]}")
    print_ts(f"Platform: {sys.platform}")
    print()

    whisper_model = load_whisper_model()

    current_input_device  = None
    current_camera_device = None
    history: list[dict]   = []

    print_ts(f"Local chat with {MODEL_NAME} started.")
    print("Commands:")
    print("  /exit     quit")
    print("  /quit     quit")
    print("  /clear    clear conversation history")
    print("  /speak    record from microphone and capture camera emotion in parallel")
    print("  /devices  list microphone input devices")
    print("  /mic      choose microphone input device")
    print("  /cams     list camera devices")
    print("  /cam      choose camera device")
    print()
    print("Typing works at all times. The camera only activates during /speak.")
    print()

    while True:
        user_text, current_input_device, current_camera_device, face_capture = (
            get_user_input(
                whisper_model=whisper_model,
                current_input_device=current_input_device,
                current_camera_device=current_camera_device,
            )
        )

        # None signals EOF / KeyboardInterrupt.
        if user_text is None:
            break

        # Empty string means a command was handled with no follow-up message.
        if not user_text:
            continue

        command = normalize_command(user_text)

        if command in {"exit", "quit"}:
            print_ts("Goodbye.")
            break

        if command == "clear":
            history = []
            print_ts("Conversation cleared.")
            print()
            continue

        print_ts(f"You: {user_text}")

        facial_hint = face_capture.summary_text if face_capture else None

        try:
            # ---- fast path: time / date questions ---------------------------
            if looks_like_time_question(user_text):
                final_reply = build_system_time_reply(user_text)
                print_ts(f"Assistant: {final_reply}")
                print()

                history.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "facial_emotion_hint": facial_hint,
                })
                history.append({
                    "role": "assistant",
                    "content": final_reply,
                    "timestamp": now_ts(),
                    "source": "system_time",
                })
                history = trim_history(history, MAX_HISTORY_MESSAGES)
                continue

            # ---- normal path: LLM analysis + reply --------------------------
            analysis = analyze_user_message(
                client=client,
                user_text=user_text,
                facial_emotion_hint=facial_hint,
            )

            final_reply = generate_assistant_reply(
                client=client,
                conversation_history=history,
                user_text=user_text,
                analysis=analysis,
            )

            print_ts(f"Assistant: {final_reply}")
            print()

            history.append({
                "role": "user",
                "content": user_text,
                "timestamp": now_ts(),
                "facial_emotion_hint": facial_hint,
            })
            history.append({
                "role": "assistant",
                "content": final_reply,
                "timestamp": now_ts(),
            })
            history = trim_history(history, MAX_HISTORY_MESSAGES)

        except Exception as exc:
            print(f"\nError talking to Ollama server: {exc}\n")


if __name__ == "__main__":
    main()