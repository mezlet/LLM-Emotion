import sys
from datetime import datetime

from ollama import Client

from audio import get_user_message_from_keyboard_or_voice, load_whisper_model
from config import OLLAMA_HOST, MODEL_NAME, INPUT_DEVICE
from emotion import detect_user_emotion, normalize_assistant_reply, build_system_prompt


def main() -> None:
    client = Client(host=OLLAMA_HOST)

    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}\n")

    whisper_model = load_whisper_model()
    current_input_device = INPUT_DEVICE
    base_messages = []

    print(f"Streaming local chat with {MODEL_NAME} started.")
    print("Commands:")
    print("  /exit    - quit")
    print("  /quit    - quit")
    print("  /clear   - clear conversation history")
    print("  /speak   - record from microphone and transcribe with WhisperX")
    print("  /devices - list microphone input devices")
    print("  /mic     - choose microphone input device\n")

    while True:
        user_text, current_input_device = get_user_message_from_keyboard_or_voice(
            whisper_model,
            current_input_device,
        )

        if user_text is None:
            break

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            break

        if user_text.lower() == "/clear":
            base_messages = []
            print("Conversation cleared.\n")
            continue

        user_emotion = detect_user_emotion(user_text)

        # Rebuild the system prompt each turn so it adapts to the current user tone
        messages = [
            {"role": "system", "content": build_system_prompt(user_emotion)},
            *base_messages,
            {"role": "user", "content": user_text},
        ]

        try:
            stream = client.chat(model=MODEL_NAME, messages=messages, stream=True)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] Assistant: ", end="", flush=True)
            raw_reply = ""
            for chunk in stream:
                raw_reply += chunk["message"]["content"]

            final_reply = normalize_assistant_reply(raw_reply, user_emotion)
            print(final_reply, flush=True)
            print()

            base_messages.append({"role": "user", "content": user_text})
            base_messages.append({"role": "assistant", "content": final_reply})

        except Exception as e:
            print(f"\nError talking to Ollama server: {e}\n")


if __name__ == "__main__":
    main()
