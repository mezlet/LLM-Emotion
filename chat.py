# Importing the Ollama client, which allows Python to talk to the local Ollama server
from ollama import Client
import re

import tempfile
import os
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisperx


# Address of the local Ollama server (running via `rrlab-ollama serve`)
# OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_HOST="https://thorough-smtp-neutral-marco.trycloudflare.com"

# The ollama model
MODEL_NAME = "llama3.2:1b"

# WhisperX settings
WHISPERX_MODEL = "small"
WHISPERX_DEVICE = "cpu"         # use "cuda" if you have a working NVIDIA setup
WHISPERX_COMPUTE_TYPE = "int8"  # good default for CPU
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10

def remove_emojis_except_faces(text: str) -> str:
    """
    Remove most emojis except common face emojis.
    """

    allowed_faces = {
        "😀","😁","😂","🤣","😃","😄","😅","😆","😉","😊","😋",
        "😎","😍","😘","🥰","😗","😙","😚","🙂","🤗","🤩","🤔",
        "🤨","😐","😑","😶","🙄","😏","😣","😥","😮","🤐","😯",
        "😪","😫","🥱","😴","😌","😛","😜","😝","🤤","😒","😓",
        "😔","😕","🙃","😲","☹","🙁","😖","😞","😟","😤",
        "😢","😭","😦","😧","😨","😩","🤯","😬","😰","😱",
        "🥵","🥶","😳","🤪","😵","🤠","🥳","😇","🤓","🧐",
        "😈","👿","🤡","🤥","🤫","🤭","🥴"
    }

    result = []
    for char in text:
        if char in allowed_faces:
            result.append(char)
            continue

        code = ord(char)

        # Emoji ranges
        if (
            0x1F300 <= code <= 0x1F5FF or
            0x1F680 <= code <= 0x1F6FF or
            0x1F700 <= code <= 0x1F77F or
            0x1F780 <= code <= 0x1F7FF or
            0x1F800 <= code <= 0x1F8FF or
            0x1F900 <= code <= 0x1F9FF or
            0x1FA70 <= code <= 0x1FAFF or
            0x2600 <= code <= 0x26FF or
            0x2700 <= code <= 0x27BF
        ):
            continue

        result.append(char)

    cleaned = "".join(result)

    # Remove invisible emoji joiners
    cleaned = re.sub(r'[\u200d\ufe0f]', '', cleaned)

    return cleaned

def record_audio_to_wav(
    sample_rate: int = SAMPLE_RATE,
    max_seconds: int = MAX_RECORD_SECONDS,
) -> Optional[str]:
    """
    Record audio from the default microphone and save it to a temporary WAV file.

    Returns:
        Path to the WAV file, or None if recording fails.
    """
    try:
        print(f"\nRecording... speak now ({max_seconds} seconds max).")
        print("Recording starts immediately.\n")

        audio = sd.rec(
            int(max_seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        # Flatten to 1D for writing
        audio = np.squeeze(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name

        sf.write(wav_path, audio, sample_rate)
        return wav_path

    except Exception as e:
        print(f"Audio recording error: {e}\n")
        return None


def transcribe_with_whisperx(
    wav_path: str,
    whisper_model,
    batch_size: int = 8,
) -> str:
    """
    Transcribe a WAV file with WhisperX and return plain text.
    """
    audio = whisperx.load_audio(wav_path)
    result = whisper_model.transcribe(audio, batch_size=batch_size)

    text = result.get("text", "").strip()
    if text:
        return text

    segments = result.get("segments", [])
    return " ".join(seg.get("text", "").strip() for seg in segments).strip()


def get_user_message_from_keyboard_or_voice(whisper_model) -> Optional[str]:
    """
    Get a user message either from keyboard input or from microphone speech.
    Commands:
      /speak  - record voice and transcribe
    """
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None

    if not user_text:
        return ""

    if user_text.lower() == "/speak":
        wav_path = record_audio_to_wav()
        if not wav_path:
            return ""

        try:
            transcript = transcribe_with_whisperx(wav_path, whisper_model=whisper_model)
        except Exception as e:
            print(f"WhisperX transcription error: {e}\n")
            transcript = ""
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if transcript:
            print(f"Transcribed: {transcript}\n")
        else:
            print("No speech detected.\n")

        return transcript

    return user_text


def main() -> None:
    client = Client(host=OLLAMA_HOST)

    # Load WhisperX once at startup
    try:
        print("Loading WhisperX model...")
        whisper_model = whisperx.load_model(
            WHISPERX_MODEL,
            WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
        )
        print("WhisperX ready.\n")
    except Exception as e:
        print(f"Failed to load WhisperX: {e}")
        print("Voice input will not work.\n")
        whisper_model = None

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Use only face emojis like 🙂 😂 😢 😡. "
                "Do not use any other emojis."
            ),
        }
    ]

    print(f"Streaming local chat with {MODEL_NAME} started.")
    print("Commands:")
    print("  /exit   - quit")
    print("  /quit   - quit")
    print("  /clear  - clear conversation history")
    print("  /speak  - record from microphone and transcribe with WhisperX\n")

    while True:
        if whisper_model is None:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
        else:
            user_text = get_user_message_from_keyboard_or_voice(whisper_model)
            if user_text is None:
                break

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("Goodbye.")
            break

        if user_text.lower() == "/clear":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Use only face emojis like 🙂 😂 😢 😡. "
                        "Do not use any other emojis."
                    ),
                }
            ]
            print("Conversation cleared.\n")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            stream = client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )

            print("\nAssistant: ", end="", flush=True)
            full_reply = ""

            for chunk in stream:
                piece = chunk["message"]["content"]
                cleaned_piece = remove_emojis_except_faces(piece)
                full_reply += cleaned_piece
                print(cleaned_piece, end="", flush=True)

            print("\n")
            messages.append({"role": "assistant", "content": full_reply})

        except Exception as e:
            print(f"\nError talking to local Ollama server: {e}\n")
            messages.pop()


if __name__ == "__main__":
    main()