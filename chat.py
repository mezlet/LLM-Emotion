from ollama import Client
import os
import re
import sys
import tempfile
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisperx


# Address of the local Ollama server
OLLAMA_HOST = "http://127.0.0.1:11434"

# The ollama model
MODEL_NAME = "llama3.2:1b"

# WhisperX settings
WHISPERX_MODEL = "small"
WHISPERX_DEVICE = "cpu"
WHISPERX_COMPUTE_TYPE = "int8"
WHISPERX_LANGUAGE = "en"   # Set None for auto-detect

# Target ASR sample rate
TARGET_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10

# Optional input device index, can be changed with /mic
INPUT_DEVICE: Optional[int] = None

MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003


ALLOWED_FACE_EMOJIS = {
    "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊", "😋",
    "😎", "😍", "😘", "🥰", "😗", "😙", "😚", "🙂", "🤗", "🤩", "🤔",
    "🤨", "😐", "😑", "😶", "🙄", "😏", "😣", "😥", "😮", "🤐", "😯",
    "😪", "😫", "🥱", "😴", "😌", "😛", "😜", "😝", "🤤", "😒", "😓",
    "😔", "😕", "🙃", "😲", "☹", "🙁", "😖", "😞", "😟", "😤",
    "😢", "😭", "😦", "😧", "😨", "😩", "🤯", "😬", "😰", "😱",
    "🥵", "🥶", "😳", "🤪", "😵", "🤠", "🥳", "😇", "🤓", "🧐",
    "😈", "👿", "🤡", "🤥", "🤫", "🤭", "🥴"
}

# A smaller curated set for controlled human-like use
POSITIVE_EMOJIS = {"🙂", "😊", "😄", "😌", "🥳"}
SUPPORTIVE_EMOJIS = {"😔", "😢", "😟", "😕"}
PLAYFUL_EMOJIS = {"😄", "😅", "😉"}
THOUGHTFUL_EMOJIS = {"🤔", "🧐"}
NEUTRAL_EMOJIS = {"🙂"}


def remove_emojis_except_faces(text: str) -> str:
    result = []
    for char in text:
        if char in ALLOWED_FACE_EMOJIS:
            result.append(char)
            continue

        code = ord(char)
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
    cleaned = re.sub(r"[\u200d\ufe0f]", "", cleaned)
    return cleaned


def extract_face_emojis(text: str) -> list[str]:
    return [ch for ch in text if ch in ALLOWED_FACE_EMOJIS]


def remove_all_face_emojis(text: str) -> str:
    return "".join(ch for ch in text if ch not in ALLOWED_FACE_EMOJIS)


def detect_user_emotion(user_text: str) -> str:
    """
    Lightweight rule-based emotion detection.
    Returns one of:
      happy, sad, playful, thoughtful, angry, anxious, neutral, factual
    """
    text = user_text.strip().lower()

    if not text:
        return "neutral"

    # Strong signal from emoji
    if any(e in user_text for e in ["😢", "😭", "😞", "😔", "☹", "🙁"]):
        return "sad"
    if any(e in user_text for e in ["😄", "😁", "😀", "😊", "🥳"]):
        return "happy"
    if any(e in user_text for e in ["😂", "🤣", "😉", "😜"]):
        return "playful"
    if any(e in user_text for e in ["🤔", "🧐"]):
        return "thoughtful"
    if any(e in user_text for e in ["😠", "😡", "😤"]):
        return "angry"
    if any(e in user_text for e in ["😟", "😰", "😨"]):
        return "anxious"

    happy_words = {
        "happy", "glad", "excited", "great", "awesome", "amazing",
        "wonderful", "fantastic", "love", "yay", "proud", "relieved"
    }
    sad_words = {
        "sad", "upset", "hurt", "cry", "crying", "lonely", "depressed",
        "down", "unhappy", "miss", "heartbroken", "tired"
    }
    playful_words = {
        "haha", "lol", "lmao", "funny", "joke", "kidding"
    }
    thoughtful_words = {
        "think", "wonder", "curious", "why", "how", "what do you think"
    }
    angry_words = {
        "angry", "mad", "annoyed", "frustrated", "furious", "irritated"
    }
    anxious_words = {
        "nervous", "anxious", "worried", "scared", "afraid", "stress", "stressed"
    }

    text_words = set(re.findall(r"\b\w+\b", text))

    if text_words & happy_words:
        return "happy"
    if text_words & sad_words:
        return "sad"
    if text_words & playful_words:
        return "playful"
    if text_words & angry_words:
        return "angry"
    if text_words & anxious_words:
        return "anxious"

    # Technical or factual queries should usually not get emojis
    factual_patterns = [
        r"\bwhat is\b",
        r"\bwhere is\b",
        r"\bwho is\b",
        r"\bwhen is\b",
        r"\bhow do i\b",
        r"\bexplain\b",
        r"\bdefine\b",
        r"\binstall\b",
        r"\berror\b",
        r"\bcode\b",
        r"\bpython\b",
        r"\bdebug\b",
    ]
    if any(re.search(pattern, text) for pattern in factual_patterns):
        return "factual"

    if text_words & thoughtful_words:
        return "thoughtful"

    return "neutral"


def choose_expected_emoji(emotion: str) -> Optional[str]:
    """
    Pick one emoji that matches the user's emotional context.
    None means no emoji is preferred.
    """
    if emotion == "happy":
        return "😊"
    if emotion == "sad":
        return "😔"
    if emotion == "playful":
        return "😄"
    if emotion == "thoughtful":
        return "🤔"
    if emotion == "angry":
        return "😕"
    if emotion == "anxious":
        return "😟"
    if emotion == "neutral":
        return None
    if emotion == "factual":
        return None
    return None


def normalize_assistant_reply(text: str, emotion: str) -> str:
    """
    Enforce emoji policy after generation:
    - facial emojis only
    - at most one facial emoji
    - remove emoji if it conflicts with the detected context
    - place emoji naturally at the end
    """
    text = remove_emojis_except_faces(text)
    found = extract_face_emojis(text)
    plain = remove_all_face_emojis(text)

    # Normalize spacing
    plain = re.sub(r"\s+", " ", plain).strip()

    if not plain:
        return "I’m here."

    expected = choose_expected_emoji(emotion)

    # If no emoji is wanted for this kind of message, return plain text
    if expected is None:
        return plain

    # Keep a matching emoji if the model already produced one that fits the category
    kept_emoji = None
    if emotion == "happy":
        for e in found:
            if e in POSITIVE_EMOJIS:
                kept_emoji = e
                break
    elif emotion == "sad":
        for e in found:
            if e in SUPPORTIVE_EMOJIS:
                kept_emoji = e
                break
    elif emotion == "playful":
        for e in found:
            if e in PLAYFUL_EMOJIS:
                kept_emoji = e
                break
    elif emotion == "thoughtful":
        for e in found:
            if e in THOUGHTFUL_EMOJIS:
                kept_emoji = e
                break
    elif emotion in {"angry", "anxious"}:
        for e in found:
            if e in SUPPORTIVE_EMOJIS or e in NEUTRAL_EMOJIS:
                kept_emoji = e
                break

    if kept_emoji is None:
        kept_emoji = expected

    # Avoid double punctuation before emoji
    plain = re.sub(r"\s+([,.;!?])", r"\1", plain)
    plain = plain.rstrip()

    return f"{plain} {kept_emoji}"


def build_system_prompt(user_emotion: str) -> str:
    emoji_guidance = {
        "happy": (
            "The user sounds happy, proud, excited, or relieved. "
            "You may use at most one warm positive facial emoji if it fits naturally, such as 😊 or 😄."
        ),
        "sad": (
            "The user sounds sad, hurt, lonely, or disappointed. "
            "Respond gently and supportively. "
            "You may use at most one soft supportive facial emoji if it fits naturally, such as 😔 or 😢. "
            "Never use laughing or playful emojis."
        ),
        "playful": (
            "The user sounds playful or lighthearted. "
            "You may use at most one light playful facial emoji if it fits naturally, such as 😄 or 😉."
        ),
        "thoughtful": (
            "The user sounds reflective or curious. "
            "You may use at most one thoughtful facial emoji if it fits naturally, such as 🤔."
        ),
        "angry": (
            "The user sounds frustrated or upset. "
            "Respond calmly and helpfully. "
            "Do not mirror anger with aggressive emojis. "
            "If you use an emoji, use at most one gentle face emoji like 😕."
        ),
        "anxious": (
            "The user sounds nervous or worried. "
            "Respond calmly and reassuringly. "
            "You may use at most one gentle supportive facial emoji if it fits naturally, such as 😟."
        ),
        "neutral": (
            "The user's tone is neutral. "
            "Do not force emojis. Use no emoji unless a small amount of warmth genuinely improves the reply."
        ),
        "factual": (
            "The user's message is mainly factual, technical, or informational. "
            "Do not use emojis unless clearly helpful. Usually use no emoji."
        ),
    }

    contextual_rule = emoji_guidance.get(user_emotion, emoji_guidance["neutral"])

    return (
        "You are a helpful, emotionally intelligent assistant. "
        "Always respond with full sentences. "
        "Use facial emojis naturally and sparingly, like a thoughtful human texter. "
        "Use at most ONE facial emoji in the whole response. "
        "Only use an emoji when it improves emotional clarity, warmth, or tone. "
        "Never reply with only an emoji. "
        "Never use non-face emojis. "
        "Never use laughing emojis for sadness, vulnerability, confusion, or serious topics. "
        "Do not ask redundant questions when the user's emotional state is already obvious. "
        f"{contextual_rule}"
    )


def list_input_devices() -> None:
    """
    Print available audio input devices with host API names.
    """
    print("\nAvailable input devices:")
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        print(f"Could not query audio devices: {e}\n")
        return

    found = False
    for idx, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            found = True
            hostapi_name = hostapis[device["hostapi"]]["name"]
            print(
                f"  [{idx}] {device['name']} | "
                f"hostapi={hostapi_name} | "
                f"inputs={device['max_input_channels']} | "
                f"default_sr={device['default_samplerate']}"
            )

    if not found:
        print("  No input devices found.")

    print()
    print(f"Current default device: {sd.default.device}\n")


def choose_input_device(current_input_device: Optional[int]) -> Optional[int]:
    list_input_devices()
    selection = input("Enter input device index (blank to keep current/default): ").strip()

    if not selection:
        return current_input_device

    try:
        device_index = int(selection)
        device_info = sd.query_devices(device_index)
        if device_info["max_input_channels"] <= 0:
            print("That device does not support input.\n")
            return current_input_device

        print(f"Using input device [{device_index}] {device_info['name']}\n")
        return device_index
    except Exception as e:
        print(f"Invalid device selection: {e}\n")
        return current_input_device


def get_effective_input_samplerate(input_device: Optional[int]) -> int:
    """
    Use the selected device's default sample rate.
    This avoids invalid sample rate errors on Linux hardware inputs.
    """
    if input_device is None:
        device_info = sd.query_devices(kind="input")
    else:
        device_info = sd.query_devices(input_device)

    default_sr = int(round(device_info["default_samplerate"]))

    # Guard against bogus/zero values
    if default_sr <= 0:
        return TARGET_SAMPLE_RATE

    return default_sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Simple linear resampling using NumPy only.
    Avoids extra SciPy dependency.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / orig_sr
    target_length = max(1, int(round(duration * target_sr)))

    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, num=target_length, endpoint=False)

    resampled = np.interp(new_times, old_times, audio)
    return resampled.astype(np.float32)


def record_audio_to_wav(
    max_seconds: int = MAX_RECORD_SECONDS,
    input_device: Optional[int] = INPUT_DEVICE,
) -> Optional[str]:
    """
    Record audio from the microphone using the device's native/default sample rate,
    then resample to TARGET_SAMPLE_RATE for WhisperX.
    """
    try:
        effective_sr = get_effective_input_samplerate(input_device)

        sd.check_input_settings(
            device=input_device,
            samplerate=effective_sr,
            channels=1,
            dtype="float32",
        )

        print(f"\nRecording... speak now ({max_seconds} seconds max).")
        print(f"Using sample rate {effective_sr} Hz for input.\n")

        audio = sd.rec(
            frames=int(max_seconds * effective_sr),
            samplerate=effective_sr,
            channels=1,
            dtype="float32",
            device=input_device,
        )
        sd.wait()

        audio = np.squeeze(audio)

        if audio.size == 0:
            print("Audio recording failed: empty audio buffer.\n")
            return None

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))

        print(f"Recorded audio level: peak={peak:.4f}, rms={rms:.4f}")

        if peak < MIN_PEAK_THRESHOLD or rms < MIN_RMS_THRESHOLD:
            print("Recorded audio is too quiet or silent.")
            print("Try a different microphone, check mute/input volume, or use /devices.\n")
            return None

        # Normalize gently
        target_peak = 0.9
        gain = min(target_peak / peak, 10.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

        # Resample to WhisperX target rate
        audio_16k = resample_audio(audio, effective_sr, TARGET_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name

        sf.write(wav_path, audio_16k, TARGET_SAMPLE_RATE)
        return wav_path

    except Exception as e:
        print(f"Audio recording error: {e}\n")
        return None


def transcribe_with_whisperx(
    wav_path: str,
    whisper_model,
    batch_size: int = 8,
) -> str:
    audio = whisperx.load_audio(wav_path)
    result = whisper_model.transcribe(audio, batch_size=batch_size)

    text = result.get("text", "").strip()
    if text:
        return text

    segments = result.get("segments", [])
    return " ".join(seg.get("text", "").strip() for seg in segments).strip()


def get_user_message_from_keyboard_or_voice(
    whisper_model,
    current_input_device: Optional[int],
) -> Tuple[Optional[str], Optional[int]]:
    """
    Commands:
      /speak    - record voice and transcribe
      /devices  - list input devices
      /mic      - choose input device
    """
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None, current_input_device

    if not user_text:
        return "", current_input_device

    lowered = user_text.lower()

    if lowered == "/devices":
        list_input_devices()
        return "", current_input_device

    if lowered == "/mic":
        new_device = choose_input_device(current_input_device)
        return "", new_device

    if lowered == "/speak":
        wav_path = record_audio_to_wav(input_device=current_input_device)
        if not wav_path:
            return "", current_input_device

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

        return transcript, current_input_device

    return user_text, current_input_device


def main() -> None:
    client = Client(host=OLLAMA_HOST)

    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}\n")

    try:
        print("Loading WhisperX model...")
        whisper_model = whisperx.load_model(
            WHISPERX_MODEL,
            WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
            language=WHISPERX_LANGUAGE,
        )
        print("WhisperX ready.\n")
    except Exception as e:
        print(f"Failed to load WhisperX: {e}")
        print("Voice input will not work.\n")
        whisper_model = None

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
        if whisper_model is None:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
        else:
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
            {
                "role": "system",
                "content": build_system_prompt(user_emotion),
            },
            *base_messages,
            {"role": "user", "content": user_text},
        ]

        try:
            stream = client.chat(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )

            print("\nAssistant: ", end="", flush=True)
            raw_reply = ""

            for chunk in stream:
                piece = chunk["message"]["content"]
                raw_reply += piece

            final_reply = normalize_assistant_reply(raw_reply, user_emotion)

            print(final_reply, flush=True)
            print()

            base_messages.append({"role": "user", "content": user_text})
            base_messages.append({"role": "assistant", "content": final_reply})

        except Exception as e:
            print(f"\nError talking to Ollama server: {e}\n")


if __name__ == "__main__":
    main()