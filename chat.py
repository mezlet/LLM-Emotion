from __future__ import annotations

# ============================================================
# Standard library imports
# ============================================================

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

# ============================================================
# Third-party imports
# ============================================================

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisperx
from ollama import Client


# ============================================================
# Configuration
# ============================================================

# Local Ollama server
# OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_HOST = "https://lucas-physicians-toilet-lenders.trycloudflare.com"

# Choose your Ollama model here
MODEL_NAME = "llama3:8b"

# WhisperX speech-to-text settings
WHISPERX_MODEL = "small"
WHISPERX_DEVICE = "cpu"
WHISPERX_COMPUTE_TYPE = "int8"
WHISPERX_LANGUAGE = "en"  # set to None if you want auto-detect

# Audio settings
TARGET_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10

# Optional microphone device index
INPUT_DEVICE: Optional[int] = None

# Silence / low-volume detection thresholds
MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003

# Limit how much conversation history goes into the prompt
MAX_HISTORY_MESSAGES = 12


# ============================================================
# Timestamp helpers
# ============================================================

def now_ts() -> str:
    """
    Return the current local time in a readable format.
    Example: 2026-04-13 05:46:17
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_ts(message: str) -> None:
    """
    Print a message with a timestamp prefix.
    """
    print(f"[{now_ts()}] {message}")


# ============================================================
# Emoji policy
# ============================================================

# Only facial emojis are allowed in the final assistant output.
# The LLM may choose an emoji, but the code will only allow one
# from this approved set.
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


# ============================================================
# Data model
# ============================================================

@dataclass
class MessageAnalysis:
    """
    This holds the LLM's analysis of the user's message.

    Important:
    - The LLM is the sole determinant of emotion and emoji suitability.
    - The code does not classify the emotion itself.
    """
    emotion_summary: str
    reply_tone: str
    should_use_emoji: bool
    emoji: Optional[str]
    reason: str


# ============================================================
# Command helpers
# ============================================================

def normalize_command(text: str) -> str:
    """
    Normalize command-like user input.

    This lets these forms behave the same:
    - /speak
    - \\speak
    - speak
    """
    normalized = text.strip().lower()
    normalized = re.sub(r"^[\\/]+", "", normalized)
    return normalized


# ============================================================
# Emoji and text cleanup helpers
# ============================================================

def remove_ascii_emoticons(text: str) -> str:
    """
    Remove text-based emoticons such as:
    :) :)) :( ;-) :D

    This matters because the emoji policy only allows real facial emojis,
    not ASCII emotions.
    """
    return re.sub(r"[:;=8][\-^]?[)(DPp/\\|]+", "", text)


def remove_emojis_except_faces(text: str) -> str:
    """
    Remove all emoji-like characters except the allowed facial emojis.
    """
    result = []

    for char in text:
        if char in ALLOWED_FACE_EMOJIS:
            result.append(char)
            continue

        code = ord(char)

        # Remove characters from common emoji/symbol ranges
        if (
            0x1F300 <= code <= 0x1F5FF or
            0x1F600 <= code <= 0x1F64F or
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

    # Remove invisible formatting leftovers used in some emoji forms
    cleaned = re.sub(r"[\u200d\ufe0f]", "", cleaned)
    return cleaned


def extract_face_emojis(text: str) -> list[str]:
    """
    Return all allowed facial emojis found in the text.
    """
    return [ch for ch in text if ch in ALLOWED_FACE_EMOJIS]


def remove_all_face_emojis(text: str) -> str:
    """
    Remove even allowed facial emojis.
    Used before re-adding at most one approved emoji.
    """
    return "".join(ch for ch in text if ch not in ALLOWED_FACE_EMOJIS)


def normalize_assistant_reply(text: str, analysis: MessageAnalysis) -> str:
    """
    Final policy enforcement layer.

    The LLM decides:
    - the emotional interpretation
    - whether emoji is appropriate
    - which emoji to use

    The code only enforces:
    - facial emojis only
    - at most one emoji
    - no ASCII emoticons
    - emoji goes at the end
    """
    # Remove ASCII emoticons first
    text = remove_ascii_emoticons(text)

    # Remove all non-approved emojis
    cleaned = remove_emojis_except_faces(text)

    # Keep only the plain text body
    plain = remove_all_face_emojis(cleaned)

    # Normalize whitespace and punctuation spacing
    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\s+([,.;!?])", r"\1", plain)

    if not plain:
        plain = "I’m here."

    # Enforce emoji usage only if the LLM said yes AND the chosen emoji is allowed
    approved_emoji = analysis.emoji if analysis.emoji in ALLOWED_FACE_EMOJIS else None

    if not analysis.should_use_emoji or not approved_emoji:
        return plain

    return f"{plain} {approved_emoji}"


# ============================================================
# LLM message analysis
# ============================================================

def build_message_analysis_prompt(user_text: str) -> str:
    """
    Ask the LLM to analyze the emotional meaning of the user's message.

    Important:
    - No hard-coded emotional labels are required here.
    - The LLM determines the emotional reading itself.
    """
    return f"""
    You are analyzing the emotional meaning of a user's message.

    Your task:
    - infer the user's emotional state from the full context of the message
    - decide the best reply tone
    - decide whether a facial emoji should be used
    - if an emoji should be used, choose exactly one facial emoji

    Important rules:
    - Base your judgment only on the user's message.
    - Do not assume future outcomes the user did not state.
    - If the message is technical, factual, task-focused, or command-like, usually set should_use_emoji to false.
    - Only choose a facial emoji.
    - If no emoji is appropriate, set emoji to null.
    - Keep emotion_summary natural and concise.
    - Keep reply_tone concise and practical.

    Return JSON only in this exact format:
    {{
    "emotion_summary": "short natural-language summary of the user's emotional state",
    "reply_tone": "short description of how the assistant should sound",
    "should_use_emoji": true,
    "emoji": "😊",
    "reason": "brief explanation"
    }}

    User message:
    {user_text}
    """.strip()


def safe_json_extract(text: str) -> Optional[dict]:
    """
    Try to parse JSON directly.
    If that fails, extract the first {...} block and try again.
    """
    text = text.strip()

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


def analyze_user_message_with_llm(client: Client, user_text: str) -> MessageAnalysis:
    """
    Use the LLM as the sole determinant of:
    - emotional meaning
    - reply tone
    - emoji suitability
    - emoji selection

    The code does not perform any hard-coded emotion classification.
    """
    prompt = build_message_analysis_prompt(user_text)

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze the emotional meaning of user messages "
                        "and return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=False,
        )

        print(response)

        raw = response["message"]["content"]
        data = safe_json_extract(raw)

        if not data:
            return MessageAnalysis(
                emotion_summary="Emotional context is unclear.",
                reply_tone="natural, neutral, helpful",
                should_use_emoji=False,
                emoji=None,
                reason="The model output could not be parsed as valid JSON.",
            )

        emotion_summary = str(
            data.get("emotion_summary", "Emotional context is unclear.")
        ).strip()

        reply_tone = str(
            data.get("reply_tone", "natural, neutral, helpful")
        ).strip()

        should_use_emoji = bool(data.get("should_use_emoji", False))

        raw_emoji = data.get("emoji")
        emoji = raw_emoji.strip() if isinstance(raw_emoji, str) else None
        if emoji == "":
            emoji = None

        reason = str(data.get("reason", "")).strip()

        # Policy enforcement only: reject any non-approved emoji
        if emoji not in ALLOWED_FACE_EMOJIS:
            emoji = None

        if emoji is None:
            should_use_emoji = False

        return MessageAnalysis(
            emotion_summary=emotion_summary or "Emotional context is unclear.",
            reply_tone=reply_tone or "natural, neutral, helpful",
            should_use_emoji=should_use_emoji,
            emoji=emoji,
            reason=reason or "Derived from the user's message.",
        )

    except Exception as e:
        return MessageAnalysis(
            emotion_summary="Emotional context is unclear.",
            reply_tone="natural, neutral, helpful",
            should_use_emoji=False,
            emoji=None,
            reason=f"The message analysis step failed: {e}",
        )


# ============================================================
# Prompt building for final assistant reply
# ============================================================

def build_system_prompt(analysis: MessageAnalysis) -> str:
    """
    Build the system prompt for response generation.

    This uses only the LLM's analysis.
    The code does not insert any hard-coded emotional category.
    """
    emoji_instruction = (
        f"Use exactly this one facial emoji at the very end of the response: {analysis.emoji}"
        if analysis.should_use_emoji and analysis.emoji
        else
        "Do not use any emoji."
    )

    return (
        "You are a warm, emotionally aware assistant. "
        "Respond to the user's message in a natural, socially appropriate way. "
        "Do not sound robotic, stiff, overly formal, or generic. "
        "Do not ask follow-up questions unless they are truly necessary. "
        "Most responses should be complete without ending in a question. "
        "Do not assume outcomes the user has not confirmed. "
        "Base your reply only on what the user actually said. "
        "Use full sentences. "
        "Never use non-face emojis. "
        "Never use more than one emoji. "
        f"Emotional interpretation: {analysis.emotion_summary}. "
        f"Reply tone: {analysis.reply_tone}. "
        f"{emoji_instruction}"
    )


def trim_history(messages: list[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    """
    Keep only the most recent messages to prevent prompt bloat.
    """
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def prompt_ready_history(messages: list[dict]) -> list[dict]:
    """
    Strip non-prompt metadata before sending history to the LLM.
    The model only needs role + content.
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ============================================================
# Audio / microphone helpers
# ============================================================

def list_input_devices() -> None:
    """
    Print available microphone input devices.
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
    """
    Let the user choose a microphone device.
    """
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

        print_ts(f"Using input device [{device_index}] {device_info['name']}")
        return device_index

    except Exception as e:
        print(f"Invalid device selection: {e}\n")
        return current_input_device


def get_effective_input_samplerate(input_device: Optional[int]) -> int:
    """
    Use the selected device's default sample rate.
    This avoids invalid sample-rate errors on some systems.
    """
    if input_device is None:
        device_info = sd.query_devices(kind="input")
    else:
        device_info = sd.query_devices(input_device)

    default_sr = int(round(device_info["default_samplerate"]))
    if default_sr <= 0:
        return TARGET_SAMPLE_RATE

    return default_sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using NumPy interpolation.
    This avoids adding a SciPy dependency.
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
    Record microphone input, normalize it, resample it to 16kHz,
    and save it to a temporary WAV file for WhisperX.
    """
    try:
        effective_sr = get_effective_input_samplerate(input_device)

        sd.check_input_settings(
            device=input_device,
            samplerate=effective_sr,
            channels=1,
            dtype="float32",
        )

        print(f"\n[{now_ts()}] Recording... speak now ({max_seconds} seconds max).")
        print(f"[{now_ts()}] Using sample rate {effective_sr} Hz for input.\n")

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

        print_ts(f"Recorded audio level: peak={peak:.4f}, rms={rms:.4f}")

        if peak < MIN_PEAK_THRESHOLD or rms < MIN_RMS_THRESHOLD:
            print("Recorded audio is too quiet or silent.")
            print("Try a different microphone, check mute/input volume, or use /devices.\n")
            return None

        # Gentle normalization
        target_peak = 0.9
        gain = min(target_peak / max(peak, 1e-6), 10.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

        # WhisperX works well at 16kHz
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
    """
    Transcribe a WAV file using WhisperX.
    """
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
    Read user input from keyboard or process special commands.

    Supported command forms:
    - /speak
    - \\speak
    - speak

    Also supports:
    - /devices
    - /mic
    """
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None, current_input_device

    if not user_text:
        return "", current_input_device

    normalized = normalize_command(user_text)

    if normalized == "devices":
        list_input_devices()
        return "", current_input_device

    if normalized == "mic":
        new_device = choose_input_device(current_input_device)
        return "", new_device

    if normalized == "speak":
        if whisper_model is None:
            print("Voice input is unavailable because WhisperX failed to load.\n")
            return "", current_input_device

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
            print_ts(f"Transcribed: {transcript}")
            print()
        else:
            print("No speech detected.\n")

        return transcript, current_input_device

    return user_text, current_input_device


# ============================================================
# Assistant reply generation
# ============================================================

def generate_assistant_reply(
    client: Client,
    conversation_history: list[dict],
    user_text: str,
    analysis: MessageAnalysis,
) -> str:
    """
    Generate the assistant's reply using:
    - conversation history
    - the current user message
    - the LLM's emotional analysis of that message
    """
    messages = [
        {"role": "system", "content": build_system_prompt(analysis)},
        *prompt_ready_history(trim_history(conversation_history)),
        {"role": "user", "content": user_text},
    ]

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
    )

    raw_reply = response["message"]["content"]
    final_reply = normalize_assistant_reply(raw_reply, analysis)
    return final_reply


# ============================================================
# Main application loop
# ============================================================

def main() -> None:
    """
    Main app entry point.

    Handles:
    - WhisperX loading
    - keyboard / voice input
    - LLM message analysis
    - assistant reply generation
    - timestamped conversation history
    """
    client = Client(host=OLLAMA_HOST)

    print_ts(f"Python version: {sys.version.split()[0]}")
    print_ts(f"Platform: {sys.platform}")
    print()

    # Load WhisperX once at startup so /speak is available
    try:
        print_ts("Loading WhisperX model...")
        whisper_model = whisperx.load_model(
            WHISPERX_MODEL,
            WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
            language=WHISPERX_LANGUAGE,
        )
        print_ts("WhisperX ready.")
        print()
    except Exception as e:
        print_ts(f"Failed to load WhisperX: {e}")
        print("Voice input will not work.\n")
        whisper_model = None

    current_input_device = INPUT_DEVICE
    history: list[dict] = []

    print_ts(f"Streaming local chat with {MODEL_NAME} started.")
    print("Commands:")
    print("  /exit    - quit")
    print("  /quit    - quit")
    print("  /clear   - clear conversation history")
    print("  /speak   - record from microphone and transcribe with WhisperX")
    print("  /devices - list microphone input devices")
    print("  /mic     - choose microphone input device")
    print()

    while True:
        # If WhisperX is loaded, allow typed input and voice commands.
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

        # Normalize command-like text for quit/clear handling
        normalized = normalize_command(user_text)

        if normalized in {"exit", "quit"}:
            print_ts("Goodbye.")
            break

        if normalized == "clear":
            history = []
            print_ts("Conversation cleared.")
            print()
            continue

        # Prevent extremely short meaningless input from being sent
        if len(user_text.strip()) < 1:
            continue

        print_ts(f"You: {user_text}")

        try:
            # Step 1: analyze the message with the LLM
            analysis = analyze_user_message_with_llm(client, user_text)

            # Step 2: generate the final response based on that analysis
            final_reply = generate_assistant_reply(
                client=client,
                conversation_history=history,
                user_text=user_text,
                analysis=analysis,
            )

            print_ts(f"Assistant: {final_reply}")
            print()

            # Store conversation with timestamps for logs/debugging
            history.append({
                "role": "user",
                "content": user_text,
                "timestamp": now_ts(),
            })
            history.append({
                "role": "assistant",
                "content": final_reply,
                "timestamp": now_ts(),
            })

            history = trim_history(history, MAX_HISTORY_MESSAGES)

        except Exception as e:
            print(f"\nError talking to Ollama server: {e}\n")


if __name__ == "__main__":
    main()