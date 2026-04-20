from __future__ import annotations

# Standard library imports
import json
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisperx
from deepface import DeepFace
from ollama import Client


# -----------------------------
# Configuration
# -----------------------------

# Local Ollama server
# OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_HOST = "https://streams-regarding-better-recorders.trycloudflare.com"

# Choose your Ollama model here
MODEL_NAME = "llama3.2:3b"
# MODEL_NAME = "llama3:8b"

# WhisperX speech-to-text settings
WHISPERX_MODEL = "small"
WHISPERX_DEVICE = "cpu"
WHISPERX_COMPUTE_TYPE = "int8"
WHISPERX_LANGUAGE = "en"

# Audio settings
TARGET_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10

# Optional default microphone device index
INPUT_DEVICE: Optional[int] = None

# Silence / low-volume detection thresholds
MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003

# Camera settings
CAMERA_DEVICE: Optional[int] = None
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480

# Sample fewer, cleaner frames
CAMERA_SAMPLE_EVERY_SECONDS = 1.0

# Ignore the first short period after camera starts
CAMERA_WARMUP_SECONDS = 1.5

# DeepFace settings
# Better than "opencv" in many practical webcam cases, though heavier.
# Try "mediapipe" if retinaface is too slow on your machine.
DEEPFACE_DETECTOR_BACKEND = "retinaface"
DEEPFACE_ACTIONS = ["emotion"]
DEEPFACE_ALIGN = True

# Facial hint reliability gates
FACE_MIN_TOP_SCORE = 45.0
FACE_MIN_MARGIN = 15.0

# Limit how much conversation history goes into the prompt
MAX_HISTORY_MESSAGES = 12

# Debug logging
DEBUG = False


# -----------------------------
# Timestamp helpers
# -----------------------------

def now_ts() -> str:
    """
    Return the current local time in a readable format.
    Example: 2026-04-16 15:04:12
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_ts(message: str) -> None:
    """
    Print a message with a timestamp prefix.
    """
    print(f"[{now_ts()}] {message}")


def get_system_datetime() -> datetime:
    """
    Get the current local date/time from the operating system.
    """
    return datetime.now()


# -----------------------------
# Time/date question helpers
# -----------------------------

def looks_like_time_question(user_text: str) -> bool:
    """
    Detect whether the user is asking for the current time/date/day.
    This is system-intent detection, not emotion detection.
    """
    text = user_text.strip().lower()

    patterns = [
        r"\bwhat(?:'s| is)? the time\b",
        r"\bcurrent time\b",
        r"\btime now\b",
        r"\bwhat time is it\b",
        r"\bcan you tell me the time\b",
        r"\btell me the time\b",
        r"\bwhat(?:'s| is)? the date\b",
        r"\bcurrent date\b",
        r"\bdate today\b",
        r"\bwhat(?:'s| is)? today'?s date\b",
        r"\btell me the date\b",
        r"\bwhat day is it\b",
        r"\bwhat day is today\b",
        r"\bcurrent day\b",
        r"\btoday is what day\b",
        r"\bwhat(?:'s| is)? the current time and date\b",
        r"\btoday'?s date and time\b",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def build_system_time_reply(user_text: str) -> str:
    """
    Build a direct answer using the current system time/date.
    """
    now = get_system_datetime()
    text = user_text.strip().lower()

    current_time = now.strftime("%I:%M %p").lstrip("0")
    current_date = now.strftime("%A, %B %d, %Y")
    current_day = now.strftime("%A")

    asks_time = "time" in text
    asks_date = "date" in text
    asks_day = "what day" in text or "day is it" in text or "day is today" in text

    if (asks_time and asks_date) or "time and date" in text:
        return f"The current date and time is {current_date} at {current_time}."

    if asks_time:
        return f"The current time is {current_time}."

    if asks_date:
        return f"Today's date is {current_date}."

    if asks_day:
        return f"Today is {current_day}."

    return f"The current date and time is {current_date} at {current_time}."


# -----------------------------
# Emoji policy
# -----------------------------

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


# -----------------------------
# Data models
# -----------------------------

@dataclass
class MessageAnalysis:
    """
    LLM-generated message interpretation.
    The LLM is the sole determinant of:
    - emotional meaning
    - reply tone
    - whether an emoji is appropriate
    - which emoji fits
    """
    emotion_summary: str
    reply_tone: str
    should_use_emoji: bool
    emoji: Optional[str]
    reason: str


@dataclass
class FaceEmotionCapture:
    """
    Aggregated facial-expression output collected during voice recording.

    This stores full per-frame score dictionaries rather than only one
    label per frame. That makes the signal much more stable.
    """
    emotion_score_samples: list[dict[str, float]]
    frame_count: int
    sampled_frame_count: int
    started_at: str
    ended_at: str
    error: Optional[str] = None

    @property
    def averaged_scores(self) -> dict[str, float]:
        """
        Average emotion scores across all sampled frames.
        DeepFace typically returns percentages per frame.
        """
        if not self.emotion_score_samples:
            return {}

        totals: dict[str, float] = {}
        for sample in self.emotion_score_samples:
            for emotion, score in sample.items():
                totals[emotion] = totals.get(emotion, 0.0) + score

        count = len(self.emotion_score_samples)
        return {emotion: total / count for emotion, total in totals.items()}

    @property
    def dominant_emotion(self) -> Optional[str]:
        """
        Return the top averaged emotion.
        """
        scores = self.averaged_scores
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]

    @property
    def is_reliable(self) -> bool:
        """
        Decide whether the visual signal is strong enough to be useful.

        We reject:
        - weak top scores
        - mixed top-two scores with a very small gap
        """
        scores = self.averaged_scores
        if not scores:
            return False

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ordered[0][1]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if top_score < FACE_MIN_TOP_SCORE:
            return False

        if (top_score - second_score) < FACE_MIN_MARGIN:
            return False

        return True

    @property
    def summary_text(self) -> str:
        """
        Human-readable summary safe to pass to the LLM as a weak hint.
        """
        if self.error:
            return f"No facial-expression hint available because: {self.error}"

        scores = self.averaged_scores
        if not scores:
            return "No reliable facial-expression hint was captured during speech."

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        if not self.is_reliable:
            top_parts = [f"{emo}={score:.0f}%" for emo, score in ordered[:3]]
            return (
                "Facial-expression hint was weak or mixed; "
                f"top_signals=({', '.join(top_parts)}); "
                f"samples={self.sampled_frame_count}"
            )

        parts = [f"{emo}={score:.0f}%" for emo, score in ordered if score >= 5.0]
        return (
            f"dominant={self.dominant_emotion}; "
            f"averaged_scores=({', '.join(parts)}); "
            f"samples={self.sampled_frame_count}"
        )


# -----------------------------
# Command helpers
# -----------------------------

def normalize_command(text: str) -> Optional[str]:
    """
    Normalize slash-style commands only.

    Supported command forms:
    - /speak
    - \\speak

    Normal plain words like 'speak' are NOT treated as commands.
    """
    stripped = text.strip().lower()
    if not stripped.startswith(("/", "\\")):
        return None
    return re.sub(r"^[\\/]+", "", stripped)


# -----------------------------
# Emoji and text cleanup helpers
# -----------------------------

def remove_ascii_emoticons(text: str) -> str:
    """
    Remove text-based emoticons such as:
    :) :)) :( ;-) :D
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
    cleaned = re.sub(r"[\u200d\ufe0f]", "", cleaned)
    return cleaned


def remove_all_face_emojis(text: str) -> str:
    """
    Remove even allowed facial emojis.
    """
    return "".join(ch for ch in text if ch not in ALLOWED_FACE_EMOJIS)


def normalize_assistant_reply(text: str, analysis: MessageAnalysis) -> str:
    """
    Final policy enforcement layer.
    """
    text = remove_ascii_emoticons(text)
    cleaned = remove_emojis_except_faces(text)
    plain = remove_all_face_emojis(cleaned)

    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\s+([,.;!?])", r"\1", plain)

    if not plain:
        plain = "I’m here."

    approved_emoji = analysis.emoji if analysis.emoji in ALLOWED_FACE_EMOJIS else None

    if not analysis.should_use_emoji or not approved_emoji:
        return plain

    return f"{plain} {approved_emoji}"


# -----------------------------
# Safe parsing helpers
# -----------------------------

def parse_bool(value) -> bool:
    """
    More robust bool parsing for LLM JSON output.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def safe_json_extract(text: str) -> Optional[dict]:
    """
    Try to parse JSON directly.
    If that fails, try again after stripping common markdown fences.
    Then extract the first non-greedy {...} block.
    """
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# -----------------------------
# Prompt builders
# -----------------------------

def build_message_analysis_prompt(
    user_text: str,
    facial_emotion_hint: Optional[str] = None,
) -> str:
    """
    Ask the LLM to analyze the emotional meaning of the message.

    Important:
    - Text is the primary signal.
    - Facial-expression analysis is a weak supporting hint only.
    """
    facial_section = (
        f"Facial-expression hint captured during speech: {facial_emotion_hint}\n"
        if facial_emotion_hint
        else "No facial-expression hint was available.\n"
    )

    return f"""
You are analyzing the emotional meaning of a user's message.

Your task:
- infer the user's emotional state primarily from the user's words
- use the facial-expression hint only as a weak supporting signal
- if the user's words clearly express happiness, excitement, pride, relief, or positive news, do not let a conflicting facial-expression hint override that
- if the words and facial-expression hint conflict, trust the words more
- decide the best reply tone
- decide whether a facial emoji should be used
- if an emoji should be used, choose exactly one facial emoji

Important rules:
- Base your judgment mainly on the user's message.
- Treat facial-expression analysis as weak, noisy supporting evidence only.
- Do not assume future outcomes the user did not state.
- Do not make strong claims based on facial-expression analysis.
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

{facial_section}
""".strip()


def build_system_prompt(analysis: MessageAnalysis) -> str:
    """
    Build the system prompt for response generation.
    """
    emoji_instruction = (
        f"Use exactly this one facial emoji at the very end of the response: {analysis.emoji}"
        if analysis.should_use_emoji and analysis.emoji
        else
        "Do not use any emoji."
    )

    return (
        "You are a warm, helpful and emotionally aware assistant. "
        "Respond to the user's message in a natural, socially appropriate way. "
        "Do not sound robotic, stiff, overly formal, or generic. "
        "Do not ask follow-up questions unless absolutely necessary. "
        "Most responses should not end with a question. "
        "Do not assume outcomes the user has not confirmed. "
        "Base your reply only on what the user actually said. "
        "Use full sentences. "
        "Never use non-face emojis. "
        "Never use more than one emoji. "
        f"Emotional interpretation: {analysis.emotion_summary}. "
        f"Reply tone: {analysis.reply_tone}. "
        f"{emoji_instruction}"
    )


# -----------------------------
# LLM helpers
# -----------------------------

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
    """
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def analyze_user_message_with_llm(
    client: Client,
    user_text: str,
    facial_emotion_hint: Optional[str] = None,
) -> MessageAnalysis:
    """
    Use the LLM as the sole determinant of:
    - emotional meaning
    - reply tone
    - emoji suitability
    - emoji selection
    """
    prompt = build_message_analysis_prompt(
        user_text=user_text,
        facial_emotion_hint=facial_emotion_hint,
    )

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

        if DEBUG:
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

        should_use_emoji = parse_bool(data.get("should_use_emoji", False))

        raw_emoji = data.get("emoji")
        emoji = raw_emoji.strip() if isinstance(raw_emoji, str) else None
        if emoji == "":
            emoji = None

        reason = str(data.get("reason", "")).strip()

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


def generate_assistant_reply(
    client: Client,
    conversation_history: list[dict],
    user_text: str,
    analysis: MessageAnalysis,
) -> str:
    """
    Generate the assistant's reply using:
    - conversation history
    - current user message
    - the LLM's emotional analysis
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
    return normalize_assistant_reply(raw_reply, analysis)


# -----------------------------
# Audio / microphone helpers
# -----------------------------

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
                f"  [mic {idx}] {device['name']} | "
                f"hostapi={hostapi_name} | "
                f"inputs={device['max_input_channels']} | "
                f"default_sr={device['default_samplerate']}"
            )

    if not found:
        print("  No input devices found.")

    print()
    print(f"Current default audio device: {sd.default.device}\n")


def choose_input_device(current_input_device: Optional[int]) -> Optional[int]:
    """
    Let the user choose a microphone device.
    """
    list_input_devices()
    selection = input("Enter microphone input device index (blank to keep current/default): ").strip()

    if not selection:
        return current_input_device

    try:
        device_index = int(selection)
        device_info = sd.query_devices(device_index)

        if device_info["max_input_channels"] <= 0:
            print("That device does not support input.\n")
            return current_input_device

        print_ts(f"Using microphone input device [{device_index}] {device_info['name']}")
        return device_index

    except Exception as e:
        print(f"Invalid microphone device selection: {e}\n")
        return current_input_device


def get_effective_input_samplerate(input_device: Optional[int]) -> int:
    """
    Use the selected device's default sample rate.
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

        print(f"\n[{now_ts()}] Recording microphone... speak now ({max_seconds} seconds max).")
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

        target_peak = 0.9
        gain = min(target_peak / max(peak, 1e-6), 10.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

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
    batch_size: int = 4,
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


# -----------------------------
# Camera / DeepFace helpers
# -----------------------------

def list_camera_devices(max_indices: int = 10) -> None:
    """
    Probe a range of camera indices and print available cameras.
    """
    print("\nAvailable camera devices (best-effort probe):")
    found = False

    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            found = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [cam {idx}] available | resolution={width}x{height}")
        finally:
            cap.release()

    if not found:
        print("  No camera devices found.")

    print()


def choose_camera_device(current_camera_device: Optional[int]) -> Optional[int]:
    """
    Let the user choose a camera device.
    """
    list_camera_devices()
    selection = input("Enter camera device index (blank to keep current/default): ").strip()

    if not selection:
        return current_camera_device

    try:
        camera_index = int(selection)
        cap = cv2.VideoCapture(camera_index)
        try:
            if not cap.isOpened():
                print("That camera could not be opened.\n")
                return current_camera_device

            ok, frame = cap.read()
            if not ok or frame is None:
                print("That camera opened but did not return a frame.\n")
                return current_camera_device

            print_ts(f"Using camera device [{camera_index}]")
            return camera_index
        finally:
            cap.release()

    except Exception as e:
        print(f"Invalid camera device selection: {e}\n")
        return current_camera_device


def open_camera(camera_device: Optional[int]) -> cv2.VideoCapture:
    """
    Open the chosen camera.
    """
    camera_index = 0 if camera_device is None else camera_device
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    return cap


def analyze_frame_emotion_scores(frame_bgr: np.ndarray) -> Optional[dict[str, float]]:
    """
    Analyze a single BGR frame with DeepFace and return the full emotion scores.

    This is more stable than returning only one dominant label per frame.
    """
    try:
        result = DeepFace.analyze(
            img_path=frame_bgr,
            actions=DEEPFACE_ACTIONS,
            enforce_detection=False,
            detector_backend=DEEPFACE_DETECTOR_BACKEND,
            align=DEEPFACE_ALIGN,
            silent=True,
        )

        if isinstance(result, list):
            if not result:
                return None
            face_result = result[0]
        else:
            face_result = result

        scores = face_result.get("emotion")
        if not isinstance(scores, dict):
            return None

        return {emotion: float(score) for emotion, score in scores.items()}

    except Exception:
        return None


def capture_face_emotion_during_recording(
    duration_seconds: int,
    camera_device: Optional[int],
    sample_every_seconds: float = CAMERA_SAMPLE_EVERY_SECONDS,
) -> FaceEmotionCapture:
    """
    Capture webcam frames while the user is speaking and estimate facial emotion
    using averaged DeepFace emotion scores across frames.

    This is intentionally treated as a weak hint only.
    """
    started_at = now_ts()
    emotion_score_samples: list[dict[str, float]] = []
    frame_count = 0
    sampled_frame_count = 0

    cap = open_camera(camera_device)
    if not cap.isOpened():
        return FaceEmotionCapture(
            emotion_score_samples=[],
            frame_count=0,
            sampled_frame_count=0,
            started_at=started_at,
            ended_at=now_ts(),
            error="Could not open camera.",
        )

    try:
        end_time = time.time() + duration_seconds
        capture_start_time = time.time()
        last_sample_time = 0.0

        print_ts("Camera emotion capture is active during voice recording.")

        while time.time() < end_time:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()

            # Give the camera/user a short warmup period
            if now - capture_start_time < CAMERA_WARMUP_SECONDS:
                continue

            if now - last_sample_time < sample_every_seconds:
                continue

            last_sample_time = now
            sampled_frame_count += 1

            scores = analyze_frame_emotion_scores(frame)
            if scores:
                emotion_score_samples.append(scores)

            time.sleep(0.02)

    finally:
        cap.release()

    return FaceEmotionCapture(
        emotion_score_samples=emotion_score_samples,
        frame_count=frame_count,
        sampled_frame_count=sampled_frame_count,
        started_at=started_at,
        ended_at=now_ts(),
        error=None,
    )


def record_audio_and_capture_face_emotion(
    max_seconds: int = MAX_RECORD_SECONDS,
    input_device: Optional[int] = INPUT_DEVICE,
    camera_device: Optional[int] = CAMERA_DEVICE,
) -> Tuple[Optional[str], Optional[FaceEmotionCapture]]:
    """
    Run microphone recording and camera emotion capture in parallel.
    Returns:
      - wav_path
      - face_capture summary
    """
    face_result: dict[str, Optional[FaceEmotionCapture]] = {"data": None}

    def face_worker() -> None:
        face_result["data"] = capture_face_emotion_during_recording(
            duration_seconds=max_seconds,
            camera_device=camera_device,
            sample_every_seconds=CAMERA_SAMPLE_EVERY_SECONDS,
        )

    thread = threading.Thread(target=face_worker, daemon=True)
    thread.start()

    wav_path = record_audio_to_wav(
        max_seconds=max_seconds,
        input_device=input_device,
    )

    thread.join(timeout=max_seconds + 5)

    return wav_path, face_result.get("data")


# -----------------------------
# Input handling
# -----------------------------

def get_user_message_from_keyboard_or_voice(
    whisper_model,
    current_input_device: Optional[int],
    current_camera_device: Optional[int],
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[FaceEmotionCapture]]:
    """
    Read user input from keyboard or process special commands.
    """
    try:
        user_text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None, current_input_device, current_camera_device, None

    if not user_text:
        return "", current_input_device, current_camera_device, None

    normalized = normalize_command(user_text)

    if normalized == "devices":
        list_input_devices()
        return "", current_input_device, current_camera_device, None

    if normalized == "mic":
        new_device = choose_input_device(current_input_device)
        return "", new_device, current_camera_device, None

    if normalized == "cams":
        list_camera_devices()
        return "", current_input_device, current_camera_device, None

    if normalized == "cam":
        new_camera = choose_camera_device(current_camera_device)
        return "", current_input_device, new_camera, None

    if normalized == "speak":
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
            transcript = transcribe_with_whisperx(
                wav_path,
                whisper_model=whisper_model,
            )
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
        else:
            print("No speech detected.\n")

        if face_capture:
            if face_capture.error:
                print_ts(f"Face emotion capture error: {face_capture.error}")
            else:
                print_ts(f"Face emotion summary: {face_capture.summary_text}")

        print()
        return transcript, current_input_device, current_camera_device, face_capture

    return user_text, current_input_device, current_camera_device, None


# -----------------------------
# Main application loop
# -----------------------------

def main() -> None:
    """
    Main app entry point.
    """
    client = Client(host=OLLAMA_HOST)

    print_ts(f"Python version: {sys.version.split()[0]}")
    print_ts(f"Platform: {sys.platform}")
    print()

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
        print("Voice input will not work, but typing still works.\n")
        whisper_model = None

    current_input_device = INPUT_DEVICE
    current_camera_device = CAMERA_DEVICE
    history: list[dict] = []

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

    while True:
        user_text, current_input_device, current_camera_device, face_capture = (
            get_user_message_from_keyboard_or_voice(
                whisper_model=whisper_model,
                current_input_device=current_input_device,
                current_camera_device=current_camera_device,
            )
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

        if len(user_text.strip()) < 1:
            continue

        print_ts(f"You: {user_text}")

        facial_hint = face_capture.summary_text if face_capture else None

        try:
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

            analysis = analyze_user_message_with_llm(
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

        except Exception as e:
            print(f"\nError talking to Ollama server: {e}\n")


if __name__ == "__main__":
    main()