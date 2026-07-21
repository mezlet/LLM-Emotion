#!/usr/bin/env python3
"""
Executable Ameca emotion-expression warm-up.

This script retains the original experiment's speech front end:
- Silero VAD at 16 kHz
- threshold=0.55
- min_silence_duration_ms=700
- speech_pad_ms=250
- pre-roll=0.35 s
- min utterance=0.60 s
- max utterance=15 s
- faster-whisper base, language=en, beam_size=1
- faster-whisper vad_filter=False
- condition_on_previous_text=False

The participant number is the profile identifier. Ameca separately asks the
participant to say and spell their name, then addresses them by that name.

DeepFace is launched through the same isolated worker-process arrangement as
the main experiment; TensorFlow/DeepFace is not imported into this process.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad


# =============================================================================
# Configuration retained from the main experiment
# =============================================================================

FAST_WHISPER_CONFIG = {
    "profile": "home_macbook_cpu",
    "model": os.environ.get("WHISPER_MODEL", "base"),
    "device": os.environ.get("WHISPER_DEVICE", "cuda"),
    "compute_type": os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
    "language": "en",
    "beam_size": 1,
    "vad_filter": False,
}

TARGET_SAMPLE_RATE = 16000
SILERO_SAMPLE_RATE = 16000
SILERO_CHUNK_SIZE = 512
SILERO_THRESHOLD = 0.55
SILERO_MIN_SILENCE_DURATION_MS = 700
SILERO_SPEECH_PAD_MS = 250
VAD_MAX_UTTERANCE_SECONDS = 15.0
VAD_MIN_UTTERANCE_SECONDS = 0.60
VAD_PRE_ROLL_SECONDS = 0.35

MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003

DATA_DIR = Path(os.environ.get("WARMUP_DATA_DIR", "conversation_data"))
USERS_FILE = DATA_DIR / "users.json"
PROFILE_IMAGE_DIR = DATA_DIR / "profile_images"

TTS_URL = os.environ.get(
    "TRITIUM_TTS_URL",
    "http://emah/tritium/text_to_speech/say?voice=Lucy",
)
TTS_TOKEN = os.environ.get("TRITIUM_TOKEN", "ZWNFuNQVIPyztWCfPPM5VLPslpj8rR")

CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "2560"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "30"))
USE_ZED_HALF_FRAME_CROP = os.environ.get("USE_ZED_HALF_FRAME_CROP", "1") == "1"
CAMERA_SAMPLE_EVERY_SECONDS = float(
    os.environ.get("CAMERA_SAMPLE_EVERY_SECONDS", "0.5")
)
FACE_MAX_CANDIDATE_FRAMES = int(
    os.environ.get("FACE_MAX_CANDIDATE_FRAMES", "5")
)

DEEPFACE_PYTHON = os.environ.get("DEEPFACE_PYTHON", "")
DEEPFACE_WORKER_SCRIPT = os.environ.get(
    "DEEPFACE_WORKER_SCRIPT", "deepface_worker.py"
)
DEEPFACE_STARTUP_TIMEOUT_SECONDS = float(
    os.environ.get("DEEPFACE_STARTUP_TIMEOUT_SECONDS", "90")
)
DEEPFACE_REQUEST_TIMEOUT_SECONDS = float(
    os.environ.get("DEEPFACE_REQUEST_TIMEOUT_SECONDS", "5")
)

EMOTION_SCRIPTS: dict[str, str] = {
    "joy": "I received wonderful news today, and I feel very happy and excited.",
    "sadness": (
        "Something important to me did not go as planned, and I feel sad "
        "and disappointed."
    ),
    "anger": (
        "Someone treated me unfairly, and I feel angry about what happened."
    ),
    "fear": (
        "I heard a sudden sound behind me, and I feel frightened because "
        "I do not know what caused it."
    ),
    "surprise": (
        "I opened the door and found an unexpected celebration waiting for me."
    ),
}

DEEPFACE_TO_PROFILE_EMOTION = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
}


# =============================================================================
# General helpers
# =============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def print_ts(message: str) -> None:
    print(f"[{now_ts()}] {message}", flush=True)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown_participant"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_users() -> dict[str, Any]:
    ensure_directories()
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_users(users: dict[str, Any]) -> None:
    ensure_directories()
    temp_path = USERS_FILE.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2, ensure_ascii=False)
    temp_path.replace(USERS_FILE)


def get_or_create_profile(participant_id: str) -> tuple[str, dict[str, Any]]:
    users = load_users()
    user_key = slugify(participant_id)
    profile = users.get(user_key)

    if not isinstance(profile, dict):
        profile = {
            "participant_id": participant_id,
            "display_name": "",
            "created_at": now_iso(),
            "last_seen": now_iso(),
            "conversation_summary": "",
            "warm_up": {
                "completed": False,
                "completed_at": None,
                "emotion_samples": {},
            },
        }
    else:
        profile["participant_id"] = participant_id
        profile["last_seen"] = now_iso()
        profile.setdefault("warm_up", {})
        profile["warm_up"].setdefault("completed", False)
        profile["warm_up"].setdefault("completed_at", None)
        profile["warm_up"].setdefault("emotion_samples", {})

    users[user_key] = profile
    save_users(users)
    return user_key, profile


def list_input_devices() -> None:
    print("\nAvailable audio input devices:")
    try:
        devices = sd.query_devices()
        for index, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) > 0:
                marker = ""
                try:
                    if index == sd.default.device[0]:
                        marker = " [default]"
                except Exception:
                    pass
                print(
                    f"  {index}: {device['name']} | "
                    f"inputs={device['max_input_channels']} | "
                    f"default_rate={device['default_samplerate']}{marker}"
                )
    except Exception as exc:
        print(f"  Could not enumerate devices: {exc}")
    print()


# =============================================================================
# Tritium TTS and echo guard
# =============================================================================

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*_`~]", "", text or "")
    return "".join(
        character
        for character in text
        if unicodedata.category(character)[0] != "C"
    ).strip()


def estimate_speech_duration_seconds(text: str) -> float:
    words = max(1, len(text.split()))
    return max(1.0, words / 150.0 * 60.0) + 0.8


class RobotSpeaker:
    def __init__(self, url: str, token: str = "") -> None:
        self.url = url
        self.token = token
        self._speaking_until = 0.0

    def is_speaking_or_cooling_down(self) -> bool:
        return time.time() < self._speaking_until

    def wait_until_finished(self) -> None:
        while self.is_speaking_or_cooling_down():
            time.sleep(0.1)

    def say(self, text: str) -> None:
        spoken = clean_text_for_tts(text)
        if not spoken:
            return

        print(f"\nAMECA: {spoken}", flush=True)
        self._speaking_until = max(
            self._speaking_until,
            time.time() + estimate_speech_duration_seconds(spoken),
        )

        headers = {"Content-Type": "text/plain; charset=utf-8"}
        if self.token:
            headers["X-Tritium-Auth-Token"] = self.token

        try:
            response = requests.put(
                self.url,
                data=spoken.encode("utf-8"),
                headers=headers,
                timeout=5,
            )
            if not 200 <= response.status_code < 300:
                print(
                    f"[TTS warning] Tritium returned {response.status_code}: "
                    f"{response.text[:200]!r}"
                )
        except requests.RequestException as exc:
            print(f"[TTS warning] Could not send speech to Ameca: {exc}")


# =============================================================================
# Original Silero VAD + faster-whisper speech pipeline
# =============================================================================

def get_input_samplerate(input_device: Optional[int]) -> int:
    info = sd.query_devices(input_device, "input")
    sample_rate = int(round(float(info["default_samplerate"])))
    return sample_rate if sample_rate > 0 else TARGET_SAMPLE_RATE


def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if original_sr == target_sr or audio.size == 0:
        return audio

    duration = len(audio) / float(original_sr)
    target_length = max(1, int(round(duration * target_sr)))
    old_times = np.linspace(0.0, duration, len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, target_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def save_audio_to_temp_wav(audio_16k: np.ndarray) -> Optional[str]:
    if audio_16k.size == 0:
        return None

    peak = float(np.max(np.abs(audio_16k)))
    rms = float(np.sqrt(np.mean(np.square(audio_16k))))
    print_ts(f"Captured utterance audio level: peak={peak:.4f}, rms={rms:.4f}")

    if peak < MIN_PEAK_THRESHOLD or rms < MIN_RMS_THRESHOLD:
        print_ts("Captured audio was too quiet or silent.")
        return None

    gain = min(0.9 / max(peak, 1e-6), 10.0)
    normalized = np.clip(audio_16k * gain, -1.0, 1.0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    sf.write(wav_path, normalized, TARGET_SAMPLE_RATE)
    return wav_path


def transcribe_with_faster_whisper(
    wav_path: str,
    whisper_model: WhisperModel,
) -> str:
    segments, _ = whisper_model.transcribe(
        wav_path,
        language=FAST_WHISPER_CONFIG["language"],
        beam_size=int(FAST_WHISPER_CONFIG["beam_size"]),
        vad_filter=bool(FAST_WHISPER_CONFIG["vad_filter"]),
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) <= 1 and len(text) < 3:
        return ""
    return text


class FrameCollector:
    """Continuously reads the camera and retains sampled frames during speech."""

    def __init__(self, camera: "Camera", label: str) -> None:
        self.camera = camera
        self.label = label
        self.frames: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_sample = 0.0
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            preview = frame.copy()
            cv2.putText(
                preview,
                self.label,
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Ameca warm-up", preview)
            cv2.waitKey(1)

            now = time.monotonic()
            if now >= next_sample:
                self.frames.append(frame.copy())
                next_sample = now + CAMERA_SAMPLE_EVERY_SECONDS

    def stop(self) -> list[np.ndarray]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.frames


def listen_for_utterance_with_silero_vad(
    input_device: Optional[int],
    silero_model: Any,
    prompt_label: str,
    robot_speaker: Optional[RobotSpeaker] = None,
    camera: Optional["Camera"] = None,
) -> tuple[Optional[str], list[np.ndarray]]:
    """Same VAD configuration and audio treatment as the main experiment."""
    if robot_speaker is not None:
        robot_speaker.wait_until_finished()

    input_sample_rate = get_input_samplerate(input_device)
    input_block_size = max(1, int(input_sample_rate * 0.05))
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    vad_iterator = VADIterator(
        silero_model,
        threshold=SILERO_THRESHOLD,
        sampling_rate=SILERO_SAMPLE_RATE,
        min_silence_duration_ms=SILERO_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SILERO_SPEECH_PAD_MS,
    )

    pre_roll_max_chunks = max(
        1,
        int(
            VAD_PRE_ROLL_SECONDS
            * SILERO_SAMPLE_RATE
            / SILERO_CHUNK_SIZE
        ),
    )
    pre_roll_chunks: deque[np.ndarray] = deque(maxlen=pre_roll_max_chunks)
    recorded_chunks: list[np.ndarray] = []

    is_recording = False
    speech_started_at: Optional[float] = None
    leftover_16k = np.array([], dtype=np.float32)
    frame_collector: Optional[FrameCollector] = None

    def audio_callback(indata, frames, callback_time, status) -> None:
        if status:
            print_ts(f"Audio callback status: {status}")
        audio_queue.put(indata[:, 0].copy())

    print_ts(
        f"Listening automatically for {prompt_label}. "
        "Speak when ready. Press Ctrl+C to quit."
    )
    print_ts(
        f"Silero VAD settings: threshold={SILERO_THRESHOLD}, "
        f"min_silence={SILERO_MIN_SILENCE_DURATION_MS}ms, "
        f"max_utterance={VAD_MAX_UTTERANCE_SECONDS}s"
    )

    try:
        sd.check_input_settings(
            device=input_device,
            samplerate=input_sample_rate,
            channels=1,
            dtype="float32",
        )

        with sd.InputStream(
            samplerate=input_sample_rate,
            channels=1,
            dtype="float32",
            device=input_device,
            blocksize=input_block_size,
            callback=audio_callback,
        ):
            while True:
                source = audio_queue.get()
                audio_16k = resample_audio(
                    source,
                    input_sample_rate,
                    SILERO_SAMPLE_RATE,
                )

                if leftover_16k.size:
                    audio_16k = np.concatenate((leftover_16k, audio_16k))
                    leftover_16k = np.array([], dtype=np.float32)

                full_length = (
                    len(audio_16k) // SILERO_CHUNK_SIZE
                ) * SILERO_CHUNK_SIZE

                if full_length == 0:
                    leftover_16k = audio_16k
                    continue

                processable = audio_16k[:full_length]
                leftover_16k = audio_16k[full_length:]

                chunks = np.split(
                    processable,
                    full_length // SILERO_CHUNK_SIZE,
                )

                for chunk in chunks:
                    chunk = chunk.astype(np.float32, copy=False)

                    if not is_recording:
                        pre_roll_chunks.append(chunk.copy())
                    else:
                        recorded_chunks.append(chunk.copy())

                    event = vad_iterator(
                        torch.from_numpy(chunk),
                        return_seconds=True,
                    )
                    now = time.time()

                    if event and "start" in event and not is_recording:
                        is_recording = True
                        speech_started_at = now
                        recorded_chunks = list(pre_roll_chunks)
                        recorded_chunks.append(chunk.copy())
                        pre_roll_chunks.clear()

                        if camera is not None:
                            frame_collector = FrameCollector(
                                camera,
                                f"Reading: {prompt_label}",
                            )
                            frame_collector.start()

                        print()
                        print_ts("Speech detected. Recording utterance...")

                    if event and "end" in event and is_recording:
                        duration = now - (speech_started_at or now)
                        if duration >= VAD_MIN_UTTERANCE_SECONDS:
                            print_ts("Speech ended. Processing utterance...")
                            raise StopIteration

                    if (
                        is_recording
                        and speech_started_at is not None
                        and now - speech_started_at
                        >= VAD_MAX_UTTERANCE_SECONDS
                    ):
                        print_ts(
                            "Maximum utterance length reached. "
                            "Processing utterance..."
                        )
                        raise StopIteration

    except StopIteration:
        pass
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print_ts(f"Silero VAD/audio error: {exc}")
        return None, []
    finally:
        try:
            vad_iterator.reset_states()
        except Exception:
            pass

    frames = frame_collector.stop() if frame_collector else []
    if not recorded_chunks:
        return None, frames

    audio = np.concatenate(recorded_chunks).astype(np.float32, copy=False)
    return save_audio_to_temp_wav(audio), frames


def capture_and_transcribe(
    whisper_model: WhisperModel,
    silero_model: Any,
    input_device: Optional[int],
    robot_speaker: RobotSpeaker,
    label: str,
    camera: Optional["Camera"] = None,
    attempts: int = 3,
) -> tuple[str, list[np.ndarray]]:
    for attempt in range(1, attempts + 1):
        wav_path, frames = listen_for_utterance_with_silero_vad(
            input_device=input_device,
            silero_model=silero_model,
            prompt_label=label,
            robot_speaker=robot_speaker,
            camera=camera,
        )
        if not wav_path:
            if attempt < attempts:
                robot_speaker.say(
                    "I did not hear that clearly. Please try again."
                )
            continue

        try:
            transcript = transcribe_with_faster_whisper(
                wav_path,
                whisper_model,
            )
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        print_ts(f"Transcript [faster-whisper]: {transcript!r}")
        if transcript:
            return transcript, frames

        if attempt < attempts:
            robot_speaker.say(
                "I could not transcribe that clearly. Please try again."
            )

    return "", []


# =============================================================================
# Participant name capture
# =============================================================================

def clean_spoken_name(text: str) -> str:
    text = text.strip()
    patterns = [
        r"^my name is\s+",
        r"^my name's\s+",
        r"^i am\s+",
        r"^i'm\s+",
        r"^this is\s+",
        r"^it is\s+",
        r"^it's\s+",
        r"^call me\s+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ' -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:40].title()


LETTER_WORDS = {
    "AY": "A", "A": "A",
    "BEE": "B", "BE": "B", "B": "B",
    "SEE": "C", "SEA": "C", "C": "C",
    "DEE": "D", "D": "D",
    "E": "E",
    "EFF": "F", "F": "F",
    "GEE": "G", "G": "G",
    "AITCH": "H", "H": "H",
    "EYE": "I", "I": "I",
    "JAY": "J", "J": "J",
    "KAY": "K", "K": "K",
    "EL": "L", "L": "L",
    "EM": "M", "M": "M",
    "EN": "N", "N": "N",
    "OH": "O", "O": "O",
    "PEE": "P", "P": "P",
    "QUEUE": "Q", "CUE": "Q", "Q": "Q",
    "ARE": "R", "R": "R",
    "ESS": "S", "S": "S",
    "TEE": "T", "TEA": "T", "T": "T",
    "YOU": "U", "U": "U",
    "VEE": "V", "V": "V",
    "DOUBLE U": "W", "W": "W",
    "EX": "X", "X": "X",
    "WHY": "Y", "Y": "Y",
    "ZEE": "Z", "ZED": "Z", "Z": "Z",
}


def clean_spelled_name(text: str) -> str:
    cleaned = text.upper()
    cleaned = re.sub(
        r"\b(MY NAME IS|THE SPELLING IS|IT IS|THAT IS|SPELLING)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"[^A-Z ]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Resolve "DOUBLE <letter>" before token parsing.
    def double_replacement(match: re.Match[str]) -> str:
        token = match.group(1)
        letter = LETTER_WORDS.get(token, token if len(token) == 1 else "")
        return f"{letter} {letter}" if letter else ""

    cleaned = re.sub(
        r"\bDOUBLE\s+([A-Z]+)\b",
        double_replacement,
        cleaned,
    )

    tokens = cleaned.split()
    letters: list[str] = []
    for token in tokens:
        if token in LETTER_WORDS:
            letters.append(LETTER_WORDS[token])
        elif len(token) == 1 and token.isalpha():
            letters.append(token)

    if len(letters) >= 2:
        return "".join(letters).title()

    words = [
        word
        for word in re.findall(r"[A-Z]+", cleaned)
        if word not in {"MY", "NAME", "IS", "IT", "THE"}
    ]
    if len(words) == 1 and len(words[0]) >= 2:
        return words[0].title()
    return ""


def capture_participant_name(
    speaker: RobotSpeaker,
    whisper_model: WhisperModel,
    silero_model: Any,
    input_device: Optional[int],
) -> tuple[str, dict[str, str]]:
    speaker.say(
        "Before we begin, please say your first name after I finish speaking."
    )
    spoken_transcript, _ = capture_and_transcribe(
        whisper_model,
        silero_model,
        input_device,
        speaker,
        "spoken name",
        attempts=3,
    )
    spoken_name = clean_spoken_name(spoken_transcript)

    speaker.say(
        "Thank you. Now please spell your first name, one letter at a time, "
        "after I finish speaking."
    )
    spelling_transcript, _ = capture_and_transcribe(
        whisper_model,
        silero_model,
        input_device,
        speaker,
        "spelled name",
        attempts=3,
    )
    spelled_name = clean_spelled_name(spelling_transcript)

    display_name = spelled_name or spoken_name
    if not display_name:
        print(
            "\nASR could not obtain a usable name after three attempts.",
            flush=True,
        )
        display_name = input(
            "Enter the participant's first name manually: "
        ).strip().title()
        display_name = display_name or "Participant"

    speaker.say(
        f"Thank you, {display_name}. I will call you {display_name}."
    )
    return display_name, {
        "spoken_name_transcript": spoken_transcript,
        "spelling_transcript": spelling_transcript,
        "spoken_name_candidate": spoken_name,
        "spelled_name": spelled_name,
    }


def save_participant_name(
    user_key: str,
    display_name: str,
    capture: dict[str, str],
) -> None:
    users = load_users()
    profile = users[user_key]
    profile["display_name"] = display_name
    profile["name_capture"] = {
        **capture,
        "captured_at": now_iso(),
        "backend": "silero-vad + faster-whisper",
        "configuration": FAST_WHISPER_CONFIG,
        "silero_threshold": SILERO_THRESHOLD,
        "silero_min_silence_ms": SILERO_MIN_SILENCE_DURATION_MS,
        "silero_speech_pad_ms": SILERO_SPEECH_PAD_MS,
    }
    profile["last_seen"] = now_iso()
    users[user_key] = profile
    save_users(users)


# =============================================================================
# Camera
# =============================================================================

class Camera:
    def __init__(self, device: int) -> None:
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(device, backend)
        self._lock = threading.Lock()

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device {device}.")

        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        for _ in range(15):
            self.read()
            time.sleep(0.03)

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print_ts(f"Camera ready on /dev/video{device}: {width}x{height}")

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        if USE_ZED_HALF_FRAME_CROP and frame.shape[1] >= 2000:
            frame = frame[:, : frame.shape[1] // 2]
        return frame

    def close(self) -> None:
        with self._lock:
            self.capture.release()
        cv2.destroyAllWindows()


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# =============================================================================
# Isolated DeepFace worker
# =============================================================================

@dataclass
class DeepFaceResult:
    ok: bool
    no_face: bool
    scores: dict[str, float]
    dominant_emotion: Optional[str] = None
    region: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class DeepFaceClient:
    def __init__(
        self,
        python_executable: str,
        worker_script: str,
        startup_timeout: float,
        request_timeout: float,
    ) -> None:
        self.python_executable = python_executable
        self.worker_script = worker_script
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.proc: Optional[subprocess.Popen[str]] = None
        self.responses: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._request_counter = 0
        self._ready = False
        self._start_worker()

    def _start_worker(self) -> None:
        if not self.python_executable:
            raise RuntimeError(
                "--deepface_python is required. Pass the Python interpreter "
                "from the separate DeepFace conda environment."
            )
        if not Path(self.python_executable).is_file():
            raise RuntimeError(
                f"DeepFace Python executable not found: {self.python_executable}"
            )
        if not Path(self.worker_script).is_file():
            raise RuntimeError(
                f"DeepFace worker script not found: {self.worker_script}"
            )

        self.proc = subprocess.Popen(
            [self.python_executable, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        ready_queue: "queue.Queue[str]" = queue.Queue()

        def read_ready() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            ready_queue.put(self.proc.stdout.readline().strip())

        threading.Thread(target=read_ready, daemon=True).start()
        try:
            ready = ready_queue.get(timeout=self.startup_timeout)
        except queue.Empty as exc:
            self.shutdown()
            raise RuntimeError(
                "DeepFace worker did not become ready within "
                f"{self.startup_timeout:.1f} seconds."
            ) from exc

        if ready != "READY":
            self.shutdown()
            raise RuntimeError(
                f"Unexpected DeepFace worker startup response: {ready!r}"
            )

        self._ready = True
        threading.Thread(target=self._read_responses, daemon=True).start()
        print_ts("DeepFace worker ready.")

    def _drain_stderr(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            line = line.rstrip()
            if line:
                print_ts(f"[DeepFace worker] {line}")

    def _read_responses(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            try:
                self.responses.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def is_alive(self) -> bool:
        return (
            self._ready
            and self.proc is not None
            and self.proc.poll() is None
        )

    def analyze(self, frame: np.ndarray) -> Optional[DeepFaceResult]:
        if not self.is_alive():
            return None

        fd, image_path = tempfile.mkstemp(
            suffix=".jpg",
            prefix="warmup_deepface_",
        )
        os.close(fd)

        try:
            write_ok = cv2.imwrite(image_path, frame)
            path = Path(image_path)
            if (
                not write_ok
                or not path.exists()
                or path.stat().st_size == 0
            ):
                print_ts("Could not create temporary DeepFace image.")
                return None

            self._request_counter += 1
            request_id = f"warmup_{self._request_counter}"
            request = {
                "request_id": request_id,
                "cmd": "analyze",
                "image_path": image_path,
            }

            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()

            deadline = time.time() + self.request_timeout
            while time.time() < deadline:
                remaining = max(0.05, deadline - time.time())
                try:
                    response = self.responses.get(timeout=remaining)
                except queue.Empty:
                    break

                if response.get("request_id") != request_id:
                    continue

                if not response.get("ok"):
                    print_ts(
                        "DeepFace worker error: "
                        f"{response.get('error', 'unknown error')}"
                    )
                    return None

                return DeepFaceResult(
                    ok=True,
                    no_face=bool(response.get("no_face")),
                    scores={
                        str(key): float(value)
                        for key, value in (
                            response.get("scores", {}) or {}
                        ).items()
                    },
                    dominant_emotion=response.get("dominant_emotion"),
                    region=response.get("region"),
                )

            print_ts(
                f"DeepFace request timed out after "
                f"{self.request_timeout:.1f} seconds."
            )
            return None
        finally:
            try:
                os.remove(image_path)
            except OSError:
                pass

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(
                    json.dumps({"cmd": "shutdown"}) + "\n"
                )
                self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self._ready = False


def detect_best_emotion_sample(
    frames: list[np.ndarray],
    deepface: DeepFaceClient,
) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    if not frames:
        return None, None

    ordered = sorted(frames, key=sharpness, reverse=True)
    ordered = ordered[:FACE_MAX_CANDIDATE_FRAMES]
    analyzed: list[tuple[np.ndarray, dict[str, Any]]] = []

    for frame in ordered:
        result = deepface.analyze(frame)
        if result is None or result.no_face or not result.scores:
            continue

        raw = (
            result.dominant_emotion
            or max(result.scores.items(), key=lambda item: item[1])[0]
        )
        raw = str(raw).strip().lower()
        mapped = DEEPFACE_TO_PROFILE_EMOTION.get(raw)
        if not mapped:
            continue

        confidence = float(result.scores.get(raw, 0.0)) / 100.0
        analyzed.append(
            (
                frame,
                {
                    "emotion": mapped,
                    "deepface_emotion": raw,
                    "confidence": confidence,
                    "scores": result.scores,
                    "region": result.region or {},
                },
            )
        )

    if not analyzed:
        return None, None

    return max(analyzed, key=lambda item: item[1]["confidence"])


def crop_face(
    frame: np.ndarray,
    region: dict[str, Any],
) -> np.ndarray:
    try:
        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        width = max(1, int(region.get("w", frame.shape[1])))
        height = max(1, int(region.get("h", frame.shape[0])))
        pad_x = int(width * 0.25)
        pad_y = int(height * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + width + pad_x)
        y2 = min(frame.shape[0], y + height + pad_y)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else frame
    except Exception:
        return frame


# =============================================================================
# Profile persistence
# =============================================================================

def save_emotion_sample(
    user_key: str,
    requested_emotion: str,
    transcript: str,
    frame: np.ndarray,
    result: dict[str, Any],
    script_text: str,
) -> Path:
    directory = PROFILE_IMAGE_DIR / user_key
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detected = result["emotion"]
    image_path = directory / (
        f"{requested_emotion}_detected-{detected}_{timestamp}.jpg"
    )

    image = crop_face(frame, result.get("region", {}))
    ok = cv2.imwrite(
        str(image_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    if (
        not ok
        or not image_path.exists()
        or image_path.stat().st_size == 0
    ):
        raise OSError(f"Failed to save profile image: {image_path}")

    users = load_users()
    profile = users[user_key]
    warm_up = profile.setdefault("warm_up", {})
    samples = warm_up.setdefault("emotion_samples", {})
    samples[requested_emotion] = {
        "requested_emotion": requested_emotion,
        "detected_emotion": detected,
        "matched_request": detected == requested_emotion,
        "confidence": round(float(result["confidence"]), 4),
        "deepface_scores": result["scores"],
        "script": script_text,
        "transcript": transcript,
        "image_path": str(image_path),
        "captured_at": now_iso(),
    }
    profile["last_seen"] = now_iso()
    users[user_key] = profile
    save_users(users)
    return image_path


def finish_warm_up(user_key: str) -> None:
    users = load_users()
    profile = users[user_key]
    warm_up = profile.setdefault("warm_up", {})
    samples = warm_up.setdefault("emotion_samples", {})
    warm_up["completed"] = True
    warm_up["completed_at"] = now_iso()

    matched = sum(
        1
        for sample in samples.values()
        if isinstance(sample, dict)
        and sample.get("matched_request")
    )
    summary = ", ".join(
        f"{target}→{sample.get('detected_emotion', 'unknown')}"
        for target, sample in samples.items()
        if isinstance(sample, dict)
    )
    profile["conversation_summary"] = (
        "The participant completed the Ameca familiarisation warm-up. "
        f"{len(samples)} facial-expression samples were saved and "
        f"{matched} matched the requested emotion. Detections: {summary}."
    )
    profile["last_seen"] = now_iso()
    users[user_key] = profile
    save_users(users)


# =============================================================================
# Warm-up
# =============================================================================

def run_warm_up(args: argparse.Namespace) -> None:
    participant_id = (
        args.name
        or input("Participant number: ").strip()
        or "unknown"
    )
    user_key, _ = get_or_create_profile(participant_id)

    speaker = RobotSpeaker(args.tts_url, args.tts_token)
    camera: Optional[Camera] = None
    deepface: Optional[DeepFaceClient] = None

    print_ts("Loading Silero VAD...")
    silero_model = load_silero_vad()
    print_ts("Silero VAD ready.")

    print_ts(
        "Loading faster-whisper "
        f"model={FAST_WHISPER_CONFIG['model']}, "
        f"device={FAST_WHISPER_CONFIG['device']}, "
        f"compute_type={FAST_WHISPER_CONFIG['compute_type']}..."
    )
    try:
        whisper_model = WhisperModel(
            FAST_WHISPER_CONFIG["model"],
            device=FAST_WHISPER_CONFIG["device"],
            compute_type=FAST_WHISPER_CONFIG["compute_type"],
        )
    except Exception as exc:
        if (
            FAST_WHISPER_CONFIG["device"] == "cuda"
            and args.allow_cpu_fallback
        ):
            print_ts(
                f"CUDA Whisper initialization failed: {exc}. "
                "Falling back to CPU int8."
            )
            FAST_WHISPER_CONFIG["device"] = "cpu"
            FAST_WHISPER_CONFIG["compute_type"] = "int8"
            whisper_model = WhisperModel(
                FAST_WHISPER_CONFIG["model"],
                device="cpu",
                compute_type="int8",
            )
        else:
            raise
    print_ts("faster-whisper ready.")

    try:
        deepface = DeepFaceClient(
            python_executable=args.deepface_python,
            worker_script=args.deepface_worker_script,
            startup_timeout=args.deepface_startup_timeout,
            request_timeout=args.deepface_timeout,
        )

        camera = Camera(args.camera)

        display_name, name_capture = capture_participant_name(
            speaker,
            whisper_model,
            silero_model,
            args.input_device,
        )
        save_participant_name(user_key, display_name, name_capture)

        speaker.say(
            f"Hello {display_name}. Before the experiment, we will do a "
            f"short warm-up with {len(args.emotions)} scripts."
        )
        speaker.say(
            "For each script, read the sentence aloud while showing the "
            "emotion written on the screen. I will listen automatically "
            "and tell you which facial emotion I detected."
        )

        for index, requested_emotion in enumerate(args.emotions, start=1):
            script_text = EMOTION_SCRIPTS[requested_emotion]

            print("\n" + "=" * 76)
            print(f"SCRIPT {index} OF {len(args.emotions)}")
            print(f"EMOTION TO EXPRESS: {requested_emotion.upper()}")
            print("-" * 76)
            print(script_text)
            print("=" * 76, flush=True)

            speaker.say(
                f"{display_name}, please read script {index} while "
                f"expressing {requested_emotion}. Begin after I finish speaking."
            )

            transcript, frames = capture_and_transcribe(
                whisper_model=whisper_model,
                silero_model=silero_model,
                input_device=args.input_device,
                robot_speaker=speaker,
                label=f"script {index}",
                camera=camera,
                attempts=args.script_attempts,
            )

            if not transcript:
                speaker.say(
                    "I could not capture that reading clearly, so I will "
                    "move to the next script."
                )
                continue

            print_ts(f"Script transcript: {transcript}")
            frame, result = detect_best_emotion_sample(frames, deepface)

            if frame is None or result is None:
                speaker.say(
                    "I heard the sentence, but I could not clearly detect "
                    "a facial emotion."
                )
                print_ts("No profile image was saved for this script.")
            else:
                image_path = save_emotion_sample(
                    user_key=user_key,
                    requested_emotion=requested_emotion,
                    transcript=transcript,
                    frame=frame,
                    result=result,
                    script_text=script_text,
                )
                detected = result["emotion"]
                confidence = round(result["confidence"] * 100)
                speaker.say(
                    f"I detected {detected}, with approximately "
                    f"{confidence} percent confidence."
                )
                print_ts(
                    f"Result: target={requested_emotion}, "
                    f"detected={detected}, confidence={confidence}%"
                )
                print_ts(f"Saved image: {image_path}")

            if index < len(args.emotions):
                speaker.say("We will now move to the next script.")

        finish_warm_up(user_key)
        speaker.say(
            f"Thank you, {display_name}. The warm-up is complete. "
            "We can now begin the experiment."
        )
        print_ts(f"Profile updated: {USERS_FILE}")
        print_ts(f"Images saved under: {PROFILE_IMAGE_DIR / user_key}")

    finally:
        if camera is not None:
            camera.close()
        if deepface is not None:
            deepface.shutdown()
            print_ts("DeepFace worker shut down.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Ameca's automatic Silero-VAD/faster-whisper facial-emotion "
            "warm-up."
        )
    )
    parser.add_argument(
        "--name",
        help=(
            "Participant number/profile identifier, for example A11320. "
            "Ameca will separately ask for the participant's spoken name."
        ),
    )
    parser.add_argument(
        "--input_device",
        type=int,
        default=None,
        help="SoundDevice microphone index. Omit to use the system default.",
    )
    parser.add_argument(
        "--list_input_devices",
        action="store_true",
        help="List microphone devices and exit.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(os.environ.get("CAMERA_DEVICE", "0")),
        help="OpenCV/ZED camera index. Default: 0.",
    )
    parser.add_argument(
        "--deepface_python",
        default=DEEPFACE_PYTHON,
        help=(
            "Python executable in the separate DeepFace/TensorFlow conda "
            "environment."
        ),
    )
    parser.add_argument(
        "--deepface_worker_script",
        default=DEEPFACE_WORKER_SCRIPT,
        help="Path to deepface_worker.py.",
    )
    parser.add_argument(
        "--deepface_startup_timeout",
        type=float,
        default=DEEPFACE_STARTUP_TIMEOUT_SECONDS,
        help="DeepFace worker startup timeout in seconds.",
    )
    parser.add_argument(
        "--deepface_timeout",
        type=float,
        default=DEEPFACE_REQUEST_TIMEOUT_SECONDS,
        help="DeepFace timeout per candidate frame in seconds.",
    )
    parser.add_argument(
        "--tts_url",
        default=TTS_URL,
        help="Tritium text-to-speech URL.",
    )
    parser.add_argument(
        "--tts_token",
        default=TTS_TOKEN,
        help=(
            "Tritium authentication token. Prefer the TRITIUM_TOKEN "
            "environment variable instead of passing it in shell history."
        ),
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=list(EMOTION_SCRIPTS),
        default=list(EMOTION_SCRIPTS),
        help="Emotion scripts to run.",
    )
    parser.add_argument(
        "--script_attempts",
        type=int,
        default=2,
        help="Maximum automatic listening attempts per emotion script.",
    )
    parser.add_argument(
        "--allow_cpu_fallback",
        action="store_true",
        help=(
            "Fall back to CPU int8 if the original CUDA Whisper "
            "configuration cannot initialize."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.list_input_devices:
        list_input_devices()
        return

    run_warm_up(args)


if __name__ == "__main__":
    main()
