"""
audio.py
--------
Microphone recording, audio normalisation, resampling, and WhisperX
transcription.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisperx

from config import (
    INPUT_DEVICE,
    MAX_RECORD_SECONDS,
    MIN_PEAK_THRESHOLD,
    MIN_RMS_THRESHOLD,
    TARGET_SAMPLE_RATE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_DEVICE,
    WHISPERX_LANGUAGE,
    WHISPERX_MODEL,
)
from utils import now_ts, print_ts


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def list_input_devices() -> None:
    """Print all available microphone input devices."""
    print("\nAvailable input devices:")
    try:
        devices  = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        print(f"Could not query audio devices: {exc}\n")
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


def choose_input_device(current_device: Optional[int]) -> Optional[int]:
    """
    Prompt the user to pick a microphone by index.

    Returns the new device index, or *current_device* if the selection
    is blank or invalid.
    """
    list_input_devices()
    selection = input(
        "Enter microphone input device index (blank to keep current/default): "
    ).strip()

    if not selection:
        return current_device

    try:
        device_index = int(selection)
        device_info  = sd.query_devices(device_index)

        if device_info["max_input_channels"] <= 0:
            print("That device does not support input.\n")
            return current_device

        print_ts(f"Using microphone input device [{device_index}] {device_info['name']}")
        return device_index

    except Exception as exc:
        print(f"Invalid microphone device selection: {exc}\n")
        return current_device


# ---------------------------------------------------------------------------
# Sample-rate helpers
# ---------------------------------------------------------------------------

def get_effective_input_samplerate(input_device: Optional[int]) -> int:
    """Return the default sample rate for *input_device* (or the system default)."""
    device_info = (
        sd.query_devices(input_device)
        if input_device is not None
        else sd.query_devices(kind="input")
    )
    default_sr = int(round(device_info["default_samplerate"]))
    return default_sr if default_sr > 0 else TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample *audio* from *orig_sr* to *target_sr* using NumPy interpolation.

    Returns a float32 array.  A no-op when rates are identical.
    """
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration      = len(audio) / orig_sr
    target_length = max(1, int(round(duration * target_sr)))

    old_times = np.linspace(0.0, duration, num=len(audio),       endpoint=False)
    new_times = np.linspace(0.0, duration, num=target_length,    endpoint=False)

    return np.interp(new_times, old_times, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_audio_to_wav(
    max_seconds: int = MAX_RECORD_SECONDS,
    input_device: Optional[int] = INPUT_DEVICE,
) -> Optional[str]:
    """
    Record microphone input, normalise it, resample to 16 kHz, and save
    to a temporary WAV file.

    Returns the path to the WAV file on success, or ``None`` on failure /
    silence.
    """
    try:
        effective_sr = get_effective_input_samplerate(input_device)

        sd.check_input_settings(
            device=input_device,
            samplerate=effective_sr,
            channels=1,
            dtype="float32",
        )

        print(
            f"\n[{now_ts()}] Recording microphone… speak now "
            f"({max_seconds} seconds max)."
        )
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
            print("Audio recording failed: empty buffer.\n")
            return None

        peak = float(np.max(np.abs(audio)))
        rms  = float(np.sqrt(np.mean(audio ** 2)))
        print_ts(f"Recorded audio level: peak={peak:.4f}, rms={rms:.4f}")

        if peak < MIN_PEAK_THRESHOLD or rms < MIN_RMS_THRESHOLD:
            print(
                "Recorded audio is too quiet or silent.\n"
                "Try a different microphone, check mute/input volume, or use /devices.\n"
            )
            return None

        # Normalise to 90 % of full scale (max gain cap: ×10).
        gain  = min(0.9 / max(peak, 1e-6), 10.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

        audio_16k = resample_audio(audio, effective_sr, TARGET_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        sf.write(wav_path, audio_16k, TARGET_SAMPLE_RATE)
        return wav_path

    except Exception as exc:
        print(f"Audio recording error: {exc}\n")
        return None


# ---------------------------------------------------------------------------
# WhisperX helpers
# ---------------------------------------------------------------------------

def load_whisper_model():
    """
    Load the WhisperX model specified in *config*.

    Returns the model on success, or ``None`` if loading fails (voice input
    will be disabled but typing continues to work).
    """
    try:
        print_ts("Loading WhisperX model…")
        model = whisperx.load_model(
            WHISPERX_MODEL,
            WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
            language=WHISPERX_LANGUAGE,
        )
        print_ts("WhisperX ready.")
        print()
        return model
    except Exception as exc:
        print_ts(f"Failed to load WhisperX: {exc}")
        print("Voice input will not work, but typing still works.\n")
        return None


def transcribe_with_whisperx(wav_path: str, whisper_model, batch_size: int = 4) -> str:
    """
    Transcribe *wav_path* using the pre-loaded *whisper_model*.

    Returns the transcribed text (may be empty if no speech was detected).
    """
    audio  = whisperx.load_audio(wav_path)
    result = whisper_model.transcribe(audio, batch_size=batch_size)

    text = result.get("text", "").strip()
    if text:
        return text

    segments = result.get("segments", [])
    return " ".join(seg.get("text", "").strip() for seg in segments).strip()