import os
import tempfile
import time
from datetime import datetime
from typing import Optional, Tuple

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


def load_whisper_model():
    """Load and return the WhisperX model, or None on failure."""
    try:
        print("Loading WhisperX model...")
        model = whisperx.load_model(
            WHISPERX_MODEL,
            WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
            language=WHISPERX_LANGUAGE,
        )
        print("WhisperX ready.\n")
        return model
    except Exception as e:
        print(f"Failed to load WhisperX: {e}")
        print("Voice input will not work.\n")
        return None


def list_input_devices() -> None:
    """Print available audio input devices with host API names."""
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
    Return the selected device's default sample rate.
    Using the native rate avoids invalid sample rate errors on Linux hardware inputs.
    """
    if input_device is None:
        device_info = sd.query_devices(kind="input")
    else:
        device_info = sd.query_devices(input_device)

    default_sr = int(round(device_info["default_samplerate"]))
    return default_sr if default_sr > 0 else TARGET_SAMPLE_RATE


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear resampling using NumPy only (no SciPy dependency)."""
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / orig_sr
    target_length = max(1, int(round(duration * target_sr)))
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def record_audio_to_wav(
    max_seconds: int = MAX_RECORD_SECONDS,
    input_device: Optional[int] = INPUT_DEVICE,
) -> Optional[str]:
    """
    Record audio from the microphone at the device's native sample rate,
    then resample to TARGET_SAMPLE_RATE for WhisperX.
    Returns the path to a temporary WAV file, or None on failure.
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

        # Gentle normalization
        gain = min(0.9 / peak, 10.0)
        audio = np.clip(audio * gain, -1.0, 1.0)

        audio_16k = resample_audio(audio, effective_sr, TARGET_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        sf.write(wav_path, audio_16k, TARGET_SAMPLE_RATE)
        return wav_path

    except Exception as e:
        print(f"Audio recording error: {e}\n")
        return None


def transcribe_with_whisperx(wav_path: str, whisper_model, batch_size: int = 8) -> str:
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
    Read a message from the user via keyboard or voice.

    Returns (text, device):
      - text is None  → caller should exit
      - text is ""    → command handled; caller should loop
      - text is str   → user message to process

    Slash commands handled here:
      /speak    record voice and transcribe
      /devices  list input devices
      /mic      choose input device
    """
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        user_text = input(f"[{ts}] You: ").strip()
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
        return "", choose_input_device(current_input_device)

    if lowered == "/speak":
        if whisper_model is None:
            print("Voice input is unavailable (WhisperX failed to load).\n")
            return "", current_input_device

        wav_path = record_audio_to_wav(input_device=current_input_device)
        if not wav_path:
            return "", current_input_device

        try:
            t0 = time.perf_counter()
            transcript = transcribe_with_whisperx(wav_path, whisper_model=whisper_model)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"WhisperX transcription error: {e}\n")
            transcript = ""
            elapsed = 0.0
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if transcript:
            print(f"Transcribed ({elapsed:.2f}s): {transcript}\n")
        else:
            print("No speech detected.\n")
        return transcript, current_input_device

    return user_text, current_input_device
