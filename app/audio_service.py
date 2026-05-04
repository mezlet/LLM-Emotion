from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from config import (
    INPUT_DEVICE,
    MAX_RECORD_SECONDS,
    MIN_PEAK_THRESHOLD,
    MIN_RMS_THRESHOLD,
    TARGET_SAMPLE_RATE,
)
from time_utils import now_ts, print_ts


def list_input_devices() -> None:
    print("\nAvailable input devices:")
    try:
        devices = sd.query_devices()
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
                f"  [mic {idx}] {device['name']} | hostapi={hostapi_name} | "
                f"inputs={device['max_input_channels']} | default_sr={device['default_samplerate']}"
            )
    if not found:
        print("  No input devices found.")
    print(f"\nCurrent default audio device: {sd.default.device}\n")


def choose_input_device(current_input_device: Optional[int]) -> Optional[int]:
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
    except Exception as exc:
        print(f"Invalid microphone device selection: {exc}\n")
        return current_input_device


def get_effective_input_samplerate(input_device: Optional[int]) -> int:
    device_info = sd.query_devices(kind="input") if input_device is None else sd.query_devices(input_device)
    default_sr = int(round(device_info["default_samplerate"]))
    return default_sr if default_sr > 0 else TARGET_SAMPLE_RATE


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / orig_sr
    target_length = max(1, int(round(duration * target_sr)))
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(np.float32)


def record_audio_to_wav(max_seconds: int = MAX_RECORD_SECONDS, input_device: Optional[int] = INPUT_DEVICE) -> Optional[str]:
    try:
        effective_sr = get_effective_input_samplerate(input_device)
        sd.check_input_settings(device=input_device, samplerate=effective_sr, channels=1, dtype="float32")

        print(f"\n[{now_ts()}] Recording microphone... speak now ({max_seconds} seconds max).")
        print(f"[{now_ts()}] Using sample rate {effective_sr} Hz for input.\n")

        audio = sd.rec(
            frames=int(max_seconds * effective_sr), samplerate=effective_sr,
            channels=1, dtype="float32", device=input_device,
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

        gain = min(0.9 / max(peak, 1e-6), 10.0)
        audio_16k = resample_audio(np.clip(audio * gain, -1.0, 1.0), effective_sr, TARGET_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
        sf.write(wav_path, audio_16k, TARGET_SAMPLE_RATE)
        return wav_path
    except Exception as exc:
        print(f"Audio recording error: {exc}\n")
        return None


def delete_file_safely(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
