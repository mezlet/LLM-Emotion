from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from config import (
    FASTER_WHISPER_COMPUTE_TYPE,
    FASTER_WHISPER_DEVICE,
    FASTER_WHISPER_LANGUAGE,
    FASTER_WHISPER_MODEL,
    TRANSCRIPTION_BACKEND,
    WHISPERX_BATCH_SIZE,
    WHISPERX_COMPUTE_TYPE,
    WHISPERX_DEVICE,
    WHISPERX_LANGUAGE,
    WHISPERX_MODEL,
)
from time_utils import print_ts

TranscriptionBackend = Literal["faster-whisper", "whisperx"]


@dataclass
class Transcriber:
    """
    Adapter around the selected speech-to-text backend.

    The rest of the app does not need to know whether transcription is handled
    by faster-whisper or WhisperX. It only calls `transcribe(wav_path)`.
    """

    backend: TranscriptionBackend
    model: Any

    def transcribe(self, wav_path: str) -> str:
        if self.backend == "faster-whisper":
            return _transcribe_with_faster_whisper(wav_path, self.model)
        if self.backend == "whisperx":
            return _transcribe_with_whisperx(wav_path, self.model)
        raise ValueError(f"Unsupported transcription backend: {self.backend}")


def load_transcriber() -> Optional[Transcriber]:
    """
    Load the speech-to-text backend selected in config.py.

    Supported values:
    - TRANSCRIPTION_BACKEND = "faster-whisper"
    - TRANSCRIPTION_BACKEND = "whisperx"
    """
    backend = TRANSCRIPTION_BACKEND.strip().lower()

    if backend == "faster-whisper":
        return _load_faster_whisper_transcriber()

    if backend == "whisperx":
        return _load_whisperx_transcriber()

    print_ts(
        "Invalid TRANSCRIPTION_BACKEND value. "
        "Use 'faster-whisper' or 'whisperx'. Voice input is disabled."
    )
    return None


def _load_faster_whisper_transcriber() -> Optional[Transcriber]:
    try:
        from faster_whisper import WhisperModel

        print_ts("Loading faster-whisper model...")
        model = WhisperModel(
            FASTER_WHISPER_MODEL,
            device=FASTER_WHISPER_DEVICE,
            compute_type=FASTER_WHISPER_COMPUTE_TYPE,
        )
        print_ts("faster-whisper ready.")
        print()
        return Transcriber(backend="faster-whisper", model=model)
    except Exception as exc:
        print_ts(f"Failed to load faster-whisper: {exc}")
        print("Voice input will not work, but typing still works.\n")
        return None


def _load_whisperx_transcriber() -> Optional[Transcriber]:
    try:
        import whisperx

        print_ts("Loading WhisperX model...")
        model = whisperx.load_model(
            WHISPERX_MODEL,
            device=WHISPERX_DEVICE,
            compute_type=WHISPERX_COMPUTE_TYPE,
            language=WHISPERX_LANGUAGE,
        )
        print_ts("WhisperX ready.")
        print()
        return Transcriber(backend="whisperx", model=model)
    except Exception as exc:
        print_ts(f"Failed to load WhisperX: {exc}")
        print("Voice input will not work, but typing still works.\n")
        return None


def _transcribe_with_faster_whisper(wav_path: str, model: Any, beam_size: int = 5) -> str:
    segments, _ = model.transcribe(
        wav_path,
        language=FASTER_WHISPER_LANGUAGE,
        beam_size=beam_size,
    )
    return " ".join(segment.text.strip() for segment in segments if segment.text).strip()


def _transcribe_with_whisperx(wav_path: str, model: Any) -> str:
    """
    Transcribe a WAV file with WhisperX.

    This performs transcription only. Alignment and diarization can be added
    later without changing the app/input layers.
    """
    result = model.transcribe(
        wav_path,
        batch_size=WHISPERX_BATCH_SIZE,
        language=WHISPERX_LANGUAGE,
    )

    segments = result.get("segments", []) if isinstance(result, dict) else []
    return " ".join(
        str(segment.get("text", "")).strip()
        for segment in segments
        if str(segment.get("text", "")).strip()
    ).strip()
