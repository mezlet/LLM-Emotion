"""
faster-whisper inference runner.
Wraps faster_whisper.WhisperModel for the benchmark.
"""

import gc
import re
from typing import Optional

import torch


class FasterWhisperRunner:
    """
    Transcribes audio files using faster-whisper (CTranslate2 backend).
    Single-file inference (no native batching in faster-whisper).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 5,
        language: str = "en",
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.cpu_threads = cpu_threads
        self._model = None

    def load_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install via: pip install faster-whisper"
            )

        print(f"  [faster-whisper] Loading model '{self.model_size}' on {self.device} ({self.compute_type})...")

        kwargs = dict(
            device=self.device,
            compute_type=self.compute_type,
        )
        if self.device == "cpu":
            kwargs["cpu_threads"] = self.cpu_threads

        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.model_size, **kwargs)
        print("  [faster-whisper] Model loaded.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file and return the normalized text."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        segments, info = self._model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=False,  # disable VAD for fair comparison with WhisperX
        )

        text = " ".join(seg.text.strip() for seg in segments)
        return self._normalize(text)

    def unload_model(self):
        """Free GPU memory."""
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [faster-whisper] Model unloaded.")

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s'\-]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text