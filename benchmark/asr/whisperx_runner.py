"""
WhisperX inference runner.
Wraps whisperx.load_model / transcribe for the benchmark.
"""

import gc
import importlib
import re

import torch


class WhisperXRunner:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 8,
        language: str = "en",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language
        self._model = None

    def _import_whisperx(self):
        """Use importlib to guarantee an absolute import, avoiding filename collisions."""
        try:
            return importlib.import_module("whisperx")
        except ModuleNotFoundError as e:
            raise ImportError(
                "whisperx is not installed. Install via: pip install whisperx"
            ) from e

    def load_model(self):
        whisperx = self._import_whisperx()
        print(f"  [WhisperX] Loading model '{self.model_size}' on {self.device} ({self.compute_type})...")
        self._model = whisperx.load_model(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        print("  [WhisperX] Model loaded.")

    def transcribe(self, audio_path: str) -> str:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        result = self._model.transcribe(
            audio_path,
            batch_size=self.batch_size,
            language=self.language,
        )
        segments = result.get("segments", [])
        text = " ".join(seg["text"].strip() for seg in segments)
        return self._normalize(text)

    def unload_model(self):
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [WhisperX] Model unloaded.")

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s'\-]", "", text)
        return re.sub(r"\s+", " ", text)
    