"""
LibriSpeech dataset loader using torchaudio.
Downloads and parses audio + transcripts for the requested split.
"""

import os
import re
from pathlib import Path
from typing import Optional

import torchaudio
import soundfile as sf


class LibriSpeechLoader:
    """
    Loads audio samples and transcripts from a LibriSpeech split.
    Uses torchaudio's built-in LibriSpeech dataset class.
    """

    VALID_SPLITS = {
        "train-clean-100", "train-clean-360", "train-other-500",
        "dev-clean", "dev-other",
        "test-clean", "test-other",
    }

    def __init__(
        self,
        root: str = "./data",
        split: str = "test-clean",
        max_samples: Optional[int] = None,
        download: bool = True,
        sample_rate: int = 16000,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from: {self.VALID_SPLITS}")
        self.root = Path(root)
        self.split = split
        self.max_samples = max_samples
        self.download = download
        self.sample_rate = sample_rate

    def load(self) -> list[dict]:
        """
        Returns a list of dicts:
          {id, audio_path, transcript, duration_s, sample_rate}
        """
        print(f"Loading LibriSpeech '{self.split}' from {self.root} ...")
        self.root.mkdir(parents=True, exist_ok=True)

        try:
            dataset = torchaudio.datasets.LIBRISPEECH(
                root=str(self.root),
                url=self.split,
                download=self.download,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LibriSpeech dataset.\n"
                f"Ensure torchaudio >= 0.9.0 is installed and the dataset is accessible.\n"
                f"Original error: {e}"
            )

        samples = []
        for i, item in enumerate(dataset):
            if self.max_samples and i >= self.max_samples:
                break

            waveform, sr, transcript, speaker_id, chapter_id, utterance_id = item

            # Persist audio to a temp WAV so runners can load from path
            audio_dir = self.root / "_audio_cache" / self.split
            audio_dir.mkdir(parents=True, exist_ok=True)
            sample_id = f"{speaker_id}-{chapter_id}-{utterance_id:04d}"
            audio_path = audio_dir / f"{sample_id}.wav"

            if not audio_path.exists():
                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    waveform = resampler(waveform)
                    sr = self.sample_rate

                torchaudio.save(str(audio_path), waveform, sr)

            duration_s = waveform.shape[-1] / sr

            samples.append({
                "id": sample_id,
                "audio_path": str(audio_path),
                "transcript": self._normalize_transcript(transcript),
                "duration_s": duration_s,
                "sample_rate": sr,
            })

        print(f"  → {len(samples)} samples ready.")
        return samples

    @staticmethod
    def _normalize_transcript(text: str) -> str:
        """Lower-case, strip punctuation (LibriSpeech convention)."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s'\-]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text