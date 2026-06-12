"""
Speech Accent Archive loader via Hugging Face streaming.
No local download required — streams on-the-fly and caches WAV files.

Dataset: ylacombe/speech-accent-archive
  - ~2,140 speakers reading the same English elicitation paragraph
  - Wide accent coverage: 177+ native languages represented
  - 16 kHz audio, Parquet-based on HF (no loading script)
  - CC BY-NC-SA 4.0

The elicitation text is fixed for every speaker:
  "Please call Stella. Ask her to bring these things with her from the store:
   Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe
   a snack for her brother Bob. We also need a small plastic snake and a big
   toy frog for the kids. She can scoop these things into three red bags, and
   we will go meet her Wednesday at the train station."

Setup:
    huggingface-cli login   # one-time, or set HF_TOKEN env var

Usage:
    loader = SpeechAccentArchiveLoader(max_samples=200)
    samples = loader.load()

Each returned sample dict:
    {id, audio_path, transcript, duration_s, sample_rate,
     speaker_id, native_language, age, sex, english_residence,
     age_of_english_onset}
"""

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


HF_DATASET_ID = "ylacombe/speech-accent-archive"

# The single elicitation paragraph every speaker reads — normalized form.
ELICITATION_TEXT = (
    "please call stella ask her to bring these things with her from the store "
    "six spoons of fresh snow peas five thick slabs of blue cheese and maybe "
    "a snack for her brother bob we also need a small plastic snake and a big "
    "toy frog for the kids she can scoop these things into three red bags and "
    "we will go meet her wednesday at the train station"
)


class SpeechAccentArchiveLoader:
    """
    Streams the Speech Accent Archive from HuggingFace and caches as WAV files.
    Supports optional filtering by native language for targeted accent benchmarking.

    Because every speaker reads the same elicitation paragraph, the reference
    transcript is identical for all samples — making WER directly reflect
    accent-induced recognition difficulty.
    """

    # Dataset has a single HF split
    VALID_SPLITS = {"train"}

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = 200,
        sample_rate: int = 16000,
        cache_dir: str = "./data/SpeechAccentArchive/_wav_cache",
        native_language_filter: Optional[list[str]] = None,
        dataset_id: str = HF_DATASET_ID,
    ):
        """
        Parameters
        ----------
        split : str
            HF split to load (only "train" exists in the dataset).
        max_samples : int | None
            Maximum number of samples to return. None = no limit.
        sample_rate : int
            Target sample rate; audio is resampled if necessary.
        cache_dir : str
            Directory for cached WAV files.
        native_language_filter : list[str] | None
            If provided, keep only speakers whose native_language matches one of
            these strings (case-insensitive). E.g. ["english", "mandarin"].
        dataset_id : str
            HuggingFace dataset identifier.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Only 'train' exists.")
        self.split = split
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir)
        self.native_language_filter = (
            [l.lower() for l in native_language_filter]
            if native_language_filter
            else None
        )
        self.dataset_id = dataset_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def load(self) -> list[dict]:
        try:
            from datasets import load_dataset, Audio
        except ImportError:
            raise ImportError("Install via: pip install 'datasets==2.19.2'")

        import os

        token = (
            os.environ.get("HF_TOKEN", "").strip()
            or self._read_cached_token()
        )

        print(f"Streaming Speech Accent Archive from HuggingFace ...")
        print(f"  Dataset : {self.dataset_id}")
        if self.native_language_filter:
            print(f"  Language filter: {self.native_language_filter}")
        if self.max_samples:
            print(f"  Samples : up to {self.max_samples}")

        try:
            ds = load_dataset(
                self.dataset_id,
                split=self.split,
                streaming=True,
                token=token or None,
            )
            # Disable HF audio decoding — we decode ourselves with soundfile
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {self.dataset_id}.\n"
                f"Try: huggingface-cli login\n"
                f"Original error: {e}"
            )

        samples, skipped = [], 0

        for i, row in enumerate(ds):
            if self.max_samples and len(samples) >= self.max_samples:
                break

            native_language = (row.get("native_language") or "").lower().strip()

            if self.native_language_filter and native_language not in self.native_language_filter:
                continue

            speaker_id = str(row.get("speakerid") or row.get("speaker_id") or f"spk_{i:05d}")
            sample_id = f"saa_{speaker_id}"
            wav_path = self.cache_dir / f"{sample_id}.wav"

            if not wav_path.exists():
                audio = row.get("audio")
                if audio is None:
                    skipped += 1
                    continue
                raw_bytes = audio.get("bytes")
                if not raw_bytes:
                    skipped += 1
                    continue
                try:
                    array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
                except Exception:
                    skipped += 1
                    continue
                if array.ndim > 1:
                    array = array.mean(axis=1)
                if sr != self.sample_rate:
                    array = self._resample(array, sr, self.sample_rate)
                sf.write(str(wav_path), array, self.sample_rate)

            try:
                info = sf.info(str(wav_path))
            except Exception:
                skipped += 1
                continue

            if not (0.5 <= info.duration <= 60.0):
                skipped += 1
                continue

            samples.append({
                "id":                    sample_id,
                "audio_path":            str(wav_path),
                "transcript":            ELICITATION_TEXT,
                "duration_s":            info.duration,
                "sample_rate":           self.sample_rate,
                "speaker_id":            speaker_id,
                "native_language":       native_language,
                "age":                   row.get("age", ""),
                "sex":                   row.get("sex", ""),
                "english_residence":     row.get("english_residence", ""),
                "age_of_english_onset":  row.get("age_of_english_onset", ""),
            })

            if len(samples) % 20 == 0 and len(samples) > 0:
                print(f"  Streamed {len(samples)} samples ...", end="\r")

        print()
        if skipped:
            print(f"  (skipped {skipped}: missing audio / out-of-range duration)")

        lang_counts: dict[str, int] = {}
        for s in samples:
            lang = s["native_language"] or "unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        top = sorted(lang_counts.items(), key=lambda x: -x[1])[:8]
        breakdown = ", ".join(f"{l}={n}" for l, n in top)
        if len(lang_counts) > 8:
            breakdown += f", ... ({len(lang_counts)} total)"
        print(f"  → {len(samples)} samples ready ({breakdown})")
        return samples

    # ------------------------------------------------------------------
    @staticmethod
    def _read_cached_token() -> str:
        p = Path.home() / ".cache" / "huggingface" / "token"
        return p.read_text().strip() if p.exists() else ""

    @staticmethod
    def _resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import torch
            import torchaudio
            t = torch.from_numpy(array).unsqueeze(0)
            t = torchaudio.functional.resample(t, orig_sr, target_sr)
            return t.squeeze(0).numpy()
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(array, target_sr // g, orig_sr // g).astype(np.float32)
