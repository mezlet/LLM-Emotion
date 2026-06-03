"""
L2-ARCTIC accented English loader via Hugging Face streaming.
No local download required — streams on-the-fly and caches WAV files.

Dataset: KoelLabs/L2Arctic
  - 24 non-native English speakers
  - 6 L1 backgrounds: Arabic, Hindi, Korean, Mandarin, Spanish, Vietnamese
  - 3599 scripted samples (~220 min) + 22 spontaneous samples (~26 min)
  - Pre-converted to 16 kHz float32, Parquet-based (no loading script)
  - CC BY-NC 4.0, auto-approved on HF (no manual terms needed)

Setup:
    huggingface-cli login   # one-time, or set HF_TOKEN env var

Usage:
    loader = L2ArcticLoader(split="scripted", max_samples=200)
    samples = loader.load()

Each returned sample dict:
    {id, audio_path, transcript, duration_s, sample_rate,
     speaker_id, native_language, gender}
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


HF_DATASET_ID = "KoelLabs/L2Arctic"

# L1 background → speaker IDs mapping
L1_SPEAKERS = {
    "arabic":     ["ABA", "YBAA", "ZHAA", "SKA"],
    "hindi":      ["HQTV", "SVBI", "TNI", "NCC"],
    "korean":     ["HJK", "HKK", "YDCK", "YKWK"],
    "mandarin":   ["BWC", "LXC", "NJS", "TXHC"],
    "spanish":    ["EBVS", "ERMS", "MBMPS", "PNV"],
    "vietnamese": ["THV", "TLV", "RRBI", "ASI"],
}


class L2ArcticLoader:
    """
    Streams L2-ARCTIC from HuggingFace and caches as WAV files.
    Supports filtering by L1 background for targeted accent benchmarking.
    """

    VALID_SPLITS = {"scripted", "spontaneous"}

    def __init__(
        self,
        split: str = "scripted",
        max_samples: Optional[int] = 200,
        sample_rate: int = 16000,
        cache_dir: str = "./data/L2Arctic/_wav_cache",
        l1_filter: Optional[list[str]] = None,   # e.g. ["arabic", "mandarin"]
        dataset_id: str = HF_DATASET_ID,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from: {self.VALID_SPLITS}")
        self.split      = split
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.cache_dir  = Path(cache_dir) / split
        self.l1_filter  = [l.lower() for l in l1_filter] if l1_filter else None
        self.dataset_id = dataset_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def load(self) -> list[dict]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install via: pip install datasets")

        import os
        token = (
            os.environ.get("HF_TOKEN", "").strip()
            or self._read_cached_token()
        )

        print(f"Streaming L2-ARCTIC '{self.split}' from HuggingFace ...")
        print(f"  Dataset : {self.dataset_id}")
        if self.l1_filter:
            print(f"  L1 filter: {self.l1_filter}")
        if self.max_samples:
            print(f"  Samples : up to {self.max_samples}")

        try:
            import datasets as hf_datasets
            from datasets import Audio
            # Load WITHOUT decoding audio — get raw bytes only
            ds = load_dataset(
                self.dataset_id,
                split=self.split,
                streaming=True,
                token=token or None,
            )
            # Disable audio decoding entirely — we'll decode ourselves with soundfile
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

            speaker_id = row.get("speaker_id", "")
            l1 = self._speaker_to_l1(speaker_id)

            if self.l1_filter and l1 not in self.l1_filter:
                continue

            transcript = self._normalize(row.get("text", "") or row.get("transcript", ""))
            if not transcript:
                skipped += 1
                continue

            sample_id = row.get("id", f"l2arctic_{self.split}_{i:05d}")
            wav_path  = self.cache_dir / f"{sample_id}.wav"

            if not wav_path.exists():
                audio = row.get("audio")
                if audio is None:
                    skipped += 1
                    continue
                # audio["bytes"] contains raw file bytes; decode with soundfile
                import io
                raw_bytes = audio.get("bytes")
                if not raw_bytes:
                    skipped += 1
                    continue
                try:
                    array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
                except Exception:
                    skipped += 1
                    continue
                if sr != self.sample_rate:
                    array = self._resample(array, sr, self.sample_rate)
                sf.write(str(wav_path), array, self.sample_rate)

            info = sf.info(str(wav_path))
            if not (0.3 <= info.duration <= 30.0):
                skipped += 1
                continue

            samples.append({
                "id":              sample_id,
                "audio_path":      str(wav_path),
                "transcript":      transcript,
                "duration_s":      info.duration,
                "sample_rate":     self.sample_rate,
                "speaker_id":      speaker_id,
                "native_language": l1,
                "gender":          row.get("gender", ""),
            })

            if len(samples) % 20 == 0 and len(samples) > 0:
                print(f"  Streamed {len(samples)} samples ...", end="\r")

        print()
        if skipped:
            print(f"  (skipped {skipped}: missing audio / empty transcript / out-of-range duration)")

        l1_counts = {}
        for s in samples:
            l1_counts[s["native_language"]] = l1_counts.get(s["native_language"], 0) + 1
        breakdown = ", ".join(f"{l}={n}" for l, n in sorted(l1_counts.items()))
        print(f"  → {len(samples)} samples ready ({breakdown})")
        return samples

    # ------------------------------------------------------------------
    def _speaker_to_l1(self, speaker_id: str) -> str:
        for l1, speakers in L1_SPEAKERS.items():
            if speaker_id.upper() in speakers:
                return l1
        return "unknown"

    @staticmethod
    def _read_cached_token() -> str:
        p = Path.home() / ".cache" / "huggingface" / "token"
        return p.read_text().strip() if p.exists() else ""

    @staticmethod
    def _resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import torch, torchaudio
            t = torch.from_numpy(array).unsqueeze(0)
            t = torchaudio.functional.resample(t, orig_sr, target_sr)
            return t.squeeze(0).numpy()
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(array, target_sr // g, orig_sr // g).astype(np.float32)

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s'\-]", "", text)
        return re.sub(r"\s+", " ", text)