"""
Mozilla Common Voice loader via the `datacollective` Python package.

Setup (one-time):
    pip install datacollective
    export MDC_API_KEY=your_api_key_here   # from mozilladatacollective.com → Create Credentials

Usage:
    loader = CommonVoiceLoader(split="test", max_samples=200)
    samples = loader.load()

The dataset ID or slug can be overridden via:
    export MDC_DATASET_ID=cmndapwry02jnmh07dyo46mot
    # or the slug: common-voice-scripted-speech-25-0-english-...
"""

import csv
import os
import re
import tarfile
from pathlib import Path
from typing import Optional

import torchaudio


MDC_API_KEY_ENV    = "MDC_API_KEY"
MDC_DATASET_ID_ENV = "MDC_DATASET_ID"

# English CV 25.0 — ID and slug are interchangeable per the MDC docs
DEFAULT_DATASET_ID = "cmndapwry02jnmh07dyo46mot"


class CommonVoiceLoader:
    """
    Loads Common Voice samples using the `datacollective` package.
    Downloads the dataset once, caches locally, then reads from cache.
    """

    VALID_SPLITS = {"train", "dev", "test", "validated", "other", "invalidated"}

    def __init__(
        self,
        split: str = "test",
        max_samples: Optional[int] = 200,
        sample_rate: int = 16000,
        data_dir: str = "./data/CommonVoice",
        cv_dir: Optional[str] = None,       # skip download; use local folder
        dataset_id: Optional[str] = None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from: {self.VALID_SPLITS}")

        self.split       = split
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.data_dir    = Path(data_dir)
        self.cv_dir      = Path(cv_dir) if cv_dir else None
        self.dataset_id  = (
            dataset_id
            or os.environ.get(MDC_DATASET_ID_ENV, DEFAULT_DATASET_ID)
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def load(self) -> list[dict]:
        # If caller pointed at a local folder, skip the download entirely
        if self.cv_dir and self.cv_dir.exists():
            print(f"Loading Common Voice from local folder: {self.cv_dir}")
            return self._load_local(self.cv_dir)

        extract_dir = self.data_dir / f"cv_{self.dataset_id[:12]}"

        if not (extract_dir / "clips").exists() and not (extract_dir / "en" / "clips").exists():
            self._download(extract_dir)
        else:
            print(f"  Using cached data: {extract_dir}")

        return self._load_local(extract_dir)

    # ------------------------------------------------------------------
    def _download(self, extract_dir: Path):
        """Download via the MDC REST API (presigned URL)."""
        import requests

        api_key = os.environ.get(MDC_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"MDC API key not set.\n"
                f"Export it with: export {MDC_API_KEY_ENV}=your_key_here\n"
                f"Get a key at: https://mozilladatacollective.com → Create Credentials"
            )

        print(f"Requesting download URL for dataset '{self.dataset_id}' ...")
        resp = requests.post(
            f"https://mozilladatacollective.com/api/datasets/{self.dataset_id}/download",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"API key rejected (HTTP {resp.status_code}).\n"
                f"Check your key at: https://mozilladatacollective.com → Create Credentials"
            )
        resp.raise_for_status()

        download_url = resp.json().get("downloadUrl")
        if not download_url:
            raise RuntimeError(f"No downloadUrl in API response: {resp.text[:200]}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        archive_path = extract_dir / "corpus.tar.gz"

        print(f"Downloading archive → {archive_path}")
        with requests.get(download_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r  {downloaded/1e6:.0f} / {total/1e6:.0f} MB", end="", flush=True)
        print()

        print(f"Extracting ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        archive_path.unlink()
        print(f"  Done → {extract_dir}")

    # ------------------------------------------------------------------
    def _load_local(self, root: Path) -> list[dict]:
        clips_dir = self._find_clips_dir(root)
        tsv_path  = self._find_tsv(root)

        wav_cache = self.data_dir / "_wav_cache" / self.split
        wav_cache.mkdir(parents=True, exist_ok=True)

        print(f"  TSV    : {tsv_path}")
        print(f"  Clips  : {clips_dir}")
        if self.max_samples:
            print(f"  Samples: up to {self.max_samples}")

        samples, skipped = [], 0

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if self.max_samples and len(samples) >= self.max_samples:
                    break

                clip = row.get("path", "")
                if not clip.endswith(".mp3"):
                    clip += ".mp3"
                mp3_path = clips_dir / clip
                if not mp3_path.exists():
                    skipped += 1
                    continue

                transcript = self._normalize(row.get("sentence", ""))
                if not transcript:
                    skipped += 1
                    continue

                wav_path = wav_cache / (mp3_path.stem + ".wav")
                if not wav_path.exists():
                    try:
                        wav, sr = torchaudio.load(str(mp3_path))
                        if sr != self.sample_rate:
                            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
                            sr = self.sample_rate
                        if wav.shape[0] > 1:
                            wav = wav.mean(dim=0, keepdim=True)
                        torchaudio.save(str(wav_path), wav, sr)
                    except Exception:
                        skipped += 1
                        continue
                else:
                    wav, sr = torchaudio.load(str(wav_path))

                duration_s = wav.shape[-1] / sr
                if not (0.5 <= duration_s <= 30.0):
                    skipped += 1
                    continue

                samples.append({
                    "id":          mp3_path.stem,
                    "audio_path":  str(wav_path),
                    "transcript":  transcript,
                    "duration_s":  duration_s,
                    "sample_rate": sr,
                    "client_id":   row.get("client_id", ""),
                    "age":         row.get("age", ""),
                    "gender":      row.get("gender", ""),
                    "accent":      row.get("accents", ""),
                })

        if skipped:
            print(f"  (skipped {skipped}: missing audio / empty transcript / out-of-range duration)")
        print(f"  → {len(samples)} samples ready.")
        return samples

    # ------------------------------------------------------------------
    def _find_clips_dir(self, root: Path) -> Path:
        for c in [root / "clips", root / "en" / "clips"]:
            if c.exists():
                return c
        raise FileNotFoundError(f"No clips/ directory found under {root}")

    def _find_tsv(self, root: Path) -> Path:
        for base in [root, root / "en"]:
            p = base / f"{self.split}.tsv"
            if p.exists():
                return p
        for base in [root, root / "en"]:
            p = base / "validated.tsv"
            if p.exists():
                print(f"  [WARNING] '{self.split}.tsv' not found; falling back to '{p}'")
                return p
        raise FileNotFoundError(
            f"No TSV for split '{self.split}' under {root}.\n"
            f"Found: {list(root.rglob('*.tsv'))}"
        )

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s'\-]", "", text)
        return re.sub(r"\s+", " ", text)