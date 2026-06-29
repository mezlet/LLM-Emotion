"""
fer_benchmark/dataset.py
Dataset loader for RAF-DB, AffectNet, FER2013.
"""
import os, random, csv
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np

RAFDB_NUM_MAP = {
    "1": "surprise", "2": "fear",    "3": "disgust",
    "4": "happiness", "5": "sadness", "6": "anger", "7": "neutral",
}
FER2013_INT_MAP = {
    0: "anger", 1: "disgust", 2: "fear",
    3: "happiness", 4: "sadness", 5: "surprise", 6: "neutral",
}
SYNONYMS = {
    "happy": "happiness", "joy": "happiness", "sad": "sadness",
    "fearful": "fear", "angry": "anger", "disgusted": "disgust",
    "surprised": "surprise", "calm": "neutral",
}

def _normalise(label: str) -> str:
    label = label.lower().strip()
    return SYNONYMS.get(label, label)

def _load_imagefolder(root, label_map=None):
    samples = []
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir(): continue
        folder = class_dir.name
        label = label_map.get(folder) if label_map else _normalise(folder)
        if label is None: continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                samples.append((img_path, label))
    return samples

def _load_fer2013_csv(csv_path, split="PublicTest"):
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="fer2013_"))
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row.get("Usage", split) != split: continue
            label = FER2013_INT_MAP.get(int(row["emotion"]), "unknown")
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(pixels, mode="L").convert("RGB")
            img_path = tmp_dir / f"{split}_{i:05d}.png"
            img.save(img_path)
            samples.append((img_path, label))
    return samples


class DatasetLoader:
    def __init__(self, root="./data", max_samples=None, seed=42):
        self.root = Path(root)
        self.max_samples = max_samples
        self.seed = seed

    def _subsample(self, samples):
        if self.max_samples and len(samples) > self.max_samples:
            rng = random.Random(self.seed)
            samples = rng.sample(samples, self.max_samples)
        return samples

    def load(self, name: str):
        loaders = {
            "rafdb":     self._load_rafdb,
            "affectnet": self._load_affectnet,
            "fer2013":   self._load_fer2013,
        }
        if name not in loaders:
            raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}")
        samples = loaders[name]()
        samples = self._subsample(samples)
        print(f"  Loaded {len(samples)} samples from {name}")
        return samples

    def _load_rafdb(self):
        ds_root = self.root / "rafdb"
        if not ds_root.exists():
            raise FileNotFoundError(f"RAF-DB not found at {ds_root}")
        test_dir = ds_root / "test"
        search = test_dir if test_dir.exists() else ds_root
        subdirs = [d.name for d in search.iterdir() if d.is_dir()]
        label_map = RAFDB_NUM_MAP if all(d.isdigit() for d in subdirs) else None
        samples = _load_imagefolder(search, label_map)
        if not samples: raise FileNotFoundError(f"RAF-DB: no images found under {search}")
        return samples

    def _load_affectnet(self):
        ds_root = self.root / "affectnet"
        if not ds_root.exists():
            raise FileNotFoundError(f"AffectNet not found at {ds_root}")
        search = (ds_root / "val") if (ds_root / "val").exists() else ds_root
        samples = _load_imagefolder(search)
        if not samples: raise FileNotFoundError("AffectNet: no images found")
        return samples

    def _load_fer2013(self):
        ds_root = self.root / "fer2013"
        if not ds_root.exists():
            raise FileNotFoundError(f"FER2013 not found at {ds_root}")
        test_dir = ds_root / "test"
        if test_dir.exists():
            samples = _load_imagefolder(test_dir)
            if samples: return samples
        csv_path = ds_root / "fer2013.csv"
        if csv_path.exists():
            return _load_fer2013_csv(csv_path)
        raise FileNotFoundError("FER2013: no test/ folder or fer2013.csv found")
