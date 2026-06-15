"""RAF-DB dataset loading utilities (kagglehub: shuvoalok/raf-db-dataset)."""

import os
import pandas as pd
from .config import RAFDB_CLASS_TO_LABEL


def download_rafdb() -> str:
    """Download (or reuse cached) RAF-DB dataset via kagglehub. Returns root path."""
    import kagglehub
    path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    print(f"RAF-DB dataset path: {path}")
    return path


def load_rafdb_index(rafdb_root: str, split: str = "test") -> pd.DataFrame:
    candidates = [
        os.path.join(rafdb_root, "DATASET", split),
        os.path.join(rafdb_root, split),
    ]
    split_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if split_dir is None:
        raise FileNotFoundError(
            f"Could not find split '{split}' under {rafdb_root}. Tried: {candidates}"
        )

    rows = []
    for class_folder in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        try:
            class_id = int(class_folder)
        except ValueError:
            continue

        label = RAFDB_CLASS_TO_LABEL.get(class_id)
        if label is None:
            continue

        for fname in sorted(os.listdir(class_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rows.append({
                "image_path": os.path.join(class_path, fname),
                "true_label": label,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under {split_dir}. Check dataset structure.")
    return df
