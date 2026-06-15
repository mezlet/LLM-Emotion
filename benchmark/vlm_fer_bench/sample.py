# ============================================================
# vlm_fer_bench/dataset.py
# ============================================================
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
    """
    Expects the kagglehub 'shuvoalok/raf-db-dataset' layout:

      rafdb_root/
        DATASET/
          train/
            1/  2/  3/  4/  5/  6/  7/   (class-numbered folders, images inside)
          test/
            1/  2/  3/  4/  5/  6/  7/

    Class folder numbers map to RAF-DB emotion labels via RAFDB_CLASS_TO_LABEL.

    Returns DataFrame with columns: image_path, true_label
    """
    # locate the split directory (handles both <root>/DATASET/<split> and <root>/<split>)
    candidates = [
        os.path.join(rafdb_root, "DATASET", split),
        os.path.join(rafdb_root, split),
    ]
    split_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if split_dir is None:
        raise FileNotFoundError(
            f"Could not find split '{split}' under {rafdb_root}. "
            f"Tried: {candidates}"
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


# ============================================================
# vlm_fer_bench/config.py  (only the changed/added parts)
# ============================================================
"""Configuration constants for VLM FER benchmarking."""

import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MODEL_NAMES = {
    "qwen2.5-vl": "qwen2.5vl:7b",
    "llama3.2-vision": "llama3.2-vision:11b",
    "llava": "llava:7b",
}

# RAF-DB class folder -> emotion label
# Per official RAF-DB EmoLabel definition (1-7), preserved in this Kaggle release
RAFDB_CLASS_TO_LABEL = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

EMOTION_SET = list(RAFDB_CLASS_TO_LABEL.values())

PROMPT_TEMPLATE = (
    "Look at the face in this image and identify the dominant emotion. "
    f"Choose exactly one word from this list: {', '.join(EMOTION_SET)}. "
    "Respond with only the single emotion word, nothing else."
)

NORMALIZATION_MAP = {
    "happy": "happiness",
    "happiness": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "scared": "fear",
    "disgust": "disgust",
    "disgusted": "disgust",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
    "calm": "neutral",
}

DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 2
DEFAULT_SAVE_EVERY = 50
DEFAULT_IMAGE_MAX_SIZE = 512


# ============================================================
# run_vlm_fer_benchmark.py  (entry point — updated to support kagglehub)
# ============================================================
"""
CLI entry point for VLM FER benchmarking on RAF-DB.

Usage:
    # auto-download via kagglehub
    python run_vlm_fer_benchmark.py --download --output_dir ./results

    # use an already-downloaded/local path
    python run_vlm_fer_benchmark.py --rafdb_root /path/to/dataset --output_dir ./results
    python run_vlm_fer_benchmark.py --rafdb_root /path/to/dataset --limit 100 --models qwen2.5-vl
"""

import os
import argparse
import pandas as pd

from vlm_fer_bench import (
    MODEL_NAMES,
    load_rafdb_index,
    run_benchmark,
    compute_metrics,
)
from vlm_fer_bench.dataset import download_rafdb


def main():
    parser = argparse.ArgumentParser(description="VLM FER Benchmark on RAF-DB")
    parser.add_argument("--rafdb_root", default=None,
                         help="Path to RAF-DB dataset root (omit if using --download)")
    parser.add_argument("--download", action="store_true",
                         help="Download RAF-DB via kagglehub (shuvoalok/raf-db-dataset)")
    parser.add_argument("--output_dir", default="./vlm_fer_results", help="Output directory")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for quick tests)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES.keys()),
                         choices=list(MODEL_NAMES.keys()))
    args = parser.parse_args()

    if args.download:
        rafdb_root = download_rafdb()
    elif args.rafdb_root:
        rafdb_root = args.rafdb_root
    else:
        parser.error("Provide --rafdb_root or pass --download")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading RAF-DB index...")
    df = load_rafdb_index(rafdb_root, split=args.split)
    print(f"Loaded {len(df)} images for split='{args.split}'")
    print(df["true_label"].value_counts())

    all_metrics = []
    for model_key in args.models:
        results_df = run_benchmark(model_key, df, args.output_dir, limit=args.limit)
        metrics = compute_metrics(results_df, model_key, args.output_dir)
        all_metrics.append(metrics)

    summary = pd.DataFrame([
        {
            "model": m["model"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "n_unknown": m["n_unknown_responses"],
            "avg_latency_sec": m["avg_latency_sec"],
        }
        for m in all_metrics
    ])
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()