"""
data_loader.py
--------------
Loads and samples the GoEmotions dataset via HuggingFace `datasets`.

GoEmotions 'raw' config: each emotion is its own binary column (not a 'labels' list).
This module reads whichever schema is present.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import logging
from datasets import load_dataset
from config import EMOTION_LABELS

log = logging.getLogger(__name__)

# GoEmotions official label ordering
GOEMOTIONS_LABEL_ORDER = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]


def load_goemotions_sample(n: int = 200, seed: int = 42, split: str = "train") -> list[dict]:
    """
    Download GoEmotions (raw config) and return `n` stratified samples.

    The 'raw' config exposes each emotion as its own binary column rather
    than a single 'labels' list — this function handles both schemas.

    Each returned sample dict:
        { "id": str, "text": str, "labels": list[str] }
    """
    log.info(f"Downloading GoEmotions ({split} split)...")
    dataset = load_dataset("go_emotions", "raw", split=split)
    log.info(f"  Raw split size: {len(dataset)}")

    # Detect schema: 'raw' config has one column per emotion label
    first_row = dataset[0]
    use_column_per_emotion = "labels" not in first_row

    if use_column_per_emotion:
        log.info("  Schema: per-column emotions (raw config)")
    else:
        log.info("  Schema: integer 'labels' list")

    samples = []
    for i, row in enumerate(dataset):
        if use_column_per_emotion:
            label_strings = [lbl for lbl in GOEMOTIONS_LABEL_ORDER if row.get(lbl, 0) == 1]
        else:
            label_indices = [j for j, v in enumerate(row["labels"]) if v == 1]
            label_strings = [GOEMOTIONS_LABEL_ORDER[j] for j in label_indices]

        if not label_strings:
            label_strings = ["neutral"]

        samples.append({
            "id": str(row.get("id", i)),
            "text": row["text"].strip(),
            "labels": label_strings,
        })

    random.seed(seed)
    if n >= len(samples):
        log.warning(f"Requested {n} samples but only {len(samples)} available; using all.")
        return samples

    sampled = _stratified_sample(samples, n, seed)
    log.info(f"  Sampled {len(sampled)} items")
    return sampled


def _stratified_sample(samples: list[dict], n: int, seed: int) -> list[dict]:
    """Pick roughly equal numbers per primary label."""
    from collections import defaultdict
    random.seed(seed)

    buckets = defaultdict(list)
    for s in samples:
        primary = s["labels"][0]
        buckets[primary].append(s)

    n_labels = len(buckets)
    per_label = max(1, n // n_labels)
    selected = []

    for label, items in buckets.items():
        random.shuffle(items)
        selected.extend(items[:per_label])

    remaining = [s for s in samples if s not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, n - len(selected))])

    random.shuffle(selected)
    return selected[:n]