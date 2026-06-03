"""
evaluator.py
------------
Computes multi-label classification metrics for benchmark results.

Metrics computed:
- Macro / Micro / Weighted F1
- Exact Match Ratio (Subset Accuracy)
- Hamming Loss
- Per-class Precision, Recall, F1
- Mean inference latency
"""

import logging
import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from config import EMOTION_LABELS

log = logging.getLogger(__name__)

LABEL_INDEX = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}
N_LABELS = len(EMOTION_LABELS)


def _to_binary_vector(label_list: list[str]) -> list[int]:
    """Convert a list of label strings to a binary indicator vector."""
    vec = [0] * N_LABELS
    for lbl in label_list:
        lbl = lbl.strip().lower()
        if lbl in LABEL_INDEX:
            vec[LABEL_INDEX[lbl]] = 1
    # If nothing matched, mark neutral
    if sum(vec) == 0:
        vec[LABEL_INDEX["neutral"]] = 1
    return vec


def compute_metrics(results: list[dict], model_label: str = "") -> dict:
    """
    Compute evaluation metrics from per-sample result rows.

    Args:
        results:     List of dicts with keys 'true_labels', 'predicted_labels',
                     and 'latency_ms'.
        model_label: Display name used in logs.

    Returns:
        Dict of metric name → value, including a 'per_class' sub-dict.
    """
    if not results:
        log.warning("No results provided to evaluator.")
        return {}

    y_true, y_pred = [], []

    for row in results:
        true = row["true_labels"].split("|") if isinstance(row["true_labels"], str) else row["true_labels"]
        pred = row["predicted_labels"].split("|") if isinstance(row["predicted_labels"], str) else row["predicted_labels"]
        y_true.append(_to_binary_vector(true))
        y_pred.append(_to_binary_vector(pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_f1    = round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4)
    micro_f1    = round(float(f1_score(y_true, y_pred, average="micro",    zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    h_loss      = round(float(hamming_loss(y_true, y_pred)), 4)
    exact_match = round(float(accuracy_score(y_true, y_pred)), 4)

    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), (int, float))]
    mean_latency = round(float(np.mean(latencies)), 2) if latencies else None
    p95_latency  = round(float(np.percentile(latencies, 95)), 2) if latencies else None

    # Per-class breakdown
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    per_class = {}
    for i, lbl in enumerate(EMOTION_LABELS):
        per_class[lbl] = {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1":        round(float(f1[i]), 4),
            "support":   int(support[i]),
        }

    log.info(classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        zero_division=0
    ))

    return {
        "model": model_label,
        "n_samples": len(results),
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "exact_match_ratio": exact_match,
        "hamming_loss": h_loss,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "per_class": per_class,
    }