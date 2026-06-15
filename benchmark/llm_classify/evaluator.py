"""
evaluator.py
------------
Computes classification metrics for benchmark results.

Works for both:
  - Multi-label datasets (GoEmotions, 28 classes): metrics computed on
    binary indicator vectors over the full label set.
  - Single-label datasets (ISEAR, DailyDialog, 7-class Ekman set): metrics
    are still computed on binary indicator vectors (each sample has exactly
    one "1"), which is the standard reduction for evaluating single-label
    classification with multi-label-style metric functions, and additionally
    a plain single-label accuracy is reported.

Metrics computed:
- Macro / Micro / Weighted F1
- Exact Match Ratio (Subset Accuracy)
- Hamming Loss
- Accuracy (single-label datasets only)
- Per-class Precision, Recall, F1
- Mean / P95 inference latency
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
from config import GOEMOTIONS_LABELS

log = logging.getLogger(__name__)


def _to_binary_vector(label_list: list[str], all_labels: list[str], label_index: dict) -> list[int]:
    """Convert a list of label strings to a binary indicator vector."""
    vec = [0] * len(all_labels)
    for lbl in label_list:
        lbl = lbl.strip().lower()
        if lbl in label_index:
            vec[label_index[lbl]] = 1
    if sum(vec) == 0 and "neutral" in label_index:
        vec[label_index["neutral"]] = 1
    return vec


def compute_metrics(results: list[dict], model_label: str = "",
                     all_labels: list[str] | None = None,
                     single_label: bool = False) -> dict:
    """
    Compute evaluation metrics from per-sample result rows.

    Args:
        results:      List of dicts with keys 'true_labels', 'predicted_labels',
                       and 'latency_ms'.
        model_label:  Display name used in logs.
        all_labels:   The full label set for this dataset/run. Defaults to
                       GOEMOTIONS_LABELS for backwards compatibility.
        single_label: If True, also compute a plain top-1 accuracy metric
                       (appropriate for ISEAR / DailyDialog).

    Returns:
        Dict of metric name -> value, including a 'per_class' sub-dict.
    """
    if not results:
        log.warning("No results provided to evaluator.")
        return {}

    if all_labels is None:
        all_labels = GOEMOTIONS_LABELS

    label_index = {lbl: i for i, lbl in enumerate(all_labels)}

    y_true, y_pred = [], []
    top1_correct = 0

    for row in results:
        true = row["true_labels"].split("|") if isinstance(row["true_labels"], str) else row["true_labels"]
        pred = row["predicted_labels"].split("|") if isinstance(row["predicted_labels"], str) else row["predicted_labels"]

        y_true.append(_to_binary_vector(true, all_labels, label_index))
        y_pred.append(_to_binary_vector(pred, all_labels, label_index))

        if single_label:
            true_lbl = true[0].strip().lower() if true else "neutral"
            pred_lbl = pred[0].strip().lower() if pred else "neutral"
            if true_lbl == pred_lbl:
                top1_correct += 1

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

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    per_class = {}
    for i, lbl in enumerate(all_labels):
        per_class[lbl] = {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1":        round(float(f1[i]), 4),
            "support":   int(support[i]),
        }

    log.info(classification_report(
        y_true, y_pred,
        target_names=all_labels,
        zero_division=0
    ))

    metrics = {
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

    if single_label:
        top1_accuracy = round(top1_correct / len(results), 4)
        metrics["top1_accuracy"] = top1_accuracy
        log.info(f"  top1_accuracy: {top1_accuracy}")

    return metrics