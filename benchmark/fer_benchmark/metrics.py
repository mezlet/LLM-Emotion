"""
fer_benchmark/metrics.py
Accuracy, F1, Cohen's kappa, per-class breakdown.
"""
from typing import List, Dict

def compute_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    labels = sorted(set(true_labels) | set(pred_labels) - {"unknown"})
    return {
        "accuracy":    round(accuracy_score(true_labels, pred_labels), 4),
        "f1_macro":    round(f1_score(true_labels, pred_labels, average="macro",
                                      labels=labels, zero_division=0), 4),
        "f1_weighted": round(f1_score(true_labels, pred_labels, average="weighted",
                                      labels=labels, zero_division=0), 4),
        "kappa":       round(cohen_kappa_score(true_labels, pred_labels), 4),
    }

def compute_per_class_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict:
    from sklearn.metrics import classification_report
    labels = sorted(set(true_labels) - {"unknown"})
    report = classification_report(true_labels, pred_labels,
                                   labels=labels, output_dict=True, zero_division=0)
    return {k: {m: round(v, 4) for m, v in val.items()}
            for k, val in report.items() if isinstance(val, dict)}
