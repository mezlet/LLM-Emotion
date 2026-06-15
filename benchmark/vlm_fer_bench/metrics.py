"""Metric computation and reporting."""

import os
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from .config import EMOTION_SET


def compute_metrics(results_df: pd.DataFrame, model_key: str, output_dir: str) -> dict:
    valid = results_df[results_df["prediction"] != "unknown"]
    n_unknown = len(results_df) - len(valid)

    y_true = valid["true_label"]
    y_pred = valid["prediction"]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, labels=EMOTION_SET, zero_division=0, output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_SET)
    cm_df = pd.DataFrame(cm, index=EMOTION_SET, columns=EMOTION_SET)
    cm_df.to_csv(os.path.join(output_dir, f"confusion_matrix_{model_key}.csv"))

    avg_latency = results_df["latency_sec"].dropna().mean()

    metrics = {
        "model": model_key,
        "n_samples": len(results_df),
        "n_unknown_responses": n_unknown,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "avg_latency_sec": avg_latency,
        "per_class_report": report,
    }

    with open(os.path.join(output_dir, f"metrics_{model_key}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n--- {model_key} ---")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {f1:.4f}")
    print(f"Macro Prec:    {precision:.4f}")
    print(f"Macro Recall:  {recall:.4f}")
    print(f"Unknown resp:  {n_unknown}/{len(results_df)}")
    print(f"Avg latency:   {avg_latency:.2f}s")

    return metrics
