"""
visualize_results.py
--------------------
Reads benchmark_summary.json and generates comparison plots:
  1. Bar chart: Macro / Micro / Weighted F1 per model
  2. Heatmap: Per-class F1 for each model
  3. Latency distribution (box plot) from per-model CSVs

Usage:
    python visualize_results.py --results results/
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import EMOTION_LABELS


def load_summary(results_dir: Path) -> dict:
    path = results_dir / "benchmark_summary.json"
    with open(path) as f:
        return json.load(f)


def load_latencies(results_dir: Path) -> dict[str, list[float]]:
    """Load per-sample latencies from per-model CSVs."""
    latencies = {}
    for csv_path in results_dir.glob("*_predictions.csv"):
        model_tag = csv_path.stem.replace("_predictions", "").replace("_", ":")
        vals = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    vals.append(float(row["latency_ms"]))
                except (KeyError, ValueError):
                    pass
        if vals:
            latencies[model_tag] = vals
    return latencies


def plot_overall_metrics(summary: dict, out_dir: Path):
    models = list(summary["models"].keys())
    metric_keys = ["macro_f1", "micro_f1", "weighted_f1", "exact_match_ratio"]
    metric_labels = ["Macro F1", "Micro F1", "Weighted F1", "Exact Match"]

    x = np.arange(len(metric_keys))
    width = 0.35
    offsets = np.linspace(-width / 2 * (len(models) - 1), width / 2 * (len(models) - 1), len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(models))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [summary["models"][model].get(k, 0) for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width * 0.9, label=model, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Emotion Classification Metrics (GoEmotions)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "overall_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_class_heatmap(summary: dict, out_dir: Path):
    models = list(summary["models"].keys())
    labels = EMOTION_LABELS

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 10), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        per_class = summary["models"][model].get("per_class", {})
        f1_vals = np.array([per_class.get(lbl, {}).get("f1", 0.0) for lbl in labels]).reshape(-1, 1)

        sns.heatmap(
            f1_vals,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            cbar=(ax == axes[-1]),
            yticklabels=labels,
            xticklabels=[model],
        )
        ax.set_title(f"{model}\nPer-Class F1", fontsize=11)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Per-Class F1 Score by Model", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "per_class_f1_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_latency_boxplot(latencies: dict[str, list[float]], out_dir: Path):
    if not latencies:
        print("No latency data found; skipping latency plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    data = list(latencies.values())
    labels = list(latencies.keys())

    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colors = sns.color_palette("Set2", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Distribution per Model")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "latency_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--results", type=str, default="results",
                        help="Directory containing benchmark outputs")
    args = parser.parse_args()

    results_dir = Path(args.results)
    summary = load_summary(results_dir)
    latencies = load_latencies(results_dir)

    plot_overall_metrics(summary, results_dir)
    plot_per_class_heatmap(summary, results_dir)
    plot_latency_boxplot(latencies, results_dir)

    print("\nAll plots saved to:", results_dir)


if __name__ == "__main__":
    main()