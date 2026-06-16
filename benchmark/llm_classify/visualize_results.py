"""
visualize_results.py
--------------------
Reads a benchmark_summary_<dataset>_<mode>.json file and generates
comparison plots:
  1. Bar chart: Macro / Micro / Weighted F1 (+ Top-1 Accuracy for
     single-label datasets) per model
  2. Heatmap: Per-class F1 for each model, using the label set recorded
     in the summary file (28-class GoEmotions or 7-class Ekman)
  3. Latency distribution (box plot) from per-model CSVs

Usage:
    python visualize_results.py --summary results/benchmark_summary_isear_zero_shot.json
    python visualize_results.py --dataset isear --mode zero_shot --results results/
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def load_summary(summary_path: Path) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def load_latencies(results_dir: Path, dataset: str, mode: str) -> dict[str, list[float]]:
    """Load per-sample latencies from per-model CSVs matching dataset/mode."""
    latencies = {}
    pattern = f"*_{dataset}_*_predictions.csv" if mode == "both" else f"*_{dataset}_{mode}_predictions.csv"

    for csv_path in results_dir.glob(pattern):
        # filename: <model>_<dataset>_<mode>_predictions.csv
        stem = csv_path.stem.replace("_predictions", "")
        # strip dataset and mode suffix to recover model + mode label
        label = stem.replace(f"_{dataset}_", " [") + "]"
        label = label.replace("_", ":", 1)  # restore model tag's colon (first underscore only)

        vals = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    vals.append(float(row["latency_ms"]))
                except (KeyError, ValueError):
                    pass
        if vals:
            latencies[label] = vals
    return latencies


def plot_overall_metrics(summary: dict, out_dir: Path, tag: str):
    models = list(summary["models"].keys())
    single_label = not summary.get("multi_label", True)

    metric_keys = ["macro_f1", "micro_f1", "weighted_f1", "exact_match_ratio"]
    metric_labels = ["Macro F1", "Micro F1", "Weighted F1", "Exact Match"]
    if single_label:
        metric_keys.append("top1_accuracy")
        metric_labels.append("Top-1 Acc")

    x = np.arange(len(metric_keys))
    width = 0.35
    offsets = np.linspace(-width / 2 * (len(models) - 1), width / 2 * (len(models) - 1), len(models))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(models))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [summary["models"][model].get(k, 0) or 0 for k in metric_keys]
        bars = ax.bar(x + offsets[i], vals, width * 0.9, label=model, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison: Emotion Classification Metrics ({summary['dataset_display_name']})")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"overall_metrics_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_class_heatmap(summary: dict, out_dir: Path, tag: str):
    models = list(summary["models"].keys())
    labels = summary["label_set"]

    n_models = len(models)
    height = max(6, 0.35 * len(labels))
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, height), sharey=True)
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
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"Per-Class F1 Score by Model ({summary['dataset_display_name']})", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / f"per_class_f1_heatmap_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_latency_boxplot(latencies: dict[str, list[float]], out_dir: Path, tag: str, dataset_display_name: str):
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

    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Inference Latency Distribution per Model ({dataset_display_name})")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"latency_boxplot_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--results", type=str, default="results",
                        help="Directory containing benchmark outputs")
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to a specific benchmark_summary_<dataset>_<mode>.json file")
    parser.add_argument("--dataset", type=str, default="goemotions",
                        help="Dataset name (used if --summary not given)")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        help="Prompt mode: zero_shot | few_shot | both (used if --summary not given)")
    args = parser.parse_args()

    results_dir = Path(args.results)

    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = results_dir / f"benchmark_summary_{args.dataset}_{args.mode}.json"

    summary = load_summary(summary_path)
    dataset = summary["dataset"]
    mode = summary["mode"]
    tag = f"{dataset}_{mode}"

    latencies = load_latencies(results_dir, dataset, mode)

    plot_overall_metrics(summary, results_dir, tag)
    plot_per_class_heatmap(summary, results_dir, tag)
    plot_latency_boxplot(latencies, results_dir, tag, summary["dataset_display_name"])

    print("\nAll plots saved to:", results_dir)


if __name__ == "__main__":
    main()