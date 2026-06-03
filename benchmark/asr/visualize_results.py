"""
Visualize benchmark results from results/benchmark_results.json.
Generates WER distribution, RTF comparison, and latency scatter plots.

Usage:
    python visualize_results.py --results-dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_results(results_dir: Path):
    path = results_dir / "benchmark_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found at {path}. Run benchmark_asr.py first.")
    with open(path) as f:
        return json.load(f)


def plot_wer_distribution(models_data, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models_data:
        wers = [s["wer"] for s in model["sample_results"]]
        ax.hist(wers, bins=30, alpha=0.6, label=model["model"], edgecolor="white")

    ax.set_xlabel("Word Error Rate (WER)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("WER Distribution per Model", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "wer_distribution.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_rtf_comparison(models_data, output_dir: Path):
    model_names = [m["model"] for m in models_data]
    mean_rtfs = [m["mean_rtf"] for m in models_data]
    median_rtfs = [m["median_rtf"] for m in models_data]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, mean_rtfs, width, label="Mean RTF", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, median_rtfs, width, label="Median RTF", color="#dd8452")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Real-time threshold (RTF=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Real-Time Factor (RTF)", fontsize=12)
    ax.set_title("RTF Comparison (lower = faster)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    out = output_dir / "rtf_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_wer_vs_duration(models_data, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    for i, model in enumerate(models_data):
        durations = [s["audio_duration_s"] for s in model["sample_results"]]
        wers = [s["wer"] for s in model["sample_results"]]
        ax.scatter(durations, wers, alpha=0.5, s=20, label=model["model"], color=colors[i % len(colors)])

    ax.set_xlabel("Audio Duration (s)", fontsize=12)
    ax.set_ylabel("WER", fontsize=12)
    ax.set_title("WER vs. Audio Duration", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "wer_vs_duration.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_latency_scatter(models_data, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c72b0", "#dd8452"]

    for i, model in enumerate(models_data):
        durations = [s["audio_duration_s"] for s in model["sample_results"]]
        latencies = [s["inference_latency_s"] for s in model["sample_results"]]
        ax.scatter(durations, latencies, alpha=0.5, s=20, label=model["model"], color=colors[i % len(colors)])

    # Reference line: RTF=1
    max_dur = max(
        max(s["audio_duration_s"] for s in m["sample_results"]) for m in models_data
    )
    ax.plot([0, max_dur], [0, max_dur], "r--", linewidth=1.2, label="RTF = 1.0")

    ax.set_xlabel("Audio Duration (s)", fontsize=12)
    ax.set_ylabel("Inference Latency (s)", fontsize=12)
    ax.set_title("Inference Latency vs. Audio Duration", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = output_dir / "latency_scatter.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = load_results(results_dir)
    models_data = data["models"]
    cfg = data["config"]

    print(f"\nGenerating plots for: {cfg['split']} | {cfg['model_size']} | {cfg['num_samples']} samples")
    plot_wer_distribution(models_data, results_dir)
    plot_rtf_comparison(models_data, results_dir)
    plot_wer_vs_duration(models_data, results_dir)
    plot_latency_scatter(models_data, results_dir)
    print("\n✓ All plots saved.")


if __name__ == "__main__":
    main()