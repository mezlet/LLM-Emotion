"""
visualize_results.py
Plots from results/ — run after benchmark_runner.py completes.

Usage:
    python visualize_results.py
    python visualize_results.py --results_dir results --figures_dir figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import MODELS, FIGURES_DIR, RESULTS_DIR

# ── Plot config ───────────────────────────────────────────────────────────────

MODEL_COLORS = {k: v["color"] for k, v in MODELS.items()}

METRIC_LABELS = {
    "bleu_mean":          "BLEU ↑",
    "rouge_l_mean":       "ROUGE-L ↑",
    "bert_score_f1_mean": "BERTScore F1 ↑",
    "empathy_score_mean": "Empathy Score ↑",
    "latency_s_mean":     "Latency (s) ↓",
}


# ── 1. Bar chart — overall metric comparison ──────────────────────────────────

def plot_bar_comparison(summary: pd.DataFrame, out: Path):
    present = [m for m in METRIC_LABELS if m in summary.columns
               and summary[m].notna().any()]
    n   = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, present):
        for i, (_, row) in enumerate(summary.iterrows()):
            color = MODEL_COLORS.get(row["model"], f"C{i}")
            ax.bar(row["model"], row[metric], color=color, alpha=0.85, width=0.5)
            std_col = metric.replace("_mean", "_std")
            if std_col in summary.columns and pd.notna(row.get(std_col)):
                ax.errorbar(row["model"], row[metric], yerr=row[std_col],
                            fmt="none", color="black", capsize=5, linewidth=1.5)
            ax.text(i, row[metric] + 0.004, f"{row[metric]:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        upper = min(1.0, summary[metric].max() * 1.25) if "latency" not in metric else None
        ax.set_ylim(0, upper)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.suptitle(
        "Llama 3 8B vs Mistral 7B — EmpatheticDialogues Response Generation",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out / "comparison_bar.png")


# ── 2. Per-emotion heatmap ────────────────────────────────────────────────────

def plot_emotion_heatmap(emotion_df: pd.DataFrame, metric_col: str, out: Path):
    col = f"{metric_col}_mean"
    if col not in emotion_df.columns or emotion_df[col].isna().all():
        return

    pivot = emotion_df.pivot(index="emotion", columns="model", values=col)
    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.42)))
    im  = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    vmax = np.nanmax(pivot.values)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val    = pivot.values[i, j]
            label  = f"{val:.3f}" if not np.isnan(val) else "N/A"
            color  = "white" if val > vmax * 0.7 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(f"Per-Emotion {METRIC_LABELS.get(col, metric_col)}", fontsize=12)
    plt.tight_layout()
    _save(fig, out / f"emotion_heatmap_{metric_col}.png")


# ── 3. Latency distribution ───────────────────────────────────────────────────

def plot_latency_distribution(results_dir: Path, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_key, color in MODEL_COLORS.items():
        csv = results_dir / f"raw_{model_key}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        ax.hist(df["latency_s"].dropna(), bins=30, alpha=0.65,
                label=MODELS[model_key]["label"], color=color, edgecolor="white")

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Latency Distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "latency_distribution.png")


# ── 4. Radar / spider chart ───────────────────────────────────────────────────

def plot_radar(summary: pd.DataFrame, out: Path):
    radar_metrics = [m for m in METRIC_LABELS
                     if "latency" not in m
                     and m in summary.columns
                     and summary[m].notna().any()]
    if len(radar_metrics) < 3:
        return

    labels = [METRIC_LABELS[m].replace(" ↑", "").replace(" ↓", "") for m in radar_metrics]
    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for _, row in summary.iterrows():
        values = [row[m] for m in radar_metrics]
        # normalise each metric to [0,1] across models
        normed = []
        for v, m in zip(values, radar_metrics):
            col   = summary[m].dropna().values
            mn, mx = col.min(), col.max()
            normed.append((v - mn) / (mx - mn + 1e-9))
        normed += normed[:1]

        color = MODEL_COLORS.get(row["model"], "gray")
        ax.plot(angles, normed, linewidth=2, label=MODELS[row["model"]]["label"], color=color)
        ax.fill(angles, normed, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Normalised Metric Profile", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    _save(fig, out / "radar_profile.png")


# ── 5. Scatter — BLEU vs ROUGE-L ─────────────────────────────────────────────

def plot_scatter_bleu_rouge(results_dir: Path, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_key, color in MODEL_COLORS.items():
        csv = results_dir / f"raw_{model_key}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        ax.scatter(df["bleu"], df["rouge_l"], alpha=0.35, s=15,
                   label=MODELS[model_key]["label"], color=color)

    ax.set_xlabel("BLEU")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("BLEU vs ROUGE-L per Sample")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out / "scatter_bleu_rouge.png")


# ── Utility ───────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved → {path.name}")


def generate_all(results_dir: Path = RESULTS_DIR, figures_dir: Path = FIGURES_DIR):
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary    = pd.read_csv(results_dir / "summary.csv")
    emotion_df = pd.read_csv(results_dir / "per_emotion_summary.csv")

    plot_bar_comparison(summary, figures_dir)
    plot_radar(summary, figures_dir)
    plot_latency_distribution(results_dir, figures_dir)
    plot_scatter_bleu_rouge(results_dir, figures_dir)

    for metric in ["bleu", "rouge_l", "empathy_score"]:
        plot_emotion_heatmap(emotion_df, metric, figures_dir)

    print(f"\n[visualize] All figures saved to → {figures_dir.resolve()}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--figures_dir", default=str(FIGURES_DIR))
    args = parser.parse_args()
    generate_all(Path(args.results_dir), Path(args.figures_dir))
