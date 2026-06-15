"""
benchmark_runner.py
Main entry point for the response generation benchmark.
Supports --mode flag to run zero-shot, few-shot, or both,
and --dataset to choose EmpatheticDialogues or DailyDialog.

Usage:
    python benchmark_runner.py                          # zero-shot, Ollama, EmpatheticDialogues
    python benchmark_runner.py --mode few_shot          # few-shot prompting
    python benchmark_runner.py --mode both              # zero-shot + few-shot
    python benchmark_runner.py --dataset daily_dialog   # use DailyDialog instead
    python benchmark_runner.py --backend hf             # HuggingFace backend
    python benchmark_runner.py --dry_run                # 10 samples, no save
    python benchmark_runner.py --skip_empathy           # skip NLI scorer
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config          import MODEL_KEYS, MODELS, RESULTS_DIR, FIGURES_DIR, GENERATION, DATASETS
from data_loader     import load_dataset_samples
from model_client    import get_client
from evaluator       import (
    evaluate_batch, aggregate, load_empathy_classifier, SCORE_COLS
)
from visualize_results import generate_all


# ── Inference loop ────────────────────────────────────────────────────────────

def run_model(
    samples:     list[dict],
    client,
    model_key:   str,
    mode:        str = "zero_shot",
    max_tokens:  int = GENERATION["max_tokens"],
    empathy_clf  = None,
) -> list[dict]:
    """
    Run generation + evaluation for one model over all samples.
    Returns a list of per-sample result dicts.
    """
    hypotheses: list[str]   = []
    references: list[str]   = []
    latencies:  list[float] = []

    for sample in tqdm(samples, desc=f"{MODELS[model_key]['label']} [{mode}]"):
        try:
            hyp, lat = client.generate(
                model_key  = model_key,
                emotion    = sample["emotion"],
                context    = sample["context"],
                utterance  = sample["utterance"],
                max_tokens = max_tokens,
            )
        except RuntimeError as exc:
            print(f"\n[runner] Skipping {sample['conv_id']}: {exc}")
            hyp, lat = "", 0.0

        hypotheses.append(hyp)
        references.append(sample["reference"])
        latencies.append(lat)

    metrics = evaluate_batch(references, hypotheses, latencies, empathy_clf)

    return [
        {
            "model":         model_key,
            "mode":          mode,
            "conv_id":       samples[i]["conv_id"],
            "emotion":       samples[i]["emotion"],
            "utterance":     samples[i]["utterance"],
            "reference":     references[i],
            "hypothesis":    hypotheses[i],
            **{col: metrics[col][i] for col in SCORE_COLS},
        }
        for i in range(len(samples))
    ]


# ── Aggregation ───────────────────────────────────────────────────────────────

def build_summary(all_records: list[dict]) -> pd.DataFrame:
    df   = pd.DataFrame(all_records)
    rows = []
    for (model_key, mode), grp in df.groupby(["model", "mode"]):
        row = {"model": model_key, "mode": mode,
               "label": MODELS[model_key]["label"]}
        for col in SCORE_COLS:
            stats          = aggregate(grp[col].tolist())
            row[f"{col}_mean"] = stats["mean"]
            row[f"{col}_std"]  = stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_emotion_summary(all_records: list[dict]) -> pd.DataFrame:
    df   = pd.DataFrame(all_records)
    rows = []
    for (model_key, mode, emotion), grp in df.groupby(["model", "mode", "emotion"]):
        row = {"model": model_key, "mode": mode, "emotion": emotion, "n": len(grp)}
        for col in SCORE_COLS:
            row[f"{col}_mean"] = aggregate(grp[col].tolist())["mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["emotion", "model", "mode"]).reset_index(drop=True)


# ── Reporting ─────────────────────────────────────────────────────────────────

METRIC_DISPLAY = {
    "bleu_mean":          "BLEU ↑",
    "rouge_l_mean":       "ROUGE-L ↑",
    "bert_score_f1_mean": "BERTScore F1 ↑",
    "empathy_score_mean": "Empathy Score ↑",
    "latency_s_mean":     "Latency (s) ↓",
}


def print_summary(summary: pd.DataFrame):
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS — EmpatheticDialogues Response Generation")
    print("="*70)
    for _, row in summary.iterrows():
        print(f"\n  {row['label'].upper()}  [{row['mode']}]")
        for col, label in METRIC_DISPLAY.items():
            if col in summary.columns and pd.notna(row[col]):
                std_col = col.replace("_mean", "_std")
                std_str = f" ± {row[std_col]:.4f}" if std_col in summary.columns else ""
                print(f"    {label:<22}: {row[col]:.4f}{std_str}")

    print("\n  --- Winner per metric (zero_shot) ---")
    zs = summary[summary["mode"] == "zero_shot"]
    for col, label in METRIC_DISPLAY.items():
        if col not in zs.columns or zs[col].isna().all():
            continue
        idx    = zs[col].idxmin() if "latency" in col else zs[col].idxmax()
        winner = zs.loc[idx]
        print(f"    {label:<22}: {winner['label'].upper()} ({winner[col]:.4f})")
    print("="*70)


# ── Save helpers ──────────────────────────────────────────────────────────────

def _nan_safe(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def save_results(
    all_records: list[dict],
    summary:     pd.DataFrame,
    emotion_df:  pd.DataFrame,
    results_dir: Path,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(results_dir / "summary.csv", index=False)
    emotion_df.to_csv(results_dir / "per_emotion_summary.csv", index=False)

    df = pd.DataFrame(all_records)
    for model_key in df["model"].unique():
        df[df["model"] == model_key].to_csv(
            results_dir / f"raw_{model_key}.csv", index=False
        )

    with open(results_dir / "all_results.json", "w") as f:
        json.dump([{k: _nan_safe(v) for k, v in r.items()} for r in all_records],
                  f, indent=2)

    print(f"\n[runner] Results saved → {results_dir.resolve()}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Response Generation Benchmark: Llama 3 8B vs Mistral 7B"
    )
    p.add_argument("--mode",        choices=["zero_shot", "few_shot", "both"],
                   default="zero_shot",
                   help="Prompting mode (default: zero_shot)")
    p.add_argument("--dataset",     choices=list(DATASETS.keys()),
                   default="empathetic_dialogues",
                   help="Dataset to benchmark on (default: empathetic_dialogues)")
    p.add_argument("--backend",     choices=["ollama", "hf"], default="ollama")
    p.add_argument("--ollama_host", default="http://localhost:11434")
    p.add_argument("--split",       default=None,
                   help="Dataset split (default: per-dataset config, usually 'test')")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Number of samples (default: per-dataset config, 200)")
    p.add_argument("--max_tokens",  type=int, default=GENERATION["max_tokens"])
    p.add_argument("--skip_empathy", action="store_true",
                   help="Disable NLI-based empathy scorer (saves ~1.5 GB RAM)")
    p.add_argument("--force_download", action="store_true",
                   help="Bypass local dataset cache and re-download from source")
    p.add_argument("--dry_run",     action="store_true",
                   help="Run on 10 samples only; skip saving outputs")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        args.num_samples = 10
        print("[runner] DRY RUN — 10 samples, no output will be saved\n")

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    # ── Setup
    samples = load_dataset_samples(
        name           = args.dataset,
        split          = args.split,
        num_samples    = args.num_samples,
        force_download = args.force_download,
    )

    backend_kwargs = {"host": args.ollama_host} if args.backend == "ollama" else {}
    client = get_client(args.backend, **backend_kwargs)

    # Health check (Ollama only)
    if args.backend == "ollama" and not client.health_check():
        raise SystemExit(
            f"[runner] Cannot reach Ollama at {args.ollama_host}. "
            "Is the server running?"
        )

    empathy_clf = None if args.skip_empathy else load_empathy_classifier()

    # ── Run
    all_records: list[dict] = []
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        print(f"{'='*60}")
        for model_key in MODEL_KEYS:
            records = run_model(
                samples     = samples,
                client      = client,
                model_key   = model_key,
                mode        = mode,
                max_tokens  = args.max_tokens,
                empathy_clf = empathy_clf,
            )
            all_records.extend(records)

    # ── Summarise + print
    summary    = build_summary(all_records)
    emotion_df = build_per_emotion_summary(all_records)
    print_summary(summary)

    if args.dry_run:
        return

    # ── Save + plot
    results_dir = RESULTS_DIR / args.dataset
    figures_dir = FIGURES_DIR / args.dataset
    save_results(all_records, summary, emotion_df, results_dir)
    generate_all(results_dir, figures_dir)


if __name__ == "__main__":
    main()
