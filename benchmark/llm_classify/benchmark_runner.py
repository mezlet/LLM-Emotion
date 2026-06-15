"""
benchmark_runner.py
-------------------
Benchmarks Llama 3 8B vs Mistral 7B on emotion classification across
three datasets: GoEmotions (28-class, multi-label), ISEAR (7-class,
single-label), and DailyDialog (7-class, single-label).

Models are queried via Ollama. Results are saved to CSV and a summary JSON.

Usage:
    # GoEmotions (default), zero-shot, 200 samples
    python benchmark_runner.py

    # ISEAR, zero-shot
    python benchmark_runner.py --dataset isear

    # DailyDialog, both prompting modes
    python benchmark_runner.py --dataset dailydialog --mode both

    # ISEAR, few-shot, single model, custom sample size
    python benchmark_runner.py --dataset isear --mode few_shot --models mistral:7b --samples 300
"""

import argparse
import json
import time
import csv
import logging
from pathlib import Path
from datetime import datetime

from data_loader import load_dataset_sample
from model_client import OllamaClient
from prompt_builder import build_prompt
from evaluator import compute_metrics
from config import MODELS, DATASETS, DEFAULT_DATASET, DEFAULT_SAMPLES, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def run_benchmark(model_name: str, samples: list[dict], output_dir: Path,
                  prompt_mode: str, dataset_name: str, dataset_labels: list[str]) -> list[dict]:
    """Run inference for one model over all samples. Returns per-sample result rows."""
    client = OllamaClient(model_name)
    results = []

    log.info(f"Starting benchmark for model: {model_name} | dataset: {dataset_name} "
             f"| mode: {prompt_mode} ({len(samples)} samples)")

    for i, sample in enumerate(samples):
        text = sample["text"]
        true_labels = sample["labels"]

        prompt = build_prompt(text, mode=prompt_mode, dataset=dataset_name)

        t0 = time.perf_counter()
        raw_response = client.generate(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        predicted_labels = parse_predicted_labels(raw_response, dataset_labels)

        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "prompt_mode": prompt_mode,
            "sample_id": sample["id"],
            "text": text,
            "true_labels": "|".join(true_labels),
            "predicted_labels": "|".join(predicted_labels),
            "raw_response": raw_response.strip(),
            "latency_ms": round(latency_ms, 2),
        })

        if (i + 1) % 20 == 0:
            log.info(f"  [{model_name}] Processed {i+1}/{len(samples)} samples")

    safe_model = model_name.replace(":", "_")
    csv_path = output_dir / f"{safe_model}_{dataset_name}_{prompt_mode}_predictions.csv"
    _save_csv(results, csv_path)
    log.info(f"Saved predictions to {csv_path}")

    return results


def parse_predicted_labels(response: str, dataset_labels: list[str]) -> list[str]:
    """
    Extract predicted emotion label(s) from raw model output.

    Scans the lowercased response for any label string from
    `dataset_labels`. Falls back to 'neutral' if nothing recognized.

    Note: for single-label datasets this may still find multiple label
    words if they appear incidentally in the response; downstream
    evaluation (single_label=True) uses only the first match for top-1
    accuracy while macro/micro F1 are computed on the full set found.
    """
    response_lower = response.lower()
    found = [lbl for lbl in dataset_labels if lbl in response_lower]
    return found if found else ["neutral"]


def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on emotion classification datasets")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=list(DATASETS.keys()),
                        help=f"Dataset to benchmark on (default: {DEFAULT_DATASET})")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Ollama model tags to benchmark")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split override (defaults to per-dataset registry value)")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "both"],
                        help="Prompting mode: zero_shot | few_shot | both (default: zero_shot)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    dataset_info = DATASETS[dataset_name]
    dataset_labels = dataset_info["labels"]
    multi_label = dataset_info["multi_label"]

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    log.info(f"Loading {dataset_info['display_name']} dataset "
             f"({args.samples} samples, seed={args.seed})")
    samples = load_dataset_sample(dataset_name, n=args.samples, seed=args.seed, split=args.split)
    log.info(f"Loaded {len(samples)} samples")

    all_metrics = {}

    for prompt_mode in modes:
        log.info(f"\n{'#'*60}")
        log.info(f"DATASET: {dataset_info['display_name']} | PROMPT MODE: {prompt_mode.upper()}")
        log.info(f"{'#'*60}")

        for model_tag in args.models:
            model_label = MODELS.get(model_tag, model_tag)
            run_label = f"{model_label} [{prompt_mode}]"

            log.info(f"\n{'='*60}")
            log.info(f"Model: {model_label} ({model_tag})")
            log.info(f"{'='*60}")

            results = run_benchmark(
                model_tag, samples, output_dir, prompt_mode, dataset_name, dataset_labels
            )
            metrics = compute_metrics(
                results, run_label,
                all_labels=dataset_labels,
                single_label=not multi_label,
            )
            all_metrics[run_label] = metrics

            log.info(f"Results for {run_label}:")
            for k, v in metrics.items():
                if k != "per_class":
                    log.info(f"  {k}: {v}")

    # Save combined summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_name,
        "dataset_display_name": dataset_info["display_name"],
        "n_samples": args.samples,
        "seed": args.seed,
        "mode": args.mode,
        "label_set": dataset_labels,
        "multi_label": multi_label,
        "models": all_metrics,
    }
    summary_path = output_dir / f"benchmark_summary_{dataset_name}_{args.mode}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary saved to {summary_path}")

    print_comparison_table(all_metrics, single_label=not multi_label)


def print_comparison_table(all_metrics: dict, single_label: bool = False):
    col_w = 22
    print("\n" + "=" * (30 + col_w * len(all_metrics)))
    print("BENCHMARK SUMMARY")
    print("=" * (30 + col_w * len(all_metrics)))
    header = f"{'Metric':<30}" + "".join(f"{m:<{col_w}}" for m in all_metrics)
    print(header)
    print("-" * (30 + col_w * len(all_metrics)))

    metric_keys = ["macro_f1", "micro_f1", "weighted_f1",
                   "exact_match_ratio", "hamming_loss"]
    if single_label:
        metric_keys.append("top1_accuracy")
    metric_keys.append("mean_latency_ms")

    for key in metric_keys:
        row = f"{key:<30}"
        for model_metrics in all_metrics.values():
            val = model_metrics.get(key, "N/A")
            row += f"{str(val):<{col_w}}"
        print(row)
    print("=" * (30 + col_w * len(all_metrics)))


if __name__ == "__main__":
    main()