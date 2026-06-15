"""
CLI entry point for VLM FER benchmarking on RAF-DB.

Usage:
    python run_vlm_fer_benchmark.py --download --split test --output_dir ./results
    python run_vlm_fer_benchmark.py --rafdb_root /path/to/dataset --output_dir ./results
    python run_vlm_fer_benchmark.py --download --limit 100 --models qwen2.5-vl
"""

import os
import argparse
import pandas as pd

from vlm_fer_bench import MODEL_NAMES, load_rafdb_index, run_benchmark, compute_metrics
from vlm_fer_bench.dataset import download_rafdb


def main():
    parser = argparse.ArgumentParser(description="VLM FER Benchmark on RAF-DB")
    parser.add_argument("--rafdb_root", default=None,
                         help="Path to RAF-DB dataset root (omit if using --download)")
    parser.add_argument("--num_gpu", type=int, default=-1,
                     help="Layers to offload to GPU per Ollama call. -1 = auto (default).")
    parser.add_argument("--download", action="store_true",
                         help="Download RAF-DB via kagglehub (shuvoalok/raf-db-dataset)")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES.keys()),
                         choices=list(MODEL_NAMES.keys()))
    args = parser.parse_args()

    if args.download:
        rafdb_root = download_rafdb()
    elif args.rafdb_root:
        rafdb_root = args.rafdb_root
    else:
        parser.error("Provide --rafdb_root or pass --download")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading RAF-DB index...")
    df = load_rafdb_index(rafdb_root, split=args.split)
    print(f"Loaded {len(df)} images for split='{args.split}'")
    print(df["true_label"].value_counts())

    all_metrics = []
    for model_key in args.models:
        results_df = run_benchmark(model_key, df, args.output_dir, limit=args.limit, num_gpu=args.num_gpu)
        metrics = compute_metrics(results_df, model_key, args.output_dir)
        all_metrics.append(metrics)

    summary = pd.DataFrame([
        {
            "model": m["model"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "n_unknown": m["n_unknown_responses"],
            "avg_latency_sec": m["avg_latency_sec"],
        }
        for m in all_metrics
    ])
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()