"""Per-model benchmark execution loop with resume support."""

import os
import pandas as pd
from .config import MODEL_NAMES, PROMPT_TEMPLATE, DEFAULT_SAVE_EVERY
from .ollama_client import query_ollama_vlm
from .parsing import normalize_prediction


def run_benchmark(model_key: str, df: pd.DataFrame, output_dir: str,
                   limit: int = None, save_every: int = DEFAULT_SAVE_EVERY,
                   num_gpu: int = -1) -> pd.DataFrame:
    model_tag = MODEL_NAMES[model_key]
    print(f"\n=== Benchmarking {model_key} ({model_tag}) on {len(df)} images (num_gpu={num_gpu}) ===")

    if limit:
        df = df.head(limit)

    out_csv = os.path.join(output_dir, f"results_{model_key}.csv")
    results = []

    start_idx = 0
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        results = existing.to_dict("records")
        start_idx = len(results)
        print(f"Resuming from row {start_idx}")

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        resp = query_ollama_vlm(model_tag, row["image_path"], PROMPT_TEMPLATE, num_gpu=num_gpu)
        prediction = normalize_prediction(resp["raw_response"])

        results.append({
            "image_path": row["image_path"],
            "true_label": row["true_label"],
            "raw_response": resp["raw_response"],
            "prediction": prediction,
            "latency_sec": resp["latency_sec"],
            "error": resp["error"],
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            print(f"  [{model_key}] {i + 1}/{len(df)} done "
                  f"(last pred='{prediction}', true='{row['true_label']}')")

        if (i + 1) % save_every == 0 or (i + 1) == len(df):
            pd.DataFrame(results).to_csv(out_csv, index=False)

    return pd.DataFrame(results)