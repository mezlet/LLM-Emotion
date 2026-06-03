"""
Writes benchmark results to disk:
  - JSON (full per-sample data)
  - CSV (aggregate metrics for easy import)
  - Markdown summary table
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING


class ResultsWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def write_all(self, aggregate_results, cfg):
        self._write_json(aggregate_results, cfg)
        self._write_csv(aggregate_results)
        self._write_markdown(aggregate_results, cfg)

    # ------------------------------------------------------------------
    def _write_json(self, results, cfg):
        payload = {
            "config": {
                "split": cfg.split,
                "num_samples": cfg.num_samples,
                "model_size": cfg.model_size,
                "device": cfg.device,
                "compute_type": cfg.compute_type,
                "batch_size": cfg.batch_size,
                "beam_size": cfg.beam_size,
            },
            "models": [asdict(r) for r in results],
        }
        path = self.output_dir / "benchmark_results.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  JSON  → {path}")

    def _write_csv(self, results):
        path = self.output_dir / "aggregate_metrics.csv"
        fieldnames = [
            "model", "model_size", "split", "num_samples",
            "mean_wer", "median_wer", "mean_cer", "median_cer",
            "mean_rtf", "median_rtf", "mean_latency_s",
            "total_audio_duration_s", "total_inference_time_s",
            "device", "compute_type",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: getattr(r, k) for k in fieldnames}
                writer.writerow(row)
        print(f"  CSV   → {path}")

    def _write_markdown(self, results, cfg):
        path = self.output_dir / "summary.md"
        lines = [
            "# ASR Benchmark Summary",
            "",
            f"**Split:** `{cfg.split}`  |  **Model size:** `{cfg.model_size}`  |  "
            f"**Samples:** `{cfg.num_samples}`  |  **Device:** `{cfg.device}` ({cfg.compute_type})",
            "",
            "## Aggregate Metrics",
            "",
            "| Model | Mean WER ↓ | Median WER | Mean CER ↓ | Mean RTF ↓ | Mean Latency (s) |",
            "|-------|-----------|------------|-----------|-----------|-----------------|",
        ]
        for r in results:
            lines.append(
                f"| {r.model} | {r.mean_wer:.4f} | {r.median_wer:.4f} | "
                f"{r.mean_cer:.4f} | {r.mean_rtf:.4f} | {r.mean_latency_s:.3f} |"
            )

        lines += [
            "",
            "## Per-sample Results",
            "",
            "| Sample ID | Model | WER | CER | RTF | Duration (s) | Latency (s) |",
            "|-----------|-------|-----|-----|-----|-------------|------------|",
        ]
        for r in results:
            for s in r.sample_results:
                lines.append(
                    f"| {s['sample_id']} | {s['model']} | {s['wer']:.4f} | "
                    f"{s['cer']:.4f} | {s['rtf']:.4f} | {s['audio_duration_s']:.2f} | "
                    f"{s['inference_latency_s']:.3f} |"
                )

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"  MD    → {path}")