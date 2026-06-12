"""
ASR Benchmark: WhisperX vs. faster-whisper
==========================================
Supports LibriSpeech, Mozilla Common Voice 11.0, L2-ARCTIC, and
Speech Accent Archive (via HF streaming).

Usage:
    # LibriSpeech (auto-downloads ~346 MB)
    python benchmark_asr.py --dataset librispeech --split test-clean \\
        --num-samples 200 --device cpu --compute-type int8

    # Common Voice 11.0 — streams from HF, no local storage needed
    # Requires: huggingface-cli login  +  accepting dataset terms on HF
    python benchmark_asr.py --dataset commonvoice --split test \\
        --num-samples 200 --device cpu --compute-type int8

    # Speech Accent Archive — streams from HF, fixed elicitation paragraph
    # Requires: huggingface-cli login
    python benchmark_asr.py --dataset speechaccent --split train \\
        --num-samples 200 --device cpu --compute-type int8
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import numpy as np
import torch

from dataset_loader import LibriSpeechLoader
from common_voice_loader import CommonVoiceLoader
from l2arctic_loader import L2ArcticLoader
from speech_accent_archive_loader import SpeechAccentArchiveLoader
from whisperx_runner import WhisperXRunner
from faster_whisper_runner import FasterWhisperRunner
from google_asr_runner import GoogleASRRunner
from metrics import compute_wer, compute_cer, compute_rtf
from results_writer import ResultsWriter


@dataclass
class BenchmarkConfig:
    dataset: str = "librispeech"        # "librispeech" | "commonvoice" | "l2arctic" | "speechaccent"
    # Google ASR options
    google_asr: bool = False            # include Google Cloud Speech-to-Text runner
    google_model: str = "latest_long"   # Google model name
    google_language: str = "en-US"      # BCP-47 language code
    split: str = "test-clean"
    num_samples: Optional[int] = 100
    model_size: str = "base"
    device: str = "cuda"
    compute_type: str = "float16"
    batch_size: int = 8
    output_dir: str = "results"
    librispeech_dir: str = "./data/LibriSpeech"
    cv_dir: Optional[str] = None
    l1_filter: Optional[list] = None          # e.g. ["arabic", "mandarin"] for l2arctic
    native_language_filter: Optional[list] = None  # e.g. ["english", "mandarin"] for speechaccent
    beam_size: int = 5
    language: str = "en"
    download: bool = True


@dataclass
class SampleResult:
    sample_id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    audio_duration_s: float
    inference_latency_s: float
    rtf: float
    model: str


@dataclass
class AggregateResult:
    model: str
    model_size: str
    split: str
    dataset: str
    num_samples: int
    mean_wer: float
    median_wer: float
    mean_cer: float
    median_cer: float
    mean_rtf: float
    median_rtf: float
    mean_latency_s: float
    total_audio_duration_s: float
    total_inference_time_s: float
    device: str
    compute_type: str
    sample_results: list = field(default_factory=list)


def load_samples(cfg: BenchmarkConfig) -> list[dict]:
    if cfg.dataset == "librispeech":
        loader = LibriSpeechLoader(
            root=cfg.librispeech_dir,
            split=cfg.split,
            max_samples=cfg.num_samples,
            download=cfg.download,
        )
        samples = loader.load()
        print(f"Loaded {len(samples)} samples from LibriSpeech ({cfg.split})\n")

    elif cfg.dataset == "l2arctic":
        cv_split = cfg.split if cfg.split in {"scripted", "spontaneous"} else "scripted"
        loader = L2ArcticLoader(
            split=cv_split,
            max_samples=cfg.num_samples,
            l1_filter=cfg.l1_filter,
        )
        samples = loader.load()
        print(f"Loaded {len(samples)} samples from L2-ARCTIC ({cv_split})\n")

    elif cfg.dataset == "commonvoice":
        hf_split = "validation" if cfg.split == "dev" else cfg.split
        loader = CommonVoiceLoader(
            split=hf_split,
            max_samples=cfg.num_samples,
            cv_dir=cfg.cv_dir,          # None = use MDC API; path = local folder
        )
        samples = loader.load()
        print(f"Loaded {len(samples)} samples from Common Voice ({hf_split})\n")

    elif cfg.dataset == "speechaccent":
        saa_split = cfg.split if cfg.split == "train" else "train"
        loader = SpeechAccentArchiveLoader(
            split=saa_split,
            max_samples=cfg.num_samples,
            native_language_filter=cfg.native_language_filter,
        )
        samples = loader.load()
        print(f"Loaded {len(samples)} samples from Speech Accent Archive ({saa_split})\n")

    else:
        raise ValueError(
            f"Unknown dataset '{cfg.dataset}'. "
            "Choose 'librispeech', 'commonvoice', 'l2arctic', or 'speechaccent'."
        )

    return samples


def run_benchmark(cfg: BenchmarkConfig):
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"
        cfg.compute_type = "int8"

    print(f"\n{'='*60}")
    print(f"  ASR Benchmark: WhisperX vs. faster-whisper")
    print(f"  Dataset: {cfg.dataset} ({cfg.split}) | Samples: {cfg.num_samples} | Model: {cfg.model_size}")
    print(f"  Device: {device} | Compute type: {cfg.compute_type}")
    print(f"{'='*60}\n")

    samples = load_samples(cfg)

    runners = {
        "whisperx": WhisperXRunner(
            model_size=cfg.model_size,
            device=device,
            compute_type=cfg.compute_type,
            batch_size=cfg.batch_size,
            language=cfg.language,
        ),
        "faster-whisper": FasterWhisperRunner(
            model_size=cfg.model_size,
            device=device,
            compute_type=cfg.compute_type,
            beam_size=cfg.beam_size,
            language=cfg.language,
        ),
    }
    if cfg.google_asr:
        runners["google-asr"] = GoogleASRRunner(
            language_code=cfg.google_language,
            model=cfg.google_model,
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ResultsWriter(output_dir)
    all_aggregate: list[AggregateResult] = []

    for model_name, runner in runners.items():
        print(f"\n--- Loading {model_name} ---")
        runner.load_model()

        sample_results: list[SampleResult] = []
        print(f"Running inference on {len(samples)} samples...")

        for i, sample in enumerate(samples):
            t0 = time.perf_counter()
            hypothesis = runner.transcribe(sample["audio_path"])
            latency = time.perf_counter() - t0

            sample_results.append(SampleResult(
                sample_id=sample["id"],
                reference=sample["transcript"],
                hypothesis=hypothesis,
                wer=compute_wer(sample["transcript"], hypothesis),
                cer=compute_cer(sample["transcript"], hypothesis),
                audio_duration_s=sample["duration_s"],
                inference_latency_s=latency,
                rtf=compute_rtf(latency, sample["duration_s"]),
                model=model_name,
            ))

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                avg_wer = np.mean([r.wer for r in sample_results])
                print(f"  [{i+1}/{len(samples)}] Running avg WER: {avg_wer:.4f}")

        runner.unload_model()

        wers      = [r.wer for r in sample_results]
        cers      = [r.cer for r in sample_results]
        rtfs      = [r.rtf for r in sample_results]
        latencies = [r.inference_latency_s for r in sample_results]
        total_audio = sum(r.audio_duration_s for r in sample_results)
        total_inf   = sum(latencies)

        agg = AggregateResult(
            model=model_name,
            model_size=cfg.model_size,
            split=cfg.split,
            dataset=cfg.dataset,
            num_samples=len(sample_results),
            mean_wer=float(np.mean(wers)),
            median_wer=float(np.median(wers)),
            mean_cer=float(np.mean(cers)),
            median_cer=float(np.median(cers)),
            mean_rtf=float(np.mean(rtfs)),
            median_rtf=float(np.median(rtfs)),
            mean_latency_s=float(np.mean(latencies)),
            total_audio_duration_s=total_audio,
            total_inference_time_s=total_inf,
            device=device,
            compute_type=cfg.compute_type,
            sample_results=[asdict(r) for r in sample_results],
        )
        all_aggregate.append(agg)

        print(f"\n[{model_name}] Results:")
        print(f"  Mean WER:     {agg.mean_wer:.4f}")
        print(f"  Median WER:   {agg.median_wer:.4f}")
        print(f"  Mean CER:     {agg.mean_cer:.4f}")
        print(f"  Mean RTF:     {agg.mean_rtf:.4f}  (lower = faster than real-time)")
        print(f"  Mean Latency: {agg.mean_latency_s:.3f}s")
        print(f"  Total audio:  {total_audio:.1f}s | Total inference: {total_inf:.1f}s")

    writer.write_all(all_aggregate, cfg)
    print(f"\n✓ Results saved to: {output_dir}/")
    print_comparison_table(all_aggregate)


def print_comparison_table(results: list[AggregateResult]):
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    header = f"{'Model':<18} {'WER':>8} {'CER':>8} {'RTF':>8} {'Latency':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model:<18} {r.mean_wer:>8.4f} {r.mean_cer:>8.4f} "
            f"{r.mean_rtf:>8.4f} {r.mean_latency_s:>9.3f}s"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperX vs faster-whisper benchmark")

    parser.add_argument("--dataset", default="librispeech",
                        choices=["librispeech", "commonvoice", "l2arctic", "speechaccent"])
    parser.add_argument("--split", default="test-clean",
                        help="LibriSpeech: test-clean/test-other/dev-clean/dev-other | "
                             "CommonVoice: test/validation/train | "
                             "L2-ARCTIC: scripted/spontaneous | "
                             "SpeechAccent: train")
    parser.add_argument("--l1-filter", nargs="+", default=None,
                        choices=["arabic", "hindi", "korean", "mandarin", "spanish", "vietnamese"],
                        help="L2-ARCTIC only: filter by speaker L1 background")
    parser.add_argument("--native-language-filter", nargs="+", default=None,
                        metavar="LANGUAGE",
                        help="Speech Accent Archive only: filter by native language "
                             "(e.g. --native-language-filter english mandarin arabic)")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--model-size", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "int8", "int8_float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cv-dir", default=None,
                        help="Path to local pre-extracted CV folder (clips/ + *.tsv). "
                             "If omitted, downloads via MDC API using MDC_API_KEY env var.")
    parser.add_argument("--librispeech-dir", default="./data/LibriSpeech")
    parser.add_argument("--no-download", action="store_true")

    # Google Cloud Speech-to-Text options
    parser.add_argument("--google-asr", action="store_true",
                        help="Include Google Cloud Speech-to-Text in the benchmark. "
                             "Requires GOOGLE_APPLICATION_CREDENTIALS env var.")
    parser.add_argument("--google-model", default="latest_long",
                        choices=["latest_long", "latest_short", "phone_call",
                                 "video", "command_and_search", "default"],
                        help="Google Speech-to-Text model (default: latest_long)")
    parser.add_argument("--google-language", default="en-US",
                        help="BCP-47 language code for Google ASR (default: en-US)")

    args = parser.parse_args()

    cfg = BenchmarkConfig(
        dataset=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        output_dir=args.output_dir,
        librispeech_dir=args.librispeech_dir,
        cv_dir=args.cv_dir,
        l1_filter=args.l1_filter,
        native_language_filter=args.native_language_filter,
        download=not args.no_download,
        google_asr=args.google_asr,
        google_model=args.google_model,
        google_language=args.google_language,
    )
    run_benchmark(cfg)