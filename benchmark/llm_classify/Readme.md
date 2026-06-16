# LLM Emotion Benchmark: Llama 3 8B vs Mistral 7B

Zero-shot and few-shot emotion classification benchmark comparing **Llama 3 8B** and
**Mistral 7B** across three datasets, using Ollama for local inference.

Part of the EMAH thesis component benchmarking phase —
*"Multimodal Emotion Recognition and Congruent Expressive Response Generation
for the Ameca Social Humanoid Robot"*

---

## Supported Datasets

| `--dataset` value | Dataset | Classes | Label type | Notes |
|---|---|---|---|---|
| `goemotions` (default) | [GoEmotions](https://huggingface.co/datasets/go_emotions) | 28 | Multi-label | Reddit comments; `raw` HF config, sampled from `train` split |
| `isear` | ISEAR | 7 (shared Ekman set) | Single-label | Self-reported emotional narratives; ISEAR's 7 native categories mapped onto the shared 7-class set (see mapping below) |
| `dailydialog` | DailyDialog | 7 (shared Ekman set) | Single-label | Scripted multi-turn dialogue, flattened to individual utterances; sampled from `test` split |

### Shared 7-class Ekman label set (ISEAR & DailyDialog)

```
neutral, anger, disgust, fear, joy, sadness, surprise
```

Using the same label set for both datasets allows direct cross-dataset
comparison of model behaviour under identical prompts and metrics.

**ISEAR → shared label mapping** (documented in `data_loader.py`):

| ISEAR category | Mapped to |
|---|---|
| joy | joy |
| fear | fear |
| anger | anger |
| sadness | sadness |
| disgust | disgust |
| shame | sadness *(closest Ekman analogue; no dedicated category)* |
| guilt | sadness *(same rationale as shame)* |

Note: ISEAR contains no native `surprise` or `neutral` examples — these
classes will have zero ground-truth support when benchmarking on ISEAR,
which should be reported as a structural limitation.

**DailyDialog → shared label mapping:** DailyDialog's native 7 categories
(`no emotion, anger, disgust, fear, happiness, sadness, surprise`) map 1:1
onto the shared set, with `no emotion → neutral` and `happiness → joy`.
Note: DailyDialog is heavily skewed toward `neutral` (~83% of utterances);
stratified sampling caps the per-class draw so rarer emotions are still
represented in the benchmark sample.

---

## File Structure

```
.
├── benchmark_runner.py     # Main entry point; supports --dataset and --mode flags
├── config.py               # Model tags, dataset registry, label sets, Ollama settings
├── data_loader.py          # Loaders for GoEmotions, ISEAR, DailyDialog + stratified sampling
├── model_client.py         # Ollama HTTP client with retry logic
├── prompt_builder.py        # Zero-shot & few-shot prompts, dataset-aware (multi- vs single-label)
├── evaluator.py            # Metrics: F1, Hamming, Exact Match, Top-1 Accuracy
├── visualize_results.py    # Plots from results/, dataset-aware label sets
└── requirements.txt
```

---

## Prerequisites

### 1. Ollama running locally
```bash
# Install from https://ollama.com, then pull both models
ollama pull llama3:8b
ollama pull mistral:7b
```

### 2. Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# GoEmotions (default), zero-shot, 200 samples
python benchmark_runner.py

# ISEAR, zero-shot
python benchmark_runner.py --dataset isear

# DailyDialog, both prompting modes
python benchmark_runner.py --dataset dailydialog --mode both

# ISEAR, few-shot, single model, custom sample size and seed
python benchmark_runner.py --dataset isear --mode few_shot --models mistral:7b --samples 300 --seed 0

# Override the dataset split
python benchmark_runner.py --dataset dailydialog --split validation

# Generate plots for a specific run
python visualize_results.py --dataset isear --mode zero_shot --results results/
python visualize_results.py --summary results/benchmark_summary_dailydialog_both.json
```

---

## Output Files

Files are namespaced by `<model>_<dataset>_<mode>` (or `benchmark_summary_<dataset>_<mode>.json`):

| File | Contents |
|------|----------|
| `results/<model>_<dataset>_<mode>_predictions.csv` | Per-sample predictions + latency |
| `results/benchmark_summary_<dataset>_<mode>.json` | Aggregated metrics, label set, and per-class breakdown |
| `results/overall_metrics_<dataset>_<mode>.png` | Bar chart: F1 (+ Top-1 Accuracy for single-label) comparison |
| `results/per_class_f1_heatmap_<dataset>_<mode>.png` | Heatmap: per-emotion F1 per model |
| `results/latency_boxplot_<dataset>_<mode>.png` | Box plot: inference latency distribution |

Each CSV row includes: `model`, `dataset`, `prompt_mode`, `sample_id`, `text`,
`true_labels`, `predicted_labels`, `raw_response`, `latency_ms`.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Macro F1** | Unweighted mean F1 across all classes |
| **Micro F1** | Global TP/FP/FN aggregated across all classes |
| **Weighted F1** | F1 weighted by class support |
| **Exact Match Ratio** | Fraction of samples where predicted label set == true label set |
| **Hamming Loss** | Average fraction of incorrectly predicted labels per sample |
| **Top-1 Accuracy** | *(ISEAR / DailyDialog only)* Plain single-label accuracy — predicted label == ground-truth label |
| **Mean / P95 Latency** | Per-request Ollama inference time in milliseconds |

---

## GoEmotions Zero-Shot Results (200 samples, seed=42)

| Metric | Llama 3 8B | Mistral 7B |
|--------|-----------|-----------|
| Macro F1 | 0.2187 | **0.3074** |
| Micro F1 | 0.2114 | **0.2925** |
| Weighted F1 | 0.2255 | **0.3134** |
| Exact Match | 0.030 | **0.105** |
| Hamming Loss | 0.0986 | **0.0804** |
| Mean Latency | 1432ms | **804ms** |

**Key observations:**
- Mistral 7B outperforms Llama 3 8B on every metric in zero-shot setting
- Llama 3 8B shows high-precision / low-recall behaviour on most labels (e.g. `nervousness`: 1.0 / 0.07), suggesting over-conservative predictions
- Mistral 7B shows more balanced precision/recall and handles nuanced emotions better (`gratitude`: 0.88 F1, `relief`: 0.71 F1)
- `neutral` scores 0.00 F1 on both models — a known zero-shot limitation
- Mistral 7B is **1.8× faster** (~800ms vs ~1430ms per sample), relevant for real-time HRI deployment on Ameca
- Few-shot prompting did not improve F1 for either model (see thesis Chapter 4 for full analysis); Mistral 7B's macro F1 dropped from 0.307 → 0.266 with few-shot

ISEAR and DailyDialog results pending — run with `--dataset isear` / `--dataset dailydialog`
and update this section once results are available.

---

## Prompting

### Zero-shot
- **GoEmotions** (multi-label): instructs the model to output a comma-separated
  list of applicable labels from the 28-class set.
- **ISEAR / DailyDialog** (single-label): instructs the model to output exactly
  one label from the shared 7-class Ekman set.

### Few-shot
- **GoEmotions**: 5 examples covering `admiration/joy`, `anger/disappointment/sadness`,
  `gratitude`, `confusion`, and `neutral`.
- **ISEAR / DailyDialog**: 7 examples, one per class in the shared Ekman set
  (`joy, anger, disgust, fear, sadness, surprise, neutral`).

To add or modify examples, edit `FEW_SHOT_EXAMPLES_GOEMOTIONS` or
`FEW_SHOT_EXAMPLES_EKMAN7` in `prompt_builder.py`.

---

## Design Notes

- **Label extraction** scans the model's free-text output for valid label names — robust to varied phrasing and formatting
- **Temperature = 0.0** for deterministic, reproducible outputs (set in `config.py`)
- **Stratified sampling**: GoEmotions stratifies by primary (first) label; ISEAR
  and DailyDialog stratify by the single ground-truth label, capping per-class
  draws so dominant classes (e.g. DailyDialog's `neutral`) don't crowd out rare ones
- **Schema auto-detection** in `data_loader.py` handles multiple HF repo
  variants for ISEAR and DailyDialog, and both the `raw`/`simplified` configs
  of GoEmotions
- The GoEmotions `raw` HuggingFace config only exposes a `train` split (211k examples); sampling is done from there
- DailyDialog is flattened from multi-turn dialogues into individual
  `(utterance, emotion)` pairs before sampling — dialogue context beyond the
  single utterance is not provided to the model