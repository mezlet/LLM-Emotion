# GoEmotions LLM Benchmark: Llama 3 8B vs Mistral 7B

Zero-shot and few-shot emotion classification benchmark comparing **Llama 3 8B** and
**Mistral 7B** on the [GoEmotions](https://huggingface.co/datasets/go_emotions) dataset
using Ollama for local inference.

Part of the EMAH thesis component benchmarking phase —
*"Embodying Affective AI: A Multimodal Emotion-Aware HRI System on Ameca"*

---

## File Structure

```
.
├── benchmark_runner.py     # Main entry point; supports --mode flag
├── config.py               # Model tags, emotion labels, Ollama settings
├── data_loader.py          # HuggingFace GoEmotions loader + stratified sampling
├── model_client.py         # Ollama HTTP client with retry logic
├── prompt_builder.py       # Zero-shot & few-shot prompt templates
├── evaluator.py            # Multi-label metrics (F1, Hamming, Exact Match)
├── visualize_results.py    # Plots from results/
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
# Zero-shot benchmark (default, 200 samples)
python benchmark_runner.py

# Few-shot benchmark
python benchmark_runner.py --mode few_shot

# Both modes in one run — produces a 4-column comparison table
python benchmark_runner.py --mode both

# Custom options
python benchmark_runner.py --mode both --samples 500 --output results/ --seed 0

# Single model only
python benchmark_runner.py --mode few_shot --models mistral:7b

# Generate plots after benchmarking
python visualize_results.py --results results/
```

---

## Output Files

| File | Contents |
|------|----------|
| `results/<model>_zero_shot_predictions.csv` | Per-sample predictions + latency (zero-shot) |
| `results/<model>_few_shot_predictions.csv` | Per-sample predictions + latency (few-shot) |
| `results/benchmark_summary_zero_shot.json` | Aggregated metrics, zero-shot run |
| `results/benchmark_summary_few_shot.json` | Aggregated metrics, few-shot run |
| `results/benchmark_summary_both.json` | Aggregated metrics, both modes combined |
| `results/overall_metrics.png` | Bar chart: F1 comparison across models |
| `results/per_class_f1_heatmap.png` | Heatmap: per-emotion F1 per model |
| `results/latency_boxplot.png` | Box plot: inference latency distribution |

Each CSV row includes: `model`, `prompt_mode`, `sample_id`, `text`,
`true_labels`, `predicted_labels`, `raw_response`, `latency_ms`.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **Macro F1** | Unweighted mean F1 across all 28 emotion classes |
| **Micro F1** | Global TP/FP/FN aggregated across all classes |
| **Weighted F1** | F1 weighted by class support |
| **Exact Match Ratio** | Fraction of samples where predicted set == true label set |
| **Hamming Loss** | Average fraction of incorrectly predicted labels per sample |
| **Mean / P95 Latency** | Per-request Ollama inference time in milliseconds |

---

## Zero-Shot Results (200 samples, seed=42)

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

---

## Prompting

### Zero-shot
Instructs the model to classify from the 28-label list with no examples.

### Few-shot
Includes 5 labelled examples covering: `admiration/joy`, `anger/disappointment`,
`gratitude`, `confusion`, and `neutral`. Examples are prepended before the target text.

To add or modify examples, edit `FEW_SHOT_EXAMPLES` in `prompt_builder.py`.

---

## Design Notes

- **Label extraction** scans the model's free-text output for valid label names — robust to varied phrasing and formatting
- **Temperature = 0.0** for deterministic, reproducible outputs (set in `config.py`)
- **Stratified sampling** by primary emotion ensures coverage across all 28 classes
- **Schema auto-detection** in `data_loader.py` handles both the `raw` config (per-column binary) and `simplified` config (integer label list) of GoEmotions
- The `raw` HuggingFace config only exposes a `train` split (211k examples); sampling is done from there