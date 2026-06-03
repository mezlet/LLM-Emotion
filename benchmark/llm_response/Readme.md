# Response Generation Benchmark
**Llama 3 8B vs Mistral 7B on EmpatheticDialogues**

Component benchmarking module for the EMAH thesis — evaluates LLM response generation quality under affective dialogue conditions.

---

## Structure

```
resp_gen_benchmark/
├── benchmark_runner.py     # Main entry point; supports --mode flag
├── config.py               # Model tags, emotion labels, Ollama settings
├── data_loader.py          # HuggingFace EmpatheticDialogues + stratified sampling
├── model_client.py         # Ollama HTTP client with retry logic
├── prompt_builder.py       # Zero-shot & few-shot prompt templates
├── evaluator.py            # BLEU, ROUGE-L, BERTScore, Empathy Score
├── visualize_results.py    # Plots from results/
├── requirements.txt
├── results/                # Auto-created — CSVs and JSON outputs
└── figures/                # Auto-created — all plots
```

---

## Setup

```bash
pip install -r requirements.txt
```

Pull models in Ollama (if using the default backend):

```bash
ollama pull llama3:8b
ollama pull mistral:7b
```

---

## Usage

```bash
# Standard run — zero-shot, Ollama, 200 samples
python benchmark_runner.py

# Few-shot prompting
python benchmark_runner.py --mode few_shot

# Compare zero-shot vs few-shot in one run
python benchmark_runner.py --mode both

# HuggingFace backend (downloads models on first run; GPU recommended)
python benchmark_runner.py --backend hf

# Larger run (adjust sample count)
python benchmark_runner.py --num_samples 500

# Skip NLI-based empathy scorer (saves ~1.5 GB RAM)
python benchmark_runner.py --skip_empathy

# Sanity check — 10 samples, no output saved
python benchmark_runner.py --dry_run
```

Plots only (after a completed run):

```bash
python visualize_results.py
python visualize_results.py --results_dir results --figures_dir figures
```

---

## Dataset

**EmpatheticDialogues** (Facebook Research, 2019)
- 25k open-domain conversations grounded in 32 emotion categories
- HuggingFace path: `facebook/empathetic_dialogues`
- Each benchmark sample = final speaker turn (prompt) + final listener turn (reference response)
- Stratified sampling ensures all 32 emotion categories are represented

---

## Models

| Key | Ollama tag | HuggingFace tag |
|---|---|---|
| `llama3` | `llama3:8b` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `mistral` | `mistral:7b` | `mistralai/Mistral-7B-Instruct-v0.3` |

Model tags and generation parameters (`temperature`, `top_p`, `max_tokens`) are centralised in `config.py`.

---

## Prompting Modes

| Mode | Description |
|---|---|
| `zero_shot` | System prompt + emotion context + utterance |
| `few_shot` | Same as zero-shot, prepended with 3 curated (emotion, utterance, response) examples |

Few-shot examples are defined in `prompt_builder.py` and can be extended or swapped.

---

## Metrics

| Metric | Description | Direction |
|---|---|---|
| **BLEU** | N-gram overlap with reference response (NLTK, smoothed) | ↑ higher is better |
| **ROUGE-L** | Longest common subsequence F1 against reference | ↑ higher is better |
| **BERTScore F1** | Contextual semantic similarity via pretrained BERT | ↑ higher is better |
| **Empathy Score** | P(empathetic) from zero-shot NLI (`facebook/bart-large-mnli`) | ↑ higher is better |
| **Latency (s)** | Wall-clock inference time per sample | ↓ lower is better |

BERTScore requires `pip install bert-score`. Empathy Score requires `transformers` and downloads `bart-large-mnli` on first use. Both degrade gracefully (skipped with a warning) if unavailable.

---

## Outputs

### `results/`
| File | Description |
|---|---|
| `summary.csv` | Mean ± std per model and mode across all metrics |
| `per_emotion_summary.csv` | Per-emotion breakdown for all metrics |
| `raw_llama3.csv` | Per-sample scores for Llama 3 8B |
| `raw_mistral.csv` | Per-sample scores for Mistral 7B |
| `all_results.json` | Combined record of all samples and scores |

### `figures/`
| File | Description |
|---|---|
| `comparison_bar.png` | Side-by-side bar chart for all metrics with error bars |
| `radar_profile.png` | Normalised multi-metric spider chart |
| `latency_distribution.png` | Latency histogram per model |
| `scatter_bleu_rouge.png` | Per-sample BLEU vs ROUGE-L scatter |
| `emotion_heatmap_bleu.png` | Per-emotion BLEU heatmap |
| `emotion_heatmap_rouge_l.png` | Per-emotion ROUGE-L heatmap |
| `emotion_heatmap_empathy_score.png` | Per-emotion empathy score heatmap |

---

## Configuration

All key parameters live in `config.py`:

```python
OLLAMA_HOST    = "http://localhost:11434"
MAX_RETRIES    = 3          # retry attempts on failed requests
RETRY_DELAY    = 2.0        # seconds between retries (exponential back-off)

GENERATION = {
    "max_tokens":  150,
    "temperature": 0.7,
    "top_p":       0.9,
}

DATASET = {
    "split":       "test",
    "num_samples": 200,     # None → full test split (~2k samples)
}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `datasets` | EmpatheticDialogues loading |
| `nltk` | BLEU tokenisation |
| `rouge-score` | ROUGE-L computation |
| `bert-score` | BERTScore F1 (optional) |
| `transformers` | Empathy NLI scorer + HF backend |
| `torch` | Required for HF backend and empathy scorer |
| `pandas`, `numpy` | Data handling and aggregation |
| `matplotlib` | All visualisations |
| `tqdm` | Progress bars |