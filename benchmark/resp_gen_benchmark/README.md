# Response Generation Benchmark
**Llama 3 8B vs Mistral 7B on EmpatheticDialogues / DailyDialog**

Component benchmarking module for the EMAH thesis — evaluates LLM response generation quality under affective dialogue conditions.

---

## Structure

```
resp_gen_benchmark/
├── benchmark_runner.py     # Main entry point; supports --mode and --dataset flags
├── config.py               # Model tags, dataset configs, emotion labels, Ollama settings
├── data_loader.py          # On-the-fly download + caching for EmpatheticDialogues & DailyDialog
├── model_client.py         # Ollama HTTP client with retry logic
├── prompt_builder.py       # Zero-shot & few-shot prompt templates
├── evaluator.py            # BLEU, ROUGE-L, BERTScore, Empathy Score
├── visualize_results.py    # Plots from results/
├── requirements.txt
├── notebooks/
│   └── resp_gen_benchmark_walkthrough.ipynb   # Interactive cell-by-cell pipeline walkthrough
├── data_cache/             # Auto-created — cached preprocessed dataset samples (JSON)
├── results/                # Auto-created — CSVs and JSON outputs, per dataset
│   ├── empathetic_dialogues/
│   └── daily_dialog/
└── figures/                # Auto-created — all plots, per dataset
    ├── empathetic_dialogues/
    └── daily_dialog/
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
# Standard run — zero-shot, Ollama, EmpatheticDialogues, 200 samples
python benchmark_runner.py

# Use DailyDialog instead
python benchmark_runner.py --dataset daily_dialog

# Few-shot prompting
python benchmark_runner.py --mode few_shot

# Compare zero-shot vs few-shot in one run
python benchmark_runner.py --mode both

# DailyDialog + few-shot + HF backend
python benchmark_runner.py --dataset daily_dialog --mode both --backend hf

# HuggingFace backend (downloads models on first run; GPU recommended)
python benchmark_runner.py --backend hf

# Larger run (adjust sample count)
python benchmark_runner.py --num_samples 500

# Skip NLI-based empathy scorer (saves ~1.5 GB RAM/VRAM)
python benchmark_runner.py --skip_empathy

# Force empathy classifier onto GPU (auto-detected by default)
python benchmark_runner.py --device gpu

# Force empathy classifier onto CPU (e.g. small/shared GPU)
python benchmark_runner.py --device cpu

# Re-download dataset, bypassing local cache
python benchmark_runner.py --dataset daily_dialog --force_download

# Sanity check — 10 samples, no output saved
python benchmark_runner.py --dry_run
```

Plots only (after a completed run):

```bash
python visualize_results.py --results_dir results/empathetic_dialogues --figures_dir figures/empathetic_dialogues
python visualize_results.py --results_dir results/daily_dialog --figures_dir figures/daily_dialog
```

### Interactive notebook

For step-by-step exploration — inspecting prompts, spot-checking individual
generations, and rendering figures inline — use
`notebooks/resp_gen_benchmark_walkthrough.ipynb`:

```bash
jupyter notebook notebooks/resp_gen_benchmark_walkthrough.ipynb
```

It mirrors `benchmark_runner.py`'s pipeline (load → prompt → generate →
evaluate → aggregate → visualise) but runs each stage in its own cell, with
configuration variables (`DATASET_NAME`, `NUM_SAMPLES`, `MODE`, etc.) at the
top. Defaults to a small sample count for fast interactive iteration; for the
full benchmark, use the CLI instead.

---

## Datasets

Select with `--dataset {empathetic_dialogues, daily_dialog}` (default: `empathetic_dialogues`).
Both loaders produce samples in a common schema: `conv_id`, `emotion`, `context`, `utterance`, `reference`.

### EmpatheticDialogues (Facebook Research, 2019)
- 25k open-domain conversations grounded in 32 emotion categories
- HuggingFace path: `facebook/empathetic_dialogues`
- Each sample = final speaker turn (`utterance`/prompt) + final listener turn (`reference`)
- `emotion` = the conversation's emotion label (one of 32)
- Stratified sampling ensures all 32 categories are represented

### DailyDialog
- ~13k multi-turn everyday conversations, annotated per-utterance with 7 emotion labels
  (`no_emotion`, `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`)
- HuggingFace path: `daily_dialog`
- Each sample = second-to-last turn (`utterance`/prompt) + final turn (`reference`)
- `emotion` = the emotion label of the **reference** (final) turn
- Since `no_emotion` dominates (~83% of utterances), set
  `exclude_reference_emotion: 0` in `config.py` under `DATASETS["daily_dialog"]`
  to focus the benchmark on emotionally-loaded responses
- Stratified sampling ensures all present emotion categories are represented

Both dataset configs (HF path, split, sample count, emotion mapping) live in
`config.py` under `DATASETS`.

### Download & caching

Datasets are **downloaded on the fly** on first use — no manual preparation
needed. Both loaders cascade through multiple sources so a single
deprecated/blocked endpoint doesn't break the pipeline:

1. **HF download** — `datasets.load_dataset(...)` fetches and caches the raw
   dataset under the standard HuggingFace cache (`~/.cache/huggingface`).
   Retried up to 3× with exponential back-off on transient failures.

   If the dataset's loading script has been deprecated (`RuntimeError:
   Dataset scripts are no longer supported`), the loader automatically
   retries against HuggingFace's auto-converted Parquet revision
   (`revision="refs/convert/parquet"`), which bypasses the script entirely.

2. **EmpatheticDialogues fallback** — if HF still fails (e.g. `refs/convert/
   parquet` is unavailable for this dataset), the loader downloads the
   official `empatheticdialogues.tar.gz` directly from
   `dl.fbaipublicfiles.com` and parses `train.csv` / `valid.csv` / `test.csv`
   itself (repairing the small number of rows with an unescaped comma in
   `prompt`, a known quirk of the original files). If that also fails, it
   falls back to the community Parquet mirror `Estwld/empathetic_dialogues_llm`.

3. **DailyDialog fallback** — DailyDialog's original HuggingFace loading
   script is deprecated/removed on newer `datasets` versions, and unlike
   namespaced datasets, `refs/convert/parquet` isn't reachable for canonical
   IDs like `daily_dialog` (HuggingFace requires `namespace/name` for that
   revision lookup) — so this failure is deterministic, not transient, and
   the HF attempt is tried only once (no wasted retries) before falling
   through. The loader then downloads a tarball of `snakeztc/NeuralDialog-LAED`
   via `codeload.github.com` and extracts `data/daily_dialog/<split>/dialogues.txt`
   and `dialogues_emotion.txt` directly (the canonical `__eou__`-delimited
   dialogue format with per-utterance emotion codes 0–6).

4. **Local sample cache** — after grouping/preprocessing, the resulting
   sample list is cached as JSON under `data_cache/<dataset>_<split>_<n>.json`.
   Subsequent runs with the same `--dataset`, `--split`, and `--num_samples`
   load instantly from this cache instead of re-downloading or re-processing.

Use `--force_download` to bypass both the HF cache check and the local
sample cache and re-fetch everything from source:

```bash
python benchmark_runner.py --dataset daily_dialog --force_download
python benchmark_runner.py --dataset empathetic_dialogues --force_download
```

> **Note:** if your environment restricts outbound network access to an
> allowlist, ensure `huggingface.co`, `dl.fbaipublicfiles.com`, and
> `codeload.github.com` are reachable for the respective fallback tiers
> to work.

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

`facebook/bart-large-mnli` needs ~1.6GB VRAM. With `--device auto` (default),
it loads on GPU if CUDA is available; on small/shared GPUs (e.g. a 3GB card
already running Ollama), a CUDA OOM during loading triggers automatic
fallback to CPU rather than aborting the run. Use `--device cpu` to skip
the GPU attempt entirely, or `--skip_empathy` to disable this metric.

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

> **Note:** the number of rows in each emotion heatmap depends on the active
> dataset — 32 for EmpatheticDialogues, up to 7 for DailyDialog (fewer if
> `exclude_reference_emotion` is set).

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

ACTIVE_DATASET = "empathetic_dialogues"   # default if --dataset not passed

DATASETS = {
    "empathetic_dialogues": {
        "hf_path":     "facebook/empathetic_dialogues",
        "split":       "test",
        "num_samples": 200,     # None → full test split (~2k samples)
    },
    "daily_dialog": {
        "hf_path":     "daily_dialog",
        "split":       "test",
        "num_samples": 200,     # None → full test split
        "github_fallback": {
            "repo":   "snakeztc/NeuralDialog-LAED",
            "branch": "master",
            "split_dirs": {"train": "train", "validation": "validation",
                           "test": "test"},
        },
        "emotion_map": {0: "no_emotion", 1: "anger", 2: "disgust",
                        3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"},
        "exclude_reference_emotion": None,  # set to 0 to drop "no_emotion" references
    },
}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `datasets` | EmpatheticDialogues / DailyDialog loading |
| `nltk` | BLEU tokenisation |
| `rouge-score` | ROUGE-L computation |
| `bert-score` | BERTScore F1 (optional) |
| `transformers` | Empathy NLI scorer + HF backend |
| `torch` | Required for HF backend and empathy scorer |
| `pandas`, `numpy` | Data handling and aggregation |
| `matplotlib` | All visualisations |
| `tqdm` | Progress bars |
