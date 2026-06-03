# ASR Benchmark: WhisperX vs. faster-whisper

Evaluates both ASR systems across **WER**, **CER**, and **RTF** on three datasets:
- **LibriSpeech** — clean read speech, auto-downloads ~346 MB
- **L2-ARCTIC** — non-native/accented English (6 L1 backgrounds), streams from HF, no storage needed
- **Mozilla Common Voice** — crowd-sourced natural speech, requires MDC API key

## File Structure

```
asr_benchmark/
├── benchmark_asr.py          # Main entry point (all three datasets)
├── dataset_loader.py         # LibriSpeech loader via torchaudio
├── l2arctic_loader.py        # L2-ARCTIC accented English loader via HF streaming
├── common_voice_loader.py    # Common Voice loader via MDC API
├── whisperx_runner.py        # WhisperX inference wrapper
├── faster_whisper_runner.py  # faster-whisper inference wrapper
├── metrics.py                # WER / CER / RTF computation (jiwer)
├── results_writer.py         # JSON + CSV + Markdown output
├── visualize_results.py      # Matplotlib plots from saved JSON
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** `whisperx` requires `ffmpeg` on PATH.
> - Linux: `apt install ffmpeg`
> - Mac (Homebrew): `brew install ffmpeg`
> - Mac (conda): `conda install -c conda-forge ffmpeg -y`

---

## Dataset Setup

### LibriSpeech
Downloaded automatically on first run. No action needed.

---

### L2-ARCTIC (recommended for accent robustness testing)

Non-native English speech from 24 speakers across 6 L1 backgrounds (Arabic, Hindi, Korean, Mandarin, Spanish, Vietnamese). Parquet-based on HF — no loading script, no manual terms acceptance, small footprint.

**Setup (one-time):**
```bash
huggingface-cli login   # or: export HF_TOKEN=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
```
Access is auto-approved when you accept the dataset terms at:
https://huggingface.co/datasets/KoelLabs/L2Arctic

**Run:**
```bash
# All speakers, scripted split
python benchmark_asr.py --dataset l2arctic --split scripted --num-samples 200 --device cpu --compute-type int8

# Filter by L1 background
python benchmark_asr.py --dataset l2arctic --split scripted --num-samples 200 \
    --l1-filter arabic mandarin hindi --device cpu --compute-type int8

# Available splits: scripted, spontaneous
# Available L1 filters: arabic, hindi, korean, mandarin, spanish, vietnamese
```

---

### Common Voice (Mozilla Data Collective)

Requires an MDC API key from [mozilladatacollective.com](https://mozilladatacollective.com).

```bash
export MDC_API_KEY=your_api_key_here
export MDC_DATASET_ID=cmndapwry02jnmh07dyo46mot   # English CV 25.0
python benchmark_asr.py --dataset commonvoice --split test --num-samples 200 --device cpu --compute-type int8
```

## Split Reference

Each dataset uses different split names — **do not mix them up**:

| Dataset | `--dataset` | Valid `--split` values |
|---------|-------------|------------------------|
| LibriSpeech | `librispeech` | `test-clean`, `test-other`, `dev-clean`, `dev-other` |
| L2-ARCTIC | `l2arctic` | `scripted`, `spontaneous` |
| Common Voice | `commonvoice` | `test`, `validation`, `train`, `validated` |

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `whisperx` | WhisperX ASR | `pip install whisperx` |
| `faster-whisper` | faster-whisper ASR | `pip install faster-whisper` |
| `jiwer` | WER/CER metrics | `pip install jiwer` |
| `datasets` | HF streaming (L2-ARCTIC) | `pip install "datasets==2.19.2"` |
| `soundfile` | Audio decoding | `pip install soundfile` |
| `ffmpeg` | Audio loading for WhisperX | `conda install -c conda-forge ffmpeg` |

> **Important:** Use `datasets==2.19.2` specifically. Newer versions require `torchcodec`/`librosa` which conflict with the environment.

---



### LibriSpeech

```bash
python benchmark_asr.py --dataset librispeech --split test-clean \
    --num-samples 200 --device cpu --compute-type int8

# Available splits: test-clean, test-other, dev-clean, dev-other
```

### L2-ARCTIC

```bash
python benchmark_asr.py --dataset l2arctic --split scripted \
    --num-samples 200 --device cpu --compute-type int8
```

### GPU run (any dataset)

```bash
python benchmark_asr.py --dataset l2arctic --split scripted \
    --num-samples 500 --model-size small --device cuda --compute-type float16 --batch-size 16
```

---

## Visualize Results

```bash
python visualize_results.py --results-dir results/
```

| Plot | Description |
|------|-------------|
| `wer_distribution.png` | WER histogram per model |
| `rtf_comparison.png` | Mean/median RTF bar chart |
| `wer_vs_duration.png` | Scatter: WER vs audio length |
| `latency_scatter.png` | Scatter: inference latency vs audio length |

---

## Output Files

| File | Contents |
|------|----------|
| `results/benchmark_results.json` | Full per-sample results + config |
| `results/aggregate_metrics.csv` | One row per model, all aggregate metrics |
| `results/per_sample_results.csv` | One row per sample |
| `results/summary.md` | Markdown summary table |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate — `(S+D+I) / N`, lower is better |
| **CER** | Character Error Rate — edit distance at character level |
| **RTF** | Real-Time Factor — `inference_time / audio_duration`; `< 1.0` = faster than real-time |

---

## Implementation Notes

- L2-ARCTIC audio streams from HF and is cached as 16 kHz WAV under `./data/L2Arctic/_wav_cache/`. First run streams only the samples you request (~a few MB per 200 samples).
- LibriSpeech audio is cached under `./data/LibriSpeech/_audio_cache/`.
- WhisperX VAD alignment is **not** applied during benchmarking to keep latency comparable with faster-whisper.
- `UserWarning` noise from torchaudio and pyannote is suppressed automatically.
- For the thesis, running LibriSpeech (clean) + L2-ARCTIC (accented) gives a direct cross-dataset WER comparison that motivates the multimodal pipeline: if ASR alone already degrades on accented speech, that strengthens the case for affective/prosodic fusion.

---

## Known Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ValueError: Invalid split 'scripted'` | Used L2-ARCTIC split with `--dataset librispeech` | Use `--split test-clean` for LibriSpeech |
| `ImportError: torchcodec` | `datasets>=4.x` requires torchcodec/librosa | Downgrade: `pip install "datasets==2.19.2"` |
| `FileNotFoundError: ffmpeg` | ffmpeg not on PATH | `conda install -c conda-forge ffmpeg -y` |
| `ModuleNotFoundError: jiwer` | jiwer not installed | `pip install jiwer` |
| `No space left on device` | Disk full during pip install | `pip cache purge` then retry |