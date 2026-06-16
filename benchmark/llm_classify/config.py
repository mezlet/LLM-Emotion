"""
config.py
---------
Central configuration for the LLM emotion-classification benchmark.

Supports three datasets:
  - GoEmotions  (28-class, multi-label, Reddit comments)
  - ISEAR       (7-class,  single-label, self-reported survey narratives)
  - DailyDialog (7-class,  single-label, scripted multi-turn dialogue)

ISEAR and DailyDialog share the same 7-class Ekman-based label set, which
allows direct cross-dataset comparison under identical prompts/metrics.
"""

# Ollama model tags -> display labels
MODELS = {
    "llama3:8b": "Llama 3 8B",
    "mistral:7b": "Mistral 7B",
}

# ──────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────

# GoEmotions: 28 emotion labels (+ neutral), multi-label
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Shared 7-class Ekman-based label set used by ISEAR and DailyDialog.
# This is the canonical DailyDialog label ordering (index 0-6); ISEAR's
# 7 categories are mapped onto this same set in data_loader.py.
EKMAN7_LABELS = [
    "neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise",
]

# Registry: maps a --dataset CLI value to its label set and whether it is
# multi-label (GoEmotions) or single-label (ISEAR, DailyDialog).
DATASETS = {
    "goemotions": {
        "display_name": "GoEmotions",
        "labels": GOEMOTIONS_LABELS,
        "multi_label": True,
        "default_split": "train",
    },
    "isear": {
        "display_name": "ISEAR",
        "labels": EKMAN7_LABELS,
        "multi_label": False,
        "default_split": "train",
    },
    "dailydialog": {
        "display_name": "DailyDialog",
        "labels": EKMAN7_LABELS,
        "multi_label": False,
        "default_split": "test",
    },
}

DEFAULT_DATASET = "goemotions"

# Backwards-compatible default export (used by older modules / scripts that
# import EMOTION_LABELS directly for the GoEmotions benchmark).
EMOTION_LABELS = GOEMOTIONS_LABELS

# Number of samples to evaluate by default
DEFAULT_SAMPLES = 200

# Default output directory
OUTPUT_DIR = "results"

# ──────────────────────────────────────────────────────────────────────────
# Ollama inference settings
# ──────────────────────────────────────────────────────────────────────────

# OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_BASE_URL = "https://hurricane-grass-bid-chelsea.trycloudflare.com"
OLLAMA_TIMEOUT_S = 120          # per-request timeout in seconds
OLLAMA_MAX_TOKENS = 128         # keep responses short for label extraction
OLLAMA_TEMPERATURE = 0.0        # deterministic outputs for benchmarking