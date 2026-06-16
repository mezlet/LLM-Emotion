"""
config.py
Model tags, Ollama settings, dataset parameters, and output paths
for the response generation benchmark (EmpatheticDialogues / DailyDialog).
"""

from pathlib import Path

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_TIMEOUT = 120   # seconds per request
MAX_RETRIES    = 3
RETRY_DELAY    = 2.0   # seconds between retries

# ── Models ────────────────────────────────────────────────────────────────────

MODELS = {
    "llama3": {
        "ollama_tag": "llama3:8b",
        "hf_tag":     "meta-llama/Meta-Llama-3-8B-Instruct",
        "label":      "Llama 3 8B",
        "color":      "#4A90D9",
    },
    "mistral": {
        "ollama_tag": "mistral:7b",
        "hf_tag":     "mistralai/Mistral-7B-Instruct-v0.3",
        "label":      "Mistral 7B",
        "color":      "#E87040",
    },
}

MODEL_KEYS = list(MODELS.keys())

# ── Generation ────────────────────────────────────────────────────────────────

GENERATION = {
    "max_tokens":  150,
    "temperature": 0.7,
    "top_p":       0.9,
}

# ── Dataset ───────────────────────────────────────────────────────────────────

# Currently active dataset. Choose "empathetic_dialogues" or "daily_dialog".
ACTIVE_DATASET = "empathetic_dialogues"

DATASETS = {
    "empathetic_dialogues": {
        "hf_path":     "facebook/empathetic_dialogues",
        "split":       "test",
        "num_samples": 200,   # None → use full split
    },
    "daily_dialog": {
        "hf_path":     "daily_dialog",
        "split":       "test",
        "num_samples": 200,   # None → use full split
        # DailyDialog's original HF loading script is deprecated/removed on
        # newer `datasets` versions. Fallback: fetch dialogues.txt and
        # dialogues_emotion.txt directly from a GitHub mirror via
        # codeload.github.com (allowlisted), which hosts the canonical
        # per-split DailyDialog text files.
        "github_fallback": {
            "repo":   "snakeztc/NeuralDialog-LAED",
            "branch": "master",
            # maps our split names -> mirror's directory names
            "split_dirs": {
                "train":      "train",
                "validation": "validation",
                "valid":      "validation",
                "dev":        "validation",
                "test":       "test",
            },
        },
        # DailyDialog uses numeric emotion ids; map to readable labels
        "emotion_map": {
            0: "no_emotion",
            1: "anger",
            2: "disgust",
            3: "fear",
            4: "happiness",
            5: "sadness",
            6: "surprise",
        },
        # Drop samples whose reference (listener) turn has this emotion id.
        # "no_emotion" dominates DailyDialog (~83%); set to None to keep everything.
        "exclude_reference_emotion": None,
    },
}

# Backwards-compatible alias used by existing code / EmpatheticDialogues loader
DATASET = DATASETS["empathetic_dialogues"]

# EmpatheticDialogues emotion categories (32 fine-grained labels)
EMOTION_LABELS = [
    "admiring", "afraid", "agreeable", "angry", "annoyed",
    "anticipating", "anxious", "apprehensive", "ashamed", "caring",
    "confident", "content", "devastated", "disappointed", "disgusted",
    "embarrassed", "excited", "faithful", "furious", "grateful",
    "guilty", "hopeful", "impressed", "jealous", "joyful",
    "lonely", "nostalgic", "prepared", "proud", "sad",
    "sentimental", "surprised", "terrified", "trusting",
]

# DailyDialog emotion categories (7 coarse labels)
DAILY_DIALOG_EMOTION_LABELS = list(DATASETS["daily_dialog"]["emotion_map"].values())

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
CACHE_DIR   = Path("data_cache")   # local cache for downloaded datasets

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
