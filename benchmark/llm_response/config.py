"""
config.py
Model tags, Ollama settings, dataset parameters, and output paths
for the response generation benchmark (EmpatheticDialogues).
"""

from pathlib import Path

# ── Ollama ────────────────────────────────────────────────────────────────────

# OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_HOST = "https://samuel-submit-pattern-criterion.trycloudflare.com"
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

DATASET = {
    "hf_path":     "facebook/empathetic_dialogues",
    "split":       "test",
    "num_samples": 200,   # None → use full split
}

# EmpatheticDialogues emotion categories
EMOTION_LABELS = [
    "admiring", "afraid", "agreeable", "angry", "annoyed",
    "anticipating", "anxious", "apprehensive", "ashamed", "caring",
    "confident", "content", "devastated", "disappointed", "disgusted",
    "embarrassed", "excited", "faithful", "furious", "grateful",
    "guilty", "hopeful", "impressed", "jealous", "joyful",
    "lonely", "nostalgic", "prepared", "proud", "sad",
    "sentimental", "surprised", "terrified", "trusting",
]

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
