"""
config.py
---------
Central configuration for the GoEmotions LLM benchmark.
"""

# Ollama model tags → display labels
MODELS = {
    "llama3:8b": "Llama 3 8B",
    "mistral:7b": "Mistral 7B",
}

# GoEmotions 28 emotion labels (+ neutral)
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Number of GoEmotions samples to evaluate by default
DEFAULT_SAMPLES = 200

# Default output directory
OUTPUT_DIR = "results"

# Ollama inference settings
# OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_BASE_URL = "https://known-builder-audit-electric.trycloudflare.com"
OLLAMA_TIMEOUT_S = 120          # per-request timeout in seconds
OLLAMA_MAX_TOKENS = 128         # keep responses short for label extraction
OLLAMA_TEMPERATURE = 0.0        # deterministic outputs for benchmarking