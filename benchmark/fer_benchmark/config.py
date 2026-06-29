"""
fer_benchmark/config.py
Central config — mirrors vlm_fer_bench/config.py style.
"""

# Models to run in the benchmark (order controls table order)
MODELS_TO_RUN = ["deepface", "qwen2.5vl", "llava"]

# Datasets to evaluate (names match DatasetLoader.load() keys)
DATASETS = ["rafdb"]   # add "affectnet", "fer2013" when available

# DeepFace face detector backend
# Options: "retinaface" (best), "mtcnn", "opencv" (fastest), "skip"
DEEPFACE_BACKEND = "retinaface"

# Ollama model tags
OLLAMA_MODEL_TAGS = {
    "qwen2.5vl": "qwen2.5vl:7b",
    "llava":     "llava:7b",
}

# Canonical 8-class emotion set (Plutchik)
EMOTION_LABELS = [
    "anger", "disgust", "fear", "happiness",
    "sadness", "surprise", "neutral", "contempt",
]
