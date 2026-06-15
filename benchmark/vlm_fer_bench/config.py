"""Configuration constants for VLM FER benchmarking."""

import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MODEL_NAMES = {
    "qwen2.5-vl": "qwen2.5vl:7b",
    "llama3.2-vision": "llama3.2-vision:11b",
    "llava": "llava:7b",
}

RAFDB_CLASS_TO_LABEL = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral",
}

EMOTION_SET = list(RAFDB_CLASS_TO_LABEL.values())

PROMPT_TEMPLATE = (
    "Look at the face in this image and identify the dominant emotion. "
    f"Choose exactly one word from this list: {', '.join(EMOTION_SET)}. "
    "Respond with only the single emotion word, nothing else."
)

NORMALIZATION_MAP = {
    "happy": "happiness",
    "happiness": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "scared": "fear",
    "disgust": "disgust",
    "disgusted": "disgust",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
    "calm": "neutral",
}

DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 2
DEFAULT_SAVE_EVERY = 50
DEFAULT_IMAGE_MAX_SIZE = 512
