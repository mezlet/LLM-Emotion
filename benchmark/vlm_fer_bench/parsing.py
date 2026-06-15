"""Response normalization / parsing utilities."""

from .config import NORMALIZATION_MAP


def normalize_prediction(raw_text: str) -> str:
    text = raw_text.strip().lower()
    text = text.strip(".,!?:;\"'")

    if text in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[text]

    for keyword, canonical in NORMALIZATION_MAP.items():
        if keyword in text:
            return canonical

    return "unknown"
