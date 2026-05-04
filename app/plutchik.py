from __future__ import annotations

from typing import Optional

PLUTCHIK_PRIMARY = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

PLUTCHIK_OPPOSITES = {
    "joy": "sadness",
    "sadness": "joy",
    "trust": "disgust",
    "disgust": "trust",
    "fear": "anger",
    "anger": "fear",
    "surprise": "anticipation",
    "anticipation": "surprise",
}

PLUTCHIK_INTENSITY_LABELS = {
    "joy": {"low": "serenity", "medium": "joy", "high": "ecstasy"},
    "trust": {"low": "acceptance", "medium": "trust", "high": "admiration"},
    "fear": {"low": "apprehension", "medium": "fear", "high": "terror"},
    "surprise": {"low": "distraction", "medium": "surprise", "high": "amazement"},
    "sadness": {"low": "pensiveness", "medium": "sadness", "high": "grief"},
    "disgust": {"low": "boredom", "medium": "disgust", "high": "loathing"},
    "anger": {"low": "annoyance", "medium": "anger", "high": "rage"},
    "anticipation": {"low": "interest", "medium": "anticipation", "high": "vigilance"},
}

# Primary dyads from adjacent Plutchik emotions.
PLUTCHIK_DYADS = {
    frozenset(("joy", "trust")): "love",
    frozenset(("trust", "fear")): "submission",
    frozenset(("fear", "surprise")): "awe",
    frozenset(("surprise", "sadness")): "disappointment",
    frozenset(("sadness", "disgust")): "remorse",
    frozenset(("disgust", "anger")): "contempt",
    frozenset(("anger", "anticipation")): "aggression",
    frozenset(("anticipation", "joy")): "optimism",
}

VALENCE_MAP = {
    "joy": 0.9,
    "trust": 0.7,
    "anticipation": 0.45,
    "surprise": 0.1,
    "fear": -0.75,
    "anger": -0.8,
    "disgust": -0.85,
    "sadness": -0.8,
}

AROUSAL_MAP = {
    "joy": 0.65,
    "trust": 0.4,
    "anticipation": 0.7,
    "surprise": 0.85,
    "fear": 0.9,
    "anger": 0.85,
    "disgust": 0.55,
    "sadness": 0.35,
}

EMOJI_BY_EXPRESSION_TYPE = {
    "warm": "😊",
    "supportive": "🙂",
    "enthusiastic": "😄",
    "playful": "😉",
    "concerned": "😟",
    "calm": "😌",
    "thoughtful": "🤔",
    "neutral": "🙂",
}

SOCIAL_INTENTS = {
    "greeting",
    "smalltalk",
    "information_request",
    "problem_statement",
    "emotional_expression",
    "gratitude",
    "farewell",
    "unclear",
}

RESPONSE_STYLES = {
    "neutral", "friendly", "empathetic", "enthusiastic", "professional", "supportive", "concise"
}


def clamp(value: object, minimum: float = 0.0, maximum: float = 1.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def normalize_primary_emotion(value: object, default: str = "trust") -> str:
    emotion = str(value or "").strip().lower()
    return emotion if emotion in PLUTCHIK_PRIMARY else default


def normalize_secondary_emotion(value: object) -> Optional[str]:
    if value is None:
        return None
    emotion = str(value).strip().lower()
    if not emotion or emotion == "null":
        return None
    return emotion if emotion in PLUTCHIK_PRIMARY else None


def intensity_level_from_score(score: float) -> str:
    if score < 0.34:
        return "low"
    if score < 0.67:
        return "medium"
    return "high"


def derive_blended_emotion(primary: str, secondary: Optional[str]) -> Optional[str]:
    if not secondary or primary == secondary:
        return None
    return PLUTCHIK_DYADS.get(frozenset((primary, secondary)))


def canonical_valence(primary: str, intensity_score: float) -> float:
    base = VALENCE_MAP.get(primary, 0.0)
    # Keep sign but scale slightly by intensity, avoiding exaggerated values for weak emotions.
    return round(base * (0.5 + 0.5 * clamp(intensity_score)), 2)


def canonical_arousal(primary: str, intensity_score: float) -> float:
    base = AROUSAL_MAP.get(primary, 0.5)
    return round(clamp((base * 0.6) + (clamp(intensity_score) * 0.4)), 2)


def derive_emoji_type(primary: str, social_intent: str, response_style: str) -> str:
    if social_intent in {"greeting", "smalltalk", "gratitude"}:
        return "warm"
    if response_style in {"empathetic", "supportive"} or primary in {"sadness", "fear"}:
        return "supportive"
    if primary == "joy":
        return "enthusiastic"
    if primary == "anticipation":
        return "thoughtful"
    return "neutral"
