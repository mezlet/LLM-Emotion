from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmotionState:
    primary: str
    secondary: Optional[str]
    blended_emotion: Optional[str]
    intensity_level: str
    intensity_label: str
    intensity_score: float
    valence: float
    arousal: float
    confidence: float


@dataclass(frozen=True)
class EmojiPolicy:
    use: bool
    type: Optional[str]
    emoji: Optional[str]


@dataclass(frozen=True)
class ResponsePolicy:
    style: str
    verbosity: str
    emoji: EmojiPolicy


@dataclass(frozen=True)
class MessageAnalysis:
    emotion: EmotionState
    social_intent: str
    response: ResponsePolicy
    reason: str


@dataclass(frozen=True)
class ModalityEmotion:
    """Emotion label produced by one perception modality."""
    modality: str
    emotion: EmotionState
    available: bool
    source: str
    summary: str


@dataclass(frozen=True)
class MultimodalEmotionContext:
    """The three separate emotion signals passed to the response LLM.

    This intentionally does not fuse the modalities. The response generator sees
    each emotion independently as context history/evidence.
    """
    text: ModalityEmotion
    prosody: ModalityEmotion
    face: ModalityEmotion


@dataclass
class FaceEmotionCapture:
    emotion_score_samples: list[dict[str, float]]
    frame_count: int
    sampled_frame_count: int
    started_at: str
    ended_at: str
    error: Optional[str] = None

    @property
    def averaged_scores(self) -> dict[str, float]:
        if not self.emotion_score_samples:
            return {}

        totals: dict[str, float] = {}
        for sample in self.emotion_score_samples:
            for emotion, score in sample.items():
                totals[emotion] = totals.get(emotion, 0.0) + score

        count = len(self.emotion_score_samples)
        return {emotion: total / count for emotion, total in totals.items()}

    @property
    def dominant_emotion(self) -> Optional[str]:
        scores = self.averaged_scores
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]

    @property
    def is_reliable(self) -> bool:
        scores = self.averaged_scores
        if not scores:
            return False

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ordered[0][1]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        return top_score >= 45.0 and (top_score - second_score) >= 15.0

    @property
    def summary_text(self) -> str:
        if self.error:
            return f"No facial-expression hint available because: {self.error}"

        scores = self.averaged_scores
        if not scores:
            return "No reliable facial-expression hint was captured during speech."

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        if not self.is_reliable:
            top_parts = [f"{emo}={score:.0f}%" for emo, score in ordered[:3]]
            return (
                "Facial-expression hint was weak or mixed; "
                f"top_signals=({', '.join(top_parts)}); "
                f"samples={self.sampled_frame_count}"
            )

        parts = [f"{emo}={score:.0f}%" for emo, score in ordered if score >= 5.0]
        return (
            f"dominant={self.dominant_emotion}; "
            f"averaged_scores=({', '.join(parts)}); "
            f"samples={self.sampled_frame_count}"
        )
