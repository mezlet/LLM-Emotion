"""
models.py
---------
Dataclasses used across the application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import FACE_MIN_TOP_SCORE, FACE_MIN_MARGIN


# ---------------------------------------------------------------------------
# LLM analysis result
# ---------------------------------------------------------------------------

@dataclass
class MessageAnalysis:
    """
    LLM-generated interpretation of a user's message.

    The LLM is the sole determinant of:
    - emotional meaning
    - reply tone
    - whether an emoji is appropriate
    - which emoji to use
    """

    emotion_summary: str
    reply_tone: str
    should_use_emoji: bool
    emoji: Optional[str]
    reason: str


# ---------------------------------------------------------------------------
# Facial-emotion capture result
# ---------------------------------------------------------------------------

@dataclass
class FaceEmotionCapture:
    """
    Aggregated facial-expression output collected during voice recording.

    Stores per-frame score *dictionaries* (not single labels) so that
    averaging across frames produces a much more stable signal.
    """

    emotion_score_samples: list[dict[str, float]]
    frame_count: int
    sampled_frame_count: int
    started_at: str
    ended_at: str
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def averaged_scores(self) -> dict[str, float]:
        """Average emotion scores across all sampled frames."""
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
        """Return the emotion with the highest averaged score."""
        scores = self.averaged_scores
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]

    @property
    def is_reliable(self) -> bool:
        """
        Return True when the visual signal is strong and unambiguous.

        Rejects:
        - weak top scores (below FACE_MIN_TOP_SCORE)
        - mixed top-two scores with a gap smaller than FACE_MIN_MARGIN
        """
        scores = self.averaged_scores
        if not scores:
            return False

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ordered[0][1]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        if top_score < FACE_MIN_TOP_SCORE:
            return False
        if (top_score - second_score) < FACE_MIN_MARGIN:
            return False

        return True

    @property
    def summary_text(self) -> str:
        """Human-readable summary safe to pass to the LLM as a weak hint."""
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