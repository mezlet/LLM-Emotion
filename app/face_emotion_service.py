from __future__ import annotations

from typing import Optional

from models import EmotionState, FaceEmotionCapture, ModalityEmotion
from plutchik import (
    PLUTCHIK_INTENSITY_LABELS,
    canonical_arousal,
    canonical_valence,
    derive_blended_emotion,
    intensity_level_from_score,
)

FACE_TO_PLUTCHIK = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "trust",
}


def _emotion_state(primary: str, intensity: float, confidence: float, secondary: Optional[str] = None) -> EmotionState:
    intensity = max(0.0, min(1.0, float(intensity)))
    confidence = max(0.0, min(1.0, float(confidence)))
    level = intensity_level_from_score(intensity)
    return EmotionState(
        primary=primary,
        secondary=secondary,
        blended_emotion=derive_blended_emotion(primary, secondary),
        intensity_level=level,
        intensity_label=PLUTCHIK_INTENSITY_LABELS[primary][level],
        intensity_score=round(intensity, 2),
        valence=canonical_valence(primary, intensity),
        arousal=canonical_arousal(primary, intensity),
        confidence=round(confidence, 2),
    )


def unavailable_face(reason: str = "No camera expression was captured; input was typed or camera unavailable.") -> ModalityEmotion:
    return ModalityEmotion(
        modality="face",
        emotion=_emotion_state("trust", 0.05, 0.0),
        available=False,
        source="deepface_or_vlm",
        summary=reason,
    )


def classify_face_capture_to_plutchik(face_capture: Optional[FaceEmotionCapture]) -> ModalityEmotion:
    """Map DeepFace/VLM facial-expression scores to Plutchik labels.

    The camera service already supports DeepFace or VLM. Both return the same
    canonical face-expression score keys, so this function only performs the
    Plutchik mapping.
    """
    if face_capture is None:
        return unavailable_face()
    if face_capture.error:
        return unavailable_face(face_capture.error)

    scores = face_capture.averaged_scores
    if not scores:
        return unavailable_face("No reliable facial-expression score was captured.")

    plutchik_scores: dict[str, float] = {}
    for face_label, score in scores.items():
        plutchik = FACE_TO_PLUTCHIK.get(str(face_label).strip().lower())
        if plutchik:
            plutchik_scores[plutchik] = plutchik_scores.get(plutchik, 0.0) + float(score)

    if not plutchik_scores:
        return unavailable_face("Facial-expression labels could not be mapped to Plutchik labels.")

    ordered = sorted(plutchik_scores.items(), key=lambda item: item[1], reverse=True)
    primary, top_score = ordered[0]
    secondary = ordered[1][0] if len(ordered) > 1 and ordered[1][1] >= 20.0 else None
    intensity = max(0.05, min(1.0, top_score / 100.0))
    confidence = intensity if face_capture.is_reliable else min(0.45, intensity)

    return ModalityEmotion(
        modality="face",
        emotion=_emotion_state(primary, intensity, confidence, secondary),
        available=True,
        source="camera_deepface_or_vlm",
        summary=face_capture.summary_text,
    )
