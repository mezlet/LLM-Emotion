from __future__ import annotations

from typing import Optional

import numpy as np

from models import EmotionState, ModalityEmotion
from plutchik import (
    PLUTCHIK_INTENSITY_LABELS,
    canonical_arousal,
    canonical_valence,
    derive_blended_emotion,
    intensity_level_from_score,
)


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


def unavailable_prosody(reason: str = "No raw audio was available; input was typed.") -> ModalityEmotion:
    return ModalityEmotion(
        modality="prosody",
        emotion=_emotion_state("trust", 0.05, 0.0),
        available=False,
        source="librosa",
        summary=reason,
    )


def classify_prosody_from_audio(wav_path: Optional[str]) -> ModalityEmotion:
    """Extract pitch and speaking-rate features with librosa and map them to Plutchik labels.

    This is a lightweight rule-based prosody classifier suitable for a thesis
    prototype. It keeps prosody separate from text and face signals.
    """
    if not wav_path:
        return unavailable_prosody()

    try:
        import librosa
    except Exception as exc:
        return unavailable_prosody(f"librosa is unavailable: {exc}")

    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if y.size == 0:
            return unavailable_prosody("Audio buffer was empty.")

        duration = max(0.001, librosa.get_duration(y=y, sr=sr))
        rms = float(np.mean(librosa.feature.rms(y=y)))

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        mean_pitch = float(np.mean(voiced)) if voiced.size else 0.0
        pitch_std = float(np.std(voiced)) if voiced.size else 0.0
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else 0.0

        intervals = librosa.effects.split(y, top_db=30)
        voiced_seconds = sum((end - start) / sr for start, end in intervals)
        speech_rate = voiced_seconds / duration

        # Mapping intuition:
        # - high pitch variability + high rate/energy => anger or fear
        # - low energy + low rate => sadness
        # - moderate stable voice => trust
        # - very high pitch + high variability, not clearly angry => surprise/fear
        if rms < 0.018 and speech_rate < 0.45:
            primary, secondary = "sadness", None
            intensity = min(1.0, 0.45 + (0.45 - speech_rate))
        elif pitch_std > 55 and speech_rate > 0.62 and rms > 0.035:
            primary, secondary = "anger", "fear"
            intensity = min(1.0, 0.55 + min(pitch_std / 180.0, 0.35))
        elif mean_pitch > 240 and pitch_std > 45:
            primary, secondary = "surprise", "fear"
            intensity = min(1.0, 0.50 + min(pitch_std / 200.0, 0.35))
        elif speech_rate > 0.68:
            primary, secondary = "anticipation", None
            intensity = min(1.0, 0.45 + (speech_rate - 0.68))
        else:
            primary, secondary = "trust", None
            intensity = 0.30

        quality = min(1.0, max(0.0, voiced_ratio) * min(1.0, duration / 3.0))
        confidence = max(0.15, quality)

        summary = (
            f"pitch_mean={mean_pitch:.1f}Hz; pitch_std={pitch_std:.1f}Hz; "
            f"speaking_rate={speech_rate:.2f}; rms={rms:.4f}; voiced_ratio={voiced_ratio:.2f}"
        )
        return ModalityEmotion(
            modality="prosody",
            emotion=_emotion_state(primary, intensity, confidence, secondary),
            available=True,
            source="librosa_pitch_speaking_rate",
            summary=summary,
        )
    except Exception as exc:
        return unavailable_prosody(f"Prosody extraction failed: {exc}")
