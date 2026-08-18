#!/usr/bin/env python3
"""Shared ARALF utilities for experiment.py and experiment_warm_up.py.

This module is deliberately dependency-light (NumPy only).  It owns the pieces
that must stay mathematically identical across the warm-up and experiment:

* Ekman-label normalization and bounded helpers
* text/DeepFace/prosody distribution construction
* smooth DeepFace reliability
* nearest-reference AU classification + AU reliability
* two-level reliability-adaptive late fusion
* acoustic feature extraction and optional participant-normalized prosody

DeepFace and Py-Feat clients remain in the caller scripts; this module accepts
plain score dictionaries and a detector object exposing ``extract_arrays``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional
import math
import time

import numpy as np

EKMAN_EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral",
]

EKMAN_EMOTION_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "joyful": "joy",
    "sad": "sadness",
    "angry": "anger",
    "afraid": "fear",
    "scared": "fear",
    "surprised": "surprise",
    "disgusted": "disgust",
    "none": "neutral",
}

DEEPFACE_TO_EKMAN = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
}


def clamp01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def normalize_ekman_emotion(emotion: str) -> str:
    value = str(emotion or "").strip().lower()
    value = EKMAN_EMOTION_ALIASES.get(value, value)
    return value if value in EKMAN_EMOTION_LABELS else "neutral"


def cosine_similarity_nonnegative(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or a.size != b.size:
        return 0.0
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return clamp01(float(np.dot(a, b) / denom))


def zero_distribution() -> dict[str, float]:
    return {emotion: 0.0 for emotion in EKMAN_EMOTION_LABELS}


def normalize_distribution(scores: Optional[dict[str, Any]]) -> dict[str, float]:
    out = zero_distribution()
    if not scores:
        return out
    for raw_label, raw_value in scores.items():
        raw = str(raw_label or "").strip().lower()
        label = EKMAN_EMOTION_ALIASES.get(raw, raw)
        if label not in EKMAN_EMOTION_LABELS:
            continue
        try:
            value = max(0.0, float(raw_value))
        except Exception:
            continue
        if math.isfinite(value):
            out[label] += value
    total = float(sum(out.values()))
    if total <= 0.0:
        return out
    return {label: value / total for label, value in out.items()}


def categorical_distribution(emotion: str, confidence: float) -> dict[str, float]:
    """Convert one categorical label to a probability-like Ekman distribution.

    Confidence is interpreted as strength above a uniform prior, so a weak
    target never becomes less probable than every non-target class.
    """
    target = normalize_ekman_emotion(emotion)
    c = clamp01(confidence)
    n = len(EKMAN_EMOTION_LABELS)
    uniform = 1.0 / n
    target_probability = uniform + c * (1.0 - uniform)
    other = (1.0 - target_probability) / max(1, n - 1)
    return {
        label: target_probability if label == target else other
        for label in EKMAN_EMOTION_LABELS
    }


def text_distribution(
    emotion: str,
    confidence: float,
    scores: Optional[dict[str, Any]],
) -> dict[str, float]:
    normalized = normalize_distribution(scores)
    if sum(normalized.values()) > 0.0:
        return normalized
    return categorical_distribution(emotion, confidence)


def deepface_distribution(scores: Optional[dict[str, Any]]) -> dict[str, float]:
    """Map DeepFace's seven native scores to the canonical seven Ekman labels."""
    out = zero_distribution()
    if not scores:
        return out
    for raw_label, raw_value in scores.items():
        raw = str(raw_label or "").strip().lower()
        mapped = DEEPFACE_TO_EKMAN.get(raw)
        if not mapped and raw in EKMAN_EMOTION_LABELS:
            # Accept already-normalized canonical score dictionaries too.
            mapped = raw
        if not mapped:
            continue
        try:
            value = max(0.0, float(raw_value))
        except Exception:
            continue
        if math.isfinite(value):
            out[mapped] += value
    total = float(sum(out.values()))
    if total <= 0.0:
        return out
    return {label: value / total for label, value in out.items()}


def deepface_reliability(
    scores: Optional[dict[str, Any]],
    *,
    usable_frame_count: int = 1,
    target_frame_count: int = 1,
) -> float:
    """Smooth DeepFace reliability with no hard reliable/unreliable cliff."""
    if not scores:
        return 0.0
    values: list[float] = []
    for value in scores.values():
        try:
            v = max(0.0, float(value))
        except Exception:
            continue
        if math.isfinite(v):
            values.append(v)
    if not values:
        return 0.0
    values.sort(reverse=True)
    top = values[0]
    second = values[1] if len(values) > 1 else 0.0
    margin = max(0.0, top - second)
    top_component = clamp01(top / 100.0)
    margin_component = clamp01(margin / 60.0)
    frame_component = clamp01(float(usable_frame_count) / max(1.0, float(target_frame_count)))
    return clamp01(
        0.45 * top_component
        + 0.35 * margin_component
        + 0.20 * frame_component
    )


def _au_rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMS Euclidean distance between two already-comparable AU vectors."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or a.size != b.size:
        return float("inf")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("inf")
    return float(np.linalg.norm(a - b) / np.sqrt(max(1, a.size)))


def _fit_au_reference_standardizer(
    reference_bank: dict[str, list[np.ndarray]],
    *,
    std_floor_fraction: float = 0.25,
    std_abs_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit participant-specific per-AU z-score statistics from warm-up references.

    The adaptive floor prevents nearly constant AU columns from exploding after
    standardization.  The floor is derived from the participant's own non-zero
    AU standard deviations, with a tiny absolute numerical fallback.
    """
    all_refs = [
        np.asarray(v, dtype=np.float32).reshape(-1)
        for refs in reference_bank.values()
        for v in refs
    ]
    matrix = np.stack(all_refs, axis=0)
    mean = np.mean(matrix, axis=0).astype(np.float32)
    raw_std = np.std(matrix, axis=0).astype(np.float32)

    abs_floor = max(1e-8, float(std_abs_floor))
    positive = raw_std[np.isfinite(raw_std) & (raw_std > abs_floor)]
    if positive.size:
        adaptive_floor = max(
            abs_floor,
            float(np.median(positive)) * max(0.0, float(std_floor_fraction)),
        )
    else:
        adaptive_floor = abs_floor

    scale = np.where(np.isfinite(raw_std), np.maximum(raw_std, adaptive_floor), adaptive_floor)
    return mean, scale.astype(np.float32), float(adaptive_floor)


def _standardize_au_vector(
    vector: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return ((vector - mean) / scale).astype(np.float32)


def _estimate_au_distance_calibration(
    reference_bank: dict[str, list[np.ndarray]],
) -> dict[str, float]:
    """Estimate distance scales from leave-one-reference-out calibration geometry.

    ``distance_tau`` is chosen so the midpoint between the typical nearest
    same-emotion and nearest other-emotion reference has exponential similarity
    0.5.  ``margin_saturation`` is the 75th percentile of positive reference
    separation margins.  These are participant/profile-specific rather than
    assuming a raw AU RMS distance of 1.0 or a universal margin of 0.25.
    """
    same_nearest: list[float] = []
    other_nearest: list[float] = []
    positive_margins: list[float] = []

    for emotion, refs in reference_bank.items():
        for idx, ref in enumerate(refs):
            same = [
                _au_rms_distance(ref, candidate)
                for j, candidate in enumerate(refs)
                if j != idx
            ]
            other = [
                _au_rms_distance(ref, candidate)
                for other_emotion, other_refs in reference_bank.items()
                if other_emotion != emotion
                for candidate in other_refs
            ]
            same = [d for d in same if math.isfinite(d)]
            other = [d for d in other if math.isfinite(d)]
            if not same or not other:
                continue
            s = float(min(same))
            o = float(min(other))
            same_nearest.append(s)
            other_nearest.append(o)
            if o > s:
                positive_margins.append(o - s)

    if same_nearest and other_nearest:
        same_median = float(np.median(same_nearest))
        other_median = float(np.median(other_nearest))
        boundary = max(1e-6, 0.5 * (same_median + other_median))
        distance_tau = boundary / math.log(2.0)
    elif same_nearest:
        same_median = float(np.median(same_nearest))
        other_median = 0.0
        # Fallback: a typical same-emotion reference should still look reasonably similar.
        distance_tau = max(1e-6, same_median / -math.log(0.70))
    else:
        same_median = 0.0
        other_median = 0.0
        distance_tau = 1.0

    if positive_margins:
        margin_saturation = max(1e-6, float(np.percentile(positive_margins, 75)))
        margin_median = float(np.median(positive_margins))
    else:
        margin_saturation = 0.25
        margin_median = 0.0

    return {
        "distance_tau": float(distance_tau),
        "margin_saturation": float(margin_saturation),
        "same_nearest_median": float(same_median),
        "other_nearest_median": float(other_median),
        "positive_margin_median": float(margin_median),
    }


def au_reference_similarity_from_distance(distance: float, tau: float = 1.0) -> float:
    """Map calibrated AU distance to similarity with an exponential kernel."""
    try:
        distance = float(distance)
        tau = float(tau)
    except Exception:
        return 0.0
    if not math.isfinite(distance) or not math.isfinite(tau) or tau <= 1e-8:
        return 0.0
    return clamp01(math.exp(-max(0.0, distance) / tau))


def verify_au_nearest_reference(
    *,
    crops: list[np.ndarray],
    detector: Any,
    calibration: Optional[dict[str, Any]],
    deepface_emotion: Optional[str] = None,
    margin_saturation: float = 0.0,
    live_frame_count: int = 3,
    expected_ref_count: Optional[int] = None,
    distance_tau: float = 0.0,
    std_floor_fraction: float = 0.25,
    std_abs_floor: float = 1e-3,
) -> dict[str, Any]:
    """Classify live AUs by nearest personalized warm-up references.

    Distances are computed in participant-specific z-scored AU space.  A positive
    ``distance_tau`` or ``margin_saturation`` overrides the automatically derived
    profile-specific value; non-positive values select automatic calibration.
    """
    started = time.time()
    base: dict[str, Any] = {
        "available": False,
        "status": "unavailable",
        "confidence": 0.0,
        "au_emotion": None,
        "deepface_emotion": (
            normalize_ekman_emotion(deepface_emotion) if deepface_emotion else None
        ),
        "analysis_seconds": 0.0,
        "au_distribution": zero_distribution(),
    }
    if not calibration:
        base["status"] = "no_calibration_profile"
        return base
    calibration_status = str(calibration.get("status", "invalid_calibration"))
    if calibration_status not in {"ready", "partial"}:
        base["status"] = f"calibration_{calibration_status}"
        base["calibration_status"] = calibration_status
        return base
    if detector is None:
        base["status"] = "au_detector_unavailable"
        return base

    au_columns = list(calibration.get("au_columns", []) or [])
    if not au_columns:
        base["status"] = "invalid_profile_vectors"
        return base

    reference_bank: dict[str, list[np.ndarray]] = {}
    neutral = calibration.get("neutral") or {}
    neutral_refs = [np.asarray(v, dtype=np.float32) for v in (neutral.get("vectors") or [])]
    neutral_refs = [v.reshape(-1) for v in neutral_refs if v.size > 0 and np.all(np.isfinite(v))]
    if neutral_refs:
        reference_bank["neutral"] = neutral_refs

    for raw_emotion, item in (calibration.get("emotions") or {}).items():
        if item.get("usable") is False:
            continue
        refs = [np.asarray(v, dtype=np.float32) for v in (item.get("vectors") or [])]
        refs = [v.reshape(-1) for v in refs if v.size > 0 and np.all(np.isfinite(v))]
        if refs:
            reference_bank[normalize_ekman_emotion(raw_emotion)] = refs

    if "neutral" not in reference_bank:
        base["status"] = "no_usable_reference_bank"
        return base

    stored_ref_count = int(calibration.get("crops_per_emotion", 0) or 0)
    if stored_ref_count <= 0:
        base["status"] = "invalid_reference_count"
        base["stored_crops_per_emotion"] = stored_ref_count
        return base

    # By default trust the reference count recorded by the warm-up calibration.
    # A caller may still provide expected_ref_count as an explicit compatibility
    # guard, but the live experiment no longer assumes exactly four references.
    resolved_ref_count = stored_ref_count
    if expected_ref_count is not None:
        requested_ref_count = max(1, int(expected_ref_count))
        if stored_ref_count != requested_ref_count:
            base["status"] = "reference_count_mismatch"
            base["stored_crops_per_emotion"] = stored_ref_count
            base["expected_crops_per_emotion"] = requested_ref_count
            return base
        resolved_ref_count = requested_ref_count

    expected_emotions = [
        normalize_ekman_emotion(e)
        for e in (calibration.get("reference_emotions") or reference_bank.keys())
    ]
    incomplete = {
        emotion: len(reference_bank.get(emotion, []))
        for emotion in reference_bank
        if len(reference_bank.get(emotion, [])) != resolved_ref_count
    }
    if incomplete:
        base["status"] = "incomplete_reference_bank"
        base["reference_counts"] = {
            emotion: len(reference_bank.get(emotion, [])) for emotion in expected_emotions
        }
        return base

    try:
        ref_mean, ref_scale, scale_floor = _fit_au_reference_standardizer(
            reference_bank,
            std_floor_fraction=std_floor_fraction,
            std_abs_floor=std_abs_floor,
        )
        standardized_bank = {
            emotion: [_standardize_au_vector(ref, ref_mean, ref_scale) for ref in refs]
            for emotion, refs in reference_bank.items()
        }
    except Exception as exc:
        base["status"] = "au_reference_normalization_failed"
        base["reason"] = str(exc)
        base["analysis_seconds"] = round(time.time() - started, 4)
        return base

    distance_calibration = _estimate_au_distance_calibration(standardized_bank)
    auto_tau = float(distance_calibration["distance_tau"])
    auto_margin_saturation = float(distance_calibration["margin_saturation"])
    effective_tau = float(distance_tau) if float(distance_tau) > 0.0 else auto_tau
    effective_margin_saturation = (
        float(margin_saturation)
        if float(margin_saturation) > 0.0
        else auto_margin_saturation
    )
    effective_tau = max(1e-6, effective_tau)
    effective_margin_saturation = max(1e-6, effective_margin_saturation)

    try:
        extracted = detector.extract_arrays(crops[: max(1, int(live_frame_count))])
    except Exception as exc:
        base["status"] = "au_extraction_failed"
        base["reason"] = str(exc)
        base["analysis_seconds"] = round(time.time() - started, 4)
        return base

    if list(getattr(detector, "au_columns", [])) != au_columns:
        base["status"] = "au_column_mismatch"
        base["reason"] = (
            f"Warm-up AU columns {au_columns!r} do not match live columns "
            f"{getattr(detector, 'au_columns', None)!r}."
        )
        base["analysis_seconds"] = round(time.time() - started, 4)
        return base

    sample_ref = next(iter(reference_bank.values()))[0]
    live_vectors: list[tuple[int, np.ndarray]] = []
    invalid_count = 0
    for frame_index, value in enumerate(extracted):
        if value is None:
            invalid_count += 1
            continue
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if (
            vector.size == 0
            or vector.size != sample_ref.size
            or not np.all(np.isfinite(vector))
        ):
            invalid_count += 1
            continue
        live_vectors.append((frame_index, _standardize_au_vector(vector, ref_mean, ref_scale)))

    if not live_vectors:
        base.update({
            "status": "invalid_live_au" if invalid_count else "no_live_au_vector",
            "invalid_live_au_count": invalid_count,
            "analysis_seconds": round(time.time() - started, 4),
        })
        return base

    per_frame: list[dict[str, Any]] = []
    aggregate_distances: dict[str, list[float]] = {
        emotion: [] for emotion in standardized_bank
    }
    for frame_index, live in live_vectors:
        matches: dict[str, Any] = {}
        for emotion, refs in standardized_bank.items():
            distances = [_au_rms_distance(live, ref) for ref in refs]
            best_idx = int(np.argmin(distances))
            best_distance = float(distances[best_idx])
            aggregate_distances[emotion].append(best_distance)
            matches[emotion] = {
                "closest_reference_index": best_idx,
                "distance": round(best_distance, 6),
                "similarity": round(
                    au_reference_similarity_from_distance(best_distance, effective_tau), 6
                ),
            }
        ranked_frame = sorted(matches.items(), key=lambda item: float(item[1]["distance"]))
        per_frame.append({
            "frame_index": frame_index,
            "winner": ranked_frame[0][0] if ranked_frame else None,
            "matches": matches,
        })

    mean_distances = {
        emotion: float(np.mean(distances))
        for emotion, distances in aggregate_distances.items()
        if distances
    }
    ranked = sorted(mean_distances.items(), key=lambda item: item[1])
    if not ranked:
        base["status"] = "no_usable_frame_decision"
        base["analysis_seconds"] = round(time.time() - started, 4)
        return base

    au_emotion, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else best_distance + effective_margin_saturation
    best_similarity = au_reference_similarity_from_distance(best_distance, effective_tau)
    distance_margin = max(0.0, float(second_distance) - float(best_distance))
    margin_component = clamp01(distance_margin / effective_margin_saturation)
    frame_winners = [item.get("winner") for item in per_frame if item.get("winner")]
    agreement_ratio = (
        sum(1 for winner in frame_winners if winner == au_emotion)
        / max(1, len(frame_winners))
    )

    # Separate *classification certainty* from *absolute reference similarity*.
    #
    # - margin_component asks: how clearly is the winning emotion separated
    #   from the second-best emotion?
    # - agreement_ratio asks: how consistently do the live frames choose the
    #   same winning emotion?
    # - best_similarity asks: how close is the live expression to this
    #   participant's stored references in absolute standardized AU space?
    #
    # A live expression can therefore be a clear relative winner even when it
    # is not very close to any stored reference.  ``confidence`` deliberately
    # reports classification certainty; ARALF reliability combines it with
    # absolute reference similarity below.
    classification_confidence = clamp01(
        0.60 * margin_component + 0.40 * agreement_ratio
    )

    similarities = {
        emotion: au_reference_similarity_from_distance(distance, effective_tau)
        for emotion, distance in mean_distances.items()
    }
    sim_total = float(sum(similarities.values()))
    au_distribution = zero_distribution()
    if sim_total > 0.0:
        for emotion, value in similarities.items():
            au_distribution[emotion] = value / sim_total

    return {
        **base,
        "available": True,
        "status": "nearest_reference_match",
        "confidence": round(classification_confidence, 6),
        "classification_confidence": round(classification_confidence, 6),
        "reference_similarity": round(best_similarity, 6),
        "au_emotion": au_emotion,
        "agreement": (
            normalize_ekman_emotion(deepface_emotion) == au_emotion
            if deepface_emotion else None
        ),
        "best_mean_distance": round(float(best_distance), 6),
        "second_best_mean_distance": round(float(second_distance), 6),
        "distance_margin": round(distance_margin, 6),
        "best_similarity": round(best_similarity, 6),
        "margin_component": round(margin_component, 6),
        "frame_agreement_ratio": round(agreement_ratio, 6),
        "per_frame": per_frame,
        "mean_distances": {emotion: round(value, 6) for emotion, value in mean_distances.items()},
        "au_distribution": {emotion: round(value, 6) for emotion, value in au_distribution.items()},
        "reference_counts": {emotion: len(refs) for emotion, refs in reference_bank.items()},
        "live_crop_count": len(live_vectors),
        "invalid_live_au_count": invalid_count,
        "calibration_status": calibration_status,
        "distance_space": "participant_reference_zscore_rms",
        "distance_tau": round(effective_tau, 6),
        "distance_tau_source": "override" if float(distance_tau) > 0.0 else "reference_auto",
        "margin_saturation_used": round(effective_margin_saturation, 6),
        "margin_saturation_source": (
            "override" if float(margin_saturation) > 0.0 else "reference_auto"
        ),
        "au_std_floor": round(scale_floor, 6),
        "reference_distance_calibration": {
            key: round(float(value), 6) for key, value in distance_calibration.items()
        },
        "analysis_seconds": round(time.time() - started, 4),
    }


def au_reliability(
    verification: Optional[dict[str, Any]],
    *,
    partial_scale: float = 0.65,
) -> float:
    """How much ARALF should trust the AU prediction.

    ``verification["confidence"]`` is classification certainty: how clearly
    the winning class separates from competitors and how consistently frames
    agree.  ``reference_similarity`` is absolute closeness to the participant's
    stored AU references.  ARALF reliability intentionally requires *both*:

        reliability = classification_confidence * reference_similarity

    A partial calibration profile receives one additional calibration-quality
    discount.  This prevents a very clear relative winner that is far from all
    stored references from receiving excessive fusion weight.
    """
    if not verification or not verification.get("available"):
        return 0.0
    status = str(verification.get("calibration_status", ""))
    if status not in {"ready", "partial"}:
        return 0.0

    classification_confidence = clamp01(float(
        verification.get("classification_confidence", verification.get("confidence", 0.0))
    ))
    reference_similarity = clamp01(float(
        verification.get("reference_similarity", verification.get("best_similarity", 0.0))
    ))
    reliability = classification_confidence * reference_similarity
    if status == "partial":
        reliability *= clamp01(partial_scale)
    return clamp01(reliability)


@dataclass
class ARALFFusionOutput:
    emotion: str
    confidence: float
    scores: dict[str, float]
    weights: dict[str, float]
    reason: str


def adaptive_reliability_fusion(
    *,
    text_dist: dict[str, float],
    text_rel: float,
    deepface_dist: Optional[dict[str, float]],
    deepface_rel: float,
    au_dist: Optional[dict[str, float]],
    au_rel: float,
    prosody_dist: Optional[dict[str, float]],
    prosody_rel: float,
    text_prior: float = 0.30,
    face_prior: float = 0.60,
    prosody_prior: float = 0.10,
) -> ARALFFusionOutput:
    """Two-level ARALF fusion with an evidence-quality confidence discount.

    Top level: text / face / prosody compete by prior×reliability.
    Face level: the single face share is split between DeepFace and AU by their
    own reliabilities, so two pipelines on the same crop do not double-count
    facial evidence.

    The relative active weights are renormalized to one for score mixing.  To
    fix the old pool leak, the final dominant confidence is multiplied by the
    weighted mean reliability of the *available* top-level channels.  Thus a
    lone low-confidence modality may get relative weight 1.0, but it cannot
    produce high fused confidence merely because every other modality dropped.
    """
    text_dist_n = normalize_distribution(text_dist)
    deepface_dist_n = normalize_distribution(deepface_dist)
    au_dist_n = normalize_distribution(au_dist)
    prosody_dist_n = normalize_distribution(prosody_dist)

    text_rel = clamp01(text_rel)
    deepface_rel = clamp01(deepface_rel)
    au_rel = clamp01(au_rel)
    prosody_rel = clamp01(prosody_rel)

    # max() is intentionally conservative: DeepFace and AU are complementary
    # feature pipelines over the same face, not independent modalities.
    face_rel = max(deepface_rel, au_rel)

    text_prior = max(0.0, float(text_prior))
    face_prior = max(0.0, float(face_prior))
    prosody_prior = max(0.0, float(prosody_prior))

    raw_text = text_prior * text_rel
    raw_face = face_prior * face_rel
    raw_prosody = prosody_prior * prosody_rel
    raw_total = raw_text + raw_face + raw_prosody

    if raw_total <= 1e-12:
        scores = zero_distribution()
        scores["neutral"] = 1.0
        return ARALFFusionOutput(
            emotion="neutral",
            confidence=0.0,
            scores=scores,
            weights={
                "prior_text": text_prior,
                "prior_face": face_prior,
                "prior_prosody": prosody_prior,
                "reliability_text": text_rel,
                "reliability_face": face_rel,
                "reliability_deepface": deepface_rel,
                "reliability_au": au_rel,
                "reliability_prosody": prosody_rel,
                "active_text": 0.0,
                "active_face": 0.0,
                "active_deepface": 0.0,
                "active_au": 0.0,
                "active_prosody": 0.0,
                "evidence_quality": 0.0,
            },
            reason="ARALF had no reliable modality evidence; neutral fallback with zero confidence.",
        )

    wt = raw_text / raw_total
    w_face = raw_face / raw_total
    wp = raw_prosody / raw_total

    face_denom = deepface_rel + au_rel
    if face_denom > 1e-12:
        w_deepface = w_face * (deepface_rel / face_denom)
        w_au = w_face * (au_rel / face_denom)
    else:
        w_deepface = 0.0
        w_au = 0.0

    scores = {
        emotion: (
            wt * text_dist_n.get(emotion, 0.0)
            + w_deepface * deepface_dist_n.get(emotion, 0.0)
            + w_au * au_dist_n.get(emotion, 0.0)
            + wp * prosody_dist_n.get(emotion, 0.0)
        )
        for emotion in EKMAN_EMOTION_LABELS
    }
    score_total = float(sum(scores.values()))
    if score_total > 0.0:
        scores = {emotion: value / score_total for emotion, value in scores.items()}

    dominant, dominant_score = max(scores.items(), key=lambda item: item[1])

    # Do not penalize a modality for being genuinely unavailable; do penalize
    # weak evidence among channels that are present.  Presence is inferred from
    # non-zero reliability because unavailable channels are hard-zeroed by caller.
    active_prior_denominator = 0.0
    if text_rel > 0.0:
        active_prior_denominator += text_prior
    if face_rel > 0.0:
        active_prior_denominator += face_prior
    if prosody_rel > 0.0:
        active_prior_denominator += prosody_prior
    evidence_quality = (
        clamp01(raw_total / active_prior_denominator)
        if active_prior_denominator > 1e-12 else 0.0
    )
    confidence = clamp01(dominant_score * evidence_quality)

    reason = (
        f"Reliability-adaptive ARALF selected {dominant}: "
        f"text rel={text_rel:.2f} weight={wt:.2f}; "
        f"face rel={face_rel:.2f} weight={w_face:.2f} "
        f"(DeepFace rel={deepface_rel:.2f} weight={w_deepface:.2f}, "
        f"AU rel={au_rel:.2f} weight={w_au:.2f}); "
        f"prosody rel={prosody_rel:.2f} weight={wp:.2f}; "
        f"evidence_quality={evidence_quality:.2f}."
    )
    return ARALFFusionOutput(
        emotion=dominant,
        confidence=confidence,
        scores=scores,
        weights={
            "prior_text": text_prior,
            "prior_face": face_prior,
            "prior_prosody": prosody_prior,
            "reliability_text": text_rel,
            "reliability_face": face_rel,
            "reliability_deepface": deepface_rel,
            "reliability_au": au_rel,
            "reliability_prosody": prosody_rel,
            "raw_text": raw_text,
            "raw_face": raw_face,
            "raw_prosody": raw_prosody,
            "active_text": wt,
            "active_face": w_face,
            "active_deepface": w_deepface,
            "active_au": w_au,
            "active_prosody": wp,
            "evidence_quality": evidence_quality,
        },
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Shared prosody implementation
# ---------------------------------------------------------------------------

def prosody_frame_features(
    audio: np.ndarray,
    sample_rate: int,
    *,
    periodicity_threshold: float = 0.30,
) -> dict[str, float]:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {}
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    duration = float(x.size / max(1, sample_rate))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x))))) if x.size > 1 else 0.0

    frame_len = max(256, int(round(0.040 * sample_rate)))
    hop = max(128, int(round(0.020 * sample_rate)))
    if x.size < frame_len:
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[:x.size] = x
        x_for_frames = padded
    else:
        x_for_frames = x

    frames: list[np.ndarray] = []
    frame_rms: list[float] = []
    for start_idx in range(0, max(1, x_for_frames.size - frame_len + 1), hop):
        frame = x_for_frames[start_idx:start_idx + frame_len]
        if frame.size < frame_len:
            break
        frames.append(frame)
        frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))
    if not frames:
        frames = [x_for_frames[:frame_len]]
        frame_rms = [float(np.sqrt(np.mean(frames[0] ** 2)))]

    rms_array = np.asarray(frame_rms, dtype=np.float32)
    energy_std = float(np.std(rms_array))
    energy_mean = float(np.mean(rms_array))
    energy_cv = float(energy_std / max(1e-8, energy_mean))
    energy_p90 = float(np.percentile(rms_array, 90))
    energy_p10 = float(np.percentile(rms_array, 10))
    energy_range = max(0.0, energy_p90 - energy_p10)

    active_threshold = max(0.008, float(np.percentile(rms_array, 35)) * 0.70)
    window = np.hanning(frame_len).astype(np.float32)
    min_lag = max(1, int(sample_rate / 350.0))
    max_lag = min(frame_len - 2, int(sample_rate / 70.0))
    nfft = 1 << ((2 * frame_len - 1).bit_length())

    pitches: list[float] = []
    periodicities: list[float] = []
    centroids: list[float] = []
    flatness_values: list[float] = []
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    for frame, frms in zip(frames, frame_rms):
        if frms < active_threshold:
            continue
        centered = frame - float(np.mean(frame))
        weighted = centered * window
        mag = np.abs(np.fft.rfft(weighted)).astype(np.float64)
        mag_sum = float(np.sum(mag))
        if mag_sum > 1e-12:
            centroids.append(float(np.sum(freqs * mag) / mag_sum))
            flatness_values.append(float(
                np.exp(np.mean(np.log(mag + 1e-12))) / (np.mean(mag) + 1e-12)
            ))
        spectrum = np.fft.rfft(weighted, n=nfft)
        ac = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:frame_len]
        ac0 = float(ac[0]) if ac.size else 0.0
        if ac0 <= 1e-10 or max_lag <= min_lag:
            continue
        search = ac[min_lag:max_lag + 1]
        rel_idx = int(np.argmax(search))
        lag = min_lag + rel_idx
        periodicity = float(ac[lag] / ac0)
        periodicities.append(periodicity)
        if periodicity >= float(periodicity_threshold):
            f0 = float(sample_rate / lag)
            if 70.0 <= f0 <= 350.0:
                pitches.append(f0)

    active_count = sum(1 for value in frame_rms if value >= active_threshold)
    voiced_ratio = float(len(pitches) / max(1, active_count))
    if pitches:
        pitch_arr = np.asarray(pitches, dtype=np.float32)
        pitch_median = float(np.median(pitch_arr))
        pitch_mean = float(np.mean(pitch_arr))
        pitch_std = float(np.std(pitch_arr))
        pitch_p90 = float(np.percentile(pitch_arr, 90))
        pitch_p10 = float(np.percentile(pitch_arr, 10))
        pitch_range = max(0.0, pitch_p90 - pitch_p10)
    else:
        pitch_median = pitch_mean = pitch_std = pitch_range = 0.0

    return {
        "peak": peak,
        "rms": rms,
        "duration": duration,
        "energy_std": energy_std,
        "energy_cv": energy_cv,
        "energy_range": energy_range,
        "zero_crossing_rate": zcr,
        "pitch_median_hz": pitch_median,
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "pitch_range_hz": pitch_range,
        "voiced_ratio": voiced_ratio,
        "spectral_centroid_hz": float(np.median(centroids)) if centroids else 0.0,
        "spectral_flatness": float(np.median(flatness_values)) if flatness_values else 0.0,
        "periodicity_median": float(np.median(periodicities)) if periodicities else 0.0,
    }


def build_prosody_baseline(
    feature_samples: Iterable[dict[str, Any]],
    *,
    minimum_ready_samples: int = 3,
) -> Optional[dict[str, Any]]:
    """Build a participant-neutral prosody reference from repeated utterances.

    A minimum of three valid neutral samples is recommended.  Fewer samples are
    retained as a ``partial`` profile for backwards compatibility, but callers
    can use ``sample_quality`` to reduce their reliability.  Across-utterance
    scale estimates use MAD where possible and retain conservative floors so a
    very steady neutral reading cannot create exaggerated z-scores.
    """
    samples = [dict(item) for item in feature_samples if item]
    valid = [
        item for item in samples
        if float(item.get("pitch_median_hz", 0.0) or 0.0) > 0.0
        and float(item.get("rms", 0.0) or 0.0) > 0.0
    ]
    if not valid:
        return None

    minimum_ready_samples = max(1, int(minimum_ready_samples))

    def arr(key: str) -> np.ndarray:
        return np.asarray(
            [float(item.get(key, 0.0) or 0.0) for item in valid],
            dtype=np.float64,
        )

    def robust_scale(values: np.ndarray) -> float:
        if values.size <= 1:
            return 0.0
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return 1.4826 * mad

    pitch = arr("pitch_median_hz")
    rms = arr("rms")
    pitch_range = arr("pitch_range_hz")
    within_pitch_std = arr("pitch_std_hz")
    within_energy_std = arr("energy_std")

    pitch_center = float(np.median(pitch))
    rms_center = float(np.median(rms))
    range_center = float(np.median(pitch_range))

    pitch_scale = max(
        15.0,
        float(np.median(within_pitch_std)) if within_pitch_std.size else 0.0,
        robust_scale(pitch),
    )
    rms_scale = max(
        0.02,
        float(np.median(within_energy_std)) if within_energy_std.size else 0.0,
        robust_scale(rms),
    )
    range_scale = max(
        20.0,
        2.0 * (float(np.median(within_pitch_std)) if within_pitch_std.size else 0.0),
        robust_scale(pitch_range),
    )

    sample_count = len(valid)
    sample_quality = clamp01(sample_count / float(minimum_ready_samples))
    return {
        "version": 2,
        "status": "ready" if sample_count >= minimum_ready_samples else "partial",
        "sample_count": sample_count,
        "minimum_ready_samples": minimum_ready_samples,
        "sample_quality": sample_quality,
        "source": "repeated_neutral_warmup_utterances",
        "pitch_median_hz": pitch_center,
        "pitch_scale_hz": pitch_scale,
        "rms": rms_center,
        "rms_scale": rms_scale,
        "pitch_range_hz": range_center,
        "pitch_range_scale_hz": range_scale,
        "samples": valid,
    }


def add_prosody_zscores(
    features: dict[str, float],
    baseline: Optional[dict[str, Any]],
) -> dict[str, float]:
    out = dict(features)
    status = str((baseline or {}).get("status", ""))
    if not baseline or status not in {"ready", "partial"}:
        out["participant_baseline_available"] = 0.0
        out["participant_baseline_quality"] = 0.0
        return out
    try:
        pitch_scale = max(1e-8, float(baseline.get("pitch_scale_hz", 15.0)))
        rms_scale = max(1e-8, float(baseline.get("rms_scale", 0.02)))
        range_scale = max(1e-8, float(baseline.get("pitch_range_scale_hz", 20.0)))
        out["pitch_level_z"] = (
            float(out.get("pitch_median_hz", 0.0))
            - float(baseline.get("pitch_median_hz", 0.0))
        ) / pitch_scale
        out["rms_z"] = (
            float(out.get("rms", 0.0)) - float(baseline.get("rms", 0.0))
        ) / rms_scale
        out["pitch_range_z"] = (
            float(out.get("pitch_range_hz", 0.0))
            - float(baseline.get("pitch_range_hz", 0.0))
        ) / range_scale
        minimum = max(1, int(baseline.get("minimum_ready_samples", 3) or 3))
        count = max(0, int(baseline.get("sample_count", 0) or 0))
        quality = baseline.get("sample_quality")
        if quality is None:
            quality = count / float(minimum)
        out["participant_baseline_available"] = 1.0
        out["participant_baseline_quality"] = clamp01(float(quality))
    except Exception:
        out["participant_baseline_available"] = 0.0
        out["participant_baseline_quality"] = 0.0
    return out


def _scaled_confidence(
    value: float,
    low: float,
    high: float,
    *,
    max_confidence: float,
    floor: float = 0.15,
) -> float:
    if high <= low:
        return min(max_confidence, floor)
    strength = clamp01((value - low) / (high - low))
    return min(max_confidence, floor + strength * (max_confidence - floor))


def analyze_prosody_shared(
    audio: np.ndarray,
    sample_rate: int,
    *,
    baseline: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Return a continuous Ekman prosody distribution plus bounded reliability.

    Prosody no longer behaves as an all-or-nothing categorical gate.  Once an
    utterance has enough voiced speech for acoustic analysis, the function
    returns a seven-class score distribution.  Ambiguous cues remain usable but
    receive low reliability, so ARALF can give them a small weight instead of
    dropping prosody entirely.
    """
    cfg = {
        "min_duration": 0.45,
        "min_voiced_ratio": 0.12,
        "max_confidence": 0.35,
        "low_rms": 0.08,
        "high_rms": 0.30,
        "very_high_rms": 0.45,
        "high_pitch_median_hz": 220.0,
        "high_pitch_range_hz": 90.0,
        "very_high_pitch_range_hz": 140.0,
        "low_pitch_range_hz": 45.0,
        "anger_zcr": 0.075,
        "anger_centroid_hz": 1200.0,
        "periodicity_threshold": 0.30,
        "z_moderate": 0.75,
        "z_high": 1.25,
        "z_very_high": 1.75,
        "z_low": -1.0,
    }
    if config:
        for key, value in config.items():
            try:
                cfg[key] = float(value)
            except Exception:
                pass

    empty = zero_distribution()
    if audio is None or np.asarray(audio).size == 0:
        return {
            "available": False,
            "emotion": "neutral",
            "confidence": 0.0,
            "reliability": 0.0,
            "scores": empty,
            "reason": "No raw utterance audio was available for prosody analysis.",
            "features": {},
        }

    try:
        features = prosody_frame_features(
            np.asarray(audio, dtype=np.float32),
            sample_rate,
            periodicity_threshold=cfg["periodicity_threshold"],
        )
        if not features:
            raise RuntimeError("no acoustic features could be extracted")
        features = add_prosody_zscores(features, baseline)

        rms = float(features.get("rms", 0.0))
        duration = float(features.get("duration", 0.0))
        zcr = float(features.get("zero_crossing_rate", 0.0))
        pitch_median = float(features.get("pitch_median_hz", 0.0))
        pitch_range = float(features.get("pitch_range_hz", 0.0))
        pitch_std = float(features.get("pitch_std_hz", 0.0))
        voiced_ratio = float(features.get("voiced_ratio", 0.0))
        centroid = float(features.get("spectral_centroid_hz", 0.0))
        periodicity = float(features.get("periodicity_median", 0.0))
        baseline_available = bool(features.get("participant_baseline_available", 0.0) >= 0.5)
        baseline_quality = clamp01(float(features.get("participant_baseline_quality", 0.0)))

        if duration < cfg["min_duration"]:
            return {
                "available": False,
                "emotion": "neutral",
                "confidence": 0.0,
                "reliability": 0.0,
                "scores": empty,
                "reason": "Utterance was too short for a stable acoustic prosody estimate.",
                "features": features,
            }
        if voiced_ratio < cfg["min_voiced_ratio"] or pitch_median <= 0.0:
            return {
                "available": False,
                "emotion": "neutral",
                "confidence": 0.0,
                "reliability": 0.0,
                "scores": empty,
                "reason": "Too little stable voiced speech was available for prosody classification.",
                "features": features,
            }

        def pos(value: float, threshold: float = 0.20, span: float = 1.50) -> float:
            return clamp01((float(value) - threshold) / max(1e-8, span))

        roughness = max(
            clamp01((zcr - cfg["anger_zcr"]) / 0.10),
            clamp01((centroid - cfg["anger_centroid_hz"]) / 1800.0),
        )

        if baseline_available:
            pitch_z = float(features.get("pitch_level_z", 0.0))
            rms_z = float(features.get("rms_z", 0.0))
            range_z = float(features.get("pitch_range_z", 0.0))

            high_pitch = pos(pitch_z)
            low_pitch = pos(-pitch_z)
            high_energy = pos(rms_z)
            low_energy = pos(-rms_z)
            expressive = pos(range_z)
            restricted = pos(-range_z)

            neutral_distance = float(np.sqrt(np.mean(np.square([
                np.clip(pitch_z, -3.0, 3.0),
                np.clip(rms_z, -3.0, 3.0),
                np.clip(range_z, -3.0, 3.0),
            ]))))
            neutral_similarity = float(np.exp(-0.9 * neutral_distance))

            raw_scores = {
                "neutral": 0.30 + 0.90 * neutral_similarity,
                "sadness": 0.10 + 1.20 * low_energy + 1.00 * restricted
                           + 0.35 * low_pitch + 0.80 * low_energy * restricted,
                "joy": 0.10 + 1.00 * high_energy + 0.85 * expressive
                       + 0.30 * high_pitch + 0.35 * high_energy * expressive,
                "anger": 0.08 + 1.10 * high_energy + 0.85 * roughness
                         + 0.25 * expressive + 0.35 * high_energy * roughness,
                "fear": 0.08 + 1.00 * high_pitch + 0.75 * expressive
                        + 0.20 * roughness,
                "surprise": 0.08 + 1.20 * expressive + 0.65 * high_pitch
                            + 0.35 * high_energy,
                "disgust": 0.06 + 0.50 * roughness + 0.30 * low_pitch
                           + 0.15 * restricted,
            }
            signal_strength = clamp01(neutral_distance / 1.75)
            calibration_factor = 0.55 + 0.45 * baseline_quality
            mode = "participant-normalized"
        else:
            # Absolute fallback is kept only for participants without a usable
            # neutral profile.  Values are transformed into smooth activations
            # instead of hard if/else labels.
            high_pitch = clamp01((pitch_median - (cfg["high_pitch_median_hz"] - 35.0)) / 120.0)
            low_pitch = clamp01(((cfg["high_pitch_median_hz"] - 45.0) - pitch_median) / 100.0)
            high_energy = clamp01((rms - cfg["low_rms"]) / max(0.05, cfg["high_rms"] - cfg["low_rms"]))
            low_energy = clamp01((cfg["low_rms"] - rms) / max(0.02, cfg["low_rms"]))
            expressive = clamp01((pitch_range - cfg["low_pitch_range_hz"]) / max(20.0, cfg["very_high_pitch_range_hz"] - cfg["low_pitch_range_hz"]))
            restricted = clamp01((cfg["low_pitch_range_hz"] - pitch_range) / max(10.0, cfg["low_pitch_range_hz"]))
            neutral_similarity = clamp01(
                1.0 - 0.45 * high_energy - 0.35 * low_energy
                - 0.35 * expressive - 0.20 * high_pitch
            )
            raw_scores = {
                "neutral": 0.35 + 0.85 * neutral_similarity,
                "sadness": 0.10 + 1.15 * low_energy + 0.90 * restricted + 0.25 * low_pitch,
                "joy": 0.10 + 0.95 * high_energy + 0.80 * expressive + 0.20 * high_pitch,
                "anger": 0.08 + 1.05 * high_energy + 0.85 * roughness + 0.20 * expressive,
                "fear": 0.08 + 0.95 * high_pitch + 0.70 * expressive + 0.15 * roughness,
                "surprise": 0.08 + 1.10 * expressive + 0.60 * high_pitch + 0.30 * high_energy,
                "disgust": 0.06 + 0.45 * roughness + 0.25 * low_pitch + 0.10 * restricted,
            }
            signal_strength = max(high_pitch, low_pitch, high_energy, low_energy, expressive, restricted)
            calibration_factor = 0.70
            mode = "absolute-threshold fallback"

        scores = normalize_distribution(raw_scores)
        dominant, dominant_score = max(scores.items(), key=lambda item: item[1])
        ordered_scores = sorted(scores.values(), reverse=True)
        margin = ordered_scores[0] - ordered_scores[1] if len(ordered_scores) > 1 else ordered_scores[0]

        duration_quality = clamp01((duration - cfg["min_duration"]) / 1.50 + 0.35)
        voicing_quality = clamp01(
            (voiced_ratio - cfg["min_voiced_ratio"])
            / max(0.10, 0.70 - cfg["min_voiced_ratio"])
        )
        periodicity_quality = clamp01(
            (periodicity - cfg["periodicity_threshold"])
            / max(0.10, 0.75 - cfg["periodicity_threshold"])
        )
        acoustic_quality = clamp01(
            0.35 * duration_quality + 0.40 * voicing_quality + 0.25 * periodicity_quality
        )

        decisiveness = clamp01(margin / 0.25)
        reliability = cfg["max_confidence"] * acoustic_quality * calibration_factor * (
            0.35 + 0.35 * signal_strength + 0.30 * decisiveness
        )
        reliability = clamp01(min(cfg["max_confidence"], reliability))

        features["prosody_distribution_margin"] = float(margin)
        features["prosody_signal_strength"] = float(signal_strength)
        features["prosody_acoustic_quality"] = float(acoustic_quality)

        return {
            "available": True,
            "emotion": dominant,
            "confidence": reliability,
            "reliability": reliability,
            "scores": scores,
            "reason": (
                f"Continuous {mode} prosody distribution selected {dominant} "
                f"(score={dominant_score:.2f}, reliability={reliability:.2f}, "
                f"margin={margin:.2f}); mixed cues are retained with low weight rather than abstained."
            ),
            "features": features,
        }
    except Exception as exc:
        return {
            "available": False,
            "emotion": "neutral",
            "confidence": 0.0,
            "reliability": 0.0,
            "scores": empty,
            "reason": f"Prosody analysis failed: {exc}",
            "features": {},
        }
