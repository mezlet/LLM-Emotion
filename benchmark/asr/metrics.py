"""
ASR evaluation metrics: WER, CER, RTF.
Uses the `jiwer` library for WER/CER computation.
"""


def _get_jiwer():
    try:
        import jiwer
        return jiwer
    except ImportError:
        raise ImportError(
            "jiwer is required for WER/CER computation. "
            "Install via: pip install jiwer"
        )


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate: (S + D + I) / N
    Returns a float in [0, ∞). Clamped at 1.0 for reporting.
    """
    jiwer = _get_jiwer()
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    wer = jiwer.wer(reference, hypothesis)
    return min(wer, 1.0)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate: edit distance at character level / len(reference).
    """
    jiwer = _get_jiwer()
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    cer = jiwer.cer(reference, hypothesis)
    return min(cer, 1.0)


def compute_rtf(inference_time_s: float, audio_duration_s: float) -> float:
    """
    Real-Time Factor: inference_time / audio_duration.
    RTF < 1.0 means faster than real-time.
    """
    if audio_duration_s <= 0:
        return float("inf")
    return inference_time_s / audio_duration_s