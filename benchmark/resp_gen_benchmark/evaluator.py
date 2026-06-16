"""
evaluator.py
Metric computation for response generation:

  - BLEU          NLTK sentence BLEU (smoothed)
  - ROUGE-L       F1 via rouge-score
  - BERTScore F1  Contextual semantic similarity (optional)
  - Empathy Score P(empathetic) via zero-shot NLI (optional)
  - Latency       Recorded by model_client, aggregated here
"""

import warnings

import nltk
import numpy as np

warnings.filterwarnings("ignore")
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib


# ── BLEU ──────────────────────────────────────────────────────────────────────

_smooth = SmoothingFunction().method1

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    if not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_smooth)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

_rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    return _rouge.score(reference, hypothesis)["rougeL"].fmeasure


# ── BERTScore ─────────────────────────────────────────────────────────────────

try:
    from bert_score import score as _bert_score_fn
    _BERTSCORE_OK = True
except ImportError:
    _BERTSCORE_OK = False
    print("[evaluator] bert_score not installed — BERTScore disabled. "
          "Run: pip install bert-score")


def batch_bertscore(
    references:  list[str],
    hypotheses:  list[str],
    lang:        str = "en",
) -> list[float | None]:
    if not _BERTSCORE_OK:
        return [None] * len(references)
    _, _, F1 = _bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return F1.tolist()


# ── Empathy Score (Zero-shot NLI) ─────────────────────────────────────────────

_NLI_LABELS   = ["empathetic", "neutral", "dismissive"]
_empathy_clf  = None    # lazily loaded


def load_empathy_classifier(device: int | str = "auto"):
    """
    Load the NLI classifier once and cache it.

    device: -1 or "cpu" → CPU; 0 or "gpu" → first GPU; "auto" → use GPU if
    available, else CPU.

    Call this explicitly before evaluate_batch() if you want the scorer.

    Note: facebook/bart-large-mnli needs ~1.6GB VRAM. On small GPUs (e.g.
    a 3GB card already running Ollama), this can cause CUDA OOM. If GPU
    loading fails, automatically falls back to CPU rather than aborting
    the whole benchmark run.
    """
    global _empathy_clf

    device = _resolve_device(device)

    try:
        from transformers import pipeline
        _empathy_clf = pipeline(
            "zero-shot-classification",
            model  = "facebook/bart-large-mnli",
            device = device,
        )
        where = "CPU" if device == -1 else f"GPU (cuda:{device})"
        print(f"[evaluator] Empathy classifier loaded on {where} (facebook/bart-large-mnli)")
    except Exception as exc:
        if device != -1:
            print(f"[evaluator] GPU load failed ({exc}); falling back to CPU …")
            try:
                from transformers import pipeline
                _empathy_clf = pipeline(
                    "zero-shot-classification",
                    model  = "facebook/bart-large-mnli",
                    device = -1,
                )
                print("[evaluator] Empathy classifier loaded on CPU (facebook/bart-large-mnli)")
            except Exception as exc2:
                print(f"[evaluator] Could not load empathy classifier: {exc2}")
                _empathy_clf = None
        else:
            print(f"[evaluator] Could not load empathy classifier: {exc}")
            _empathy_clf = None

    return _empathy_clf


def _pick_device() -> int:
    """
    Return 0 if a CUDA GPU is available, else -1 (CPU).
    Does not check free VRAM — callers should handle OOM via fallback.
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def _resolve_device(device: int | str) -> int:
    """
    Normalize a user-supplied device spec into the int form `pipeline()`
    expects (-1 for CPU, 0+ for GPU index).

    Accepts: -1, 0, 1, ... (already valid ints) and the strings
    "auto", "cpu", "gpu"/"cuda" (case-insensitive).
    """
    if isinstance(device, str):
        key = device.strip().lower()
        if key == "auto":
            return _pick_device()
        if key == "cpu":
            return -1
        if key in ("gpu", "cuda"):
            return 0
        raise ValueError(
            f"Unrecognized device spec '{device}'. Use -1, 0, 'auto', 'cpu', or 'gpu'."
        )
    return device


def compute_empathy_score(text: str, clf=None) -> float | None:
    """Return P(empathetic) ∈ [0, 1], or None if classifier unavailable."""
    clf = clf or _empathy_clf
    if clf is None or not text.strip():
        return None
    result      = clf(text, _NLI_LABELS, multi_label=False)
    label_probs = dict(zip(result["labels"], result["scores"]))
    return label_probs.get("empathetic", 0.0)


# ── Batch evaluation (main interface used by benchmark_runner) ────────────────

def evaluate_batch(
    references:  list[str],
    hypotheses:  list[str],
    latencies:   list[float],
    empathy_clf  = None,
) -> dict[str, list]:
    """
    Compute all metrics for a batch of (reference, hypothesis) pairs.

    Returns a dict:
        bleu, rouge_l, bert_score_f1, empathy_score, latency_s
        — each a list of per-sample scores (float or None).
    """
    n = len(references)
    assert len(hypotheses) == len(latencies) == n, \
        "references, hypotheses, and latencies must have the same length."

    clf = empathy_clf or _empathy_clf

    bleu    = [compute_bleu(r, h)            for r, h in zip(references, hypotheses)]
    rouge   = [compute_rouge_l(r, h)         for r, h in zip(references, hypotheses)]
    bert    = batch_bertscore(references, hypotheses)
    empathy = [compute_empathy_score(h, clf) for h in hypotheses]

    return {
        "bleu":          bleu,
        "rouge_l":       rouge,
        "bert_score_f1": bert,
        "empathy_score": empathy,
        "latency_s":     list(latencies),
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate(scores: list[float | None]) -> dict[str, float]:
    """Mean and std, ignoring None / NaN."""
    clean = [v for v in scores if v is not None and not np.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(clean)), "std": float(np.std(clean))}


SCORE_COLS = ["bleu", "rouge_l", "bert_score_f1", "empathy_score", "latency_s"]
