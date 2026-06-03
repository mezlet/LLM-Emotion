"""
data_loader.py
HuggingFace EmpatheticDialogues loader with stratified sampling.

Each sample produced:
  conv_id   : conversation identifier
  emotion   : emotion label for the conversation
  context   : up to 4 preceding turns joined by ' | '
  utterance : final speaker turn  (used as the prompt)
  reference : final listener turn (gold response for evaluation)
"""

import pandas as pd
from datasets import load_dataset

from config import DATASET


def load_empathetic_dialogues(
    split:       str  = DATASET["split"],
    num_samples: int  = DATASET["num_samples"],
) -> list[dict]:
    """
    Load and preprocess EmpatheticDialogues.

    Returns a list of sample dicts ready for inference.
    If num_samples is None or 0, the full split is returned.
    """
    print(f"[data_loader] Loading '{DATASET['hf_path']}' ({split} split) …")
    ds = load_dataset(DATASET["hf_path"], split=split, trust_remote_code=True)

    samples = _group_conversations(ds)

    if num_samples:
        samples = _stratified_sample(samples, num_samples)

    n_emotions = len({s["emotion"] for s in samples})
    print(f"[data_loader] {len(samples)} samples | {n_emotions} emotion categories")
    return samples


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_conversations(ds) -> list[dict]:
    """
    Group dataset rows by conv_id and extract one sample per conversation:
      - last speaker turn  → utterance (prompt)
      - last listener turn → reference (gold response)
      - preceding turns    → context (up to 4 turns)
    """
    samples      = []
    current_id   = None
    current_emo  = None
    history: list[str] = []

    for row in ds:
        conv_id   = row["conv_id"]
        utterance = row["utterance"].strip()
        emotion   = row["context"]          # HF field 'context' = emotion label

        if conv_id != current_id:
            if len(history) >= 2:
                samples.append(_make_sample(current_id, current_emo, history))
            current_id  = conv_id
            current_emo = emotion
            history     = []

        history.append(utterance)

    # flush last conversation
    if len(history) >= 2:
        samples.append(_make_sample(current_id, current_emo, history))

    return samples


def _make_sample(conv_id: str, emotion: str, history: list[str]) -> dict:
    *ctx, speaker_turn, listener_turn = history
    context_str = " | ".join(ctx[-4:]) if ctx else ""
    return {
        "conv_id":   conv_id,
        "emotion":   emotion,
        "context":   context_str,
        "utterance": speaker_turn,
        "reference": listener_turn,
    }


def _stratified_sample(samples: list[dict], n: int) -> list[dict]:
    """
    Stratified sample across emotion categories so all emotions are represented.
    """
    df         = pd.DataFrame(samples)
    n_emotions = df["emotion"].nunique()
    per_emo    = max(1, n // n_emotions)

    sampled = (
        df.groupby("emotion", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), per_emo), random_state=42))
    )
    return sampled.head(n).to_dict("records")
