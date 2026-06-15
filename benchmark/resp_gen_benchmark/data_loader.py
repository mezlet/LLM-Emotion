"""
data_loader.py
HuggingFace dataset loaders with on-the-fly download, retries, fallback
mirrors, local on-disk caching, and stratified sampling.

Supported datasets:
  - EmpatheticDialogues (load_empathetic_dialogues)
  - DailyDialog         (load_daily_dialog)

Both loaders produce samples in a common schema:
  conv_id   : conversation identifier
  emotion   : emotion label (reference-turn emotion for DailyDialog)
  context   : up to 4 preceding turns joined by ' | '
  utterance : final speaker turn  (used as the prompt)
  reference : final listener turn (gold response for evaluation)

A unified entry point `load_dataset_samples(name, ...)` dispatches to the
correct loader based on config.ACTIVE_DATASET or an explicit name.

DOWNLOAD BEHAVIOUR
  On first call, the raw HF dataset is downloaded (datasets library handles
  caching under ~/.cache/huggingface by default). To avoid re-downloading
  on every run AND to avoid re-running the (slower) grouping/preprocessing
  step, the *preprocessed* sample list is additionally cached as a local
  JSON file under config.CACHE_DIR. Pass `force_download=True` to bypass
  both caches and re-fetch from the source.
"""

import json
import time

import pandas as pd
from datasets import load_dataset as hf_load_dataset

from config import DATASET, DATASETS, ACTIVE_DATASET, CACHE_DIR

DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_DELAY = 3.0  # seconds, doubled each retry


def _hf_load_single(hf_path: str, split: str):
    """
    Single download attempt, handling two known failure modes on recent
    `datasets` versions:

      1. `trust_remote_code` is no longer an accepted kwarg for script-based
         datasets — retry without it.
      2. Even without that kwarg, script-based datasets now raise
         `RuntimeError: Dataset scripts are no longer supported`. HuggingFace
         auto-converts every dataset (including script-based ones) to Parquet
         on the `refs/convert/parquet` branch — retry against that revision,
         which bypasses the loading script entirely.
    """
    try:
        return hf_load_dataset(hf_path, split=split, trust_remote_code=True)
    except TypeError:
        pass  # newer `datasets` versions don't accept trust_remote_code
    except (ValueError, RuntimeError) as e:
        if "trust_remote_code" not in str(e) and "Dataset scripts" not in str(e):
            raise

    try:
        return hf_load_dataset(hf_path, split=split)
    except RuntimeError as e:
        if "Dataset scripts" not in str(e):
            raise
        # Fall back to the auto-converted Parquet revision (bypasses the
        # deprecated loading script entirely).
        print(f"[data_loader] '{hf_path}' uses a deprecated loading script; "
              f"retrying via refs/convert/parquet …")
        return hf_load_dataset(hf_path, split=split, revision="refs/convert/parquet")


def _load_with_retries(load_fn, label: str):
    """
    Call `load_fn()` with retries and exponential back-off.
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"[data_loader] Loading '{label}' [attempt {attempt}/{DOWNLOAD_RETRIES}] …")
            return load_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < DOWNLOAD_RETRIES:
                wait = DOWNLOAD_RETRY_DELAY * (2 ** (attempt - 1))
                print(f"[data_loader] Failed ({exc!r}). Retrying in {wait:.1f}s …")
                time.sleep(wait)
    raise last_exc


# ── Local sample cache (preprocessed JSON) ────────────────────────────────────

def _cache_path(dataset_name: str, split: str, num_samples: int | None) -> "Path":
    from pathlib import Path
    tag = num_samples if num_samples else "full"
    return Path(CACHE_DIR) / f"{dataset_name}_{split}_{tag}.json"


def _load_from_cache(dataset_name: str, split: str, num_samples: int | None):
    path = _cache_path(dataset_name, split, num_samples)
    if path.exists():
        with open(path) as f:
            samples = json.load(f)
        print(f"[data_loader] Loaded {len(samples)} cached samples ← {path}")
        return samples
    return None


def _save_to_cache(samples: list[dict], dataset_name: str, split: str, num_samples: int | None):
    path = _cache_path(dataset_name, split, num_samples)
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[data_loader] Cached {len(samples)} samples → {path}")


def _load_empathetic_dialogues_fbai_csv(split: str) -> list[dict]:
    """
    Fallback: download the official EmpatheticDialogues CSV directly from
    Facebook AI's public file server (the source `datasets.load_dataset`
    itself downloads from). Bypasses the `datasets` library entirely.

    CSV columns: conv_id, utterance_idx, context, prompt, speaker_idx,
                 utterance, selfeval, tags

    The original files have two known row-malformation patterns:

    1. A handful of rows have an unescaped comma inside `prompt`, splitting
       it into two fields (9 total instead of 8, extra field at index 4).
       Detected when field[4] is NOT a small integer (speaker_idx must be
       0 or 1) — in that case fields 3 and 4 are merged back together.

    2. Some rows (notably in test.csv) have a massive corrupted `tags`
       field containing pipe-delimited junk appended after an empty `tags`,
       producing >8 fields with the overflow at the END. Detected when
       field[4] IS a valid speaker_idx — in that case all fields from
       index 7 onward are joined back into a single `tags` value.

    Only `conv_id`, `context` (emotion label), and `utterance` are used
    downstream, but all 8 columns are returned for compatibility with the
    HF dataset row schema.

    Returns a list of dicts with the same keys as the HF dataset rows.
    """
    import csv
    import io
    import urllib.request

    split_map = {"train": "train", "validation": "valid", "valid": "valid",
                  "dev": "valid", "test": "test"}
    fname = split_map.get(split)
    if fname is None:
        raise ValueError(f"Unknown EmpatheticDialogues split '{split}'")

    url = f"https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    print(f"[data_loader] Fetching EmpatheticDialogues ({split}) from "
          f"dl.fbaipublicfiles.com …")

    import tarfile
    with urllib.request.urlopen(url, timeout=180) as resp:
        tar_bytes = resp.read()

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        member = tar.extractfile(f"empatheticdialogues/{fname}.csv")
        if member is None:
            raise FileNotFoundError(f"empatheticdialogues/{fname}.csv not found in tarball")
        text = member.read().decode("utf-8")

    rows = []
    malformed = 0
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    n_cols = len(header)  # 8: conv_id, utterance_idx, context, prompt, speaker_idx, utterance, selfeval, tags

    for fields in reader:
        if not fields:
            continue

        if len(fields) == n_cols:
            row = dict(zip(header, fields))

        elif len(fields) > n_cols:
            malformed += 1
            # Disambiguate by checking whether field[4] (speaker_idx) looks
            # like a valid speaker index (0 or 1).
            if fields[4] in ("0", "1"):
                # Pattern 2: overflow is trailing junk in `tags` (field 7+).
                fixed = fields[:7] + ["|".join(fields[7:])]
            else:
                # Pattern 1: unescaped comma split `prompt` (fields 3,4).
                # Merge fields 3..(len-n_cols+1) back into `prompt`, then
                # continue with the remaining fields in their normal slots.
                overflow = len(fields) - n_cols
                merged_prompt = ",".join(fields[3:4 + overflow])
                fixed = fields[:3] + [merged_prompt] + fields[4 + overflow:]

            if len(fixed) != n_cols:
                # Still malformed after repair attempt — skip rather than
                # risk silently corrupting conv_id/context/utterance.
                continue
            row = dict(zip(header, fixed))

        else:
            # Fewer fields than expected — skip.
            malformed += 1
            continue

        rows.append(row)

    if malformed:
        print(f"[data_loader] Repaired/skipped {malformed} malformed row(s) in {fname}.csv")
    print(f"[data_loader] Loaded {len(rows)} rows from FBAI CSV ({fname}.csv)")
    return rows


def _load_empathetic_dialogues_parquet_mirror(split: str) -> list[dict]:
    """
    Last-resort fallback: a community Parquet re-upload of EmpatheticDialogues
    with the same schema as the original (conv_id, utterance_idx, context,
    prompt, speaker_idx, utterance, selfeval, tags).
    """
    mirror = "Estwld/empathetic_dialogues_llm"
    print(f"[data_loader] Trying Parquet mirror '{mirror}' ({split}) …")
    ds = hf_load_dataset(mirror, split=split)
    return [dict(r) for r in ds]


def load_empathetic_dialogues(
    split:          str  = DATASET["split"],
    num_samples:    int  = DATASET["num_samples"],
    force_download: bool = False,
) -> list[dict]:
    """
    Load and preprocess EmpatheticDialogues, downloading on the fly.

    Returns a list of sample dicts ready for inference.
    If num_samples is None or 0, the full split is returned.

    Results are cached locally under config.CACHE_DIR; pass
    force_download=True to bypass the cache and re-download.
    """
    cfg = DATASETS["empathetic_dialogues"]

    if not force_download:
        cached = _load_from_cache("empathetic_dialogues", split, num_samples)
        if cached is not None:
            return cached

    # 1) HuggingFace `datasets` (with retries; includes refs/convert/parquet
    #    fallback inside _hf_load_single for deprecated loading scripts)
    ds = None
    try:
        ds = _load_with_retries(
            lambda: _hf_load_single(cfg["hf_path"], split), label=cfg["hf_path"]
        )
    except Exception as exc:
        print(f"[data_loader] HuggingFace EmpatheticDialogues load failed: {exc!r}")

    # 2) Fallback: official CSV from Facebook AI's file server
    if ds is None:
        try:
            ds = _load_with_retries(
                lambda: _load_empathetic_dialogues_fbai_csv(split),
                label="dl.fbaipublicfiles.com",
            )
        except Exception as exc:
            print(f"[data_loader] FBAI CSV fallback failed: {exc!r}")

    # 3) Last resort: community Parquet mirror
    if ds is None:
        try:
            ds = _load_with_retries(
                lambda: _load_empathetic_dialogues_parquet_mirror(split),
                label="Estwld/empathetic_dialogues_llm",
            )
        except Exception as exc:
            raise RuntimeError(
                "[data_loader] Could not load EmpatheticDialogues from any "
                "source (HuggingFace, dl.fbaipublicfiles.com, or community "
                f"Parquet mirror). Last error: {exc!r}"
            ) from exc

    samples = _group_ed_conversations(ds)

    if num_samples:
        samples = _stratified_sample(samples, num_samples, key="emotion")

    n_emotions = len({s["emotion"] for s in samples})
    print(f"[data_loader] {len(samples)} samples | {n_emotions} emotion categories")

    _save_to_cache(samples, "empathetic_dialogues", split, num_samples)
    return samples


# ── Internal helpers: EmpatheticDialogues ─────────────────────────────────────

def _group_ed_conversations(ds) -> list[dict]:
    """
    Group EmpatheticDialogues rows by conv_id and extract one sample per
    conversation:
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
                samples.append(_make_ed_sample(current_id, current_emo, history))
            current_id  = conv_id
            current_emo = emotion
            history     = []

        history.append(utterance)

    # flush last conversation
    if len(history) >= 2:
        samples.append(_make_ed_sample(current_id, current_emo, history))

    return samples


def _make_ed_sample(conv_id: str, emotion: str, history: list[str]) -> dict:
    *ctx, speaker_turn, listener_turn = history
    context_str = " | ".join(ctx[-4:]) if ctx else ""
    return {
        "conv_id":   conv_id,
        "emotion":   emotion,
        "context":   context_str,
        "utterance": speaker_turn,
        "reference": listener_turn,
    }


# ── Internal helpers: shared ─────────────────────────────────────────────────

def _stratified_sample(samples: list[dict], n: int, key: str = "emotion") -> list[dict]:
    """
    Stratified sample across `key` categories so all categories are represented.

    Note: on pandas >= 3.0, groupby(..., group_keys=False).apply(...) drops
    the grouping column from the result, so we sample indices per group and
    reselect from the original DataFrame to guarantee `key` is preserved.
    """
    df       = pd.DataFrame(samples)
    n_groups = df[key].nunique()
    per_grp  = max(1, n // n_groups)

    sampled_idx = (
        df.groupby(key, group_keys=False)
          .apply(lambda x: x.sample(min(len(x), per_grp), random_state=42).index,
                 include_groups=False)
    )
    # sampled_idx may be a Series of Index objects (one per group); flatten
    all_idx = [i for group_idx in sampled_idx for i in group_idx]

    sampled = df.loc[all_idx]
    return sampled.head(n).to_dict("records")


# ── DailyDialog ────────────────────────────────────────────────────────────────

def _load_daily_dialog_hf(split: str):
    """
    Try loading DailyDialog via the HuggingFace `datasets` library.
    May fail on newer `datasets` versions where the original script-based
    loader has been deprecated/removed.

    Returns an iterable of rows with 'dialog' (list[str]) and
    'emotion' (list[int]) fields, or raises on failure.
    """
    cfg = DATASETS["daily_dialog"]
    ds = _hf_load_single(cfg["hf_path"], split)
    return ds


def _load_daily_dialog_github(split: str) -> list[dict]:
    """
    Fallback loader: download a tarball of the configured GitHub repo via
    codeload.github.com (allowlisted) and extract
    data/daily_dialog/<split>/dialogues.txt and dialogues_emotion.txt.

    Each line in dialogues.txt is one dialogue, with utterances separated
    by '__eou__'. The corresponding line in dialogues_emotion.txt contains
    one space-separated integer emotion code (0-6) per utterance.

    Returns a list of dicts: {'dialog': list[str], 'emotion': list[int]}
    """
    import io
    import tarfile
    import urllib.request

    gh = DATASETS["daily_dialog"]["github_fallback"]
    split_dir = gh["split_dirs"].get(split)
    if split_dir is None:
        raise ValueError(
            f"Unknown DailyDialog split '{split}' for GitHub fallback. "
            f"Available: {list(gh['split_dirs'])}"
        )

    repo, branch = gh["repo"], gh["branch"]
    url = f"https://codeload.github.com/{repo}/tar.gz/{branch}"
    print(f"[data_loader] Fetching DailyDialog ({split}) from GitHub fallback "
          f"({repo}@{branch}) …")

    with urllib.request.urlopen(url, timeout=120) as resp:
        tar_bytes = resp.read()

    repo_name = repo.split("/")[-1]
    base = f"{repo_name}-{branch}/data/daily_dialog/{split_dir}"
    dialogues_path = f"{base}/dialogues.txt"
    emotions_path  = f"{base}/dialogues_emotion.txt"

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        dialogues_member = tar.extractfile(dialogues_path)
        emotions_member  = tar.extractfile(emotions_path)
        if dialogues_member is None or emotions_member is None:
            raise FileNotFoundError(
                f"Could not find '{dialogues_path}' or '{emotions_path}' "
                f"in {repo}@{branch} tarball"
            )
        dialogues_lines = dialogues_member.read().decode("utf-8").splitlines()
        emotions_lines  = emotions_member.read().decode("utf-8").splitlines()

    rows = []
    for dlg_line, emo_line in zip(dialogues_lines, emotions_lines):
        utterances = [u.strip() for u in dlg_line.split("__eou__") if u.strip()]
        emotions   = [int(e) for e in emo_line.split() if e.strip()]
        if not utterances:
            continue
        rows.append({"dialog": utterances, "emotion": emotions})

    print(f"[data_loader] Loaded {len(rows)} dialogues from GitHub fallback")
    return rows


def load_daily_dialog(
    split:          str  = None,
    num_samples:    int  = None,
    force_download: bool = False,
) -> list[dict]:
    """
    Load and preprocess DailyDialog for response generation benchmarking,
    downloading on the fly (HuggingFace first, GitHub tarball fallback,
    local caching).

    DailyDialog provides, per conversation:
      - dialog  : list[str]  utterances in turn order
      - emotion : list[int]  emotion id per utterance (0-6, see config)

    Each conversation contributes one sample:
      - second-to-last turn  → utterance (prompt)
      - up to 3 turns before that → context (up to 4 turns joined by ' | ')
      - the FINAL turn       → reference (gold response)

    Note the turn-taking convention differs from EmpatheticDialogues: the
    "prompt" here is the second-to-last turn and the "reference" is the
    final turn, i.e. (context..., utterance) -> reference.

    `emotion` in the returned sample is the emotion label of the REFERENCE
    (final) turn, mapped via config.DATASETS["daily_dialog"]["emotion_map"].

    If `exclude_reference_emotion` is set in config, samples whose reference
    turn carries that emotion id are dropped (e.g. drop "no_emotion" to focus
    on emotionally-loaded responses).

    Results are cached locally under config.CACHE_DIR; pass
    force_download=True to bypass the cache and re-download.
    """
    cfg         = DATASETS["daily_dialog"]
    split       = split or cfg["split"]
    num_samples = num_samples if num_samples is not None else cfg["num_samples"]
    emotion_map = cfg["emotion_map"]
    exclude_id  = cfg.get("exclude_reference_emotion")

    if not force_download:
        cached = _load_from_cache("daily_dialog", split, num_samples)
        if cached is not None:
            return cached

    # 1) Try HuggingFace `datasets` first (with retries)
    rows = None
    try:
        ds = _load_with_retries(lambda: _load_daily_dialog_hf(split), label=cfg["hf_path"])
        rows = [{"dialog": r["dialog"], "emotion": r["emotion"]} for r in ds]
    except Exception as exc:
        print(f"[data_loader] HuggingFace DailyDialog load failed: {exc!r}")
        print("[data_loader] Falling back to GitHub mirror …")

    # 2) Fallback: GitHub tarball
    if rows is None:
        rows = _load_with_retries(
            lambda: _load_daily_dialog_github(split), label="GitHub fallback"
        )

    samples = []
    for i, row in enumerate(rows):
        dialog  = [str(u).strip() for u in row["dialog"]]
        emotion = [int(e) for e in row["emotion"]]

        if len(dialog) < 2:
            continue
        if len(emotion) != len(dialog):
            emotion = (emotion + [0] * len(dialog))[: len(dialog)]

        *ctx, prompt_turn, reference_turn = dialog
        ref_emotion_id = emotion[-1]

        if exclude_id is not None and ref_emotion_id == exclude_id:
            continue

        context_str = " | ".join(ctx[-4:]) if ctx else ""

        samples.append({
            "conv_id":   f"dd_{i}",
            "emotion":   emotion_map.get(ref_emotion_id, "unknown"),
            "context":   context_str,
            "utterance": prompt_turn,
            "reference": reference_turn,
        })

    if num_samples:
        samples = _stratified_sample(samples, num_samples, key="emotion")

    n_emotions = len({s["emotion"] for s in samples})
    print(f"[data_loader] {len(samples)} samples | {n_emotions} emotion categories")
    if exclude_id is not None:
        print(f"[data_loader] Excluded samples with reference emotion id={exclude_id} "
              f"({emotion_map.get(exclude_id, 'unknown')})")

    _save_to_cache(samples, "daily_dialog", split, num_samples)
    return samples


# ── Unified dispatcher ────────────────────────────────────────────────────────

_LOADERS = {
    "empathetic_dialogues": load_empathetic_dialogues,
    "daily_dialog":         load_daily_dialog,
}


def load_dataset_samples(
    name:           str  = ACTIVE_DATASET,
    split:          str  = None,
    num_samples:    int  = None,
    force_download: bool = False,
) -> list[dict]:
    """
    Unified entry point. Dispatches to the correct loader based on `name`
    (defaults to config.ACTIVE_DATASET).

    All loaders return samples in the common schema:
        conv_id, emotion, context, utterance, reference

    Datasets are downloaded on the fly on first use and cached locally
    under config.CACHE_DIR. Pass force_download=True to re-download.
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_LOADERS)}"
        )

    cfg = DATASETS[name]
    split       = split or cfg["split"]
    num_samples = num_samples if num_samples is not None else cfg["num_samples"]

    return _LOADERS[name](split=split, num_samples=num_samples, force_download=force_download)
