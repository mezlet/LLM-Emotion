"""
data_loader.py
--------------
Loads and samples datasets for the LLM emotion-classification benchmark.

Supported datasets:
  - GoEmotions  (28-class, multi-label)   -> load_goemotions_sample()
  - ISEAR       (7-class,  single-label)  -> load_isear_sample()
  - DailyDialog (7-class,  single-label)  -> load_dailydialog_sample()

A generic dispatcher `load_dataset_sample(dataset_name, ...)` is provided
for use by benchmark_runner.py via the --dataset flag.

Every loader returns a list of sample dicts with the same shape:
    { "id": str, "text": str, "labels": list[str] }

For single-label datasets (ISEAR, DailyDialog), "labels" contains exactly
one string from EKMAN7_LABELS. For GoEmotions, "labels" may contain one
or more strings from GOEMOTIONS_LABELS.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import logging
from collections import defaultdict

from datasets import load_dataset
from config import GOEMOTIONS_LABELS, EKMAN7_LABELS, DATASETS

log = logging.getLogger(__name__)

# GoEmotions official label ordering (raw config, per-column schema)
GOEMOTIONS_LABEL_ORDER = GOEMOTIONS_LABELS


# ──────────────────────────────────────────────────────────────────────────
# Generic dispatcher
# ──────────────────────────────────────────────────────────────────────────

def load_dataset_sample(dataset_name: str, n: int = 200, seed: int = 42,
                         split: str | None = None) -> list[dict]:
    """
    Dispatch to the correct dataset loader.

    Args:
        dataset_name: one of "goemotions", "isear", "dailydialog"
        n:            number of samples to return
        seed:         random seed for reproducible sampling
        split:        dataset split override; if None, uses the registry default

    Returns:
        List of sample dicts: { "id", "text", "labels" }
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid options: {list(DATASETS)}"
        )

    resolved_split = split or DATASETS[dataset_name]["default_split"]

    if dataset_name == "goemotions":
        return load_goemotions_sample(n=n, seed=seed, split=resolved_split)
    elif dataset_name == "isear":
        return load_isear_sample(n=n, seed=seed, split=resolved_split)
    elif dataset_name == "dailydialog":
        return load_dailydialog_sample(n=n, seed=seed, split=resolved_split)

    raise ValueError(f"No loader implemented for '{dataset_name}'")


# ──────────────────────────────────────────────────────────────────────────
# GoEmotions (28-class, multi-label)
# ──────────────────────────────────────────────────────────────────────────

def load_goemotions_sample(n: int = 200, seed: int = 42, split: str = "train") -> list[dict]:
    """
    Download GoEmotions (raw config) and return `n` stratified samples.

    The 'raw' config exposes each emotion as its own binary column rather
    than a single 'labels' list -- this function handles both schemas.
    """
    log.info(f"Downloading GoEmotions ({split} split)...")
    dataset = load_dataset("go_emotions", "raw", split=split)
    log.info(f"  Raw split size: {len(dataset)}")

    first_row = dataset[0]
    use_column_per_emotion = "labels" not in first_row

    if use_column_per_emotion:
        log.info("  Schema: per-column emotions (raw config)")
    else:
        log.info("  Schema: integer 'labels' list")

    samples = []
    for i, row in enumerate(dataset):
        if use_column_per_emotion:
            label_strings = [lbl for lbl in GOEMOTIONS_LABEL_ORDER if row.get(lbl, 0) == 1]
        else:
            label_indices = [j for j, v in enumerate(row["labels"]) if v == 1]
            label_strings = [GOEMOTIONS_LABEL_ORDER[j] for j in label_indices]

        if not label_strings:
            label_strings = ["neutral"]

        samples.append({
            "id": str(row.get("id", i)),
            "text": row["text"].strip(),
            "labels": label_strings,
        })

    random.seed(seed)
    if n >= len(samples):
        log.warning(f"Requested {n} samples but only {len(samples)} available; using all.")
        return samples

    sampled = _stratified_sample_multilabel(samples, n, seed)
    log.info(f"  Sampled {len(sampled)} items")
    return sampled


# ──────────────────────────────────────────────────────────────────────────
# ISEAR (7-class, single-label, self-reported emotional narratives)
# ──────────────────────────────────────────────────────────────────────────

# ISEAR's 7 original categories -> shared EKMAN7_LABELS.
# ISEAR categories: joy, fear, anger, sadness, disgust, shame, guilt
#
# Mapping rationale (documented for thesis write-up):
#   joy     -> joy
#   fear    -> fear
#   anger   -> anger
#   sadness -> sadness
#   disgust -> disgust
#   shame   -> sadness   (self-conscious negative affect; closest Ekman
#                          analogue lacking a dedicated category)
#   guilt   -> sadness   (same rationale as shame)
#
# "surprise" and "neutral" are not present in ISEAR's label space and will
# have zero support in the ground truth for this dataset -- this is a known
# structural gap and should be reported as a limitation.
ISEAR_TO_EKMAN7 = {
    "joy": "joy",
    "fear": "fear",
    "anger": "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    "shame": "sadness",
    "guilt": "sadness",
    # Numeric encodings sometimes used in alternate ISEAR distributions
    # (1=joy, 2=fear, 3=anger, 4=sadness, 5=disgust, 6=shame, 7=guilt)
    1: "joy", 2: "fear", 3: "anger", 4: "sadness",
    5: "disgust", 6: "sadness", 7: "sadness",
}

# Candidate HuggingFace repos for ISEAR (tried in order; dataset has been
# mirrored under several repo names over time, and several previously-used
# repo names have since been removed from the Hub).
ISEAR_HF_CANDIDATES = [
    ("gsri-18/ISEAR-dataset-complete", None),  # 7,516 rows, columns: emotion, content
    ("Maxximilliann/isear", None),
    ("metaeval/isear", None),
]

# Direct CSV fallback (used if no HF repo can be loaded). This is the
# canonical ISEAR CSV redistributed for ML use via the py_isear_dataset
# project. Columns of interest: 'SIT' (text) and 'EMOT' (numeric 1-7 label,
# see ISEAR_TO_EKMAN7 numeric mapping below).
ISEAR_CSV_URL = (
    "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv"
)


def load_isear_sample(n: int = 200, seed: int = 42, split: str = "train") -> list[dict]:
    """
    Download ISEAR and return `n` stratified samples mapped onto the
    shared 7-class Ekman label set (EKMAN7_LABELS).

    ISEAR is single-label: each returned sample has exactly one label
    in "labels".
    """
    dataset = None
    used_repo = None
    last_err = None

    for repo, config_name in ISEAR_HF_CANDIDATES:
        try:
            log.info(f"Attempting to download ISEAR from '{repo}'...")
            if config_name:
                dataset = load_dataset(repo, config_name, split=split)
            else:
                dataset = load_dataset(repo, split=split)
            used_repo = repo
            break
        except Exception as e:
            last_err = e
            log.warning(f"  Failed to load '{repo}': {e}")
            continue

    if dataset is None:
        log.warning(
            f"All HuggingFace repo attempts failed (last error: {last_err}). "
            f"Falling back to direct CSV download from {ISEAR_CSV_URL}"
        )
        return _load_isear_from_csv(ISEAR_CSV_URL, n=n, seed=seed)

    log.info(f"  Loaded ISEAR from '{used_repo}' ({split} split): {len(dataset)} rows")
    log.info(f"  Features: {dataset.features}")

    text_col, label_col = _detect_isear_columns(dataset)
    log.info(f"  Using text column '{text_col}', label column '{label_col}'")

    label_names = None
    try:
        feat = dataset.features[label_col]
        if hasattr(feat, "names"):
            label_names = [n.lower() for n in feat.names]
    except Exception:
        pass

    samples = []
    skipped = 0
    for i, row in enumerate(dataset):
        text = str(row[text_col]).strip()
        if not text or text.lower() in {"no response", "nan", "none"}:
            skipped += 1
            continue

        raw_label = row[label_col]

        # Resolve raw_label -> ISEAR category string
        if label_names is not None and isinstance(raw_label, int):
            category = label_names[raw_label] if raw_label < len(label_names) else None
        elif isinstance(raw_label, str):
            category = raw_label.strip().lower()
        else:
            category = raw_label  # numeric code, handled by ISEAR_TO_EKMAN7

        mapped = ISEAR_TO_EKMAN7.get(category)
        if mapped is None:
            skipped += 1
            continue

        samples.append({
            "id": str(row.get("id", i)),
            "text": text,
            "labels": [mapped],
        })

    if skipped:
        log.info(f"  Skipped {skipped} rows (empty text or unmapped label)")

    random.seed(seed)
    if n >= len(samples):
        log.warning(f"Requested {n} samples but only {len(samples)} available; using all.")
        return samples

    sampled = _stratified_sample_singlelabel(samples, n, seed)
    log.info(f"  Sampled {len(sampled)} items")
    _log_label_distribution(sampled)
    return sampled


def _detect_isear_columns(dataset) -> tuple[str, str]:
    """Detect the text and label column names across known ISEAR variants."""
    cols = set(dataset.column_names)

    text_candidates = ["text", "content", "SIT", "sentence", "Sentence"]
    label_candidates = ["label", "emotion", "EMOT", "Field1"]

    text_col = next((c for c in text_candidates if c in cols), None)
    label_col = next((c for c in label_candidates if c in cols), None)

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/label columns in ISEAR dataset. "
            f"Available columns: {dataset.column_names}"
        )
    return text_col, label_col


def _load_isear_from_csv(url: str, n: int, seed: int) -> list[dict]:
    """
    Fallback loader: download the ISEAR CSV directly (no HF `datasets`
    dependency) and parse 'SIT' (text) / 'EMOT' (numeric 1-7 label) columns.
    """
    import csv as csv_module
    import urllib.request
    import io

    log.info(f"  Downloading ISEAR CSV from {url} ...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        raw_bytes = resp.read()

    # The py_isear CSV uses '|' as a delimiter and latin-1 encoding.
    text_data = raw_bytes.decode("latin-1")
    sniffed_delim = "|" if text_data.count("|") > text_data.count(",") else ","

    reader = csv_module.DictReader(io.StringIO(text_data), delimiter=sniffed_delim)
    rows = list(reader)
    if not rows:
        raise RuntimeError("ISEAR CSV fallback returned no rows.")

    cols = set(rows[0].keys())
    text_col = next((c for c in ["SIT", "text", "content"] if c in cols), None)
    label_col = next((c for c in ["EMOT", "label", "emotion"] if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns in ISEAR CSV. Columns: {sorted(cols)}")

    log.info(f"  Using text column '{text_col}', label column '{label_col}' (CSV fallback)")

    samples = []
    skipped = 0
    for i, row in enumerate(rows):
        text = (row.get(text_col) or "").strip()
        if not text or text.lower() in {"no response", "nan", "none", "[ no response.]"}:
            skipped += 1
            continue

        raw_label = (row.get(label_col) or "").strip()
        try:
            raw_label_key: int | str = int(raw_label)
        except ValueError:
            raw_label_key = raw_label.lower()

        mapped = ISEAR_TO_EKMAN7.get(raw_label_key)
        if mapped is None:
            skipped += 1
            continue

        samples.append({"id": str(i), "text": text, "labels": [mapped]})

    if skipped:
        log.info(f"  Skipped {skipped} rows (empty text or unmapped label)")

    random.seed(seed)
    if n >= len(samples):
        log.warning(f"Requested {n} samples but only {len(samples)} available; using all.")
        return samples

    sampled = _stratified_sample_singlelabel(samples, n, seed)
    log.info(f"  Sampled {len(sampled)} items")
    _log_label_distribution(sampled)
    return sampled


# ──────────────────────────────────────────────────────────────────────────
# DailyDialog (7-class, single-label, scripted multi-turn dialogue)
# ──────────────────────────────────────────────────────────────────────────

# DailyDialog's official emotion label ordering (integer-encoded):
#   0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness,
#   5: sadness, 6: surprise
#
# Mapped onto the shared EKMAN7_LABELS (note: "no emotion" -> "neutral",
# "happiness" -> "joy").
DAILYDIALOG_LABEL_ORDER = [
    "neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise",
]


def load_dailydialog_sample(n: int = 200, seed: int = 42, split: str = "test") -> list[dict]:
    """
    Download DailyDialog and return `n` stratified samples, flattened from
    multi-turn dialogues into individual utterances, mapped onto the
    shared 7-class Ekman label set (EKMAN7_LABELS).

    DailyDialog is single-label per utterance: each returned sample has
    exactly one label in "labels".

    Note: DailyDialog is heavily skewed toward "neutral" (~83% of
    utterances). Stratified sampling ensures the 6 non-neutral emotion
    classes are represented despite this imbalance.

    Loading strategy:
      1. Try HuggingFace `datasets` repos (kept for forward-compatibility,
         in case HF re-publishes a script-free version).
      2. Fall back to the canonical `dialogues.txt` / `dialogues_emotion.txt`
         pair (the original Li et al. 2017 release format, `__eou__`-delimited
         utterances + space-separated 0-6 emotion codes per dialogue),
         fetched from a GitHub mirror of the dataset.
    """
    last_err = None
    dataset = None
    used_repo = None

    for repo in ["li2017dailydialog/daily_dialog", "daily_dialog"]:
        try:
            log.info(f"Attempting to download DailyDialog from '{repo}'...")
            dataset = load_dataset(repo, split=split)
            used_repo = repo
            break
        except Exception as e:
            last_err = e
            log.warning(f"  Failed to load '{repo}': {e}")
            continue

    if dataset is None:
        log.warning(
            f"All HuggingFace repo attempts failed (last error: {last_err}). "
            f"Falling back to GitHub mirror of dialogues_text.txt / dialogues_emotion.txt"
        )
        return _load_dailydialog_from_github(n=n, seed=seed, split=split)

    log.info(f"  Loaded DailyDialog from '{used_repo}' ({split} split): {len(dataset)} dialogues")
    log.info(f"  Features: {dataset.features}")

    samples = []
    for dialog_idx, row in enumerate(dataset):
        utterances = row.get("dialog") or row.get("utterances") or []
        emotions = row.get("emotion") or row.get("emotions") or []

        for utt_idx, (text, emo) in enumerate(zip(utterances, emotions)):
            text = str(text).strip()
            if not text:
                continue

            if isinstance(emo, int):
                if emo < 0 or emo >= len(DAILYDIALOG_LABEL_ORDER):
                    continue
                label = DAILYDIALOG_LABEL_ORDER[emo]
            elif isinstance(emo, str):
                label = emo.strip().lower()
                if label == "happiness":
                    label = "joy"
                if label == "no emotion":
                    label = "neutral"
            else:
                continue

            if label not in EKMAN7_LABELS:
                continue

            samples.append({
                "id": f"{dialog_idx}_{utt_idx}",
                "text": text,
                "labels": [label],
            })

    log.info(f"  Flattened to {len(samples)} individual utterances")

    random.seed(seed)
    if n >= len(samples):
        log.warning(f"Requested {n} samples but only {len(samples)} available; using all.")
        return samples

    sampled = _stratified_sample_singlelabel(samples, n, seed)
    log.info(f"  Sampled {len(sampled)} items")
    _log_label_distribution(sampled)
    return sampled


# GitHub mirror providing the original Li et al. 2017 dialogues.txt /
# dialogues_emotion.txt pair under data/daily_dialog/<split>/. Fetched as a
# tarball via codeload.github.com (allowlisted), extracting only the two
# files needed for the requested split.
DAILYDIALOG_GITHUB_REPO = "snakeztc/NeuralDialog-LAED"
DAILYDIALOG_GITHUB_BRANCH = "master"
DAILYDIALOG_GITHUB_TARBALL_URL = (
    f"https://codeload.github.com/{DAILYDIALOG_GITHUB_REPO}/tar.gz/{DAILYDIALOG_GITHUB_BRANCH}"
)

# Map our split names onto this mirror's directory names.
DAILYDIALOG_GITHUB_SPLIT_DIRS = {
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "dev": "validation",
    "test": "test",
}


def _load_dailydialog_from_github(n: int, seed: int, split: str) -> list[dict]:
    """
    Fallback loader: download a tarball of DAILYDIALOG_GITHUB_REPO and
    extract data/daily_dialog/<split>/dialogues.txt and dialogues_emotion.txt.

    Each line in dialogues.txt is one dialogue, with utterances separated by
    '__eou__'. The corresponding line in dialogues_emotion.txt contains one
    space-separated integer emotion code (0-6) per utterance.
    """
    import tarfile
    import urllib.request
    import io

    split_dir = DAILYDIALOG_GITHUB_SPLIT_DIRS.get(split)
    if split_dir is None:
        raise ValueError(
            f"Unknown DailyDialog split '{split}'. "
            f"Valid options: {list(DAILYDIALOG_GITHUB_SPLIT_DIRS)}"
        )

    log.info(f"  Downloading DailyDialog mirror tarball from {DAILYDIALOG_GITHUB_TARBALL_URL} ...")
    with urllib.request.urlopen(DAILYDIALOG_GITHUB_TARBALL_URL, timeout=120) as resp:
        tar_bytes = resp.read()
    log.info(f"  Downloaded {len(tar_bytes) / 1e6:.1f} MB")

    text_member_suffix = f"data/daily_dialog/{split_dir}/dialogues.txt"
    emotion_member_suffix = f"data/daily_dialog/{split_dir}/dialogues_emotion.txt"

    dialogues_raw = None
    emotions_raw = None

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(text_member_suffix):
                dialogues_raw = tar.extractfile(member).read().decode("utf-8")
            elif member.name.endswith(emotion_member_suffix):
                emotions_raw = tar.extractfile(member).read().decode("utf-8")

    if dialogues_raw is None or emotions_raw is None:
        raise RuntimeError(
            f"Could not find dialogues.txt / dialogues_emotion.txt for split "
            f"'{split_dir}' inside {DAILYDIALOG_GITHUB_TARBALL_URL}"
        )

    dialogue_lines = [ln for ln in dialogues_raw.splitlines() if ln.strip()]
    emotion_lines = [ln for ln in emotions_raw.splitlines() if ln.strip()]

    if len(dialogue_lines) != len(emotion_lines):
        log.warning(
            f"  Line count mismatch: {len(dialogue_lines)} dialogues vs "
            f"{len(emotion_lines)} emotion lines (using min length)"
        )

    n_dialogues = min(len(dialogue_lines), len(emotion_lines))
    log.info(f"  Loaded {n_dialogues} dialogues from '{split_dir}' split")

    samples = []
    for dialog_idx in range(n_dialogues):
        utterances = [u.strip() for u in dialogue_lines[dialog_idx].split("__eou__") if u.strip()]
        emotion_codes = [c for c in emotion_lines[dialog_idx].split() if c.strip()]

        if len(utterances) != len(emotion_codes):
            # Misaligned line; skip rather than guess.
            continue

        for utt_idx, (text, code_str) in enumerate(zip(utterances, emotion_codes)):
            try:
                code = int(code_str)
            except ValueError:
                continue
            if code < 0 or code >= len(DAILYDIALOG_LABEL_ORDER):
                continue

            samples.append({
                "id": f"{dialog_idx}_{utt_idx}",
                "text": text,
                "labels": [DAILYDIALOG_LABEL_ORDER[code]],
            })

    log.info(f"  Flattened to {len(samples)} individual utterances")

    random.seed(seed)
    if n >= len(samples):
        log.warning(f"Requested {n} samples but only {len(samples)} available; using all.")
        return samples

    sampled = _stratified_sample_singlelabel(samples, n, seed)
    log.info(f"  Sampled {len(sampled)} items")
    _log_label_distribution(sampled)
    return sampled


# ──────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────────────────────

def _stratified_sample_multilabel(samples: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by primary (first) label -- used for GoEmotions."""
    random.seed(seed)

    buckets = defaultdict(list)
    for s in samples:
        primary = s["labels"][0]
        buckets[primary].append(s)

    n_labels = len(buckets)
    per_label = max(1, n // n_labels)
    selected = []

    for label, items in buckets.items():
        random.shuffle(items)
        selected.extend(items[:per_label])

    remaining = [s for s in samples if s not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, n - len(selected))])

    random.shuffle(selected)
    return selected[:n]


def _stratified_sample_singlelabel(samples: list[dict], n: int, seed: int) -> list[dict]:
    """
    Stratify by the single ground-truth label -- used for ISEAR and
    DailyDialog. Caps per-class draws at the requested `n // n_classes`
    so that no single class (e.g. DailyDialog's dominant "neutral")
    crowds out the rarer classes, then tops up randomly from the
    remaining pool to reach exactly `n`.
    """
    random.seed(seed)

    buckets = defaultdict(list)
    for s in samples:
        buckets[s["labels"][0]].append(s)

    n_labels = len(buckets)
    per_label = max(1, n // n_labels)
    selected = []

    for label, items in buckets.items():
        random.shuffle(items)
        take = items[:per_label]
        selected.extend(take)
        log.info(f"    class '{label}': {len(items)} available, {len(take)} sampled")

    remaining = [s for s in samples if s not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[: max(0, n - len(selected))])

    random.shuffle(selected)
    return selected[:n]


def _log_label_distribution(samples: list[dict]) -> None:
    counts = defaultdict(int)
    for s in samples:
        for lbl in s["labels"]:
            counts[lbl] += 1
    dist = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    log.info(f"  Label distribution: {dist}")