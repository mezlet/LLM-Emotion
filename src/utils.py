"""
utils.py
--------
Stateless helpers used across the application:
  - timestamp formatting
  - time / date question detection and answering
  - emoji policy (allowed set, filters)
  - assistant reply normalisation
  - safe JSON extraction
  - bool coercion for LLM output
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from models import MessageAnalysis


# ---------------------------------------------------------------------------
# Emoji policy
# ---------------------------------------------------------------------------

ALLOWED_FACE_EMOJIS: frozenset[str] = frozenset({
    "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊", "😋",
    "😎", "😍", "😘", "🥰", "😗", "😙", "😚", "🙂", "🤗", "🤩", "🤔",
    "🤨", "😐", "😑", "😶", "🙄", "😏", "😣", "😥", "😮", "🤐", "😯",
    "😪", "😫", "🥱", "😴", "😌", "😛", "😜", "😝", "🤤", "😒", "😓",
    "😔", "😕", "🙃", "😲", "☹", "🙁", "😖", "😞", "😟", "😤",
    "😢", "😭", "😦", "😧", "😨", "😩", "🤯", "😬", "😰", "😱",
    "🥵", "🥶", "😳", "🤪", "😵", "🤠", "🥳", "😇", "🤓", "🧐",
    "😈", "👿", "🤡", "🤥", "🤫", "🤭", "🥴",
})

# Unicode ranges that cover non-face emoji (Misc Symbols, Transport, etc.)
_NON_FACE_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600,  0x26FF),
    (0x2700,  0x27BF),
)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def now_ts() -> str:
    """Return the current local time as a readable string (e.g. 2026-04-16 15:04:12)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_ts(message: str) -> None:
    """Print *message* prefixed with the current timestamp."""
    print(f"[{now_ts()}] {message}")


# ---------------------------------------------------------------------------
# Time / date question detection
# ---------------------------------------------------------------------------

_TIME_DATE_PATTERNS: tuple[str, ...] = (
    r"\bwhat(?:'s| is)? the time\b",
    r"\bcurrent time\b",
    r"\btime now\b",
    r"\bwhat time is it\b",
    r"\bcan you tell me the time\b",
    r"\btell me the time\b",
    r"\bwhat(?:'s| is)? the date\b",
    r"\bcurrent date\b",
    r"\bdate today\b",
    r"\bwhat(?:'s| is)? today'?s date\b",
    r"\btell me the date\b",
    r"\bwhat day is it\b",
    r"\bwhat day is today\b",
    r"\bcurrent day\b",
    r"\btoday is what day\b",
    r"\bwhat(?:'s| is)? the current time and date\b",
    r"\btoday'?s date and time\b",
)


def looks_like_time_question(user_text: str) -> bool:
    """Return True if the user appears to be asking for the current time or date."""
    text = user_text.strip().lower()
    return any(re.search(pattern, text) for pattern in _TIME_DATE_PATTERNS)


def build_system_time_reply(user_text: str) -> str:
    """Construct a direct answer to a time / date question using the OS clock."""
    now = datetime.now()
    text = user_text.strip().lower()

    current_time = now.strftime("%I:%M %p").lstrip("0")
    current_date = now.strftime("%A, %B %d, %Y")
    current_day  = now.strftime("%A")

    asks_time = "time" in text
    asks_date = "date" in text
    asks_day  = "what day" in text or "day is it" in text or "day is today" in text

    if (asks_time and asks_date) or "time and date" in text:
        return f"The current date and time is {current_date} at {current_time}."
    if asks_time:
        return f"The current time is {current_time}."
    if asks_date:
        return f"Today's date is {current_date}."
    if asks_day:
        return f"Today is {current_day}."

    return f"The current date and time is {current_date} at {current_time}."


# ---------------------------------------------------------------------------
# Text / emoji cleanup
# ---------------------------------------------------------------------------

def remove_ascii_emoticons(text: str) -> str:
    """Strip text-based emoticons such as :) :D ;-) :\\."""
    return re.sub(r"[:;=8][\-^]?[)(DPp/\\|]+", "", text)


def remove_emojis_except_faces(text: str) -> str:
    """Remove all emoji-like characters *except* the approved facial emoji set."""
    result: list[str] = []
    for char in text:
        if char in ALLOWED_FACE_EMOJIS:
            result.append(char)
            continue
        code = ord(char)
        if any(lo <= code <= hi for lo, hi in _NON_FACE_EMOJI_RANGES):
            continue
        result.append(char)

    cleaned = "".join(result)
    cleaned = re.sub(r"[\u200d\ufe0f]", "", cleaned)
    return cleaned


def remove_all_face_emojis(text: str) -> str:
    """Remove even the approved facial emoji."""
    return "".join(ch for ch in text if ch not in ALLOWED_FACE_EMOJIS)


def normalize_assistant_reply(text: str, analysis: MessageAnalysis) -> str:
    """
    Final policy-enforcement layer applied to every assistant reply.

    Steps:
    1. Remove ASCII emoticons.
    2. Remove non-face emoji.
    3. Strip trailing whitespace and fix spacing before punctuation.
    4. Re-append the approved emoji when the analysis says to.
    """
    text    = remove_ascii_emoticons(text)
    cleaned = remove_emojis_except_faces(text)
    plain   = remove_all_face_emojis(cleaned)

    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\s+([,.;!?])", r"\1", plain)

    if not plain:
        plain = "I'm here."

    approved_emoji = (
        analysis.emoji
        if analysis.emoji in ALLOWED_FACE_EMOJIS
        else None
    )

    if not analysis.should_use_emoji or not approved_emoji:
        return plain

    return cleaned


# ---------------------------------------------------------------------------
# Safe JSON parsing
# ---------------------------------------------------------------------------

def safe_json_extract(text: str) -> Optional[dict]:
    """
    Parse a JSON object from *text*, tolerating common LLM formatting quirks.

    Attempts (in order):
    1. Direct parse after stripping markdown code fences.
    2. Extract the first ``{...}`` block and parse that.
    """
    text = text.strip()
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Bool coercion
# ---------------------------------------------------------------------------

def parse_bool(value) -> bool:
    """
    Robustly coerce LLM JSON output to a Python bool.

    Handles: actual bools, numeric 0/1, and string representations
    ("true", "false", "yes", "1", etc.).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


# ---------------------------------------------------------------------------
# Command normalisation
# ---------------------------------------------------------------------------

def normalize_command(text: str) -> Optional[str]:
    """
    Return the bare command name if *text* starts with ``/`` or ``\\``.

    Plain words (e.g. ``speak``) are **not** treated as commands.
    Returns ``None`` when the input is not a slash-command.
    """
    stripped = text.strip().lower()
    if not stripped.startswith(("/", "\\")):
        return None
    return re.sub(r"^[\\/]+", "", stripped)