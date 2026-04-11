import re
from typing import Optional


ALLOWED_FACE_EMOJIS = {
    "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊", "😋",
    "😎", "😍", "😘", "🥰", "😗", "😙", "😚", "🙂", "🤗", "🤩", "🤔",
    "🤨", "😐", "😑", "😶", "🙄", "😏", "😣", "😥", "😮", "🤐", "😯",
    "😪", "😫", "🥱", "😴", "😌", "😛", "😜", "😝", "🤤", "😒", "😓",
    "😔", "😕", "🙃", "😲", "☹", "🙁", "😖", "😞", "😟", "😤",
    "😢", "😭", "😦", "😧", "😨", "😩", "🤯", "😬", "😰", "😱",
    "🥵", "🥶", "😳", "🤪", "😵", "🤠", "🥳", "😇", "🤓", "🧐",
    "😈", "👿", "🤡", "🤥", "🤫", "🤭", "🥴",
}

POSITIVE_EMOJIS = {"🙂", "😊", "😄", "😌", "🥳"}
SUPPORTIVE_EMOJIS = {"😔", "😢", "😟", "😕"}
PLAYFUL_EMOJIS = {"😄", "😅", "😉"}
THOUGHTFUL_EMOJIS = {"🤔", "🧐"}
NEUTRAL_EMOJIS = {"🙂"}

# Emoji category → set used when matching model output
_EMOTION_EMOJI_SETS = {
    "happy": POSITIVE_EMOJIS,
    "sad": SUPPORTIVE_EMOJIS,
    "playful": PLAYFUL_EMOJIS,
    "thoughtful": THOUGHTFUL_EMOJIS,
    "angry": SUPPORTIVE_EMOJIS | NEUTRAL_EMOJIS,
    "anxious": SUPPORTIVE_EMOJIS | NEUTRAL_EMOJIS,
}

_EMOJI_SIGNALS = {
    "sad": {"😢", "😭", "😞", "😔", "☹", "🙁"},
    "happy": {"😄", "😁", "😀", "😊", "🥳"},
    "playful": {"😂", "🤣", "😉", "😜"},
    "thoughtful": {"🤔", "🧐"},
    "angry": {"😠", "😡", "😤"},
    "anxious": {"😟", "😰", "😨"},
}

_WORD_SIGNALS = {
    "happy": {
        "happy", "glad", "excited", "great", "awesome", "amazing",
        "wonderful", "fantastic", "love", "yay", "proud", "relieved",
    },
    "sad": {
        "sad", "upset", "hurt", "cry", "crying", "lonely", "depressed",
        "down", "unhappy", "miss", "heartbroken", "tired",
    },
    "playful": {"haha", "lol", "lmao", "funny", "joke", "kidding"},
    "thoughtful": {"think", "wonder", "curious", "why", "how", "what do you think"},
    "angry": {"angry", "mad", "annoyed", "frustrated", "furious", "irritated"},
    "anxious": {"nervous", "anxious", "worried", "scared", "afraid", "stress", "stressed"},
}

_FACTUAL_PATTERNS = [
    r"\bwhat is\b", r"\bwhere is\b", r"\bwho is\b", r"\bwhen is\b",
    r"\bhow do i\b", r"\bexplain\b", r"\bdefine\b", r"\binstall\b",
    r"\berror\b", r"\bcode\b", r"\bpython\b", r"\bdebug\b",
]

_EXPECTED_EMOJI: dict[str, Optional[str]] = {
    "happy": "😊",
    "sad": "😔",
    "playful": "😄",
    "thoughtful": "🤔",
    "angry": "😕",
    "anxious": "😟",
    "neutral": None,
    "factual": None,
}

_EMOJI_GUIDANCE = {
    "happy": (
        "The user sounds happy, proud, excited, or relieved. "
        "You may use at most one warm positive facial emoji if it fits naturally, such as 😊 or 😄."
    ),
    "sad": (
        "The user sounds sad, hurt, lonely, or disappointed. "
        "Respond gently and supportively. "
        "You may use at most one soft supportive facial emoji if it fits naturally, such as 😔 or 😢. "
        "Never use laughing or playful emojis."
    ),
    "playful": (
        "The user sounds playful or lighthearted. "
        "You may use at most one light playful facial emoji if it fits naturally, such as 😄 or 😉."
    ),
    "thoughtful": (
        "The user sounds reflective or curious. "
        "You may use at most one thoughtful facial emoji if it fits naturally, such as 🤔."
    ),
    "angry": (
        "The user sounds frustrated or upset. "
        "Respond calmly and helpfully. "
        "Do not mirror anger with aggressive emojis. "
        "If you use an emoji, use at most one gentle face emoji like 😕."
    ),
    "anxious": (
        "The user sounds nervous or worried. "
        "Respond calmly and reassuringly. "
        "You may use at most one gentle supportive facial emoji if it fits naturally, such as 😟."
    ),
    "neutral": (
        "The user's tone is neutral. "
        "Do not force emojis. Use no emoji unless a small amount of warmth genuinely improves the reply."
    ),
    "factual": (
        "The user's message is mainly factual, technical, or informational. "
        "Do not use emojis unless clearly helpful. Usually use no emoji."
    ),
}

_SYSTEM_PROMPT_BASE = (
    "You are a helpful, emotionally intelligent assistant. "
    "Always respond with full sentences. "
    "Use facial emojis naturally and sparingly, like a thoughtful human texter. "
    "Use at most ONE facial emoji in the whole response. "
    "Only use an emoji when it improves emotional clarity, warmth, or tone. "
    "Never reply with only an emoji. "
    "Never use non-face emojis. "
    "Never use laughing emojis for sadness, vulnerability, confusion, or serious topics. "
    "Do not ask redundant questions when the user's emotional state is already obvious. "
)


# ---------------------------------------------------------------------------
# Emoji utilities
# ---------------------------------------------------------------------------

def remove_emojis_except_faces(text: str) -> str:
    result = []
    for char in text:
        if char in ALLOWED_FACE_EMOJIS:
            result.append(char)
            continue
        code = ord(char)
        if (
            0x1F300 <= code <= 0x1F5FF
            or 0x1F680 <= code <= 0x1F6FF
            or 0x1F700 <= code <= 0x1F77F
            or 0x1F780 <= code <= 0x1F7FF
            or 0x1F800 <= code <= 0x1F8FF
            or 0x1F900 <= code <= 0x1F9FF
            or 0x1FA70 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x26FF
            or 0x2700 <= code <= 0x27BF
        ):
            continue
        result.append(char)

    cleaned = "".join(result)
    return re.sub(r"[\u200d\ufe0f]", "", cleaned)


def extract_face_emojis(text: str) -> list[str]:
    return [ch for ch in text if ch in ALLOWED_FACE_EMOJIS]


def remove_all_face_emojis(text: str) -> str:
    return "".join(ch for ch in text if ch not in ALLOWED_FACE_EMOJIS)


# ---------------------------------------------------------------------------
# Emotion detection
# ---------------------------------------------------------------------------

def detect_user_emotion(user_text: str) -> str:
    """
    Lightweight rule-based emotion detection.
    Returns one of: happy, sad, playful, thoughtful, angry, anxious, neutral, factual.
    """
    text = user_text.strip().lower()
    if not text:
        return "neutral"

    # Strong signal from emoji
    for emotion, emojis in _EMOJI_SIGNALS.items():
        if any(e in user_text for e in emojis):
            return emotion

    text_words = set(re.findall(r"\b\w+\b", text))

    for emotion in ("happy", "sad", "playful", "angry", "anxious"):
        if text_words & _WORD_SIGNALS[emotion]:
            return emotion

    if any(re.search(pattern, text) for pattern in _FACTUAL_PATTERNS):
        return "factual"

    if text_words & _WORD_SIGNALS["thoughtful"]:
        return "thoughtful"

    return "neutral"


# ---------------------------------------------------------------------------
# System prompt & reply normalization
# ---------------------------------------------------------------------------

def build_system_prompt(user_emotion: str) -> str:
    contextual_rule = _EMOJI_GUIDANCE.get(user_emotion, _EMOJI_GUIDANCE["neutral"])
    return _SYSTEM_PROMPT_BASE + contextual_rule


def normalize_assistant_reply(text: str, emotion: str) -> str:
    """
    Enforce emoji policy after generation:
    - facial emojis only, at most one
    - remove emoji if it conflicts with the detected context
    - place emoji naturally at the end
    """
    text = remove_emojis_except_faces(text)
    found = extract_face_emojis(text)
    plain = re.sub(r"\s+", " ", remove_all_face_emojis(text)).strip()

    if not plain:
        return "I'm here."

    expected = _EXPECTED_EMOJI.get(emotion)
    if expected is None:
        return plain

    allowed_set = _EMOTION_EMOJI_SETS.get(emotion, set())
    kept_emoji = next((e for e in found if e in allowed_set), None)
    if kept_emoji is None:
        kept_emoji = expected

    plain = re.sub(r"\s+([,.;!?])", r"\1", plain).rstrip()
    return f"{plain} {kept_emoji}"
