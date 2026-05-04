from __future__ import annotations

import re


ALLOWED_FACE_EMOJIS = {
    "😀", "😁", "😂", "🤣", "😃", "😄", "😅", "😆", "😉", "😊", "😋",
    "😎", "😍", "😘", "🥰", "😗", "😙", "😚", "🙂", "🤗", "🤩", "🤔",
    "🤨", "😐", "😑", "😶", "🙄", "😏", "😣", "😥", "😮", "🤐", "😯",
    "😪", "😫", "🥱", "😴", "😌", "😛", "😜", "😝", "🤤", "😒", "😓",
    "😔", "😕", "🙃", "😲", "☹", "🙁", "😖", "😞", "😟", "😤",
    "😢", "😭", "😦", "😧", "😨", "😩", "🤯", "😬", "😰", "😱",
    "🥵", "🥶", "😳", "🤪", "😵", "🤠", "🥳", "😇", "🤓", "🧐",
    "😈", "👿", "🤡", "🤥", "🤫", "🤭", "🥴"
}


def remove_ascii_emoticons(text: str) -> str:
    return re.sub(r"[:;=8][\-^]?[)(DPp/\\|]+", "", text)


def remove_emojis_except_faces(text: str) -> str:
    result = []
    for char in text:
        if char in ALLOWED_FACE_EMOJIS:
            result.append(char)
            continue

        code = ord(char)
        if (
            0x1F300 <= code <= 0x1F5FF or
            0x1F600 <= code <= 0x1F64F or
            0x1F680 <= code <= 0x1F6FF or
            0x1F700 <= code <= 0x1F77F or
            0x1F780 <= code <= 0x1F7FF or
            0x1F800 <= code <= 0x1F8FF or
            0x1F900 <= code <= 0x1F9FF or
            0x1FA70 <= code <= 0x1FAFF or
            0x2600 <= code <= 0x26FF or
            0x2700 <= code <= 0x27BF
        ):
            continue

        result.append(char)

    cleaned = "".join(result)
    return re.sub(r"[\u200d\ufe0f]", "", cleaned)


def keep_at_most_one_face_emoji_at_end(text: str) -> str:
    chars = list(text)
    emojis = [ch for ch in chars if ch in ALLOWED_FACE_EMOJIS]
    plain = "".join(ch for ch in chars if ch not in ALLOWED_FACE_EMOJIS)
    plain = re.sub(r"\s+", " ", plain).strip()
    plain = re.sub(r"\s+([,.;!?])", r"\1", plain)

    if not plain:
        plain = "I’m here."

    if not emojis:
        return plain

    # The LLM is allowed exactly one facial emoji. If it produced more, keep the last one.
    return f"{plain} {emojis[-1]}"


def normalize_assistant_reply(text: str) -> str:
    """Allow facial emojis only; remove all non-face emoji and ASCII emoticons."""
    text = remove_ascii_emoticons(text)
    cleaned = remove_emojis_except_faces(text)
    return keep_at_most_one_face_emoji_at_end(cleaned)
