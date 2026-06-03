"""
prompt_builder.py
-----------------
Constructs zero-shot and few-shot prompts for GoEmotions emotion classification.
"""

from config import EMOTION_LABELS

_LABEL_LIST = ", ".join(EMOTION_LABELS)

# ── Zero-shot prompt ──────────────────────────────────────────────────────────

ZERO_SHOT_TEMPLATE = """\
You are an emotion classifier. Your task is to identify the emotions expressed \
in the following text.

Choose only from this exact list of emotions:
{label_list}

A text may express more than one emotion. Output ONLY the applicable emotion \
labels as a comma-separated list, with no additional text, explanation, or \
punctuation beyond the commas.

If no specific emotion is expressed, output: neutral

Text: "{text}"

Emotions:"""

# ── Few-shot prompt ───────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    ("I absolutely love how this turned out!", "admiration, joy"),
    ("I can't believe they canceled the show. I'm so upset.", "anger, disappointment, sadness"),
    ("Thanks for helping me out, I really appreciate it.", "gratitude"),
    ("I have no idea what's going on here.", "confusion"),
    ("The weather is fine.", "neutral"),
]

_FEW_SHOT_BLOCK = "\n".join(
    f'Text: "{ex}"\nEmotions: {lbl}' for ex, lbl in FEW_SHOT_EXAMPLES
)

FEW_SHOT_TEMPLATE = """\
You are an emotion classifier. Your task is to identify the emotions expressed \
in the following text.

Choose only from this exact list of emotions:
{label_list}

A text may express more than one emotion. Output ONLY the applicable emotion \
labels as a comma-separated list, with no additional text.

If no specific emotion is expressed, output: neutral

Here are some examples:

{examples}

Now classify this text:
Text: "{text}"
Emotions:"""


def build_prompt(text: str, mode: str = "zero_shot") -> str:
    """
    Build a classification prompt for the given text.

    Args:
        text:  The input text to classify.
        mode:  "zero_shot" (default) or "few_shot".

    Returns:
        Formatted prompt string.
    """
    text = text.replace('"', '\\"')  # escape inner quotes

    if mode == "few_shot":
        return FEW_SHOT_TEMPLATE.format(
            label_list=_LABEL_LIST,
            examples=_FEW_SHOT_BLOCK,
            text=text,
        )
    else:
        return ZERO_SHOT_TEMPLATE.format(
            label_list=_LABEL_LIST,
            text=text,
        )