"""
prompt_builder.py
-----------------
Constructs zero-shot and few-shot prompts for emotion classification,
parameterized by the active dataset's label set.

GoEmotions uses the 28-class multi-label set (GOEMOTIONS_LABELS) and
allows multiple comma-separated labels in the output.

ISEAR and DailyDialog use the shared 7-class Ekman set (EKMAN7_LABELS)
and are single-label: the prompt instructs the model to output exactly
one label.
"""

from config import GOEMOTIONS_LABELS, EKMAN7_LABELS, DATASETS

# ──────────────────────────────────────────────────────────────────────────
# Few-shot example banks, keyed by label set
# ──────────────────────────────────────────────────────────────────────────

# Examples for the 28-class GoEmotions label set (multi-label output)
FEW_SHOT_EXAMPLES_GOEMOTIONS = [
    ("I absolutely love how this turned out!", "admiration, joy"),
    ("I can't believe they canceled the show. I'm so upset.", "anger, disappointment, sadness"),
    ("Thanks for helping me out, I really appreciate it.", "gratitude"),
    ("I have no idea what's going on here.", "confusion"),
    ("The weather is fine.", "neutral"),
]

# Examples for the shared 7-class Ekman set (single-label output)
FEW_SHOT_EXAMPLES_EKMAN7 = [
    ("I just got promoted at work, I can't stop smiling!", "joy"),
    ("He slammed the door and stormed out without a word.", "anger"),
    ("The smell from the bin made me feel sick.", "disgust"),
    ("I heard footsteps behind me in the dark alley and froze.", "fear"),
    ("My grandmother passed away last week.", "sadness"),
    ("I opened the door and there was a surprise party waiting for me!", "surprise"),
    ("The train leaves at 9am and arrives at noon.", "neutral"),
]


# ──────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────

# Multi-label template (GoEmotions)
ZERO_SHOT_MULTI_TEMPLATE = """\
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

FEW_SHOT_MULTI_TEMPLATE = """\
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

# Single-label template (ISEAR, DailyDialog)
ZERO_SHOT_SINGLE_TEMPLATE = """\
You are an emotion classifier. Your task is to identify the single, \
dominant emotion expressed in the following text.

Choose exactly ONE label from this exact list of emotions:
{label_list}

Output ONLY that one emotion label, with no additional text, explanation, \
or punctuation.

If no specific emotion is expressed, output: neutral

Text: "{text}"

Emotion:"""

FEW_SHOT_SINGLE_TEMPLATE = """\
You are an emotion classifier. Your task is to identify the single, \
dominant emotion expressed in the following text.

Choose exactly ONE label from this exact list of emotions:
{label_list}

Output ONLY that one emotion label, with no additional text.

If no specific emotion is expressed, output: neutral

Here are some examples:

{examples}

Now classify this text:
Text: "{text}"
Emotion:"""


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def build_prompt(text: str, mode: str = "zero_shot", dataset: str = "goemotions") -> str:
    """
    Build a classification prompt for the given text.

    Args:
        text:    The input text to classify.
        mode:    "zero_shot" (default) or "few_shot".
        dataset: "goemotions" (28-class multi-label) or
                 "isear" / "dailydialog" (7-class single-label, shared
                 Ekman label set).

    Returns:
        Formatted prompt string.
    """
    text = text.replace('"', '\\"')  # escape inner quotes

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {list(DATASETS)}")

    multi_label = DATASETS[dataset]["multi_label"]
    label_list = ", ".join(DATASETS[dataset]["labels"])

    if multi_label:
        examples_block = "\n".join(
            f'Text: "{ex}"\nEmotions: {lbl}' for ex, lbl in FEW_SHOT_EXAMPLES_GOEMOTIONS
        )
        if mode == "few_shot":
            return FEW_SHOT_MULTI_TEMPLATE.format(
                label_list=label_list, examples=examples_block, text=text,
            )
        return ZERO_SHOT_MULTI_TEMPLATE.format(label_list=label_list, text=text)

    else:
        examples_block = "\n".join(
            f'Text: "{ex}"\nEmotion: {lbl}' for ex, lbl in FEW_SHOT_EXAMPLES_EKMAN7
        )
        if mode == "few_shot":
            return FEW_SHOT_SINGLE_TEMPLATE.format(
                label_list=label_list, examples=examples_block, text=text,
            )
        return ZERO_SHOT_SINGLE_TEMPLATE.format(label_list=label_list, text=text)