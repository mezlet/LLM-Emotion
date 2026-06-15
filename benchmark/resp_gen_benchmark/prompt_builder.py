"""
prompt_builder.py
Zero-shot and few-shot prompt templates for empathetic response generation.
"""

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a compassionate conversational AI. "
    "Given the emotional context and the conversation history, "
    "generate an empathetic and contextually appropriate response. "
    "Keep the response concise (1-3 sentences) and natural."
)

# ── Few-shot examples (one per emotion cluster) ───────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "emotion":    "sad",
        "context":    "I lost my job last week",
        "utterance":  "I just don't know how I'm going to pay my bills.",
        "response":   "That sounds incredibly stressful — losing a job is hard enough "
                      "without the financial pressure on top. You're not alone in this, "
                      "and it's okay to take it one day at a time.",
    },
    {
        "emotion":    "joyful",
        "context":    "I just got engaged!",
        "utterance":  "I can't believe it's finally happening, I'm so happy!",
        "response":   "That's absolutely wonderful news — congratulations! "
                      "This is such an exciting chapter, enjoy every moment of it.",
    },
    {
        "emotion":    "anxious",
        "context":    "I have a big presentation tomorrow",
        "utterance":  "I keep going over it but I still feel so unprepared.",
        "response":   "Feeling nervous before something important is completely normal. "
                      "The fact that you're preparing so carefully shows how much you care — "
                      "trust yourself and the work you've put in.",
    },
]


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_user_prompt(emotion: str, context: str, utterance: str) -> str:
    """Constructs the user-facing portion of the prompt (zero-shot)."""
    return (
        f"Emotional context: {emotion}\n"
        f"Conversation so far: {context}\n"
        f"User says: {utterance}\n"
        f"Your empathetic response:"
    )


def build_full_prompt(
    emotion:   str,
    context:   str,
    utterance: str,
    mode:      str = "zero_shot",
) -> str:
    """
    Single-string prompt for backends that don't use chat templates.

    mode: 'zero_shot' | 'few_shot'
    """
    if mode == "few_shot":
        examples_block = _format_few_shot_examples()
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Here are some examples of empathetic responses:\n"
            f"{examples_block}\n"
            f"Now respond to the following:\n"
            f"{build_user_prompt(emotion, context, utterance)}"
        )
    # zero_shot (default)
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(emotion, context, utterance)}"


def build_chat_messages(
    emotion:   str,
    context:   str,
    utterance: str,
    mode:      str = "zero_shot",
) -> list[dict]:
    """
    Chat-template messages list for HF apply_chat_template / OpenAI-style APIs.

    mode: 'zero_shot' | 'few_shot'
    """
    if mode == "few_shot":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": build_user_prompt(ex["emotion"], ex["context"], ex["utterance"]),
            })
            messages.append({"role": "assistant", "content": ex["response"]})
        messages.append({"role": "user", "content": build_user_prompt(emotion, context, utterance)})
        return messages

    # zero_shot (default)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(emotion, context, utterance)},
    ]


# ── Internal ──────────────────────────────────────────────────────────────────

def _format_few_shot_examples() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(
            f"Emotional context: {ex['emotion']}\n"
            f"Conversation so far: {ex['context']}\n"
            f"User says: {ex['utterance']}\n"
            f"Response: {ex['response']}\n"
        )
    return "\n".join(lines)