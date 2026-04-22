"""
llm.py
------
All functions that talk to the Ollama LLM:
  - message analysis (emotion, tone, emoji)
  - response generation
  - conversation history management
"""

from __future__ import annotations

from typing import Optional

from ollama import Client

from config import DEBUG, MAX_HISTORY_MESSAGES, MODEL_NAME
from models import MessageAnalysis
from prompts import build_message_analysis_prompt, build_system_prompt
from utils import normalize_assistant_reply, parse_bool, safe_json_extract, ALLOWED_FACE_EMOJIS


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def trim_history(
    messages: list[dict],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict]:
    """Return the *max_messages* most recent entries from *messages*."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def prompt_ready_history(messages: list[dict]) -> list[dict]:
    """Strip non-LLM metadata (timestamps, hints) before sending history."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ---------------------------------------------------------------------------
# Message analysis
# ---------------------------------------------------------------------------

def analyze_user_message(
    client: Client,
    user_text: str,
    facial_emotion_hint: Optional[str] = None,
) -> MessageAnalysis:
    """
    Ask the LLM to interpret the emotional context of *user_text*.

    Returns a :class:`MessageAnalysis` with sensible defaults on any failure.
    """
    _FALLBACK = MessageAnalysis(
        emotion_summary="Emotional context is unclear.",
        reply_tone="natural, neutral, helpful",
        should_use_emoji=False,
        emoji=None,
        reason="",
    )

    prompt = build_message_analysis_prompt(
        user_text=user_text,
        facial_emotion_hint=facial_emotion_hint,
    )

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze the emotional meaning of user messages "
                        "and return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        if DEBUG:
            print(response)

        raw  = response["message"]["content"]
        data = safe_json_extract(raw)

        if not data:
            return MessageAnalysis(
                **{**vars(_FALLBACK),
                   "reason": "The model output could not be parsed as valid JSON."}
            )

        emotion_summary = str(
            data.get("emotion_summary", _FALLBACK.emotion_summary)
        ).strip() or _FALLBACK.emotion_summary

        reply_tone = str(
            data.get("reply_tone", _FALLBACK.reply_tone)
        ).strip() or _FALLBACK.reply_tone

        should_use_emoji = parse_bool(data.get("should_use_emoji", False))

        raw_emoji = data.get("emoji")
        emoji = raw_emoji.strip() if isinstance(raw_emoji, str) else None
        if emoji == "":
            emoji = None
        if emoji not in ALLOWED_FACE_EMOJIS:
            emoji = None
        if emoji is None:
            should_use_emoji = False

        reason = str(data.get("reason", "Derived from the user's message.")).strip()

        return MessageAnalysis(
            emotion_summary=emotion_summary,
            reply_tone=reply_tone,
            should_use_emoji=should_use_emoji,
            emoji=emoji,
            reason=reason,
        )

    except Exception as exc:
        return MessageAnalysis(
            **{**vars(_FALLBACK),
               "reason": f"The message analysis step failed: {exc}"}
        )


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_assistant_reply(
    client: Client,
    conversation_history: list[dict],
    user_text: str,
    analysis: MessageAnalysis,
) -> str:
    """
    Generate the assistant's reply, incorporating conversation history
    and the emotional context encoded in *analysis*.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(analysis)},
        *prompt_ready_history(trim_history(conversation_history)),
        {"role": "user", "content": user_text},
    ]

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
    )

    raw_reply = response["message"]["content"]
    return normalize_assistant_reply(raw_reply, analysis)