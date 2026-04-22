"""
prompts.py
----------
Functions that build prompt strings sent to the LLM.
Kept separate so prompt wording can be iterated independently.
"""

from __future__ import annotations

from typing import Optional

from models import MessageAnalysis


def build_message_analysis_prompt(
    user_text: str,
    facial_emotion_hint: Optional[str] = None,
) -> str:
    """
    Ask the LLM to analyse the emotional meaning of a user message.

    Design notes:
    - Text is the *primary* signal.
    - The facial-expression hint is a weak, noisy supporting signal only.
    - The LLM decides emoji suitability and selection.
    """
    facial_section = (
        f"Facial-expression hint captured during speech: {facial_emotion_hint}\n"
        if facial_emotion_hint
        else "No facial-expression hint was available.\n"
    )

    return f"""
You are analyzing the emotional meaning of a user's message.

Your task:
- infer the user's emotional state primarily from the user's words
- use the facial-expression hint only as a weak supporting signal
- if the user's words clearly express happiness, excitement, pride, relief, or positive news, do not let a conflicting facial-expression hint override that
- if the words and facial-expression hint conflict, trust the words more
- decide the best reply tone
- decide whether a facial emoji should be used
- if an emoji should be used, choose exactly one facial emoji

Important rules:
- Base your judgment mainly on the user's message.
- Treat facial-expression analysis as weak, noisy supporting evidence only.
- Do not assume future outcomes the user did not state.
- Do not make strong claims based on facial-expression analysis.
- If the message is technical, factual, task-focused, or command-like, usually set should_use_emoji to false.
- Only choose a facial emoji.
- If no emoji is appropriate, set emoji to null.
- Keep emotion_summary natural and concise.
- Keep reply_tone concise and practical.

Return JSON only in this exact format:
{{
  "emotion_summary": "short natural-language summary of the user's emotional state",
  "reply_tone": "short description of how the assistant should sound",
  "should_use_emoji": true,
  "emoji": "😊",
  "reason": "brief explanation"
}}

User message:
{user_text}

{facial_section}
""".strip()


def build_system_prompt(analysis: MessageAnalysis) -> str:
    """
    Build the system prompt used when generating the assistant's reply.

    Encodes the emotional tone and emoji decision from *analysis*.
    """
    emoji_instruction = (
        f"Use exactly this one facial emoji at the very end of the response: {analysis.emoji}"
        if analysis.should_use_emoji and analysis.emoji
        else "Do not use any emoji."
    )

    return (
        "You are a warm, helpful and emotionally aware assistant. "
        "Respond to the user's message in a natural, socially appropriate way. "
        "Do not sound robotic, stiff, overly formal, or generic. "
        "Do not assume outcomes the user has not confirmed. "
        "Base your reply only on what the user actually said. "
        "Use face emojis to convey the tone. "
        "Do not stay too long on a particular context. "
        "Use full sentences. "
        "Never use non-face emojis. "
        "Never use more than one emoji. "
        f"Emotional interpretation: {analysis.emotion_summary}. "
        f"Reply tone: {analysis.reply_tone}. "
        f"{emoji_instruction}"
    )