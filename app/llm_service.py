from __future__ import annotations

from typing import Optional

from ollama import Client

from config import DEBUG, MAX_HISTORY_MESSAGES, MODEL_NAME
from emoji_policy import normalize_assistant_reply
from json_utils import parse_bool, safe_json_extract
from models import EmotionState, EmojiPolicy, MessageAnalysis, ModalityEmotion, MultimodalEmotionContext, ResponsePolicy
from plutchik import (
    EMOJI_BY_EXPRESSION_TYPE,
    PLUTCHIK_INTENSITY_LABELS,
    RESPONSE_STYLES,
    SOCIAL_INTENTS,
    canonical_arousal,
    canonical_valence,
    clamp,
    derive_blended_emotion,
    derive_emoji_type,
    intensity_level_from_score,
    normalize_primary_emotion,
    normalize_secondary_emotion,
)
from prompts import build_message_analysis_prompt, build_system_prompt


def trim_history(messages: list[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    return messages if len(messages) <= max_messages else messages[-max_messages:]


def prompt_ready_history(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _default_analysis(reason: str = "Fallback neutral analysis.") -> MessageAnalysis:
    emotion = EmotionState(
        primary="trust",
        secondary=None,
        blended_emotion=None,
        intensity_level="low",
        intensity_label="acceptance",
        intensity_score=0.2,
        valence=0.35,
        arousal=0.25,
        confidence=0.3,
    )
    emoji = EmojiPolicy(use=True, type="neutral", emoji="🙂")
    response = ResponsePolicy(style="friendly", verbosity="short", emoji=emoji)
    return MessageAnalysis(emotion=emotion, social_intent="unclear", response=response, reason=reason)


def _analysis_from_json(data: dict) -> MessageAnalysis:
    emotion_data = data.get("emotion") if isinstance(data.get("emotion"), dict) else {}
    response_data = data.get("response") if isinstance(data.get("response"), dict) else {}
    emoji_data = response_data.get("emoji") if isinstance(response_data.get("emoji"), dict) else {}

    primary = normalize_primary_emotion(emotion_data.get("primary"), default="trust")
    secondary = normalize_secondary_emotion(emotion_data.get("secondary"))

    intensity_score = clamp(emotion_data.get("intensity_score"), default=0.35)
    intensity_level = str(emotion_data.get("intensity_level") or intensity_level_from_score(intensity_score)).strip().lower()
    if intensity_level not in {"low", "medium", "high"}:
        intensity_level = intensity_level_from_score(intensity_score)

    intensity_label = PLUTCHIK_INTENSITY_LABELS[primary][intensity_level]

    blended_emotion = emotion_data.get("blended_emotion")
    if not isinstance(blended_emotion, str) or not blended_emotion.strip():
        blended_emotion = derive_blended_emotion(primary, secondary)
    else:
        blended_emotion = blended_emotion.strip().lower()

    valence = canonical_valence(primary, intensity_score)
    arousal = canonical_arousal(primary, intensity_score)
    confidence = clamp(emotion_data.get("confidence"), default=0.6)

    social_intent = str(data.get("social_intent") or "unclear").strip().lower()
    if social_intent not in SOCIAL_INTENTS:
        social_intent = "unclear"

    response_style = str(response_data.get("style") or "friendly").strip().lower()
    if response_style not in RESPONSE_STYLES:
        response_style = "friendly"

    verbosity = str(response_data.get("verbosity") or "short").strip().lower()
    if verbosity not in {"short", "medium", "detailed"}:
        verbosity = "short"

    emoji_use = parse_bool(emoji_data.get("use", True))
    emoji_type = emoji_data.get("type")
    emoji_type = str(emoji_type).strip().lower() if emoji_type else derive_emoji_type(primary, social_intent, response_style)
    emoji_char = EMOJI_BY_EXPRESSION_TYPE.get(emoji_type)
    if not emoji_char:
        emoji_char = "🙂"
        emoji_type = "neutral"

    emotion = EmotionState(
        primary=primary,
        secondary=secondary,
        blended_emotion=blended_emotion,
        intensity_level=intensity_level,
        intensity_label=intensity_label,
        intensity_score=round(intensity_score, 2),
        valence=valence,
        arousal=arousal,
        confidence=round(confidence, 2),
    )
    emoji = EmojiPolicy(use=emoji_use, type=emoji_type, emoji=emoji_char)
    response = ResponsePolicy(style=response_style, verbosity=verbosity, emoji=emoji)
    reason = str(data.get("reason") or "Derived from the user's message.").strip()

    return MessageAnalysis(emotion=emotion, social_intent=social_intent, response=response, reason=reason)


def analyze_user_message_with_llm(
    client: Client,
    user_text: str,
    facial_emotion_hint: Optional[str] = None,
) -> MessageAnalysis:
    prompt = build_message_analysis_prompt(user_text=user_text, facial_emotion_hint=facial_emotion_hint)

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return valid JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        
        if DEBUG:
            print(response)

        raw = response["message"]["content"]
        data = safe_json_extract(raw)
        if not data:
            return _default_analysis("The model output could not be parsed as valid JSON.")

        return _analysis_from_json(data)

    except Exception as e:
        return _default_analysis(f"The message analysis step failed: {e}")


def modality_from_text_analysis(text_analysis: MessageAnalysis) -> ModalityEmotion:
    return ModalityEmotion(
        modality="text",
        emotion=text_analysis.emotion,
        available=True,
        source="ollama_llm_text_classifier",
        summary=f"social_intent={text_analysis.social_intent}; reason={text_analysis.reason}",
    )


def build_multimodal_context(text_analysis: MessageAnalysis, prosody_emotion: ModalityEmotion, face_emotion: ModalityEmotion) -> MultimodalEmotionContext:
    return MultimodalEmotionContext(
        text=modality_from_text_analysis(text_analysis),
        prosody=prosody_emotion,
        face=face_emotion,
    )


def generate_assistant_reply(
    client: Client,
    conversation_history: list[dict],
    user_text: str,
    text_analysis: MessageAnalysis,
    multimodal_context: MultimodalEmotionContext,
) -> str:
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                text_analysis=text_analysis,
                multimodal_context=multimodal_context,
            ),
        },
        *prompt_ready_history(trim_history(conversation_history)),
        {"role": "user", "content": user_text},
    ]

    response = client.chat(model=MODEL_NAME, messages=messages, stream=False)
    raw_reply = response["message"]["content"]
    return normalize_assistant_reply(raw_reply)
