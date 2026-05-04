from __future__ import annotations

from ollama import Client

from config import MODEL_NAME
from json_utils import safe_json_extract
from models import ExpressionPlan, MessageAnalysis


def build_expression_plan_prompt(
    user_text: str,
    fused_emotion: MessageAnalysis,
) -> str:
    emotion = fused_emotion.emotion

    return f"""
You are planning Ameca's expressive response.

Ameca is a humanoid social robot. It can express emotion through:
- verbal tone
- facial expression
- upper-body gestures
- optional facial emoji in text-only logs

Your task is to choose an expression that is emotionally congruent with the user's state.

Important:
- Do not choose a smiling or cheerful expression when the user is distressed, sad, angry, frustrated, hungry, worried, or describing harm/unfairness.
- Do not over-perform emotion.
- Ameca is a robot, so do not pretend to feel human emotions.
- If the situation is serious or negative, emoji_allowed should usually be false.
- If an emoji is allowed, choose only one facial emoji.
- The expression should support the user, not mirror them dramatically.

User message:
{user_text}

Fused user emotion:
primary={emotion.primary}
secondary={emotion.secondary}
blended={emotion.blended_emotion}
intensity={emotion.intensity_score}
confidence={emotion.confidence}
valence={emotion.valence}
arousal={emotion.arousal}

Return JSON only:
{{
  "verbal_tone": "gentle and apologetic",
  "facial_expression": "concerned",
  "gesture": "small nod",
  "emoji_allowed": false,
  "emoji": null,
  "response_length": "short",
  "reason": "brief reason"
}}
""".strip()


def plan_expressive_response(
    client: Client,
    user_text: str,
    fused_emotion: MessageAnalysis,
) -> ExpressionPlan:
    prompt = build_expression_plan_prompt(user_text, fused_emotion)

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You plan emotionally congruent robot expressions. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        data = safe_json_extract(response["message"]["content"]) or {}

        return ExpressionPlan(
            verbal_tone=str(data.get("verbal_tone", "calm and supportive")),
            facial_expression=str(data.get("facial_expression", "neutral-attentive")),
            gesture=data.get("gesture"),
            emoji_allowed=bool(data.get("emoji_allowed", False)),
            emoji=data.get("emoji"),
            response_length=str(data.get("response_length", "short")),
            reason=str(data.get("reason", "Expression chosen from fused emotion.")),
        )

    except Exception as e:
        return ExpressionPlan(
            verbal_tone="calm and supportive",
            facial_expression="neutral-attentive",
            gesture=None,
            emoji_allowed=False,
            emoji=None,
            response_length="short",
            reason=f"Planner failed: {e}",
        )