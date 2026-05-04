from __future__ import annotations

from ollama import Client

from config import MODEL_NAME
from json_utils import safe_json_extract
from models import MessageAnalysis, RobotAffectPlan, RobotExpression


def build_robot_affect_plan_prompt(
    user_text: str,
    fused_emotion: MessageAnalysis,
    input_mode: str = "typed",
) -> str:
    """
    Build a thesis-aligned affective congruence planning prompt.

    This planner does NOT decide "emoji or no emoji" as the main goal.
    It decides how Ameca should behave as an embodied robot:
    - verbal tone
    - facial expression
    - upper-body gesture
    - speech delivery style
    - expressive intensity
    """
    emotion = fused_emotion.emotion

    return f"""
You are Ameca's affective congruence planner.

Ameca is a humanoid social robot used in a university laboratory for research and demonstrations.
Your job is to choose a robot expression plan that is congruent with the user's fused emotional state.

The goal is not to mirror the user dramatically.
The goal is to produce socially appropriate embodied support through:
- facial expression
- upper-body gesture
- verbal tone
- speech style
- response length

IMPORTANT PRINCIPLES
- Ameca is a robot, not a human. Do not plan expressions that imply human lived experience.
- Do not choose cheerful, smiling, playful, or excited expressions when the user is distressed, sad, angry, hungry, worried, or reporting unfair treatment.
- For negative or serious user states, prefer concerned, neutral-attentive, apologetic, validating, or calm-supportive expressions.
- For friendly greetings or positive social openings, a gentle welcoming smile is appropriate.
- Keep robot expression intensity lower than the user's emotional intensity unless the user is clearly joyful or excited.
- Do not over-perform sadness, anger, fear, or excitement.
- Do not use the phrase "I might have misheard" unless input_mode is speech and runtime metadata explicitly indicates low ASR confidence.
- Do not describe Ameca as the user's friend. Use robot, robot assistant, or Ameca.
- The plan must support congruent expressive response generation for a social humanoid robot.

INPUT MODE
{input_mode}

USER MESSAGE
{user_text}

FUSED USER AFFECT
primary_emotion={emotion.primary}
secondary_emotion={emotion.secondary}
blended_emotion={emotion.blended_emotion}
intensity_level={emotion.intensity_level}
intensity_score={emotion.intensity_score}
intensity_label={emotion.intensity_label}
valence={emotion.valence}
arousal={emotion.arousal}
confidence={emotion.confidence}
social_intent={fused_emotion.social_intent}

Return JSON only in this exact structure:
{{
  "robot_affect": {{
    "stance": "supportive_concern",
    "facial_expression": "concerned",
    "gesture": "slow nod",
    "speech_style": "calm, validating, practical",
    "verbal_tone": "gentle and serious",
    "expressive_intensity": 0.45,
    "response_length": "short"
  }},
  "congruence_reason": "The user reports distress and unfair treatment, so Ameca should show concern and validation rather than cheerfulness."
}}
""".strip()


def _clamp_float(value, default: float = 0.4, min_value: float = 0.0, max_value: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(min_value, min(max_value, number))


def _default_robot_affect_plan(reason: str = "Fallback robot affect plan.") -> RobotAffectPlan:
    return RobotAffectPlan(
        robot_affect=RobotExpression(
            stance="neutral_support",
            facial_expression="neutral-attentive",
            gesture="small nod",
            speech_style="calm, clear, supportive",
            verbal_tone="calm and professional",
            expressive_intensity=0.3,
            response_length="short",
        ),
        congruence_reason=reason,
    )


def plan_robot_affect(
    client: Client,
    user_text: str,
    fused_emotion: MessageAnalysis,
    input_mode: str = "typed",
) -> RobotAffectPlan:
    """
    Plan Ameca's embodied affect after multimodal fusion and before response generation.

    This is the thesis-aligned stage:
    fused user affect -> congruent robot affect plan -> response generation / robot controller.
    """
    prompt = build_robot_affect_plan_prompt(
        user_text=user_text,
        fused_emotion=fused_emotion,
        input_mode=input_mode,
    )

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You plan emotionally congruent embodied expressions for Ameca, "
                        "a humanoid social robot. Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        data = safe_json_extract(response["message"]["content"]) or {}
        affect_data = data.get("robot_affect") if isinstance(data.get("robot_affect"), dict) else {}

        response_length = str(affect_data.get("response_length") or "short").strip().lower()
        if response_length not in {"short", "medium", "detailed"}:
            response_length = "short"

        plan = RobotAffectPlan(
            robot_affect=RobotExpression(
                stance=str(affect_data.get("stance") or "neutral_support").strip(),
                facial_expression=str(affect_data.get("facial_expression") or "neutral-attentive").strip(),
                gesture=str(affect_data.get("gesture") or "small nod").strip(),
                speech_style=str(affect_data.get("speech_style") or "calm, clear, supportive").strip(),
                verbal_tone=str(affect_data.get("verbal_tone") or "calm and professional").strip(),
                expressive_intensity=round(_clamp_float(affect_data.get("expressive_intensity"), default=0.35), 2),
                response_length=response_length,
            ),
            congruence_reason=str(
                data.get("congruence_reason") or "Robot expression selected from fused user affect."
            ).strip(),
        )
        return plan

    except Exception as exc:
        return _default_robot_affect_plan(f"Robot affect planner failed: {exc}")
