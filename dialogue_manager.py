from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# =========================
# Lesson state
# =========================

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "please", "affirmative", "okay", "ok"}
_NO_WORDS = {"no", "nope", "nah", "negative", "skip"}

VALID_INTENTS = {"continue", "clarify", "switch_topic", "resume_topic", "new"}


_MINIMAL_ACK_WORDS = {
    "okay", "ok", "yes", "yeah", "yep", "yup", "sure", "alright",
    "go on", "goon", "continue", "next", "got it", "gotit", "mhm", "mmhm",
    "uh huh", "uhhuh", "right", "fine", "cool", "great",
    "mm hmm", "mmhmm", "hmm", "hm", "hmm hmm", "mm", "uh", "um", "erm",
}


def is_minimal_acknowledgement(text: str) -> bool:
    stripped = re.sub(r"[^a-z\s]", " ", text.strip().lower())
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped:
        return False
    return stripped in _MINIMAL_ACK_WORDS


_JUNK_CONCEPT_VALUES = {"none", "null", "n/a", "na", "undefined"}


def _is_junk_concept_value(value: Optional[str]) -> bool:
    if not value:
        return True
    return value.strip().lower() in _JUNK_CONCEPT_VALUES


def parse_yes_no(text: str, default: bool = False) -> bool:
    normalized = re.sub(r"[^a-z\s]", " ", text.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return default

    words = set(normalized.split())
    if "not" in words or (words & _NO_WORDS):
        return False
    if words & _YES_WORDS:
        return True
    return default


@dataclass
class LessonState:
    """
    Explicit dialogue/teaching state, carried across turns for a single
    user (persisted alongside their users.json profile -- see
    to_json()/from_json()). This is what the model is missing when it
    "forgets" it was mid-explanation and restarts from the beginning.
    """

    current_topic: Optional[str] = None
    current_subtopic: Optional[str] = None
    covered_concepts: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    current_teaching_goal: Optional[str] = None
    last_intent: str = "new"

    awaiting_retention_check: bool = False
    last_new_concept: Optional[str] = None

    # ---- persistence ----

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Optional[dict[str, Any]]) -> "LessonState":
        if not isinstance(data, dict):
            return cls()

        current_topic = data.get("current_topic")
        if _is_junk_concept_value(current_topic):
            current_topic = None

        current_subtopic = data.get("current_subtopic")
        if _is_junk_concept_value(current_subtopic):
            current_subtopic = None

        last_new_concept = data.get("last_new_concept")
        if _is_junk_concept_value(last_new_concept):
            last_new_concept = None

        covered_concepts = [
            c for c in (data.get("covered_concepts") or []) if not _is_junk_concept_value(c)
        ]

        return cls(
            current_topic=current_topic,
            current_subtopic=current_subtopic,
            covered_concepts=covered_concepts,
            pending_questions=list(data.get("pending_questions") or []),
            current_teaching_goal=data.get("current_teaching_goal"),
            last_intent=data.get("last_intent", "new"),
            awaiting_retention_check=bool(data.get("awaiting_retention_check", False)),
            last_new_concept=last_new_concept,
        )

    # ---- mutation helpers ----

    def mark_concept_covered(self, concept: str) -> None:
        concept = (concept or "").strip()
        if _is_junk_concept_value(concept):
            return
        if concept and concept not in self.covered_concepts:
            self.covered_concepts.append(concept)
        self.covered_concepts = self.covered_concepts[-12:]

    def add_pending_question(self, question: str) -> None:
        question = (question or "").strip()
        if question and question not in self.pending_questions:
            self.pending_questions.append(question)
        self.pending_questions = self.pending_questions[-6:]

    def resolve_pending_questions(self, resolved: list[str]) -> None:
        if not resolved or not self.pending_questions:
            return

        def _norm(text: str) -> str:
            return re.sub(r"\s+", " ", text.strip().lower())

        resolved_norms = [_norm(r) for r in resolved if r and r.strip()]
        if not resolved_norms:
            return

        kept: list[str] = []
        for pending in self.pending_questions:
            pending_norm = _norm(pending)
            matched = any(
                pending_norm in rn or rn in pending_norm
                for rn in resolved_norms
            )
            if not matched:
                kept.append(pending)
        self.pending_questions = kept

    def reset_for_new_topic(self, new_topic: str, new_subtopic: Optional[str] = None) -> None:
        self.current_topic = new_topic
        self.current_subtopic = new_subtopic
        self.covered_concepts = []
        self.current_teaching_goal = None

    def maybe_flag_retention_check(self, introduced_new_concept: bool, concept: Optional[str]) -> None:
        if introduced_new_concept and concept:
            self.awaiting_retention_check = True
            self.last_new_concept = concept
        else:
            self.awaiting_retention_check = False
            self.last_new_concept = None

    # ---- prompt rendering ----

    def to_prompt_block(self) -> str:
        if not self.current_topic and not self.covered_concepts and not self.pending_questions:
            return "LESSON STATE\nNo lesson has started yet; this is the first substantive topic of the session."

        lines = ["LESSON STATE"]
        if self.current_topic:
            topic_line = f"Current topic: {self.current_topic}"
            if self.current_subtopic:
                topic_line += f" -- current subtopic: {self.current_subtopic}"
            lines.append(topic_line)

        if self.covered_concepts:
            lines.append("Already explained (do not repeat unless asked): " + ", ".join(self.covered_concepts))

        if self.current_teaching_goal:
            lines.append(f"Current teaching goal: {self.current_teaching_goal}")

        if self.pending_questions:
            lines.append(
                "Unresolved questions still owed to the learner (address these before "
                "moving on, or explicitly say you're coming back to them): "
                + "; ".join(self.pending_questions)
            )
        else:
            lines.append("No unresolved questions are currently owed to the learner.")

        if self.awaiting_retention_check and self.last_new_concept:
            lines.append(
                f"RETENTION CHECK DUE: you just introduced '{self.last_new_concept}' and the "
                "learner's last reply was only a bare acknowledgement (e.g. 'okay'), not a "
                "real question or explanation in their own words. Before introducing the next "
                "new concept, ask ONE short check -- e.g. a quick recall question or 'does that "
                "make sense so far?' -- rather than moving straight on to new material."
            )

        block = "\n".join(lines)
        return block[:1100]


# =========================
# Curriculum ordering
# =========================

CURRICULUM_ORDER: dict[str, list[str]] = {
    "large_language_models": [
        "tokens",
        "next-token prediction",
        "context window",
        "embeddings",
        "attention",
        "transformer architecture",
        "training (pretraining/fine-tuning)",
        "hallucination and limitations",
        "robotics use case",
    ],
    "machine_learning_basics": [
        "what learning from data means",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "training vs. inference",
        "generalization and overfitting",
    ],
    "neural_networks": [
        "neurons and layers",
        "weights and activation functions",
        "forward pass",
        "loss / error",
        "backpropagation and gradient descent",
        "training over many examples",
    ],
}

_CURRICULUM_TOPIC_KEYWORDS: dict[str, str] = {
    "large language model": "large_language_models",
    "large-level model": "large_language_models",
    "llm": "large_language_models",
    "transformer": "large_language_models",
    "machine learning": "machine_learning_basics",
    "supervised": "machine_learning_basics",
    "unsupervised": "machine_learning_basics",
    "neural network": "neural_networks",
    "backpropagation": "neural_networks",
    "activation function": "neural_networks",
}


def match_curriculum_family(topic: Optional[str], subtopic: Optional[str] = None) -> Optional[str]:
    haystack = f"{topic or ''} {subtopic or ''}".lower()
    for keyword, family in _CURRICULUM_TOPIC_KEYWORDS.items():
        if keyword in haystack:
            return family
    return None


def next_curriculum_subtopic(
    topic: Optional[str],
    covered_concepts: list[str],
    subtopic: Optional[str] = None,
) -> Optional[str]:
    family = match_curriculum_family(topic, subtopic)
    if not family:
        return None

    covered_norm = {c.strip().lower() for c in covered_concepts}
    for step in CURRICULUM_ORDER[family]:
        if step.lower() not in covered_norm:
            return step

    return None


def build_understanding_check_prompt() -> str:
    return "Did that make sense so far?"


def build_confusion_followup_prompt() -> str:
    return "No problem -- what part felt confusing, or would another example help?"


def build_topic_choice_followup_prompt() -> str:
    return "Would you like to move on to another topic, or is there something specific you'd like to talk about?"


def generate_topic_check_question(
    client: Any,
    model_name: str,
    topic_title: str,
    explanation_text: str,
    level: str,
    safe_json_extract,
    print_ts=print,
) -> str:
    fallback = f"In your own words, what's the main idea behind {topic_title}?"
    prompt = f"""
        You are the comprehension-check component of a tutoring robot. The
        tutor just explained the following at {level} level:

        Topic: {topic_title}
        Explanation given: {explanation_text}

        Write ONE short question (under 20 words) that tests whether the
        learner actually understood this specific explanation -- not a
        generic definition question, but something that requires having
        followed what was just said.

        Return JSON only: {{"question": "..."}}
        """.strip()
    try:
        response = client.chat(
            model=model_name, format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 100, "num_ctx": 2048},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if isinstance(data, dict) and str(data.get("question", "")).strip():
            return str(data["question"]).strip()
    except Exception as exc:
        print_ts(f"[COMPREHENSION] Check-question generation failed ({exc}); using fallback.")
    return fallback


def grade_comprehension_check_answer(
    client: Any,
    model_name: str,
    question: str,
    answer: str,
    explanation_text: str,
    safe_json_extract,
    print_ts=print,
) -> str:
    fallback = "Thanks for sharing that -- let's keep going."
    prompt = f"""
        You are a warm, encouraging tutor. You just asked the learner:
        "{question}"
        based on this explanation you gave: "{explanation_text}"

        They answered: "{answer}"

        Write ONE short, natural spoken response (max 30 words) that:
        - if their answer is correct or close enough, warmly affirms it
        - if their answer is wrong or off-target, gently corrects it by
          restating the key point in one sentence, without making them
          feel bad

        Do not use the literal words "correct" or "incorrect". Plain text
        only, no markdown, no preamble.
        """.strip()
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 90, "num_ctx": 2048},
            stream=False,
        )
        text = response["message"]["content"].strip()
        text = re.sub(r"\s+", " ", text)
        return text or fallback
    except Exception as exc:
        print_ts(f"[COMPREHENSION] Grading call failed ({exc}); using fallback.")
        return fallback


# =========================
# Dialogue planner
# =========================

def build_planner_prompt(
    user_text: str,
    lesson_state: LessonState,
    recent_history: list[dict],
) -> str:
    recent_turns = "\n".join(
        f"{item.get('role', '?')}: {item.get('content', '')}" for item in (recent_history or [])[-6:]
    )

    suggested_next_subtopic = next_curriculum_subtopic(
        topic=lesson_state.current_topic,
        covered_concepts=lesson_state.covered_concepts,
        subtopic=lesson_state.current_subtopic,
    )
    curriculum_guidance = (
        f"\n        This topic has a known beginner-friendly teaching order. The next "
        f"not-yet-covered step in that order is: \"{suggested_next_subtopic}\". Prefer "
        "this as the subtopic/teaching_goal UNLESS the user explicitly asked about a "
        "different, specific subtopic themselves -- in that case honor their request "
        "instead, even if it's out of this suggested order.\n"
        if suggested_next_subtopic
        else ""
    )

    retention_note = (
        "\n        The learner's latest message is a bare acknowledgement (e.g. 'okay') "
        "with no real content, and new material was just introduced last turn. Do NOT "
        "treat this as a request to advance to a new concept -- classify intent as "
        "'continue' and set teaching_goal to a brief retention check on what was just "
        "covered, not a new topic.\n"
        if lesson_state.awaiting_retention_check and is_minimal_acknowledgement(user_text)
        else ""
    )

    return f"""
        You are the dialogue-planning component of a tutoring robot. You do NOT
        answer the user -- you only decide how their latest message relates to
        the ongoing lesson, before a separate component writes the actual reply.

        Current lesson state:
        - current_topic: {lesson_state.current_topic or "none"}
        - current_subtopic: {lesson_state.current_subtopic or "none"}
        - already_explained: {lesson_state.covered_concepts or "none"}
        - unresolved_questions: {lesson_state.pending_questions or "none"}
        {curriculum_guidance}{retention_note}
        Recent conversation:
        {recent_turns or "(none yet)"}

        Latest user message:
        {user_text}

        Classify the latest message as exactly one of:
        - "continue": it follows on from the current topic/subtopic; keep building
          on what's already been explained, do not restart from the beginning.
        - "clarify": the user is asking to re-explain or clarify something already
          covered.
        - "switch_topic": the user is moving to a genuinely new topic unrelated to
          the current one.
        - "resume_topic": the user is returning to an earlier topic or an
          unresolved question from before.
        - "new": there is no current lesson yet (this is the first real topic).

        Also identify:
        - topic: the general subject of the latest message (short phrase).
        - subtopic: the specific sub-point being asked about, if any (short phrase
          or null).
        - teaching_goal: one short phrase describing what the next reply should
          accomplish.
        - resolved_questions: which of the unresolved_questions above (if any) does
          the LATEST USER MESSAGE now make it possible to fully answer? List their
          text (or a close paraphrase), or an empty list if none.
        - new_pending_questions: if the latest message asks more than one distinct
          thing, list every sub-question EXCEPT the primary one that should be
          answered right now, so the others aren't lost. Empty list if the message
          only asks one thing.

        Return JSON only, in exactly this shape:
        {{
          "intent": "continue|clarify|switch_topic|resume_topic|new",
          "topic": "...",
          "subtopic": "..." ,
          "teaching_goal": "...",
          "resolved_questions": ["..."],
          "new_pending_questions": ["..."]
        }}
        """.strip()


def _clean_str_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def classify_dialogue_turn(
    client: Any,
    user_text: str,
    lesson_state: LessonState,
    history: list[dict],
    safe_json_extract,
    model_name: str,
    print_ts=print,
) -> dict[str, Any]:
    fallback = {
        "intent": "continue",
        "topic": lesson_state.current_topic,
        "subtopic": lesson_state.current_subtopic,
        "teaching_goal": lesson_state.current_teaching_goal,
        "resolved_questions": [],
        "new_pending_questions": [],
    }

    try:
        prompt = build_planner_prompt(user_text, lesson_state, history)
        response = client.chat(
            model=model_name,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 220, "num_ctx": 2048},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return fallback

        intent = str(data.get("intent", "")).strip().lower()
        if intent not in VALID_INTENTS:
            intent = "continue"

        if lesson_state.awaiting_retention_check and is_minimal_acknowledgement(user_text):
            intent = "continue"
            topic = lesson_state.current_topic
            subtopic = None
            teaching_goal = (
                f"Briefly check the learner's understanding of "
                f"'{lesson_state.last_new_concept}' before introducing anything new."
            )
        elif is_minimal_acknowledgement(user_text):
            suggested_next_subtopic = next_curriculum_subtopic(
                topic=lesson_state.current_topic,
                covered_concepts=lesson_state.covered_concepts,
                subtopic=lesson_state.current_subtopic,
            )
            intent = "continue"
            topic = lesson_state.current_topic
            if suggested_next_subtopic:
                subtopic = suggested_next_subtopic
                teaching_goal = (
                    f"The learner gave only a brief acknowledgement with no new "
                    f"question. Do NOT repeat previous content verbatim. Teach the "
                    f"next step in this topic: '{suggested_next_subtopic}'."
                )
            else:
                subtopic = _clean_str_field(data.get("subtopic")) or None
                teaching_goal = (
                    "The learner gave only a brief acknowledgement with no new "
                    "question, and every planned step in this topic has already "
                    "been covered. Do NOT repeat previous content verbatim -- "
                    "briefly invite them to ask a follow-up question or suggest "
                    "moving to a different lesson topic."
                )
        else:
            topic = _clean_str_field(data.get("topic")) or lesson_state.current_topic
            subtopic = _clean_str_field(data.get("subtopic")) or None
            teaching_goal = _clean_str_field(data.get("teaching_goal")) or lesson_state.current_teaching_goal

        return {
            "intent": intent,
            "topic": topic,
            "subtopic": subtopic,
            "teaching_goal": teaching_goal,
            "resolved_questions": [
                cleaned for cleaned in (
                    _clean_str_field(q) for q in (data.get("resolved_questions") or [])
                ) if cleaned
            ],
            "new_pending_questions": [
                cleaned for cleaned in (
                    _clean_str_field(q) for q in (data.get("new_pending_questions") or [])
                ) if cleaned
            ],
        }
    except Exception as exc:
        print_ts(f"[PLANNER] Dialogue planner call failed ({exc}); defaulting to intent='continue'.")
        return fallback


def apply_planner_output(lesson_state: LessonState, planner_output: dict[str, Any]) -> LessonState:
    intent = planner_output.get("intent", "continue")
    topic = planner_output.get("topic")
    subtopic = planner_output.get("subtopic")

    if intent in ("switch_topic", "new") and topic:
        lesson_state.reset_for_new_topic(topic, subtopic)
    else:
        if topic:
            lesson_state.current_topic = topic
        if subtopic:
            lesson_state.current_subtopic = subtopic

    if planner_output.get("teaching_goal"):
        lesson_state.current_teaching_goal = planner_output["teaching_goal"]

    lesson_state.last_intent = intent

    lesson_state.awaiting_retention_check = False
    lesson_state.last_new_concept = None

    for extra_question in planner_output.get("new_pending_questions", []):
        lesson_state.add_pending_question(extra_question)

    lesson_state.resolve_pending_questions(planner_output.get("resolved_questions", []))

    return lesson_state


def finalize_lesson_state_after_reply(
    lesson_state: LessonState,
    planner_output: dict[str, Any],
) -> LessonState:
    concept = planner_output.get("subtopic") or planner_output.get("topic")
    if _is_junk_concept_value(concept):
        concept = None

    concept_already_covered = False
    if concept:
        concept_norm = concept.strip().lower()
        concept_already_covered = any(
            concept_norm == covered.strip().lower() for covered in lesson_state.covered_concepts
        )
        lesson_state.mark_concept_covered(concept)

    lesson_state.maybe_flag_retention_check(
        introduced_new_concept=bool(concept) and not concept_already_covered,
        concept=concept,
    )
    return lesson_state


# =========================
# Dynamic reply length
# =========================

REPLY_LENGTH_BY_CASE = {
    "clarify": 40,
    "new_topic": 130,
    "continue": 90,
    "multi_part": 220,
    "retention_check": 55,
}


def resolve_reply_word_budget(
    intent: str,
    is_multi_part: bool,
    awaiting_retention_check: bool = False,
    default_min: int = 25,
) -> int:
    if is_multi_part:
        return REPLY_LENGTH_BY_CASE["multi_part"]
    if awaiting_retention_check:
        return REPLY_LENGTH_BY_CASE["retention_check"]
    if intent == "clarify":
        return REPLY_LENGTH_BY_CASE["clarify"]
    if intent in ("new", "switch_topic"):
        return REPLY_LENGTH_BY_CASE["new_topic"]
    return max(default_min, REPLY_LENGTH_BY_CASE["continue"])
