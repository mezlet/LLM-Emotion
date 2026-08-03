"""
dialogue_manager.py

Implements the top-priority recommendations from the prompt/architecture
review:

  1. An explicit LessonState object (current topic/subtopic, covered
     concepts, pending/unresolved questions, current teaching goal) --
     replacing "the model reconstructs teaching state from raw history".
  2. A lightweight dialogue planner (classify_dialogue_turn) that decides,
     BEFORE the teaching reply is generated, whether the turn continues
     the lesson, asks for clarification, switches topic, or resumes an
     earlier topic -- output goes straight into the response-generation
     system prompt instead of leaving that judgment implicit.
  3. Pending-question tracking across turns, so a multi-part question
     ("What is RLHF and why does it matter?") doesn't silently lose its
     second half once the first is answered.
  4. A short (~150 word) rolling LESSON summary block, separate from the
     generic free-text conversation_summary already used for the
     returning-user greeting -- this one is teaching-state-shaped, not a
     narrative recap.
  5. Dynamic reply-length guidance derived from the planner's intent
     instead of a single fixed MAX_REPLY_SENTENCES that conflicts with
     "be elaborate, teach thoroughly".

Drop this file next to ameca_demo.py and see INTEGRATION.md for the exact
call sites to wire it in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# =========================
# Lesson state
# =========================

VALID_INTENTS = {"continue", "clarify", "switch_topic", "resume_topic", "new"}


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

    # ---- persistence ----

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Optional[dict[str, Any]]) -> "LessonState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            current_topic=data.get("current_topic"),
            current_subtopic=data.get("current_subtopic"),
            covered_concepts=list(data.get("covered_concepts") or []),
            pending_questions=list(data.get("pending_questions") or []),
            current_teaching_goal=data.get("current_teaching_goal"),
            last_intent=data.get("last_intent", "new"),
        )

    # ---- mutation helpers ----

    def mark_concept_covered(self, concept: str) -> None:
        concept = (concept or "").strip()
        if concept and concept not in self.covered_concepts:
            self.covered_concepts.append(concept)
        # Keep this bounded; it's a rolling teaching log, not a full
        # transcript -- the last ~12 concepts are what's actually useful
        # for "avoid repeating what's already been explained".
        self.covered_concepts = self.covered_concepts[-12:]

    def add_pending_question(self, question: str) -> None:
        question = (question or "").strip()
        if question and question not in self.pending_questions:
            self.pending_questions.append(question)
        self.pending_questions = self.pending_questions[-6:]

    def resolve_pending_questions(self, resolved: list[str]) -> None:
        """
        Remove pending questions the planner says this turn's reply
        addressed. `resolved` entries are free-text descriptions from the
        planner call, matched loosely (substring / normalized) against
        stored pending_questions, since exact string equality between two
        separate LLM calls' phrasing is unreliable.
        """
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
        # Deliberately NOT clearing pending_questions on a topic switch --
        # an unresolved question from an earlier topic can still be
        # revisited later (see "resume_topic" intent).

    # ---- prompt rendering ----

    def to_prompt_block(self) -> str:
        """
        Short, teaching-state-shaped rolling summary (~150 words max),
        meant to replace "reconstruct lesson progress from raw message
        history" with an explicit, compact object passed every turn.
        """
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

        block = "\n".join(lines)
        # Hard cap so a long-running session's lesson state can't balloon
        # the system prompt the way the old free-text conversation_summary
        # could.
        return block[:900]


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

    return f"""
        You are the dialogue-planning component of a tutoring robot. You do NOT
        answer the user -- you only decide how their latest message relates to
        the ongoing lesson, before a separate component writes the actual reply.

        Current lesson state:
        - current_topic: {lesson_state.current_topic or "none"}
        - current_subtopic: {lesson_state.current_subtopic or "none"}
        - already_explained: {lesson_state.covered_concepts or "none"}
        - unresolved_questions: {lesson_state.pending_questions or "none"}

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


def classify_dialogue_turn(
    client: Any,
    user_text: str,
    lesson_state: LessonState,
    history: list[dict],
    safe_json_extract,
    model_name: str, 
    print_ts=print,
) -> dict[str, Any]:
    """
    One extra LLM call, run concurrently alongside emotion detection /
    Self-RAG / question-level classification (they're independent of each
    other). `client` is the existing ollama.Client instance; `safe_json_extract`
    and `print_ts` are passed in so this module has no import-time
    dependency on ameca_demo.py (avoids a circular import -- ameca_demo.py
    imports this module, not the reverse).

    Fails open: on any error, returns intent="continue" with no state
    changes, so a planner hiccup degrades gracefully to "just answer the
    question" rather than blocking the turn.
    """
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
            model=model_name,  # caller's MODEL_NAME is injected via functools.partial in ameca_demo.py, see INTEGRATION.md
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

        return {
            "intent": intent,
            "topic": str(data.get("topic", "")).strip() or lesson_state.current_topic,
            "subtopic": str(data.get("subtopic", "")).strip() or None,
            "teaching_goal": str(data.get("teaching_goal", "")).strip() or lesson_state.current_teaching_goal,
            "resolved_questions": [
                str(q).strip() for q in (data.get("resolved_questions") or []) if str(q).strip()
            ],
            "new_pending_questions": [
                str(q).strip() for q in (data.get("new_pending_questions") or []) if str(q).strip()
            ],
        }
    except Exception as exc:
        print_ts(f"[PLANNER] Dialogue planner call failed ({exc}); defaulting to intent='continue'.")
        return fallback


def apply_planner_output(lesson_state: LessonState, planner_output: dict[str, Any]) -> LessonState:
    """
    Mutates `lesson_state` in place according to the planner's decision,
    and also returns it for convenience. Call this BEFORE
    generate_response(), so the updated state (topic/subtopic/teaching
    goal for THIS turn) is what actually goes into the system prompt --
    then call resolve/mark-covered helpers again AFTER the reply is known,
    to fold in what was just answered.
    """
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

    for extra_question in planner_output.get("new_pending_questions", []):
        lesson_state.add_pending_question(extra_question)

    lesson_state.resolve_pending_questions(planner_output.get("resolved_questions", []))

    return lesson_state


def finalize_lesson_state_after_reply(
    lesson_state: LessonState,
    planner_output: dict[str, Any],
) -> LessonState:
    """
    Call once the teaching reply for this turn has actually been
    generated: marks the turn's subtopic (or topic, if no subtopic) as a
    covered concept, so the next turn's "already_explained" list is
    accurate.
    """
    concept = planner_output.get("subtopic") or planner_output.get("topic")
    if concept:
        lesson_state.mark_concept_covered(concept)
    return lesson_state


# =========================
# Dynamic reply length
# =========================
#
# Replaces the single fixed MAX_REPLY_SENTENCES (which directly
# contradicted "be elaborate, teach thoroughly, answer all questions")
# with per-turn guidance driven by the planner's intent plus whether the
# message is a compound/multi-part question.

REPLY_LENGTH_BY_CASE = {
    "clarify": 40,       # simple clarification: short, direct
    "new_topic": 130,    # new concept / new_topic intent: room to actually teach
    "continue": 90,      # continuing the current lesson: moderate
    "multi_part": 220,   # multi-part question: enough room for every sub-part
}


def resolve_reply_word_budget(
    intent: str,
    is_multi_part: bool,
    default_min: int = 25,
) -> int:
    if is_multi_part:
        return REPLY_LENGTH_BY_CASE["multi_part"]
    if intent == "clarify":
        return REPLY_LENGTH_BY_CASE["clarify"]
    if intent in ("new", "switch_topic"):
        return REPLY_LENGTH_BY_CASE["new_topic"]
    return max(default_min, REPLY_LENGTH_BY_CASE["continue"])