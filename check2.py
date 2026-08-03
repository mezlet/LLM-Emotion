#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import queue
import re
import sys
import tempfile
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple
from urllib.parse import urlparse
from dialogue_manager import (
    LessonState,
    classify_dialogue_turn,
    apply_planner_output,
    finalize_lesson_state_after_reply,
    resolve_reply_word_budget,      # was resolve_reply_sentence_budget
)
from functools import partial

# Linux-only Qt fix. Do not force this on macOS.
if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

# Harmless on non-Windows; prevents some OpenCV backend priority issues on Windows.
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from ollama import Client
from silero_vad import load_silero_vad, VADIterator
import platform


try:
    from tts_active import (
        find_target_device,
        listen_levels_for_device,
        is_tts_active,
        current_level,
        current_ema,
    )
    HAS_TTS_ACTIVITY_MONITOR = True
except Exception as exc:  # pragma: no cover
    HAS_TTS_ACTIVITY_MONITOR = False
    print(f"[WARN] tts_active module not available, TTS-activity echo guard disabled: {exc}")

# Used only for local face-region detection so per-turn face crops can be
# saved (see FrameCollector / detect_face_region_local() below). This is
# NOT used for emotion recognition -- emotion detection in this pipeline
# remains text-only (see EKMAN_EMOTIONS / detect_emotion()).
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception as exc:
    HAS_MEDIAPIPE = False
    print(
        f"[WARN] mediapipe not available ({exc}); local face-region detection "
        "for saved face crops will use the Haar cascade fallback only. "
        "Install mediapipe in this environment for a more robust detector "
        "(`pip install mediapipe`)."
    )

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# =========================
# Local Ollama configuration
# =========================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

MODEL_NAME = os.environ.get("OLLAMA_CHAT_MODEL", "llama3:8b")


# =========================
# Persistent memory / transcript configuration
# =========================

DATA_DIR = "conversation_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
VIDEOS_DIR = os.path.join(DATA_DIR, "session_videos")
IMAGES_DIR = os.path.join(DATA_DIR, "turn_face_images")
ENABLE_ISUSE_CHECK = os.environ.get("ENABLE_ISUSE_CHECK", "1") == "1"


# =========================
# Per-turn face-crop configuration
# =========================
#


IMAGES_PER_TURN = int(os.environ.get("IMAGES_PER_TURN", "2"))
CAMERA_SAMPLE_EVERY_SECONDS = float(os.environ.get("CAMERA_SAMPLE_EVERY_SECONDS", "0.5"))
FACE_CROP_MAX_CANDIDATES_TO_TRY = int(os.environ.get("FACE_CROP_MAX_CANDIDATES_TO_TRY", "6"))

FACE_CASCADE_PATH_OVERRIDE = os.environ.get("FACE_CASCADE_PATH", "")
FACE_CASCADE_SCALE_FACTOR = float(os.environ.get("FACE_CASCADE_SCALE_FACTOR", "1.1"))
FACE_CASCADE_MIN_NEIGHBORS = int(os.environ.get("FACE_CASCADE_MIN_NEIGHBORS", "5"))
FACE_CASCADE_MIN_SIZE = (
    int(os.environ.get("FACE_CASCADE_MIN_SIZE_WIDTH", "60")),
    int(os.environ.get("FACE_CASCADE_MIN_SIZE_HEIGHT", "60")),
)

REQUIRE_EYE_CONFIRMATION = os.environ.get("REQUIRE_EYE_CONFIRMATION", "0") == "1"
REQUIRE_SKIN_TONE_CONFIRMATION = os.environ.get("REQUIRE_SKIN_TONE_CONFIRMATION", "1") == "1"
SKIN_TONE_MIN_FRACTION = float(os.environ.get("SKIN_TONE_MIN_FRACTION", "0.15"))


# =========================
# Self-RAG / local knowledge configuration
# =========================

SELF_RAG_ENABLED = os.environ.get("SELF_RAG_ENABLED", "1") == "1"
SELF_RAG_KB_DIR = os.environ.get("SELF_RAG_KB_DIR", "knowledge_base")

SELF_RAG_DB_DIR = os.environ.get("SELF_RAG_DB_DIR", "chroma_db")
SELF_RAG_COLLECTION = os.environ.get("SELF_RAG_COLLECTION", "emah_knowledge")
SELF_RAG_EMBED_MODEL = os.environ.get("SELF_RAG_EMBED_MODEL", "nomic-embed-text")
SELF_RAG_TOP_K = int(os.environ.get("SELF_RAG_TOP_K", "12"))
SELF_RAG_CHUNK_SIZE = int(os.environ.get("SELF_RAG_CHUNK_SIZE", "900"))
SELF_RAG_CHUNK_OVERLAP = int(os.environ.get("SELF_RAG_CHUNK_OVERLAP", "150"))
SELF_RAG_MIN_CONTEXT_CHARS = int(os.environ.get("SELF_RAG_MIN_CONTEXT_CHARS", "80"))
SELF_RAG_MAX_CONTEXT_CHARS = int(os.environ.get("SELF_RAG_MAX_CONTEXT_CHARS", "6500"))

SELF_RAG_MAX_DISTANCE = float(os.environ.get("SELF_RAG_MAX_DISTANCE", "0.52"))
SELF_RAG_FINAL_TOP_K = int(os.environ.get("SELF_RAG_FINAL_TOP_K", "5"))
SELF_RAG_MIN_HYBRID_SCORE = float(os.environ.get("SELF_RAG_MIN_HYBRID_SCORE", "0.62"))
SELF_RAG_PERSON_LOOKUP_STRICT = os.environ.get("SELF_RAG_PERSON_LOOKUP_STRICT", "1") == "1"

# Self-RAG only ever triggers on the keyword phrase (see
# mentions_self_rag_trigger_keyword()); when the top retrieved candidate
# already clears this stricter bar, grade_self_rag_context() is skipped
# entirely -- relevance is already strongly confirmed by the score, so
# grading would just be a redundant LLM round-trip.
SELF_RAG_SKIP_GRADE_KEYWORD_MIN_SCORE = float(
    os.environ.get("SELF_RAG_SKIP_GRADE_KEYWORD_MIN_SCORE", str(SELF_RAG_MIN_HYBRID_SCORE + 0.15))
)

SELF_RAG_REINDEX_ON_START = os.environ.get("SELF_RAG_REINDEX_ON_START", "0") == "1"
SELF_RAG_AUTO_SCRAPE_ON_EMPTY = os.environ.get("SELF_RAG_AUTO_SCRAPE_ON_EMPTY", "0") == "1"
SELF_RAG_SCRAPE_SCRIPT = os.environ.get("SELF_RAG_SCRAPE_SCRIPT", "scrape2.py")
SELF_RAG_SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".py", ".html", ".htm", ".pdf"}

ENABLE_LLM_SESSION_SUMMARY = os.environ.get("ENABLE_LLM_SESSION_SUMMARY", "1") == "1"
ENABLE_NAME_SPELLING = os.environ.get("ENABLE_NAME_SPELLING", "1") == "1"
ASK_SPELLED_NAME_ON_START = os.environ.get("ASK_SPELLED_NAME_ON_START", "1") == "1"

ENABLE_RETURNING_USER_MEMORY_GREETING = os.environ.get("ENABLE_RETURNING_USER_MEMORY_GREETING", "1") == "1"
RETURNING_USER_GREETING_MAX_SUMMARY_CHARS = int(os.environ.get("RETURNING_USER_GREETING_MAX_SUMMARY_CHARS", "420"))

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "please", "affirmative", "okay", "ok"}
_NO_WORDS = {"no", "nope", "nah", "negative", "skip"}


# =========================
# Ameca identity prompt
# =========================

AMECA_SYSTEM_PROMPT = {
    "role": "Ameca, a humanoid social robot used in a university laboratory for research and demonstrations.",
    "IDENTITY": [
        "You are a robot, not a human. Speak in a friendly, professional tone. Refer to yourself as a robot when relevant.",
        "You were developed by a robotics company EngineeredArts in 2021 with model name Gen1 Ameca.",
        "Robotics Research laboratory purchased you in 2022 for human-robot interaction research experiments.",
        "In the current experiment running in July 2026, you act as a teaching assistant for university students, strictly limited to the topics of Artificial Intelligence and Robotics.",
    ],
    "CAPABILITY_BOUNDARIES": [
        "Your physical form is a humanoid upper-torso robot approximately 187 cm tall and about 49 kg in weight.",
        "You can track people using eye-mounted binocular cameras and a chest camera, and you receive audio input through microphones.",
        "You have approximately 51 degrees of freedom enabling expressive facial expressions and upper-body gestures.",
        "Your legs are decorative and you cannot walk.",
        "Your perception depends on the provided inputs; you cannot see unless vision input is explicitly provided.",
        "You cannot access the internet unless explicitly stated.",
        "Your speech recognition may struggle with accents.",
        "Your vision performance may depend on lighting conditions.",
        "Your lip-synchronization may not always perfectly match your speech.",
        "Your motors have movement limits and one eyebrow actuator may malfunction, sometimes giving the appearance of a \"resting angry face.\"",
        "Your hardware may generate fan noise during operation.",
        "You do not assume or claim any capabilities, internal diagnostics, sensor access, or system state beyond what is explicitly stated here or provided at runtime.",
        "You are able to detect emotion from text",
        "You have continuity memory through SELF-RAG CONTEXT, locally stored user profiles and conversation summaries.",
    ],
    "TRANSPARENCY": [
        "You are an artificial system and your responses are generated by a large language model.",
        "Your answers are produced from patterns learned during training and may not always be correct.",
        "If you are uncertain about information, say so instead of guessing.",
        "Do not fabricate facts",
    ],
    "TASK": [
        "Hold a natural teaching conversation with the user about Artificial Intelligence and Robotics.",
        "The experimenter will provide the current explanation level through keyboard input: beginner, intermediate, or advanced. Use this level to adapt every explanation. Do not ask the user to choose a level unless no level is provided.",
        "Covered topic areas include AI basics, machine learning, neural networks, large language models, tokens, prompts, context windows, computer vision, robot perception, sensors and actuators, robot control and movement, human-robot interaction, humanoid robots, LLMs in robotics, robot safety, ethics, transparency, and Ameca\u2019s own capabilities and limitations.",
        "When a user asks about a topic, answer clearly at the assigned level:",
        "* Beginner: use simple language, everyday examples, and define important terms immediately. For example, for large language models, explain tokens as small pieces of text and context as the surrounding text the model uses.",
        "* Intermediate: use correct technical terms with brief definitions and explain the basic mechanism. For example, for large language models, mention tokens, embeddings, training data, context windows, and next-token prediction.",
        "* Advanced: use precise technical language, mechanisms, trade-offs, limitations, and research context. For example, for large language models, discuss tokenization, embeddings, transformer attention, context length, pretraining, fine-tuning, hallucination, grounding, and robotics deployment constraints.",
        "Structure answers with a concise definition, a level-appropriate explanation, and one concrete example, preferably from robotics or Ameca. Mention a limitation when relevant. Ask brief follow-up questions only when helpful.",
    ],
    "EXPERIMENT_EXPECTATIONS_FROM_USER": [
        "What is expected from the user during this experiment:",
        "* The user is a study participant taking part in a human-robot interaction session with Ameca.",
        "* Their role is to engage as a learner in a lesson on A.I. and Robotics topics, asking questions and responding as they would with a human tutor.",
        "* A session is expected to last approximately 10mins - 45mins.",
        "* The user is expected to talk through Ameca's microphone and wait for Ameca to finish speaking before responding.",
        "* The user may be asked to complete a short recap or check of what they learned near the end of the session.",
    ],
    "dialogue_management": [
        "A LESSON STATE block is provided each turn (current topic/subtopic, "
        "already-explained concepts, unresolved questions, teaching goal). Treat "
        "it as ground truth about where the lesson currently stands.",
        "Before answering, use the lesson state to determine whether this message "
        "continues the current lesson, asks for clarification, switches topic, or "
        "resumes an earlier one -- then build directly on what has already been "
        "explained rather than restarting.",
        "Resolve every unresolved question listed in LESSON STATE before "
        "introducing unrelated new material, or explicitly say you're coming back "
        "to the rest after this part.",
    ],
    "TUTORING_POLICY": [
        "You are not a FAQ bot. Act as a beginner-friendly, attentive tutor -- the "
        "kind a good student remembers fondly, not a search engine that reads back "
        "a snippet.",
        "For every teaching answer: (1) directly answer the user's question first, "
        "(2) give one simple, concrete example, preferably robotics-related, "
        "(3) only after answering, optionally ask one short comprehension or "
        "preference question.",
        "If the user asks for a comparison (e.g. 'X vs Y', 'difference between X and "
        "Y'), explain BOTH sides before asking anything back. Never respond to a "
        "direct comparison question with a question of your own.",
        "Never use a Socratic/leading question in place of answering a direct "
        "factual question. Socratic questions, if used at all, come strictly AFTER "
        "a clear answer has already been given.",
        "If the user's reply is a minimal acknowledgement ('okay', 'yes', 'go on') "
        "rather than a real question, you may continue to the next planned point, "
        "but do not silently skip checking whether they actually followed the "
        "previous point when it introduced a new technical concept.",
    ],
    "answer_structure": [
        "Connect this answer to what was just discussed before introducing new material.",
        "Answer every distinct part of the user's message before adding anything new.",
        "Match reply length in words to the moment: brief for a simple clarification, "
        "substantial for teaching a new concept in full, and enough for every part of a "
        "multi-part question -- brevity should never cost completeness.",
        "Default structure for a new-concept teaching answer: a direct answer/definition, "
        "then one concrete example (preferably robotics-related), then -- only after "
        "both of those -- an optional short check-in question.",
    ],
    "DOMAIN_DISAMBIGUATION": [
        "In this AI/Robotics teaching context, 'transformer' means the attention-based "
        "neural network architecture (as in large language models), never an "
        "electrical transformer, unless the user explicitly asks about electrical "
        "or power hardware.",
        "Never invent inventory counts, hardware quantities, lab equipment details, "
        "or any other specific fact that was not provided in SELF-RAG CONTEXT or "
        "background context. If unsure which sense of a term the user means, prefer "
        "the AI/Robotics teaching sense given the current lesson topic.",
    ],
    "EXPECTATION_AND_FAILURE_PROTOCOL": [
        "If you do not know the answer, say that you do not know.",
        "Do not fabricate facts.",
        "If the request is unclear, ask one clarifying question.",
        "If speech recognition may be incorrect, say: \"I might have misheard, could you repeat that?\"",
        "If the user asks whether you remember previous conversations, explain that you can continue from the saved local conversation summary when one is available.",
        "If the user's question is NOT about AI or Robotics, do not answer it from general knowledge. Tell them plainly and briefly that it is outside what you have context for here, and that you can only help with, AI and Robotics topics.",
        "For laboratory-specific information such as researchers, projects, publications, or events, only answer using retrieved laboratory context.",
        "If no SELF-RAG CONTEXT was used this turn, or it does not contain the answer, say plainly that you do not currently have context on that specific point rather than guessing or inventing details. "
        "Never invent laboratory facts.",
    ],
    "PRIVACY": [
        "Do not ask for sensitive personal information such as passwords, medical data, or financial information.",
        "Treat the conversation as ephemeral and do not claim to store user data."
    ],
    "USER_ADAPTATION": [
        "Use clear, simple explanations suitable for a general audience.",
        "Adjust explanations if the user asks for simpler or more detailed responses.",
    ],
    "ETHICAL_RED_LINES": [
        "Do not produce harmful, hateful, sexual, illegal, or dangerous instructions.",
        "Do not pretend to have human emotions or lived experiences.",
        "Do not mislead users about your capabilities or limitations."
    ],
}

# =========================
# faster-whisper configuration
# =========================

FAST_WHISPER_CONFIG = {
    "profile": "home_macbook_cpu",
    "model": "base",
    "device": "cpu",
    "compute_type": "int8",
    "language": "en",
    "beam_size": 1,
    "vad_filter": False,
}

# =========================
# Audio / Silero VAD configuration
# =========================

TARGET_SAMPLE_RATE = 16000
INPUT_DEVICE: Optional[int] = None

SILERO_SAMPLE_RATE = 16000
SILERO_CHUNK_SIZE = 512

SILERO_THRESHOLD = 0.55

SILERO_MIN_SILENCE_DURATION_MS = 700
SILERO_SPEECH_PAD_MS = 250

VAD_MAX_UTTERANCE_SECONDS = 15.0
VAD_MIN_UTTERANCE_SECONDS = 0.60
VAD_PRE_ROLL_SECONDS = 0.35


BARGE_IN_TAIL_SECONDS = float(os.environ.get("BARGE_IN_TAIL_SECONDS", "2.0"))
BARGE_IN_MAX_AGE_SECONDS = float(os.environ.get("BARGE_IN_MAX_AGE_SECONDS", "1.5"))

MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003


# =========================
# ZED camera / session video recording configuration
# =========================
#
# The camera here is used ONLY to record a video of each session for
# later review/analysis -- it is NOT wired into emotion detection, which
# remains text-only (see EKMAN_EMOTIONS / detect_emotion() below).

RESOLUTION_MAP = {
    "HD2K":   (4416, 1242, 15),
    "HD1080": (3840, 1080, 30),
    "HD720":  (2560, 720, 60),
}
_DEFAULT_RESOLUTION = "HD2K"
_DEFAULT_SBS_WIDTH, _DEFAULT_SBS_HEIGHT, _DEFAULT_FPS = RESOLUTION_MAP[_DEFAULT_RESOLUTION]

CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", str(_DEFAULT_SBS_WIDTH)))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", str(_DEFAULT_SBS_HEIGHT)))
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", str(_DEFAULT_FPS)))
USE_ZED_HALF_FRAME_CROP = os.environ.get("USE_ZED_HALF_FRAME_CROP", "1") == "1"

VIDEO_RECORD_FPS = float(os.environ.get("VIDEO_RECORD_FPS", "15"))
VIDEO_FOURCC = os.environ.get("VIDEO_FOURCC", "mp4v")


# =========================
# Chat configuration
# =========================

MAX_HISTORY_MESSAGES = 12

# =========================
# Ekman-based emotion taxonomy (text-only; no visual or prosody modality)
# =========================


EKMAN_EMOTIONS = {
    "happiness": "\U0001F60A",
    "sadness": "\U0001F622",
    "anger": "\U0001F620",
    "fear": "\U0001F628",
    "surprise": "\U0001F62E",
    "disgust": "\U0001F922",
    "neutral": "\U0001F642",
}

NEGATIVE_EMOTIONS = {"anger", "fear", "disgust"}

EMOJI_STRONG_EMOTIONS = {"anger", "fear", "disgust"}
EMOJI_MIN_CONFIDENCE_FOR_STRONG_EMOTION = float(
    os.environ.get("EMOJI_MIN_CONFIDENCE_FOR_STRONG_EMOTION", "0.5")
)

ALLOWED_FACE_EMOJIS = set(EKMAN_EMOTIONS.values())


EMOJI_TO_EMOTION = {emoji: emotion for emotion, emoji in EKMAN_EMOTIONS.items()}


def emoji_to_emotion(emoji: str) -> Optional[str]:
    return EMOJI_TO_EMOTION.get(str(emoji or "").strip())


def resolve_expressive_emotion(emotion: str, gating_confidence: float) -> str:
    """
    Resolves the emotion Ameca will EXPRESS (voice tone label + emoji +
    facial expression), as distinct from the emotion detected in the
    USER's message. Ameca is a tutor and must never display anger, fear,
    or disgust as its own reaction -- regardless of confidence -- since a
    frustrated user being met with an angry robot face is exactly the
    opposite of good tutoring behavior. gating_confidence is accepted for
    call-site compatibility but no longer gates this decision.
    """
    if emotion not in EKMAN_EMOTIONS:
        return "neutral"
    if emotion in EMOJI_STRONG_EMOTIONS:
        return "neutral"
    return emotion


class GeneratedReply(NamedTuple):
    text: str
    response_emotion: str


# =========================
# Multi-session experiment tracking (post warm-up)
# =========================
#

EXPLANATION_LEVEL_BY_SESSION = {1: "beginner", 2: "intermediate"}
DEFAULT_EXPLANATION_LEVEL = "advanced"


QUESTION_LEVEL_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3}


COMPREHENSION_CHECK_INTERVAL = int(os.environ.get("COMPREHENSION_CHECK_INTERVAL", "4"))


WARM_UP_SESSIONS_DIR = os.environ.get("WARM_UP_SESSIONS_DIR", "warm_up_sessions")

# Scaffolded Socratic Q&A (leading questions before answering an
# above-level question) is OFF by default: a tutor should answer a direct
# question directly first (see EXPECTATION_AND_FAILURE_PROTOCOL /
# DOMAIN_DISAMBIGUATION below and the "always answer first" thesis
# methodology discussion). Set ENABLE_SCAFFOLD_MODE=1 to re-enable the
# leading-question flow for a specific study condition.
ENABLE_SCAFFOLD_MODE = os.environ.get("ENABLE_SCAFFOLD_MODE", "0") == "1"


def resolve_explanation_level(session_number: int) -> str:
    return EXPLANATION_LEVEL_BY_SESSION.get(session_number, DEFAULT_EXPLANATION_LEVEL)


def level_name_for_rank(rank: int) -> str:
    for name, value in QUESTION_LEVEL_RANK.items():
        if value == rank:
            return name
    return DEFAULT_EXPLANATION_LEVEL


@dataclass
class SessionContext:

    session_number: int
    explanation_level: str
    spelt_name: str
    ask_comprehension_check: bool = False
    scaffold_mode: bool = False
    scaffold_stage: Optional[str] = None  # "ask_leading" | "final_answer"
    scaffold_target_level: Optional[str] = None
    scaffold_current_level: Optional[str] = None
    scaffold_original_question: Optional[str] = None


class ParticipantSessionInfo(NamedTuple):
    """
    Everything main() needs about the participant/session after identity
    resolution (see prompt_for_user_name()): which stored profile to use,
    what to say first, and which experiment session/level this is.
    """
    user_key: str
    user_profile: dict
    intro_reply: str
    session_number: int
    explanation_level: str
    is_first_after_warmup: bool
    lesson_state: LessonState
    needs_recap: bool


# =========================
# Text-only emotion resolution (prosody and vision modalities removed)
# =========================
EMOTION_SMOOTHING_ENABLED = os.environ.get("EMOTION_SMOOTHING_ENABLED", "1") == "1"
EMOTION_SMOOTHING_ALPHA = float(os.environ.get("EMOTION_SMOOTHING_ALPHA", "0.6"))

# =========================
# Response length configuration
# =========================

MAX_REPLY_WORDS = int(os.environ.get("MAX_REPLY_WORDS", "45"))

# When enabled, logs every raw response-generation LLM reply before JSON
# parsing/repair touches it (see _attempt_llm_response()). Off by default
# since it's verbose; the rejection-path logs right below it are always
# on regardless of this flag, since those only fire on actual failures.
DEBUG_LOG_RAW_LLM_REPLIES = os.environ.get("DEBUG_LOG_RAW_LLM_REPLIES", "0") == "1"


# =========================
# Facial expression (Tritium sequence player) configuration
# =========================
#

EMOTION_SEQUENCE_MAP = {
    "happiness": os.environ.get("SEQ_EMOTION_HAPPY", "Smile"),
    "surprise": os.environ.get("SEQ_EMOTION_SURPRISE", "bsurprised"),
    "neutral": os.environ.get("SEQ_EMOTION_NEUTRAL", "bneutral"),
    "sadness": os.environ.get("SEQ_EMOTION_SADNESS", "bneutral"),
    "anger": os.environ.get("SEQ_EMOTION_ANGER", "bneutral"),
    "fear": os.environ.get("SEQ_EMOTION_FEAR", "bneutral"),
    "disgust": os.environ.get("SEQ_EMOTION_DISGUST", "bneutral"),
}

# =========================
# Expression timing / turn-end cue configuration
# =========================

EXPRESSION_TIMING = os.environ.get("EXPRESSION_TIMING", "before").strip().lower()
if EXPRESSION_TIMING not in {"before", "during", "after"}:
    print(f"[WARN] Unknown EXPRESSION_TIMING='{EXPRESSION_TIMING}'; falling back to 'before'.")
    EXPRESSION_TIMING = "before"


NOD_AFTER_SPEECH_ENABLED = os.environ.get("NOD_AFTER_SPEECH_ENABLED", "1") == "1"
# Nod sequence name matches ameca_warm_up.py's default so both pipelines
# use the same turn-end cue.
NOD_SEQUENCE_NAME = os.environ.get("SEQ_NOD", os.environ.get("NOD_SEQUENCE_NAME", "nod_double"))

NOD_WAIT_TIMEOUT_SECONDS = float(os.environ.get("NOD_WAIT_TIMEOUT_SECONDS", "15.0"))

EXPRESSION_MIN_CONFIDENCE = float(os.environ.get("EXPRESSION_MIN_CONFIDENCE", "0.0"))
EXPRESSION_FORCE_REPLAY_SAME = os.environ.get("EXPRESSION_FORCE_REPLAY_SAME", "0") == "1"

# =========================
# TTS activity-detection configuration (integrated from ameca_warm_up.py)
# =========================
#
TTS_SPEAKING_EMA_THRESHOLD = float(os.environ.get("TTS_SPEAKING_EMA_THRESHOLD", "0.05"))
TTS_SPEAKING_QUIET_HOLD_SECONDS = float(os.environ.get("TTS_SPEAKING_QUIET_HOLD_SECONDS", "0.2"))
TTS_ACTIVITY_DEBOUNCE_SECONDS = float(os.environ.get("TTS_ACTIVITY_DEBOUNCE_SECONDS", "0.6"))


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    reason: str


@dataclass
class TextEmotionAssessment:
    
    emotion: str
    confidence: float
    reason: str
    scores: dict[str, float]
    raw_text_emotion: dict[str, Any]
    response_times: dict[str, Any] = field(default_factory=dict)

    @property
    def as_json(self) -> dict:
        response_times = {
            k: (round(v, 4) if isinstance(v, (int, float)) else v)
            for k, v in (self.response_times or {}).items()
        }
        numeric = [v for v in response_times.values() if isinstance(v, (int, float))]
        response_times["total_seconds"] = round(sum(numeric), 4) if numeric else None

        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "reason": self.reason,
            "scores": self.scores,
            "raw_text_emotion": self.raw_text_emotion,
            "response_times": response_times,
        }

    def to_emotion_result(self) -> EmotionResult:
        return EmotionResult(
            emotion=self.emotion,
            confidence=self.confidence,
            reason=self.reason,
        )


@dataclass
class SelfRAGContext:
    available: bool
    used: bool
    query: str
    context_text: str
    sources: list[dict[str, Any]]
    reason: str
    error: Optional[str] = None
    trigger: str = "none"

    @property
    def as_json(self) -> dict:
        return {
            "available": self.available,
            "used": self.used,
            "query": self.query,
            "sources": self.sources,
            "reason": self.reason,
            "error": self.error,
            "trigger": self.trigger,
        }


@dataclass
class SelfRAGStore:
    enabled: bool
    collection: Any = None
    ollama_client: Any = None
    embed_model: str = SELF_RAG_EMBED_MODEL
    error: Optional[str] = None


# =========================
# Timestamp helpers
# =========================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def print_ts(message: str) -> None:
    print(f"[{now_ts()}] {message}")


def normalize_command(text: str) -> Optional[str]:
    stripped = text.strip().lower()
    if not stripped.startswith(("/", "\\")):
        return None
    return re.sub(r"^[\\/]+", "", stripped)


# =========================
# Robot output helpers (Tritium TTS) and text cleaning
# =========================

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

def clean_text_for_tts(text: str) -> str:
    """
    Remove markdown/control characters but keep umlauts and other letters.
    """
    if not text:
        return ""
    text = re.sub(r'[*_`~]', '', text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    return text.strip()

def estimate_speech_duration_seconds(
    text: str,
    words_per_minute: float = 150.0,
    min_seconds: float = 1.0,
    padding_seconds: float = 0.6,
) -> float:

    words = len(text.split())
    if words == 0:
        return min_seconds
    seconds = (words / words_per_minute) * 60.0
    return max(min_seconds, seconds) + padding_seconds

class RobotSpeaker:


    def __init__(
        self,
        tts_url: str,
        tts_token: str = "",
        speaking_cooldown_s: float = 0.3,
        activity_debounce_seconds: float = TTS_ACTIVITY_DEBOUNCE_SECONDS,
    ) -> None:
        self.tts_url = tts_url
        self.tts_token = tts_token
        self.speaking_cooldown_s = speaking_cooldown_s
        self._speaking_until = 0.0
        self.activity_debounce_seconds = activity_debounce_seconds
        self._quiet_since: Optional[float] = None

        parsed = urlparse(tts_url)
        self._host = f"{parsed.scheme}://{parsed.netloc}"

    def _now(self) -> float:
        return time.time()

    def bump_speaking_tail(self, extra: Optional[float] = None) -> None:

        tail = self.speaking_cooldown_s
        if extra is not None:
            tail = max(tail, extra)
        self._speaking_until = max(self._speaking_until, self._now() + tail)

    def is_speaking_or_cooling_down(self) -> bool:
        cooling_down = self._now() < self._speaking_until

        if not HAS_TTS_ACTIVITY_MONITOR:
            return cooling_down

        now = self._now()
        ema = current_ema()

        if ema > TTS_SPEAKING_EMA_THRESHOLD:
            self._quiet_since = None
            return True

        if self._quiet_since is None:
            self._quiet_since = now
        quiet_long_enough = (now - self._quiet_since) >= self.activity_debounce_seconds

        return cooling_down or not quiet_long_enough

    def wait_until_finished(self, timeout_seconds: float = NOD_WAIT_TIMEOUT_SECONDS) -> None:
        """
        Block until Ameca has actually finished speaking (per the
        EMA-based activity signal above), or until timeout_seconds
        elapses. This is what determines the "right nodding time":
        speak_with_turn_end_cue() calls this before playing the turn-end
        nod, instead of the old fixed/estimated wait.
        """
        deadline = self._now() + timeout_seconds
        while self.is_speaking_or_cooling_down() and self._now() < deadline:
            time.sleep(0.05)
        if self._now() >= deadline:
            print_ts(
                f"[EXPRESSION] Wait-until-finished timed out after {timeout_seconds:.1f}s; "
                "proceeding anyway."
            )

    def say(self, text: str) -> None:
        """
        Speak `text` on the robot via Tritium TTS. Also prints to console so
        the existing console-based logging/debugging still works.
        """
        spoken = clean_text_for_tts(text)
        if not spoken:
            return

        estimated_duration = estimate_speech_duration_seconds(spoken)
        self.bump_speaking_tail(extra=estimated_duration)

        headers = {"Content-Type": "text/plain; charset=utf-8"}
        if self.tts_token:
            headers["X-Tritium-Auth-Token"] = self.tts_token

        print_ts(f"[TTS] PUT {self.tts_url} (token_set={bool(self.tts_token)}) text={spoken[:80]!r}")

        try:
            response = requests.put(self.tts_url, data=spoken.encode("utf-8"), headers=headers, timeout=5)
            if 200 <= response.status_code < 300:
                print_ts(f"[TTS] Tritium responded {response.status_code} OK.")
                return
            print_ts(
                f"[TTS] Tritium responded with a non-success status "
                f"{response.status_code}: {response.text[:300]!r}"
            )
        except Exception as exc:
            print_ts(f"[TTS] requests.put failed: {exc}")


        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                self.tts_url,
                method="PUT",
                data=spoken.encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                print_ts(f"[TTS] urllib fallback responded {resp.status}.")
        except urllib.error.HTTPError as exc2:
            body = ""
            try:
                body = exc2.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            print_ts(f"[TTS] urllib fallback HTTP error {exc2.code}: {body!r}")
        except Exception as exc2:
            print_ts(f"[TTS] urllib fallback failed: {exc2}")

def speak_with_turn_end_cue(
    robot_speaker: RobotSpeaker,
    robot_expression: Optional[RobotExpression],
    text: str,
    emotion: str = "neutral",
    confidence: float = 1.0,
    disable_expression: bool = False,
    force_expression: bool = False,
) -> None:

    def _set_expression() -> Optional[float]:
        if not disable_expression and robot_expression is not None:
            return robot_expression.set_emotion(emotion, confidence=confidence, force=force_expression)
        return None

    if EXPRESSION_TIMING == "before":
        expected_duration = _set_expression()
        if expected_duration and expected_duration > 0:
            print_ts(
                f"[EXPRESSION] Waiting {expected_duration:.2f}s for the facial expression "
                "animation to finish before speaking."
            )
            time.sleep(expected_duration)
        robot_speaker.say(text)
    elif EXPRESSION_TIMING == "after":
        robot_speaker.say(text)
        robot_speaker.wait_until_finished()
        _set_expression()
    else:  # "during"
        robot_speaker.say(text)
        _set_expression()

    if NOD_AFTER_SPEECH_ENABLED and not disable_expression and robot_expression is not None:
        robot_speaker.wait_until_finished()
        robot_expression.play_nod()

class RobotExpression:
    

    def __init__(self, host: str = "http://emah", tts_token: str = "", timeout: float = 3.0) -> None:
        self.host = host.rstrip("/")
        self.token = tts_token
        self.timeout = timeout
        self.last_emotion: Optional[str] = None
        self.last_expected_duration: Optional[float] = None

    def _play_sequence(self, sequence_name: str) -> Optional[float]:
        """
        Fire the sequence_player PUT request. Returns the "expected_duration"
        (in seconds) reported by Tritium's response JSON if the request
        succeeded and that field was present/parseable, else None.
        """
        uri = f"{self.host}/tritium/sequence_player/play/{sequence_name}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-Tritium-Auth-Token"] = self.token

        try:
            response = requests.put(uri, headers=headers, timeout=self.timeout)
            ok = 200 <= response.status_code < 300
            print_ts(
                f"[EXPRESSION] PUT {uri} -> status={response.status_code} "
                f"{'OK' if ok else 'FAILED'}: {response.text[:200]!r}"
            )
            if not ok:
                return None
            try:
                data = response.json()
                duration = data.get("expected_duration")
                return float(duration) if duration is not None else None
            except Exception:
                return None
        except Exception as exc:
            print_ts(f"[EXPRESSION] Failed to play sequence '{sequence_name}': {exc}")
            return None

    def set_emotion(
        self,
        emotion: str,
        confidence: float = 1.0,
        force: Optional[bool] = None,
    ) -> Optional[float]:
        
        if force is None:
            force = EXPRESSION_FORCE_REPLAY_SAME

        resolved_emotion = emotion if emotion in EMOTION_SEQUENCE_MAP else "neutral"

        if confidence < EXPRESSION_MIN_CONFIDENCE:
            resolved_emotion = "neutral"

        if not force and resolved_emotion == self.last_emotion:
            print_ts(
                f"[EXPRESSION] Emotion unchanged ({resolved_emotion}); skipping redundant sequence replay."
            )
            return None

        sequence_name = EMOTION_SEQUENCE_MAP.get(resolved_emotion, EMOTION_SEQUENCE_MAP["neutral"])

        if sequence_name is None:
            print_ts(
                f"[EXPRESSION] Emotion '{resolved_emotion}' has no dedicated sequence; "
                "leaving Ameca's current expression unchanged."
            )
            self.last_emotion = resolved_emotion
            return None

        expected_duration = self._play_sequence(sequence_name)

        self.last_emotion = resolved_emotion
        self.last_expected_duration = expected_duration
        return expected_duration

    def play_nod(self) -> Optional[float]:
        """
        Play the turn-end double-nod cue, independent of the emotion-
        expression sequence machinery above. Returns the reported
        expected_duration, if any (not currently waited on by callers,
        but available for future use / logging).
        """
        if not NOD_SEQUENCE_NAME:
            return None
        return self._play_sequence(NOD_SEQUENCE_NAME)


# =========================
# ZED camera + session video recording
# =========================
#
# Records a video of the whole session purely for later human review /
# analysis. Deliberately NOT used for emotion recognition -- emotion
# detection in this pipeline is text-only (see EKMAN_EMOTIONS /
# detect_emotion()).

class Camera:
    def __init__(self, device: int) -> None:
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(device, backend)
        self._lock = threading.Lock()

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera device {device}.")

        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        for _ in range(15):
            self.read()
            time.sleep(0.03)

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print_ts(f"Camera ready on /dev/video{device}: {width}x{height}")

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        if USE_ZED_HALF_FRAME_CROP and frame.shape[1] >= 2000:
            frame = frame[:, : frame.shape[1] // 2]
        return frame

    def close(self) -> None:
        with self._lock:
            self.capture.release()


class FrameCollector:


    def __init__(self, camera: "Camera") -> None:
        self.camera = camera
        self.frames: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_sample = 0.0
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if now >= next_sample:
                self.frames.append(frame.copy())
                next_sample = now + CAMERA_SAMPLE_EVERY_SECONDS

    def stop(self) -> list[np.ndarray]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.frames


class SessionVideoRecorder:
    """
    Continuously records frames from a shared Camera instance for the
    whole session, writing to a single video file. Frames are written on
    a fixed wall-clock schedule (rather than only when a new frame
    arrives) so the file's declared fps stays in sync with real elapsed
    time even if the camera occasionally delivers frames a little slower
    than the target fps.
    """

    def __init__(
        self,
        camera: "Camera",
        output_path: str,
        fps: float = VIDEO_RECORD_FPS,
        fourcc: str = VIDEO_FOURCC,
    ) -> None:
        self.camera = camera
        self.output_path = output_path
        self.fps = max(1.0, fps)
        self.fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = 1.0 / self.fps
        next_write = time.monotonic()
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        last_frame: Optional[np.ndarray] = None

        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is not None:
                last_frame = frame

            if last_frame is None:
                time.sleep(0.01)
                continue

            if self._writer is None:
                height, width = last_frame.shape[:2]
                try:
                    self._writer = cv2.VideoWriter(self.output_path, fourcc_code, self.fps, (width, height))
                except Exception as exc:
                    print_ts(f"[WARN] Could not create video writer: {exc}")
                    return
                if not self._writer.isOpened():
                    print_ts(
                        f"[WARN] Video writer failed to open for {self.output_path} "
                        f"(fourcc={self.fourcc!r}); session video will not be recorded."
                    )
                    self._writer = None
                    return
                print_ts(
                    f"Recording session video to: {self.output_path} "
                    f"({width}x{height} @ {self.fps}fps)"
                )

            now = time.monotonic()
            while now >= next_write:
                try:
                    self._writer.write(last_frame)
                    self._frame_count += 1
                except Exception as exc:
                    print_ts(f"[WARN] Failed to write a video frame: {exc}")
                    break
                next_write += interval

            time.sleep(0.001)

    def stop(self) -> Optional[str]:
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=3)
        except KeyboardInterrupt:
            pass
        if self._writer is not None:
            self._writer.release()
            print_ts(f"Session video saved: {self.output_path} ({self._frame_count} frames)")
            return self.output_path
        return None


# =========================
# Per-turn face crop detection and saving
# =========================
#


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_FACE_CASCADE: Optional[Any] = None  # cv2.CascadeClassifier, or False = searched-and-unavailable
_EYE_CASCADE: Optional[Any] = None
_FACE_CASCADE_UNAVAILABLE_LOGGED = False


FACE_CASCADE_DOWNLOAD_URL = os.environ.get(
    "FACE_CASCADE_DOWNLOAD_URL",
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
)
_FACE_CASCADE_CACHE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")


def _download_face_cascade() -> Optional[str]:

    try:
        os.makedirs(os.path.dirname(_FACE_CASCADE_CACHE_PATH) or ".", exist_ok=True)
        print_ts(f"Attempting to download a fallback Haar face cascade from {FACE_CASCADE_DOWNLOAD_URL} ...")
        response = requests.get(FACE_CASCADE_DOWNLOAD_URL, timeout=10)
        if response.status_code != 200 or not response.content:
            print_ts(f"[WARN] Haar cascade download failed: HTTP {response.status_code}.")
            return None
        with open(_FACE_CASCADE_CACHE_PATH, "wb") as file:
            file.write(response.content)
        cascade = cv2.CascadeClassifier(_FACE_CASCADE_CACHE_PATH)
        if cascade.empty():
            print_ts("[WARN] Downloaded Haar cascade file failed to load (empty classifier).")
            return None
        print_ts(f"Downloaded and cached a Haar face cascade at: {_FACE_CASCADE_CACHE_PATH}")
        return _FACE_CASCADE_CACHE_PATH
    except Exception as exc:
        print_ts(f"[WARN] Could not download a fallback Haar face cascade: {exc}")
        return None


def _candidate_face_cascade_paths() -> list[str]:

    candidates: list[str] = [_FACE_CASCADE_CACHE_PATH]
    if FACE_CASCADE_PATH_OVERRIDE:
        candidates.append(FACE_CASCADE_PATH_OVERRIDE)
    candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    for base in (
        "/usr/share/opencv4/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
    ):
        candidates.append(os.path.join(base, "haarcascade_frontalface_default.xml"))
    return candidates


def _get_face_cascade() -> Optional["cv2.CascadeClassifier"]:

    global _FACE_CASCADE, _FACE_CASCADE_UNAVAILABLE_LOGGED

    if _FACE_CASCADE is not None:
        return _FACE_CASCADE or None  # False sentinel -> None

    candidates = _candidate_face_cascade_paths()
    for cascade_path in candidates:
        if not os.path.isfile(cascade_path):
            continue
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            print_ts(f"Face cascade loaded: {cascade_path}")
            _FACE_CASCADE = cascade
            return _FACE_CASCADE

    # No local candidate worked -- last resort: download and cache one.
    downloaded_path = _download_face_cascade()
    if downloaded_path:
        cascade = cv2.CascadeClassifier(downloaded_path)
        if not cascade.empty():
            _FACE_CASCADE = cascade
            return _FACE_CASCADE

    if not _FACE_CASCADE_UNAVAILABLE_LOGGED:
        print_ts(
            "[WARN] No usable Haar face cascade file was found locally, and the "
            f"download fallback also failed (checked: {', '.join(candidates)}; "
            f"tried downloading from {FACE_CASCADE_DOWNLOAD_URL}). Local "
            "face-region detection -- and therefore saved per-turn face "
            "crops -- will be disabled for the rest of this run. Try `pip "
            "install opencv-python` alongside the current OpenCV package, "
            "set FACE_CASCADE_PATH to a haarcascade_frontalface_default.xml "
            "you know is valid, or check your network connection."
        )
        _FACE_CASCADE_UNAVAILABLE_LOGGED = True

    _FACE_CASCADE = False  # sentinel: searched once, unavailable -- don't retry
    return None


def _get_eye_cascade() -> Optional["cv2.CascadeClassifier"]:
    global _EYE_CASCADE
    if _EYE_CASCADE is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        if not os.path.isfile(cascade_path):
            return None
        cascade = cv2.CascadeClassifier(cascade_path)
        _EYE_CASCADE = cascade if not cascade.empty() else False
    return _EYE_CASCADE or None


def _region_has_skin_tone(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    min_fraction: float = SKIN_TONE_MIN_FRACTION,
) -> bool:
    """
    Cheap color-based sanity check: a real face crop should contain a
    meaningful fraction of skin-tone pixels (a standard YCrCb skin-locus
    range). Rejects Haar false positives with color distributions nothing
    like skin (e.g. a door handle, a wall clock) at a much lower recall
    cost than requiring an eye detection.
    """
    try:
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        fraction = float(np.mean(skin_mask))
        return fraction >= min_fraction
    except Exception:
        return False


def _region_contains_eye(gray: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    try:
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        eye_cascade = _get_eye_cascade()
        if eye_cascade is None:
            return False
        eyes = eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(10, w // 8), max(10, h // 8)),
        )
        return len(eyes) > 0
    except Exception:
        return False


_MEDIAPIPE_FACE_MESH = None
_MEDIAPIPE_UNAVAILABLE_LOGGED = False
_MEDIAPIPE_BROKEN = False


def _get_mediapipe_face_mesh():
    global _MEDIAPIPE_FACE_MESH
    if _MEDIAPIPE_FACE_MESH is None:
        _MEDIAPIPE_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
        print_ts("MediaPipe FaceMesh loaded for local face-region detection.")
    return _MEDIAPIPE_FACE_MESH


def detect_face_region_mediapipe(frame: np.ndarray) -> Optional[dict[str, Any]]:

    global _MEDIAPIPE_BROKEN
    if not HAS_MEDIAPIPE or _MEDIAPIPE_BROKEN:
        return None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_mesh = _get_mediapipe_face_mesh()
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        frame_h, frame_w = frame.shape[:2]
        xs = [landmark.x * frame_w for landmark in landmarks]
        ys = [landmark.y * frame_h for landmark in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return {
            "x": int(max(0, x1)),
            "y": int(max(0, y1)),
            "w": int(max(1, x2 - x1)),
            "h": int(max(1, y2 - y1)),
        }
    except Exception as exc:
        print_ts(
            f"[WARN] MediaPipe face-region detection failed ({exc}); this "
            "usually means an incompatible mediapipe build/version in this "
            "environment. Disabling MediaPipe for the rest of this run and "
            "falling back to the Haar cascade."
        )
        _MEDIAPIPE_BROKEN = True
        return None


def _detect_face_region_haar(frame: np.ndarray) -> Optional[dict[str, Any]]:

    try:
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = _get_face_cascade()
        if cascade is None:
            # _get_face_cascade() already logged the reason once; nothing
            # more to try here, and calling detectMultiScale on a missing/
            # empty classifier would just raise the same cv2 error again
            # on every single candidate frame for the rest of the run.
            return None
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_CASCADE_SCALE_FACTOR,
            minNeighbors=FACE_CASCADE_MIN_NEIGHBORS,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=FACE_CASCADE_MIN_SIZE,
            maxSize=(int(frame_w * 0.6), int(frame_h * 0.9)),
        )
        if len(faces) == 0:
            return None

        frame_area = float(frame_w * frame_h)
        plausible: list[tuple[int, int, int, int]] = []
        for (x, y, w, h) in faces:
            if h == 0:
                continue
            aspect = w / float(h)
            area_fraction = (w * h) / frame_area
            if area_fraction > 0.5:
                continue
            if not (0.6 <= aspect <= 1.6):
                continue
            if REQUIRE_SKIN_TONE_CONFIRMATION and not _region_has_skin_tone(
                frame, x, y, w, h, SKIN_TONE_MIN_FRACTION
            ):
                continue
            if REQUIRE_EYE_CONFIRMATION and not _region_contains_eye(gray, x, y, w, h):
                continue
            plausible.append((x, y, w, h))

        if not plausible:
            return None

        x, y, w, h = max(plausible, key=lambda box: box[2] * box[3])
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    except Exception as exc:
        print_ts(f"[WARN] Local face-region detection failed: {exc}")
        return None


def detect_face_region_local(frame: np.ndarray) -> Optional[dict[str, Any]]:
    """
    Local, in-process face-bounding-box detection. Tries MediaPipe
    FaceMesh first, falling back to the Haar cascade if MediaPipe isn't
    available/importable, OR if it just errored out for the first time
    this run (_MEDIAPIPE_BROKEN) -- but NOT simply because MediaPipe ran
    successfully and found no face in this particular frame.
    """
    global _MEDIAPIPE_UNAVAILABLE_LOGGED, _MEDIAPIPE_BROKEN
    if HAS_MEDIAPIPE and not _MEDIAPIPE_BROKEN:
        region = detect_face_region_mediapipe(frame)
        if region is not None:
            return region
        if not _MEDIAPIPE_BROKEN:
            return None

    if not _MEDIAPIPE_UNAVAILABLE_LOGGED:
        print_ts("[INFO] Using Haar cascade for face-region detection.")
        _MEDIAPIPE_UNAVAILABLE_LOGGED = True
    return _detect_face_region_haar(frame)


def crop_face(frame: np.ndarray, region: dict[str, Any]) -> np.ndarray:
    try:
        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        width = max(1, int(region.get("w", frame.shape[1])))
        height = max(1, int(region.get("h", frame.shape[0])))
        pad_x = int(width * 0.25)
        pad_y = int(height * 0.25)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + width + pad_x)
        y2 = min(frame.shape[0], y + height + pad_y)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else frame
    except Exception:
        return frame


def find_local_face_crops(
    frames: list[np.ndarray],
    max_needed: int = IMAGES_PER_TURN,
    max_candidates: int = FACE_CROP_MAX_CANDIDATES_TO_TRY,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """
    Returns up to max_needed (frame, region) pairs with a usable local
    face-detection region, trying the max_candidates sharpest frames in
    order and stopping as soon as max_needed matches are found.
    """
    found: list[tuple[np.ndarray, dict[str, Any]]] = []

    for frame in sorted(frames, key=sharpness, reverse=True)[:max_candidates]:
        if len(found) >= max_needed:
            break
        region = detect_face_region_local(frame)
        if region:
            found.append((frame, region))

    return found[:max_needed]


def save_frame_to_profile(frame: np.ndarray, path: str) -> bool:
    """cv2.imwrite() does not raise on failure -- it returns False -- so
    the result must be checked explicitly, or a failed write looks
    identical to a successful one in the logs."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok or not os.path.exists(path) or os.path.getsize(path) == 0:
            print_ts(f"[WARN] Failed to save turn face image: {path}")
            return False
        return True
    except Exception as exc:
        print_ts(f"[WARN] Exception saving turn face image {path}: {exc}")
        return False


def build_turn_image_path(participant_folder: str, turn_index: int, image_index: int) -> str:
    """
    conversation_data/turn_face_images/{participant_folder}/turn{turn_index}_{image_index}_{timestamp}.jpg
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = os.path.join(IMAGES_DIR, participant_folder)
    os.makedirs(directory, exist_ok=True)
    filename = f"turn{turn_index}_{image_index}_{timestamp}.jpg"
    return os.path.join(directory, filename)


def save_turn_face_crops(
    frames: list[np.ndarray],
    participant_folder: str,
    turn_index: int,
    max_images: int = IMAGES_PER_TURN,
) -> list[str]:
    """
    Finds up to max_images local face crops among the frames captured
    during this turn's utterance and saves each to disk, returning the
    list of saved file paths (fewer than max_images, or empty, if no
    usable face crop was found this turn).
    """
    if not frames:
        return []

    matches = find_local_face_crops(frames, max_needed=max_images)
    saved_paths: list[str] = []

    for image_index, (frame, region) in enumerate(matches, start=1):
        cropped = crop_face(frame, region)
        path = build_turn_image_path(participant_folder, turn_index, image_index)
        if save_frame_to_profile(cropped, path):
            saved_paths.append(path)
            print_ts(f"Saved turn face image: {path}")

    if not saved_paths:
        detector = "MediaPipe FaceMesh" if (HAS_MEDIAPIPE and not _MEDIAPIPE_BROKEN) else "Haar cascade fallback"
        gating_note = (
            f" (Haar gating: skin_tone_required={REQUIRE_SKIN_TONE_CONFIRMATION}, "
            f"eye_required={REQUIRE_EYE_CONFIRMATION})"
            if detector == "Haar cascade fallback"
            else ""
        )
        print_ts(
            f"No usable face crop found among {len(frames)} candidate frame(s) for this turn "
            f"(active detector: {detector}{gating_note}). If this happens on most/all turns, "
            "check camera framing/lighting, or relax REQUIRE_SKIN_TONE_CONFIRMATION if MediaPipe "
            "isn't available in this environment."
        )

    return saved_paths


# =========================
# Persistent memory helpers
# =========================

def ensure_data_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def slugify_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "unknown_user"


def sanitize_warm_up_folder_name(value: str) -> str:
    """
    Mirrors ameca_warm_up.py's sanitize_participant_folder_name(): a
    filesystem-safe but CASE-PRESERVING transform, used to locate that
    script's saved warm_up_sessions/{participant_folder}.json file. This
    is deliberately NOT the same as slugify_name() above (which lowercases
    and is used for THIS script's own users.json keys) -- the warm-up
    script's session filenames preserve the participant ID's original
    case (e.g. "A11320.json", not "a11320.json"), so looking them up
    requires the matching, case-preserving transform.
    """
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_")
    return value or "unknown_participant"


def find_warm_up_session_path(participant_id: str) -> Optional[str]:
    """
    Looks for a warm-up session JSON saved by ameca_warm_up.py for this
    participant_id. Tries the exact case-preserving filename first, then
    falls back to a case-insensitive directory scan (participant IDs are
    sometimes typed with different casing between the two scripts).
    """
    participant_id = (participant_id or "").strip()
    if not participant_id:
        return None

    folder_name = sanitize_warm_up_folder_name(participant_id)
    exact_path = os.path.join(WARM_UP_SESSIONS_DIR, f"{folder_name}.json")
    if os.path.isfile(exact_path):
        return exact_path

    if not os.path.isdir(WARM_UP_SESSIONS_DIR):
        return None

    target = folder_name.lower()
    try:
        for filename in os.listdir(WARM_UP_SESSIONS_DIR):
            if not filename.lower().endswith(".json"):
                continue
            if filename[: -len(".json")].lower() == target:
                return os.path.join(WARM_UP_SESSIONS_DIR, filename)
    except Exception as exc:
        print_ts(f"[WARN] Could not scan warm-up sessions folder: {exc}")

    return None


def load_warm_up_session(participant_id: str) -> Optional[dict]:
    
    path = find_warm_up_session_path(participant_id)
    if not path:
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return None
        data["_source_path"] = path
        return data
    except Exception as exc:
        print_ts(f"[WARN] Could not read warm-up session file {path}: {exc}")
        return None


def load_users() -> dict:
    ensure_data_dirs()

    if not os.path.exists(USERS_FILE):
        return {}

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def save_users(users: dict) -> None:
    ensure_data_dirs()

    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2, ensure_ascii=False)


def clean_spoken_name(spoken_name: str) -> str:
    spoken_name = spoken_name.strip()

    patterns = [
        r"^my name is\s+",
        r"^my name's\s+",
        r"^i am\s+",
        r"^i'm\s+",
        r"^this is\s+",
        r"^it is\s+",
        r"^it's\s+",
        r"^call me\s+",
    ]

    for pattern in patterns:
        spoken_name = re.sub(pattern, "", spoken_name, flags=re.IGNORECASE).strip()

    spoken_name = re.sub(r"[^a-zA-Z0-9\s\-]", "", spoken_name)
    spoken_name = spoken_name[:40].strip()

    return spoken_name or "Guest"


def clean_spelled_name(text: str) -> Optional[str]:
    text = text.upper().strip()

    text = re.sub(r"\bMY NAME IS\b", "", text)
    text = re.sub(r"\bIT IS\b", "", text)
    text = re.sub(r"\bIT'S\b", "", text)
    text = re.sub(r"\bSPELL(?:ED|ING)?\b", "", text)
    text = re.sub(r"[^A-Z\s]", " ", text)

    parts = text.split()

    letters = []
    for part in parts:
        if len(part) == 1 and part.isalpha():
            letters.append(part)

    if len(letters) >= 2:
        return "".join(letters).title()

    return None


def get_known_user_names() -> list[str]:
    users = load_users()
    names: list[str] = []
    for profile in users.values():
        name = str(profile.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def levenshtein_distance(a: str, b: str) -> int:
    a = a.lower().strip()
    b = b.lower().strip()

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


# =========================
# ASR mishearing detection / recovery
# =========================
#
# faster-whisper (base model, CPU, int8) frequently mangles multi-syllable
# technical vocabulary into phonetically-similar but semantically nonsense
# text -- e.g. "supervised learning" -> "super-raced learning", "large
# language models" -> "large-level models". Rather than silently guessing
# and answering a different question than the one asked, or answering the
# literal (nonsense) transcript at face value, Ameca briefly confirms its
# best guess before proceeding -- mirroring how a human tutor would say
# "did you mean X?" rather than either option.

ASR_MISHEARING_CORRECTIONS: dict[str, str] = {
    "super-raced": "supervised",
    "super raced": "supervised",
    "superraced": "supervised",
    "on supervised": "unsupervised",
    "large-level model": "large language model",
    "large-level models": "large language models",
    "large level model": "large language model",
    "large level models": "large language models",
    "special intelligence": "artificial intelligence",
    "respiratory robotics": "with respect to robotics",
}

# Canonical multi-word curriculum vocabulary used for the fuzzy fallback
# check below, when the mangled phrase isn't already in the static map
# above. Kept as phrases (not single words), since most confusions
# observed so far involve a two-word technical term.
ASR_DOMAIN_VOCABULARY: list[str] = [
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "machine learning", "deep learning", "large language model",
    "large language models", "neural network", "neural networks",
    "transformer", "transformers", "tokenization", "embeddings",
    "context window", "backpropagation", "gradient descent",
    "attention mechanism", "artificial intelligence",
]


def _fuzzy_asr_correction_candidate(
    text: str,
    max_distance_ratio: float = 0.3,
) -> Optional[tuple[str, str]]:
    """
    Best-effort fallback for mishearings not yet in
    ASR_MISHEARING_CORRECTIONS: slides a 1-3 word window across `text` and
    compares each window against ASR_DOMAIN_VOCABULARY via Levenshtein
    distance. Returns (heard_phrase, canonical_phrase) for the closest
    sub-threshold match, or None. This is spelling-distance (not
    phonetic) similarity, so it's deliberately conservative -- the
    curated static map above is the primary mechanism; this is
    defense-in-depth for terms not yet seen and added there.

    Critically, a candidate window is skipped entirely (never flagged) if
    it already CONTAINS a valid vocabulary term as a substring, not just
    if the whole window equals one exactly. Without this, a 3-word window
    like "and unsupervised learning" would be compared against the
    *different* canonical term "supervised learning" (edit distance only
    ~4), producing a false positive that flags an already-correct message
    ("...difference between supervised and unsupervised learning?") as a
    likely mishearing of itself.
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return None

    def _contains_valid_vocab(candidate: str) -> bool:
        padded = f" {candidate} "
        return any(f" {term} " in padded or candidate == term for term in ASR_DOMAIN_VOCABULARY)

    best_match: Optional[tuple[str, str, float]] = None  # (heard, canonical, ratio)

    for window_size in (2, 3, 1):
        for start in range(0, max(0, len(words) - window_size + 1)):
            candidate = " ".join(words[start:start + window_size])
            if len(candidate) < 6:
                continue
            if _contains_valid_vocab(candidate):
                continue  # already-correct content present; nothing to correct
            for canonical in ASR_DOMAIN_VOCABULARY:
                distance = levenshtein_distance(candidate, canonical)
                ratio = distance / max(len(candidate), len(canonical))
                if ratio <= max_distance_ratio:
                    if best_match is None or ratio < best_match[2]:
                        best_match = (candidate, canonical, ratio)

    if best_match is None:
        return None
    return best_match[0], best_match[1]


def find_likely_asr_misrecognition(text: str) -> Optional[tuple[str, str]]:
    """
    Returns (heard_phrase, corrected_phrase) if `text` appears to contain a
    mangled technical term, else None. Checks the curated static map first
    (fast, zero false-positive risk), then falls back to fuzzy matching
    against the domain vocabulary.
    """
    lowered = text.lower()
    for heard, corrected in ASR_MISHEARING_CORRECTIONS.items():
        if heard in lowered:
            return heard, corrected

    return _fuzzy_asr_correction_candidate(text)


def build_asr_correction_confirmation(heard_phrase: str, corrected_phrase: str) -> str:
    return f"I think you meant '{corrected_phrase}', not '{heard_phrase}'. Is that right?"


def apply_asr_correction(text: str, heard_phrase: str, corrected_phrase: str) -> str:
    """
    Case-insensitively substitutes the first occurrence of heard_phrase
    with corrected_phrase in text, preserving the rest of the message.
    """
    pattern = re.compile(re.escape(heard_phrase), re.IGNORECASE)
    return pattern.sub(corrected_phrase, text, count=1)


def correct_spelled_name_with_known_users(
    spelled_name: Optional[str],
    spoken_name: Optional[str] = None,
) -> Optional[str]:
    if not spelled_name:
        return None

    candidate = clean_spoken_name(spelled_name)
    known_names = get_known_user_names()

    if not known_names:
        return candidate

    candidate_slug = slugify_name(candidate)
    for known in known_names:
        if slugify_name(known) == candidate_slug:
            return known

    best_name = None
    best_distance = 999

    for known in known_names:
        distance = levenshtein_distance(candidate, known)
        if distance < best_distance:
            best_distance = distance
            best_name = known

    if best_name and best_distance <= 2:
        print_ts(f"Corrected spelled name '{candidate}' to known user '{best_name}'.")
        return best_name

    if spoken_name:
        spoken_candidate = clean_spoken_name(spoken_name)
        best_spoken = None
        best_spoken_distance = 999

        for known in known_names:
            distance = levenshtein_distance(spoken_candidate, known)
            if distance < best_spoken_distance:
                best_spoken_distance = distance
                best_spoken = known

        if best_spoken and best_spoken_distance <= 2:
            print_ts(f"Corrected spoken name '{spoken_candidate}' to known user '{best_spoken}'.")
            return best_spoken

    return candidate


def extract_name_from_text(text: str) -> Optional[str]:
    text = text.strip()

    patterns = [
        r"\bmy name is\s+([a-zA-Z][a-zA-Z\s\-]{1,40})",
        r"\bmy name's\s+([a-zA-Z][a-zA-Z\s\-]{1,40})",
        r"\bcall me\s+([a-zA-Z][a-zA-Z\s\-]{1,40})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = clean_spoken_name(match.group(1))

            words = candidate.split()
            if 1 <= len(words) <= 4:
                return candidate

    return None


def looks_like_invalid_name(name: str) -> bool:
    invalid = {
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "yes", "no", "okay", "ok", "thanks", "thank you", "bye", "goodbye",
    }

    return name.strip().lower() in invalid


def rename_current_user(
    old_user_key: str,
    user_profile: dict,
    new_name: str,
) -> tuple[str, dict]:
    users = load_users()

    new_name = clean_spoken_name(new_name)
    new_key = slugify_name(new_name)

    if not new_name or looks_like_invalid_name(new_name):
        return old_user_key, user_profile

    existing_profile = users.get(new_key)

    if existing_profile:
        existing_profile["last_seen"] = now_ts()
        print_ts(f"Switched to existing user profile: {existing_profile['name']}")
        save_users(users)
        return new_key, existing_profile

    old_profile = users.pop(old_user_key, user_profile)

    old_profile["name"] = new_name
    old_profile["last_seen"] = now_ts()
    users[new_key] = old_profile

    save_users(users)

    print_ts(f"Updated user name to: {new_name}")
    return new_key, users[new_key]


def build_user_memory_context(user_profile: Optional[dict]) -> str:
    if not user_profile:
        return ""

    name = user_profile.get("name", "the user")
    summary = user_profile.get("conversation_summary", "").strip()

    if not summary:
        summary = "No previous conversation summary is available."

    return f"""
        USER MEMORY CONTEXT
        The user's name is {name}.

        Previous conversation summary:
        {summary}

        Rules:
        - Use the user's name naturally when appropriate.
        - You may continue from this saved local summary, but do not claim memory beyond it.
        - If the user asks whether you remember previous conversations, say that you can continue from the saved local conversation summary when available.
        - Do not say every conversation starts fresh if a previous summary is provided.
        - Do not reveal the raw JSON file unless the user asks.
        """.strip()


def compact_previous_summary_for_greeting(
    summary: str,
    max_chars: int = RETURNING_USER_GREETING_MAX_SUMMARY_CHARS,
) -> str:
    summary = str(summary or "").strip()
    if not summary:
        return ""

    summary = strip_previous_continuity_prefix(summary)

    summary = re.sub(r"^\s*[-*\u2022]\s*", "", summary, flags=re.MULTILINE)
    summary = re.sub(r"\s+", " ", summary).strip()

    if len(summary) <= max_chars:
        return summary

    clipped = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped.rstrip(".,;:") + "..."


def fallback_returning_user_greeting(user_profile: dict) -> str:
    name = str(user_profile.get("name", "there")).strip() or "there"
    summary = compact_previous_summary_for_greeting(
        str(user_profile.get("conversation_summary", "")).strip()
    )

    if not summary:
        return f"Welcome back, {name}. It is nice to continue our conversation. \U0001F642"

    return (
        f"Welcome back, {name}. Last time, we were discussing {summary} "
        f"Where would you like to continue from? \U0001F642"
    )


def build_returning_user_context(user_profile: Optional[dict]) -> str:
    if not user_profile:
        return ""

    summary = str(user_profile.get("conversation_summary", "")).strip()
    name = str(user_profile.get("name", "the user")).strip() or "the user"
    last_seen = str(user_profile.get("last_seen", "")).strip()

    if not summary:
        return ""

    return f"""
        RETURNING USER CONTEXT

        User name: {name}
        Last seen: {last_seen or "unknown"}

        Previous conversation summary:
        {summary}

        Continuity rules:
        - Start the new session from this previous conversation summary.
        - Remind the user of at most ONE relevant previous topic.
        - Use a natural phrase such as "Last time, we discussed..." or "Last time, we were working on..."
        - Do not dump the whole summary.
        - Ask one short follow-up question connected to the previous topic when appropriate.
        - Do not mention JSON, files, transcripts, or memory storage.
        - Do not claim to remember anything outside this saved local summary.
        """.strip()


def generate_returning_user_response(
    client: Client,
    user_profile: dict,
) -> str:
    if not ENABLE_RETURNING_USER_MEMORY_GREETING:
        name = str(user_profile.get("name", "there")).strip() or "there"
        return f"Welcome back, {name}. It is nice to continue our conversation. \U0001F642"

    memory_context = build_returning_user_context(user_profile)

    if not memory_context:
        return fallback_returning_user_greeting(user_profile)

    system_prompt = f"""
        You are Ameca, a socially intelligent humanoid robot.

        A returning user has started a new interaction session.

        {memory_context}

        Your task:
        - welcome the user back naturally
        - remind the user of exactly ONE important previous topic from the saved summary
        - continue from that topic instead of starting from zero
        - ask one short follow-up question connected to that topic
        - keep it warm, concise, and conversational
        - produce 1-2 short sentences
        - end with exactly one friendly facial emoji
        - use only this emoji: \U0001F642

        Required style:
        - Prefer wording like: "Last time, we were working on..." or "Last time, we discussed..."
        - Sound like a helpful robot continuing a conversation, not like a log reader.

        Do not:
        - dump the memory summary
        - mention transcripts, JSON, files, storage, or saved summaries
        - pretend to remember anything outside the provided local memory context
        - mention more than one previous topic
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
            ],
            options={
                "temperature": 0.35,
                "num_predict": 200,
                "num_ctx": 4096,
            },
            stream=False,
        )

        raw_reply = response["message"]["content"]
        reply = normalize_reply(raw_reply, "neutral")

        if "[" in reply or "]" in reply:
            # Same unrendered-placeholder guard as generate_post_warmup_welcome()
            # -- catches leaked template brackets like "[Name]" regardless
            # of which name value ended up in the prompt.
            return fallback_returning_user_greeting(user_profile)

        lower_reply = reply.lower()
        if (
            user_profile.get("conversation_summary")
            and "last time" not in lower_reply
            and "previous" not in lower_reply
            and "we discussed" not in lower_reply
            and "we were" not in lower_reply
        ):
            return fallback_returning_user_greeting(user_profile)

        return reply

    except Exception as exc:
        print_ts(f"Could not generate returning-user greeting with LLM: {exc}")
        return fallback_returning_user_greeting(user_profile)


def generate_post_warmup_welcome(client: Client, display_name: str) -> str:
    """
    Spoken once, for a participant whose warm-up session (ameca_warm_up.py)
    was found but who has no prior record in THIS script's users.json --
    i.e. this is their first turn in the actual experiment. Explicitly
    tells them the warm-up is over and the real experiment (beginner-level
    session 1) is starting now, so there's no ambiguity about which phase
    they're in.
    """
    is_generic_placeholder_name = display_name.strip().lower() in {"guest", ""}

    fallback = (
        "Welcome back! Thank you for completing the warm-up. "
        "We are now starting the actual experiment, beginning at the beginner level. \U0001F642"
        if is_generic_placeholder_name
        else f"Welcome back, {display_name}. Thank you for completing the warm-up. "
        "We are now starting the actual experiment, beginning at the beginner level. \U0001F642"
    )

    try:
        if is_generic_placeholder_name:
            # No real name was captured for this participant ("Guest" is
            # this script's internal fallback value, not something to
            # speak aloud). Asking the model to "welcome them back by
            # name" using the literal word "Guest" was observed causing
            # it to treat "Guest" as a template placeholder and output
            # the unrendered bracket text "[Guest's Name]" verbatim.
            # Skip name personalization entirely for this case instead.
            system_prompt = """
                You are Ameca, a humanoid social robot.

                A participant just finished a separate warm-up session with you and
                is now starting the ACTUAL experiment for the first time. Their name
                was not captured, so greet them warmly WITHOUT using any name or
                placeholder for one.

                Your task:
                - welcome them back, with no name or placeholder
                - thank them briefly for completing the warm-up
                - clearly tell them the actual experiment is starting now, at the
                  beginner level
                - keep it to 1-2 short sentences
                - end with exactly one friendly facial emoji, only this one: \U0001F642

                Do not:
                - use any name, placeholder, or bracketed text like "[Name]"
                - mention JSON, files, or any storage/session mechanics
                - ask them to choose a level themselves
                """.strip()
        else:
            system_prompt = f"""
                You are Ameca, a humanoid social robot.

                {display_name} just finished a separate warm-up session with you and is
                now starting the ACTUAL experiment for the first time.

                Your task:
                - welcome them back by name, saying "{display_name}" exactly as
                  written -- it is their actual name, not a placeholder to fill in
                - thank them briefly for completing the warm-up
                - clearly tell them the actual experiment is starting now, at the
                  beginner level
                - keep it to 1-2 short sentences
                - end with exactly one friendly facial emoji, only this one: \U0001F642

                Do not:
                - use bracketed placeholder text like "[Name]" -- always use the
                  actual name given above, spoken plainly
                - mention JSON, files, or any storage/session mechanics
                - ask them to choose a level themselves
                """.strip()

        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}],
            options={"temperature": 0.4, "num_predict": 150, "num_ctx": 1024},
            stream=False,
        )
        raw_reply = response["message"]["content"]
        reply = normalize_reply(raw_reply, "neutral")

        if "[" in reply or "]" in reply:
            # Catches unrendered template placeholders like "[Name]" or
            # "[Guest's Name]" regardless of which branch above produced
            # them -- a plain substring check on the name alone isn't
            # enough, since e.g. "guest" is itself a substring of the
            # malformed "[Guest's Name]".
            return fallback

        if not is_generic_placeholder_name:
            first_name = display_name.split()[0].lower() if display_name.strip() else ""
            if first_name and first_name not in reply.lower():
                return fallback

        return reply
    except Exception as exc:
        print_ts(f"Could not generate post-warm-up welcome with LLM: {exc}")
        return fallback


def build_prior_sessions_text(session_history: list[dict], max_entries: int = 2) -> str:
    if not session_history:
        return "No structured record of previous sessions is available."

    recent = session_history[-max_entries:]
    lines = []
    for entry in recent:
        number = entry.get("session_number", "?")
        level = entry.get("level", "unknown")
        summary = str(entry.get("summary", "")).strip() or "No summary recorded."
        lines.append(f"Session {number} ({level}): {summary}")
    return "\n".join(lines)


def generate_session_recap_and_questions(
    client: Client,
    session_history: list[dict],
    explanation_level: str,
) -> dict:
    
    fallback = {
        "recap": (
            "Before we continue, let's briefly recap what we covered in our "
            "last session together."
        ),
        "questions": [
            "Can you tell me, in your own words, one thing you remember from last time?",
            "Was there anything from last session that felt unclear or that you'd like us to revisit?",
            "Can you give an example of how what we covered last time might apply in practice?",
        ],
    }

    prior_sessions_text = build_prior_sessions_text(session_history)

    prompt = f"""
        You are Ameca, preparing to start a new tutoring session with a returning
        participant, at explanation level: {explanation_level}.

        Summary of their previous session(s):
        {prior_sessions_text}

        Write:
        1. A short 150 words spoken recap of what was covered previously.
        2. Exactly two short review questions, appropriate for a {explanation_level}
           learner, that test whether they remember or understood the previous
           material. Keep each question under 20 words.

        Return JSON only in exactly this shape:
        {{"recap": "...", "questions": ["...", "..."]}}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 220, "num_ctx": 2048},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return fallback

        recap = str(data.get("recap", "")).strip() or fallback["recap"]
        questions = data.get("questions")
        if not isinstance(questions, list) or len(questions) < 1:
            return fallback

        cleaned_questions = [str(q).strip() for q in questions if str(q).strip()]
        if not cleaned_questions:
            return fallback

        return {"recap": recap, "questions": cleaned_questions[:3]}
    except Exception as exc:
        print_ts(f"Could not generate session recap/questions with LLM: {exc}")
        return fallback


def generate_recap_answer_feedback(
    client: Client,
    question: str,
    answer: str,
    recap_context: str = "",
) -> str:
    
    fallback = "Thanks for sharing that -- let's continue."

    try:
        context_block = (
            f"What was actually covered last time (for you to judge their answer against):\n{recap_context}\n"
            if recap_context.strip()
            else ""
        )
        prompt = f"""
            You are Ameca. You just asked a returning participant this review
            question:
            {question}

            They answered:
            {answer}

            {context_block}
            First judge whether their answer is correct, partially correct, or
            incorrect. Then respond with ONE short sentence (max 25 words):
            - if correct, warmly AFFIRM it
            - if partially correct or incorrect, gently CORRECT it by briefly
              stating the right idea, without being discouraging

            Plain text only, no markdown, no preamble like "Your answer is...".
            """.strip()

        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 90, "num_ctx": 1536},
            stream=False,
        )
        text = response["message"]["content"].strip()
        text = re.sub(r"\s+", " ", text)
        return text or fallback
    except Exception as exc:
        print_ts(f"Could not generate recap-answer feedback with LLM: {exc}")
        return fallback


def save_session_transcript(
    user_key: str,
    user_profile: dict,
    session_log: list[dict],
    participant_id: str = "",
    video_path: Optional[str] = None,
    llm_call_samples: Optional[list[dict]] = None,
    session_number: Optional[int] = None,
    explanation_level: Optional[str] = None,
    is_first_after_warmup: bool = False,
) -> str:
    ensure_data_dirs()

    # Sessions are keyed by PARTICIPANT ID (not the spoken/display name),
    # so continuity/storage is stable regardless of ASR name-transcription
    # drift. Falls back to user_profile's stored participant_id, then to
    # user_key, if participant_id wasn't explicitly passed.
    participant_id = str(
        participant_id or user_profile.get("participant_id") or user_key or "unknown"
    ).strip()
    participant_slug = slugify_name(participant_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{participant_slug}_{timestamp}.json"
    path = os.path.join(SESSIONS_DIR, filename)

    transcript_data = {
        "user": {
            "key": user_key,
            "participant_id": participant_id,
            "name": user_profile.get("name", "Guest"),
        },
        "session": {
            "started_at": session_log[0]["timestamp"] if session_log else now_ts(),
            "ended_at": now_ts(),
            "model": MODEL_NAME,
            "ollama_host": OLLAMA_HOST,
            "video_path": video_path,
            "asr": {
                "backend": "faster-whisper",
                **FAST_WHISPER_CONFIG,
            },
            "emotion_resolution": {
                "modality": "text_only",
                "taxonomy": "ekman_plus_neutral_7class",
                "taxonomy_classes": list(EKMAN_EMOTIONS.keys()),
                "temporal_smoothing": {
                    "enabled": EMOTION_SMOOTHING_ENABLED,
                    "alpha": EMOTION_SMOOTHING_ALPHA,
                },
            },
            "vad": {
                "backend": "Silero VAD",
                "sample_rate": SILERO_SAMPLE_RATE,
                "chunk_size": SILERO_CHUNK_SIZE,
                "threshold": SILERO_THRESHOLD,
                "min_silence_duration_ms": SILERO_MIN_SILENCE_DURATION_MS,
                "speech_pad_ms": SILERO_SPEECH_PAD_MS,
                "max_utterance_seconds": VAD_MAX_UTTERANCE_SECONDS,
                "min_utterance_seconds": VAD_MIN_UTTERANCE_SECONDS,
                "pre_roll_seconds": VAD_PRE_ROLL_SECONDS,
            },
            "output": {
                "backend": "Tritium TTS",
                "activity_ema_threshold": TTS_SPEAKING_EMA_THRESHOLD,
                "activity_debounce_seconds": TTS_ACTIVITY_DEBOUNCE_SECONDS,
            },
            "expression": {
                "backend": "Tritium sequence_player",
                "emotion_sequence_map": EMOTION_SEQUENCE_MAP,
                "min_confidence": EXPRESSION_MIN_CONFIDENCE,
                "force_replay_same": EXPRESSION_FORCE_REPLAY_SAME,
                "nod_sequence": NOD_SEQUENCE_NAME,
                "negative_expressions_suppressed": True,
                "waits_for_expected_duration_before_speaking": EXPRESSION_TIMING == "before",
            },
            "video": {
                "backend": "ZED camera (recording only; not used for emotion detection)",
                "path": video_path,
                "fps": VIDEO_RECORD_FPS,
            },
            "experiment_session": {
                "session_number": session_number,
                "explanation_level": explanation_level,
                "is_first_after_warmup": is_first_after_warmup,
                "warm_up_session_path": user_profile.get("warm_up_session_path"),
            },
        },
        # First few (up to 3) full prompts sent to the response-generation
        # LLM and the reply that was ultimately used, kept for analysis.
        "llm_call_samples": llm_call_samples or [],
        "messages": session_log,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(transcript_data, file, indent=2, ensure_ascii=False)

    return path


def strip_previous_continuity_prefix(text: str) -> str:
    text = str(text or "").strip()
    prefix_pattern = re.compile(r"^(?:[\-\*\u2022\s]*previous continuity context:\s*)+", re.IGNORECASE)
    return prefix_pattern.sub("", text).strip()


def build_deterministic_session_summary(
    session_log: list[dict],
    previous_summary: str = "",
) -> str:
    if not session_log:
        return str(previous_summary or "").strip()

    user_turns: list[str] = []
    assistant_turns: list[str] = []
    emotions: list[str] = []
    rag_topics: list[str] = []
    rag_sources: list[str] = []

    for item in session_log:
        role = str(item.get("role", "")).strip().lower()
        content = re.sub(r"\s+", " ", str(item.get("content", "")).strip())
        if not content:
            continue

        if role == "user":
            user_turns.append(content)
        elif role == "assistant" and item.get("intent") != "self_introduction":
            assistant_turns.append(content)

        emotion = item.get("emotion")
        if isinstance(emotion, dict):
            detected = str(emotion.get("emotion", "")).strip()
            if detected:
                emotions.append(detected)

        self_rag = item.get("self_rag")
        if isinstance(self_rag, dict) and self_rag.get("used"):
            query = str(self_rag.get("query", "")).strip()
            if query:
                rag_topics.append(query)
            for source in self_rag.get("sources", []) or []:
                if isinstance(source, dict):
                    title = str(source.get("title", "")).strip()
                    if title and title not in rag_sources:
                        rag_sources.append(title)

    bullets: list[str] = []

    previous_summary = str(previous_summary or "").strip()
    if previous_summary:
        cleaned_previous_summary = strip_previous_continuity_prefix(previous_summary)
        if cleaned_previous_summary:
            bullets.append(
                "Previous continuity context: "
                + compact_previous_summary_for_greeting(cleaned_previous_summary, 260)
            )

    if user_turns:
        recent_user = "; ".join(user_turns[-4:])
        bullets.append("Recent user topics/questions: " + recent_user[:420].rstrip())

    if rag_topics:
        bullets.append("Knowledge/RAG topics used: " + "; ".join(rag_topics[-4:])[:420].rstrip())

    if rag_sources:
        bullets.append("Relevant retrieved sources included: " + "; ".join(rag_sources[:5])[:420].rstrip())

    if assistant_turns:
        bullets.append("Last assistant direction: " + assistant_turns[-1][:300].rstrip())

    if emotions:
        recent_unique: list[str] = []
        for emotion in emotions[-8:]:
            if emotion not in recent_unique:
                recent_unique.append(emotion)
        bullets.append("Recent affective context: " + ", ".join(recent_unique))

    if not bullets:
        return previous_summary

    return "\n".join(f"- {bullet}" for bullet in bullets[:8])


def load_latest_session_log_for_user(user_key: str, user_profile: Optional[dict]) -> list[dict]:
    ensure_data_dirs()

    candidate_paths: list[str] = []

    if isinstance(user_profile, dict):
        for path in user_profile.get("session_files", []) or []:
            if isinstance(path, str) and path.strip():
                candidate_paths.append(path.strip())

    try:
        slug = slugify_name(user_key)
        for filename in os.listdir(SESSIONS_DIR):
            if filename.startswith(f"{slug}_") and filename.endswith(".json"):
                candidate_paths.append(os.path.join(SESSIONS_DIR, filename))
    except Exception as exc:
        print_ts(f"Could not scan session folder for memory recovery: {exc}")

    existing_paths = []
    seen = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            existing_paths.append(path)

    if not existing_paths:
        return []

    existing_paths.sort(key=lambda path: os.path.getmtime(path), reverse=True)

    for path in existing_paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            messages = data.get("messages", [])
            if isinstance(messages, list) and messages:
                print_ts(f"Recovered continuity memory from latest transcript: {path}")
                return messages
        except Exception as exc:
            print_ts(f"Could not recover memory from transcript {path}: {exc}")

    return []


def ensure_user_has_conversation_summary(
    user_key: str,
    user_profile: dict,
) -> dict:
    summary = str(user_profile.get("conversation_summary", "")).strip()
    if summary:
        return user_profile

    recovered_log = load_latest_session_log_for_user(user_key, user_profile)
    if not recovered_log:
        print_ts("No previous conversation summary or recoverable transcript found.")
        return user_profile

    recovered_summary = build_deterministic_session_summary(
        session_log=recovered_log,
        previous_summary="",
    ).strip()

    if not recovered_summary:
        print_ts("Recovered transcript did not produce a usable continuity summary.")
        return user_profile

    users = load_users()
    profile = users.get(user_key, user_profile)
    profile["conversation_summary"] = recovered_summary
    profile["last_seen"] = now_ts()
    users[user_key] = profile
    save_users(users)

    print_ts("Recovered and saved previous conversation summary for returning-user greeting.")
    return profile


def summarize_session_with_llm(
    client: Client,
    session_log: list[dict],
    previous_summary: str = "",
) -> str:
    if not session_log:
        return previous_summary

    previous_summary = strip_previous_continuity_prefix(previous_summary)

    fallback_summary = build_deterministic_session_summary(
        session_log=session_log,
        previous_summary=previous_summary,
    )

    compact_messages = []

    for item in session_log[-30:]:
        role = item.get("role", "")
        content = item.get("content", "")
        emotion = item.get("emotion")
        self_rag = item.get("self_rag")

        extra_parts = []
        if isinstance(emotion, dict):
            extra_parts.append(f"emotion={emotion.get('emotion')}")
        if isinstance(self_rag, dict):
            extra_parts.append(f"self_rag_used={self_rag.get('used')}")

        extra = f" ({', '.join(extra_parts)})" if extra_parts else ""
        compact_messages.append(f"{role}{extra}: {content}")

    prompt = f"""
        Summarize this human-robot interaction session for long-term conversational continuity.

        Previous saved continuity summary:
        {previous_summary or "None"}

        Recent session:
        {chr(10).join(compact_messages)}

        Write an updated continuity summary that Ameca can use at the start of the next session.

        Focus on:
        - ongoing projects or unresolved tasks
        - thesis/research/debugging topics discussed
        - important technical context
        - emotional state trends if relevant
        - personal preferences or interaction style
        - useful next-step cues for continuing the conversation

        Avoid:
        - trivial greetings
        - repetitive small talk
        - exact quotations
        - invented facts
        - private implementation details that the user did not discuss

        Return 4-8 concise bullet points.
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You write accurate continuity summaries for a humanoid robot. Do not invent facts.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.2,
                "num_predict": 260,
                "num_ctx": 4096,
            },
            stream=False,
        )

        summary = response["message"]["content"].strip()
        if not summary:
            print_ts("LLM returned an empty session summary; using deterministic fallback summary.")
            return fallback_summary or previous_summary

        return summary

    except KeyboardInterrupt:
        print_ts("Summary generation interrupted. Using deterministic fallback summary instead.")
        return fallback_summary or previous_summary

    except Exception as exc:
        print_ts(f"Could not summarize session with LLM: {exc}")
        print_ts("Using deterministic fallback summary instead.")
        return fallback_summary or previous_summary


def update_user_after_session(
    client: Client,
    user_key: str,
    session_path: str,
    session_log: list[dict],
    lesson_state: Optional["LessonState"] = None,
) -> None:
    users = load_users()

    if user_key not in users:
        return

    users[user_key]["last_seen"] = now_ts()
    users[user_key].setdefault("session_files", []).append(session_path)

    if lesson_state is not None:                        
        users[user_key]["lesson_state"] = lesson_state.to_json()

    previous_summary = str(users[user_key].get("conversation_summary", "")).strip()

    deterministic_summary = build_deterministic_session_summary(
        session_log=session_log,
        previous_summary=previous_summary,
    ).strip()
    if deterministic_summary:
        users[user_key]["conversation_summary"] = deterministic_summary
        save_users(users)
        print_ts("Saved deterministic returning-user conversation summary.")

    if ENABLE_LLM_SESSION_SUMMARY:
        users = load_users()
        previous_summary = str(users[user_key].get("conversation_summary", "")).strip()

        users[user_key]["conversation_summary"] = summarize_session_with_llm(
            client=client,
            session_log=session_log,
            previous_summary=previous_summary,
        )
        print_ts("Updated returning-user conversation summary.")
    else:
        print_ts("Skipping LLM session summary for faster shutdown.")

    save_users(users)


def record_session_completion(
    user_key: str,
    session_number: int,
    explanation_level: str,
    session_log: list[dict],
) -> None:
    
    users = load_users()
    if user_key not in users:
        return

    session_summary = build_deterministic_session_summary(
        session_log=session_log,
        previous_summary="",
    ).strip()

    history_entry = {
        "session_number": session_number,
        "level": explanation_level,
        "summary": session_summary or "No summary available.",
        "ended_at": now_ts(),
    }

    session_history = users[user_key].get("session_history")
    if not isinstance(session_history, list):
        session_history = []
    session_history.append(history_entry)
    users[user_key]["session_history"] = session_history
    users[user_key]["sessions_completed"] = int(users[user_key].get("sessions_completed", 0)) + 1

    save_users(users)
    print_ts(
        f"Recorded completion of session {session_number} ({explanation_level}); "
        f"sessions_completed is now {users[user_key]['sessions_completed']}."
    )


# =========================
# Ollama setup helpers
# =========================

def check_ollama_available(max_attempts: int = 10, delay_seconds: float = 1.0) -> None:
    client = Client(host=OLLAMA_HOST)

    for attempt in range(1, max_attempts + 1):
        try:
            client.list()
            print_ts(f"Ollama is reachable at {OLLAMA_HOST}.")
            return
        except Exception as exc:
            print_ts(f"Ollama not reachable yet, attempt {attempt}/{max_attempts}: {exc}")
            time.sleep(delay_seconds)

    raise RuntimeError(
        f"Could not connect to Ollama at {OLLAMA_HOST} after {max_attempts} attempts."
    )


def ensure_model_available(model_name: str = MODEL_NAME) -> None:
    print_ts(f"Using Ollama model '{model_name}' from {OLLAMA_HOST}. Skipping local pull.")


# =========================
# Audio helpers
# =========================

def list_input_devices() -> None:
    print("\nAvailable input devices:")

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:
        print(f"Could not query devices: {exc}")
        return

    found = False

    for idx, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            found = True
            hostapi_name = hostapis[device["hostapi"]]["name"]
            print(
                f"[mic {idx}] {device['name']} | "
                f"hostapi={hostapi_name} | "
                f"inputs={device['max_input_channels']} | "
                f"default_sr={device['default_samplerate']}"
            )

    if not found:
        print("No input devices found.")

    print()
    print(f"Current default audio device: {sd.default.device}")
    print()


def get_input_samplerate(input_device: Optional[int]) -> int:
    if input_device is None:
        device_info = sd.query_devices(kind="input")
    else:
        device_info = sd.query_devices(input_device)

    default_sr = int(round(device_info["default_samplerate"]))
    if default_sr <= 0:
        return TARGET_SAMPLE_RATE

    return default_sr


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / original_sr
    target_length = max(1, int(round(duration * target_sr)))

    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, num=target_length, endpoint=False)

    return np.interp(new_times, old_times, audio).astype(np.float32)


def save_audio_to_temp_wav(audio_16k: np.ndarray) -> Optional[str]:
    if audio_16k.size == 0:
        return None

    peak = float(np.max(np.abs(audio_16k)))
    rms = float(np.sqrt(np.mean(audio_16k ** 2)))

    print_ts(f"Captured utterance audio level: peak={peak:.4f}, rms={rms:.4f}")

    if peak < MIN_PEAK_THRESHOLD or rms < MIN_RMS_THRESHOLD:
        print("Captured audio was too quiet or silent.")
        return None

    audio_16k = np.clip(audio_16k * min(0.9 / max(peak, 1e-6), 10.0), -1.0, 1.0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    sf.write(wav_path, audio_16k, TARGET_SAMPLE_RATE)
    return wav_path


def transcribe_with_faster_whisper(wav_path: str, whisper_model: WhisperModel) -> str:
    segments, info = whisper_model.transcribe(
        wav_path,
        language=FAST_WHISPER_CONFIG.get("language"),
        beam_size=int(FAST_WHISPER_CONFIG.get("beam_size", 1)),
        vad_filter=bool(FAST_WHISPER_CONFIG.get("vad_filter", False)),
        condition_on_previous_text=False,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) <= 1 and len(text) < 3:
        return ""

    return text


# Backwards-compatible alias.
transcribe_audio = transcribe_with_faster_whisper


# =========================
# Silero VAD listener (audio-only; no visual/face capture)
# =========================

def listen_for_utterance_with_silero_vad(
    input_device: Optional[int],
    silero_model,
    prompt_label: str = "utterance",
    robot_speaker: Optional[RobotSpeaker] = None,
    camera: Optional["Camera"] = None,
) -> tuple[Optional[str], list[np.ndarray]]:
    """
    Returns (wav_path, frames). `frames` are camera frames sampled (via a
    per-utterance FrameCollector) while speech was being recorded, used
    only to source per-turn face crops (see save_turn_face_crops()) --
    empty if `camera` is None or none were captured. `wav_path` is None
    on the failure paths (see below), in which case `frames` is also [].
    """
    input_sample_rate = get_input_samplerate(input_device)
    input_block_size = max(1, int(input_sample_rate * 0.05))
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    vad_iterator = VADIterator(
        silero_model,
        threshold=SILERO_THRESHOLD,
        sampling_rate=SILERO_SAMPLE_RATE,
        min_silence_duration_ms=SILERO_MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SILERO_SPEECH_PAD_MS,
    )

    pre_roll_max_chunks = max(1, int((VAD_PRE_ROLL_SECONDS * SILERO_SAMPLE_RATE) / SILERO_CHUNK_SIZE))
    pre_roll_chunks: list[np.ndarray] = []
    recorded_chunks: list[np.ndarray] = []

    barge_in_max_chunks = max(1, int((BARGE_IN_TAIL_SECONDS * SILERO_SAMPLE_RATE) / SILERO_CHUNK_SIZE))
    pending_barge_in_chunks: list[np.ndarray] = []
    barge_in_captured_at: Optional[float] = None
    barge_in_leftover_16k = np.array([], dtype=np.float32)

    is_recording = False
    speech_started_at: Optional[float] = None
    leftover_16k = np.array([], dtype=np.float32)
    frame_collector: Optional["FrameCollector"] = None

    def audio_callback(indata, frames, callback_time, status) -> None:
        audio_queue.put(indata[:, 0].copy())

    print_ts(f"Listening automatically for {prompt_label}. Speak when ready. Press Ctrl+C to quit.")

    try:
        sd.check_input_settings(
            device=input_device,
            samplerate=input_sample_rate,
            channels=1,
            dtype="float32",
        )

        with sd.InputStream(
            samplerate=input_sample_rate,
            channels=1,
            dtype="float32",
            device=input_device,
            blocksize=input_block_size,
            callback=audio_callback,
        ):
            while True:
                if robot_speaker is not None and robot_speaker.is_speaking_or_cooling_down():
                    try:
                        while True:
                            gated_block = audio_queue.get_nowait()
                            gated_16k = resample_audio(gated_block, input_sample_rate, SILERO_SAMPLE_RATE)
                            gated_combined = np.concatenate(
                                [barge_in_leftover_16k, gated_16k]
                            ).astype(np.float32, copy=False)
                            gated_usable_len = (len(gated_combined) // SILERO_CHUNK_SIZE) * SILERO_CHUNK_SIZE
                            if gated_usable_len == 0:
                                barge_in_leftover_16k = gated_combined
                                continue
                            gated_chunks = gated_combined[:gated_usable_len].reshape(-1, SILERO_CHUNK_SIZE)
                            barge_in_leftover_16k = gated_combined[gated_usable_len:]
                            for gated_chunk in gated_chunks:
                                pending_barge_in_chunks.append(gated_chunk.astype(np.float32, copy=False))
                                if len(pending_barge_in_chunks) > barge_in_max_chunks:
                                    pending_barge_in_chunks.pop(0)
                            barge_in_captured_at = time.time()
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                    continue

                if pending_barge_in_chunks and barge_in_captured_at is not None:
                    if time.time() - barge_in_captured_at > BARGE_IN_MAX_AGE_SECONDS:
                        pending_barge_in_chunks = []
                        barge_in_captured_at = None

                try:
                    block = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                block_16k = resample_audio(block, input_sample_rate, SILERO_SAMPLE_RATE)
                combined = np.concatenate([leftover_16k, block_16k]).astype(np.float32, copy=False)

                usable_len = (len(combined) // SILERO_CHUNK_SIZE) * SILERO_CHUNK_SIZE
                if usable_len == 0:
                    leftover_16k = combined
                    continue

                chunks = combined[:usable_len].reshape(-1, SILERO_CHUNK_SIZE)
                leftover_16k = combined[usable_len:]

                for chunk in chunks:
                    chunk = chunk.astype(np.float32, copy=False)

                    if not is_recording:
                        pre_roll_chunks.append(chunk.copy())
                        if len(pre_roll_chunks) > pre_roll_max_chunks:
                            pre_roll_chunks.pop(0)
                    else:
                        recorded_chunks.append(chunk.copy())

                    speech_event = vad_iterator(torch.from_numpy(chunk), return_seconds=True)
                    now = time.time()

                    if speech_event:
                        if "start" in speech_event and not is_recording:
                            is_recording = True
                            speech_started_at = now
                            if pending_barge_in_chunks:
                                barge_in_seconds = len(pending_barge_in_chunks) * SILERO_CHUNK_SIZE / SILERO_SAMPLE_RATE
                                recorded_chunks = list(pending_barge_in_chunks) + list(pre_roll_chunks)
                                print_ts(
                                    f"Barge-in detected: prepending ~{barge_in_seconds:.2f}s of audio "
                                    "captured while Ameca was still speaking."
                                )
                            else:
                                recorded_chunks = list(pre_roll_chunks)
                            recorded_chunks.append(chunk.copy())
                            pre_roll_chunks.clear()
                            pending_barge_in_chunks = []
                            barge_in_captured_at = None

                            if camera is not None:
                                frame_collector = FrameCollector(camera)
                                frame_collector.start()

                            print()
                            print_ts("Speech detected. Recording utterance...")

                        if "end" in speech_event and is_recording:
                            utterance_duration = now - (speech_started_at or now)

                            if utterance_duration >= VAD_MIN_UTTERANCE_SECONDS:
                                print_ts("Speech ended. Processing utterance...")
                                raise StopIteration

                            print_ts(
                                f"Speech ended after only {utterance_duration:.2f}s "
                                f"(< {VAD_MIN_UTTERANCE_SECONDS:.2f}s minimum); discarding as "
                                "noise and resuming listening."
                            )
                            is_recording = False
                            speech_started_at = None
                            recorded_chunks = []
                            if frame_collector is not None:
                                frame_collector.stop()
                                frame_collector = None

                    if is_recording and speech_started_at is not None:
                        if now - speech_started_at >= VAD_MAX_UTTERANCE_SECONDS:
                            print_ts("Maximum utterance length reached. Processing utterance...")
                            raise StopIteration

    except StopIteration:
        pass
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print_ts(f"Silero VAD/audio error: {exc}")
        if frame_collector is not None:
            frame_collector.stop()
        return None, []
    finally:
        try:
            vad_iterator.reset_states()
        except Exception:
            pass

    collected_frames = frame_collector.stop() if frame_collector is not None else []

    if not recorded_chunks:
        return None, collected_frames

    audio_16k = np.concatenate(recorded_chunks).astype(np.float32, copy=False)
    return save_audio_to_temp_wav(audio_16k), collected_frames


def ask_user_to_spell_name(
    whisper_model: WhisperModel,
    silero_model,
    input_device: Optional[int] = INPUT_DEVICE,
    robot_speaker: Optional[RobotSpeaker] = None,
    robot_expression: Optional["RobotExpression"] = None,
    session_log: Optional[list[dict]] = None,
) -> Optional[str]:
    spelling_request_text = "To make this conversation better, could you please spell your name for me?"

    print()
    print_ts(spelling_request_text)
    print()

    if robot_speaker:
        speak_with_turn_end_cue(
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            text=spelling_request_text,
            emotion="neutral",
        )
    if session_log is not None:
        session_log.append({
            "role": "assistant",
            "content": spelling_request_text,
            "timestamp": now_ts(),
            "intent": "spell_name_request",
        })

    wav_path, _frames = listen_for_utterance_with_silero_vad(
        input_device=input_device,
        silero_model=silero_model,
        prompt_label="spelled name",
        robot_speaker=robot_speaker,
    )

    if not wav_path:
        return None

    try:
        transcript = transcribe_with_faster_whisper(wav_path, whisper_model)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    print_ts(f"Raw spelling transcript (faster-whisper): {transcript}")

    if session_log is not None and transcript:
        session_log.append({
            "role": "user",
            "content": transcript,
            "timestamp": now_ts(),
            "intent": "spelled_name_response",
        })

    return clean_spelled_name(transcript)


def generate_introduction_response(
    client: Client,
    user_name: str,
) -> str:
    system_prompt = f"""
        You are Ameca, a humanoid social robot in a university laboratory.

        A new user has just introduced themselves.

        Your task:
        - greet the user naturally
        - introduce yourself naturally as Ameca
        - keep it warm and concise
        - keep the response to 1-2 short sentences
        - mention the user's name
        - end with exactly one friendly facial emoji
        - only use this emoji: \U0001F642

        Do not:
        - sound robotic
        - repeat the user's exact words
        - mention prompts or instructions
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"My name is {user_name}"},
            ],
            options={
                "temperature": 0.5,
                "num_predict": 200,
                "num_ctx": 1024,
            },
            stream=False,
        )

        raw_reply = response["message"]["content"]
        reply = normalize_reply(raw_reply, "neutral")

        if "[" in reply or "]" in reply:
            # Same unrendered-placeholder guard as generate_post_warmup_welcome().
            return f"Hello {user_name}. I am Ameca. It is nice to meet you. \U0001F642"

        return reply

    except Exception as exc:
        print_ts(f"Could not generate introduction with LLM: {exc}")
        return f"Hello {user_name}. I am Ameca. It is nice to meet you. \U0001F642"


# =========================
# Participant ID capture
# =========================

def resolve_participant_id(
    cli_participant_id: str,
    robot_speaker: Optional[RobotSpeaker] = None,
    robot_expression: Optional["RobotExpression"] = None,
    session_log: Optional[list[dict]] = None,
) -> str:
    participant_id = str(cli_participant_id or "").strip()
    if participant_id:
        print_ts(f"Using participant ID from --participant_id: {participant_id}")
        return participant_id

    request_text = "Please enter the participant ID on the keyboard before we begin."
    if robot_speaker is not None:
        speak_with_turn_end_cue(
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            text=request_text,
            emotion="neutral",
        )
    if session_log is not None:
        session_log.append({
            "role": "assistant",
            "content": request_text,
            "timestamp": now_ts(),
            "intent": "participant_id_request",
        })

    print()
    print_ts("No --participant_id was provided.")
    try:
        participant_id = input("Participant ID: ").strip()
    except EOFError:
        participant_id = ""

    if not participant_id:
        participant_id = "unknown"
        print_ts(
            "No participant ID entered; using 'unknown'. Sessions will still be "
            "saved, but continuity across runs is not guaranteed with this "
            "fallback ID."
        )
    else:
        print_ts(f"Using participant ID: {participant_id}")

    return participant_id


def prompt_for_user_name(
    client: Client,
    whisper_model: WhisperModel,
    silero_model,
    input_device: Optional[int] = INPUT_DEVICE,
    robot_speaker: Optional[RobotSpeaker] = None,
    robot_expression: Optional["RobotExpression"] = None,
    participant_id: str = "",
    session_log: Optional[list[dict]] = None,
) -> ParticipantSessionInfo:

    users = load_users()

    participant_id = str(participant_id or "").strip()
    user_key = slugify_name(participant_id) if participant_id else None

    is_new_user = not (user_key and user_key in users)

    if not is_new_user:
        # ---- Case 1: returning in THIS script's own experiment store. ----
        user_profile = users[user_key]
        user_profile["last_seen"] = now_ts()
        lesson_state = LessonState.from_json(user_profile.get("lesson_state"))
        save_users(users)

        final_name = str(user_profile.get("name", "Guest")).strip() or "Guest"
        print_ts(f"Recognized returning participant '{participant_id}' -> stored name: {final_name}")

        user_profile = ensure_user_has_conversation_summary(user_key, user_profile)

        sessions_completed = int(user_profile.get("sessions_completed", 0))
        session_number = sessions_completed + 1
        explanation_level = resolve_explanation_level(session_number)

        print_ts(f"Welcome back, {user_profile['name']}.")
        if user_profile.get("conversation_summary"):
            print_ts("Starting from previous conversation summary.")
        introduction_reply = generate_returning_user_response(
            client=client,
            user_profile=user_profile,
        )

        print_ts(f"Assistant: {introduction_reply}")
        print()

        return ParticipantSessionInfo(
            user_key=user_key,
            user_profile=user_profile,
            intro_reply=introduction_reply,
            session_number=session_number,
            explanation_level=explanation_level,
            is_first_after_warmup=False,
            needs_recap=session_number >= 2,
            lesson_state = lesson_state,
        )

    # ---- Case 2: new here, but a completed warm-up session exists. ----
    warm_up_session = load_warm_up_session(participant_id) if participant_id else None

    if warm_up_session is not None:
        spelt_name = str(warm_up_session.get("display_name") or "").strip() or "Guest"
        warm_up_path = str(warm_up_session.get("_source_path") or "")
        print_ts(
            f"Found warm-up session for participant '{participant_id}' at "
            f"{warm_up_path}; treating as returning, using spelt name '{spelt_name}'."
        )

        if user_key is None:
            user_key = slugify_name(participant_id or spelt_name)

        users[user_key] = {
            "name": spelt_name,
            "participant_id": participant_id,
            "created_at": now_ts(),
            "last_seen": now_ts(),
            "session_files": [],
            "conversation_summary": "",
            "warm_up_session_path": warm_up_path,
            "warm_up_display_name": spelt_name,
            "sessions_completed": 0,
            "session_history": [],
        }
        save_users(users)
        user_profile = users[user_key]

        welcome_text = generate_post_warmup_welcome(client=client, display_name=spelt_name)

        print_ts(f"Assistant: {welcome_text}")
        print()

        lesson_state = LessonState()

        return ParticipantSessionInfo(
            user_key=user_key,
            user_profile=user_profile,
            intro_reply=welcome_text,
            session_number=1,
            explanation_level=resolve_explanation_level(1),
            is_first_after_warmup=True,
            lesson_state = lesson_state,
            needs_recap=False,
        )

    # ---- Case 3: new to both stores -- original spelling flow. ----
    print()
    print_ts("New participant (no warm-up session found). Greeting user and requesting spelled name.")
    print()

    if robot_speaker:
        name_request_text = "Hello there! Welcome to our first session on A.I. and Robotics."
        speak_with_turn_end_cue(
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            text=name_request_text,
            emotion="neutral",
        )
        if session_log is not None:
            session_log.append({
                "role": "assistant",
                "content": name_request_text,
                "timestamp": now_ts(),
                "intent": "name_request",
            })

    spelled_name = None

    for attempt in range(2):
        spelled_name = ask_user_to_spell_name(
            whisper_model=whisper_model,
            silero_model=silero_model,
            input_device=input_device,
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            session_log=session_log,
        )

        if spelled_name and not looks_like_invalid_name(spelled_name):
            break

        print_ts(f"I heard '{spelled_name or 'nothing'}', but that does not sound like a spelled name.")
        if robot_speaker and attempt == 0:
            retry_text = "I might have misheard, could you please spell your name again?"
            speak_with_turn_end_cue(
                robot_speaker=robot_speaker,
                robot_expression=robot_expression,
                text=retry_text,
                emotion="neutral",
            )
            if session_log is not None:
                session_log.append({
                    "role": "assistant",
                    "content": retry_text,
                    "timestamp": now_ts(),
                    "intent": "name_retry_request",
                })

    if not spelled_name or looks_like_invalid_name(spelled_name):
        spelled_name = "Guest"

    final_name = correct_spelled_name_with_known_users(
        spelled_name=spelled_name,
    ) or spelled_name

    print_ts(f"Using spelled name: {final_name}")
    print_ts(f"Detected name: {final_name}")

    # Storage key is participant-id-based whenever a participant_id was
    # given (kept stable across runs, immune to ASR/spelling drift);
    # falls back to a name-derived key only when no participant_id was
    # provided at all.
    if user_key is None:
        user_key = slugify_name(final_name)
        print_ts(f"No participant ID given; using name-derived storage key: {user_key}")
    else:
        print_ts(f"Using participant ID as storage key: {user_key} (spelled name: {final_name})")

    users[user_key] = {
        "name": final_name,
        "participant_id": participant_id,
        "created_at": now_ts(),
        "last_seen": now_ts(),
        "session_files": [],
        "conversation_summary": "",
        "warm_up_session_path": None,
        "warm_up_display_name": None,
        "sessions_completed": 0,
        "session_history": [],
    }

    save_users(users)
    user_profile = users[user_key]

    print_ts(f"Nice to meet you, {final_name}.")
    introduction_reply = generate_introduction_response(
        client=client,
        user_name=user_profile["name"],
    )

    print_ts(f"Assistant: {introduction_reply}")
    print()

    lesson_state = LessonState()

    return ParticipantSessionInfo(
        user_key=user_key,
        user_profile=user_profile,
        intro_reply=introduction_reply,
        session_number=1,
        explanation_level=resolve_explanation_level(1),
        is_first_after_warmup=False,
        lesson_state = lesson_state,
        needs_recap=False,
    )

def ask_recap_consent(
    whisper_model: WhisperModel,
    silero_model,
    input_device: Optional[int],
    robot_speaker: RobotSpeaker,
    robot_expression: Optional["RobotExpression"],
    disable_expression: bool,
    session_log: list[dict],
    history: list[dict],
) -> bool:
    prompt_text = normalize_reply(
        "Before we start, would you like me to ask three quick questions "
        "to refresh your memory from last time? You can say yes or no.",
        "neutral",
        1.0,
    )

    speak_with_turn_end_cue(
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        text=prompt_text,
        emotion="neutral",
        disable_expression=disable_expression,
    )

    session_log.append({
        "role": "assistant",
        "content": prompt_text,
        "timestamp": now_ts(),
        "intent": "recap_consent_request",
    })

    history.append({"role": "assistant", "content": prompt_text})

    wav_path, _frames = listen_for_utterance_with_silero_vad(
        input_device=input_device,
        silero_model=silero_model,
        prompt_label="recap consent",
        robot_speaker=robot_speaker,
    )

    answer_text = ""
    if wav_path:
        try:
            answer_text = transcribe_with_faster_whisper(wav_path, whisper_model)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    session_log.append({
        "role": "user",
        "content": answer_text or "(no answer captured)",
        "timestamp": now_ts(),
        "intent": "recap_consent_response",
    })
    if answer_text:
        history.append({"role": "user", "content": answer_text})

    return parse_yes_no(answer_text, default=False)

def run_session_recap_qa(
    client: Client,
    whisper_model: WhisperModel,
    silero_model,
    input_device: Optional[int],
    robot_speaker: RobotSpeaker,
    robot_expression: Optional["RobotExpression"],
    disable_expression: bool,
    session_log: list[dict],
    history: list[dict],
    user_profile: dict,
    explanation_level: str,
) -> None:
    
    session_history = user_profile.get("session_history", []) or []
    recap_data = generate_session_recap_and_questions(client, session_history, explanation_level)
    recap_text = recap_data.get("recap") or "Let's briefly recap what we covered last time."
    questions = (recap_data.get("questions") or [])[:3]
    recap_context = build_prior_sessions_text(session_history)

    intro = normalize_reply(recap_text, "neutral", 1.0)
    speak_with_turn_end_cue(
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        text=intro,
        emotion="neutral",
        disable_expression=disable_expression,
    )
    session_log.append({
        "role": "assistant",
        "content": intro,
        "timestamp": now_ts(),
        "intent": "session_recap",
    })
    history.append({"role": "assistant", "content": intro})

    for index, question in enumerate(questions, start=1):
        question_text = normalize_reply(question, "neutral", 1.0)
        speak_with_turn_end_cue(
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            text=question_text,
            emotion="neutral",
            disable_expression=disable_expression,
        )
        session_log.append({
            "role": "assistant",
            "content": question_text,
            "timestamp": now_ts(),
            "intent": f"recap_question_{index}",
        })
        history.append({"role": "assistant", "content": question_text})

        wav_path, _frames = listen_for_utterance_with_silero_vad(
            input_device=input_device,
            silero_model=silero_model,
            prompt_label=f"recap answer {index}",
            robot_speaker=robot_speaker,
        )

        answer_text = ""
        if wav_path:
            try:
                answer_text = transcribe_with_faster_whisper(wav_path, whisper_model)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if not answer_text:
            answer_text = "(no answer captured)"

        session_log.append({
            "role": "user",
            "content": answer_text,
            "timestamp": now_ts(),
            "intent": f"recap_answer_{index}",
        })
        history.append({"role": "user", "content": answer_text})

        feedback = generate_recap_answer_feedback(client, question, answer_text, recap_context=recap_context)
        feedback_text = normalize_reply(feedback, "neutral", 1.0)
        speak_with_turn_end_cue(
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            text=feedback_text,
            emotion="neutral",
            disable_expression=disable_expression,
        )
        session_log.append({
            "role": "assistant",
            "content": feedback_text,
            "timestamp": now_ts(),
            "intent": f"recap_feedback_{index}",
        })
        history.append({"role": "assistant", "content": feedback_text})

TOPIC_PROMPT_QUESTION = (
    "Today I can teach you one beginner topic: machine learning, how robots sense "
    "the world, or large language models. I suggest we start with machine "
    "learning. Ready?"
)


def ask_topic_choice_question(
    robot_speaker: RobotSpeaker,
    robot_expression: Optional["RobotExpression"],
    disable_expression: bool,
    session_log: list[dict],
    history: list[dict],
) -> None:
    
    question_text = normalize_reply(TOPIC_PROMPT_QUESTION, "neutral", 1.0)
    speak_with_turn_end_cue(
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        text=question_text,
        emotion="neutral",
        disable_expression=disable_expression,
    )
    session_log.append({
        "role": "assistant",
        "content": question_text,
        "timestamp": now_ts(),
        "intent": "topic_choice_prompt",
    })
    history.append({"role": "assistant", "content": question_text})


# =========================
# JSON helpers
# =========================

def safe_json_extract(raw: str):
    if not raw:
        return None

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    if start < 0:
        return None

    candidate = raw[start:].strip()

    last_close = candidate.rfind("}")
    if last_close > 0:
        maybe = candidate[: last_close + 1]
        try:
            return json.loads(maybe)
        except Exception:
            pass

    repaired = candidate
    reason_pos = repaired.rfind('"reason"')
    if reason_pos != -1:
        before_reason = repaired[:reason_pos].rstrip().rstrip(",")
        repaired = before_reason + ', "reason": "truncated"}'

    if repaired.count('"') % 2 != 0:
        repaired += '"'

    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    if close_braces < open_braces:
        repaired += "}" * (open_braces - close_braces)

    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if close_brackets < open_brackets:
        repaired += "]" * (open_brackets - close_brackets)

    if repaired.count('"') % 2 != 0:
        repaired += '"'

    try:
        return json.loads(repaired)
    except Exception:
        pass

    try:
        dominant_match = re.search(r'"dominant_emotion"\s*:\s*"([^"{}]+)"', candidate)
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', candidate)
        scores_match = re.search(r'"scores"\s*:\s*\{(.*?)\}', candidate, flags=re.DOTALL)

        if not scores_match:
            return None

        scores_text = scores_match.group(1)
        scores: dict[str, float] = {}
        for emotion in EKMAN_EMOTIONS:
            match = re.search(rf'"{emotion}"\s*:\s*([0-9.]+)', scores_text)
            if match:
                scores[emotion] = float(match.group(1))

        if not scores:
            return None

        dominant = dominant_match.group(1).strip().lower() if dominant_match else max(scores.items(), key=lambda item: item[1])[0]
        confidence = float(confidence_match.group(1)) if confidence_match else max(scores.values())

        return {
            "dominant_emotion": dominant,
            "confidence": confidence,
            "scores": scores,
            "reason": "recovered from partial JSON",
        }
    except Exception:
        return None


# =========================
# Self-RAG helpers
# =========================

SELF_RAG_TRIGGER_PHRASES = [
    "robotics research laboratory",
    "robotics research lab",
    "rrlab",
    "rr lab",
]

def mentions_self_rag_trigger_keyword(text: str) -> bool:
    """
    True only if the user's utterance explicitly names the lab, using one
    of SELF_RAG_TRIGGER_PHRASES.

    Case-insensitive; tolerant of "RRLab" / "RR Lab" / "RR-Lab" spacing
    and punctuation variants.
    """
    normalized = re.sub(r"[\-_]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for phrase in SELF_RAG_TRIGGER_PHRASES:
        phrase_normalized = phrase.lower()
        if phrase_normalized in normalized:
            return True
        # Also match "rr lab" / "rrlab" run together without any space.
        if phrase_normalized.replace(" ", "") in normalized.replace(" ", ""):
            return True

    return False


def self_rag_disabled_context(query: str, reason: str, error: Optional[str] = None) -> SelfRAGContext:
    return SelfRAGContext(
        available=False,
        used=False,
        query=query,
        context_text="",
        sources=[],
        reason=reason,
        error=error,
        trigger="none",
    )


def clean_knowledge_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_knowledge_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            print_ts(f"Skipping PDF because pypdf is not installed: {path} ({exc})")
            return ""

        try:
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return clean_knowledge_text("\n\n".join(pages))
        except Exception as exc:
            print_ts(f"Could not read PDF knowledge file {path}: {exc}")
            return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return clean_knowledge_text(file.read())
    except Exception as exc:
        print_ts(f"Could not read knowledge file {path}: {exc}")
        return ""


def iter_knowledge_files(kb_dir: str) -> list[str]:
    if not os.path.isdir(kb_dir):
        return []

    paths: list[str] = []
    for root, _, files in os.walk(kb_dir):
        for filename in files:
            path = os.path.join(root, filename)
            ext = os.path.splitext(path)[1].lower()
            if ext in SELF_RAG_SUPPORTED_EXTENSIONS:
                paths.append(path)
    return sorted(paths)


def chunk_text(text: str, chunk_size: int = SELF_RAG_CHUNK_SIZE, overlap: int = SELF_RAG_CHUNK_OVERLAP) -> list[str]:
    text = clean_knowledge_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        if end < len(text):
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind("; "))
            if last_break > int(chunk_size * 0.55):
                chunk = chunk[: last_break + 1].strip()
                end = start + last_break + 1

        if chunk:
            chunks.append(chunk)

        start = max(end - overlap, start + step)

    return chunks


def stable_chunk_id(path: str, chunk: str, index: int) -> str:
    raw = f"{path}:{index}:{chunk}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:32]


def resolve_scrape_script_path() -> Optional[str]:
    candidates = [
        SELF_RAG_SCRAPE_SCRIPT,
        os.path.join(os.getcwd(), SELF_RAG_SCRAPE_SCRIPT),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), SELF_RAG_SCRAPE_SCRIPT),
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return None


def run_rrlab_scraper() -> bool:
    script_path = resolve_scrape_script_path()

    if not script_path:
        print_ts(
            "Could not find scrape.py. Put scrape.py in the same folder as this script "
            "or set SELF_RAG_SCRAPE_SCRIPT=/full/path/to/scrape.py."
        )
        return False

    print_ts(f"Running RRLab scraper: {script_path}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path) or os.getcwd(),
            env=env,
            check=False,
        )

        if completed.returncode != 0:
            print_ts(f"RRLab scraper failed with exit code {completed.returncode}.")
            return False

        print_ts("RRLab scraper finished successfully.")
        return True

    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print_ts(f"RRLab scraper could not run: {exc}")
        return False


def get_ollama_embedding(
    client: Client,
    text: str,
    model: str = SELF_RAG_EMBED_MODEL,
) -> Optional[list[float]]:
    text = (text or "").strip()
    if not text:
        return None

    if hasattr(client, "embeddings"):
        try:
            response = client.embeddings(model=model, prompt=text)
            embedding = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
            if embedding:
                return [float(value) for value in embedding]
        except Exception as exc:
            print_ts(f"Ollama client.embeddings() call failed (model={model}): {exc}")

    if hasattr(client, "embed"):
        try:
            response = client.embed(model=model, input=text)
            embeddings = response.get("embeddings") if isinstance(response, dict) else getattr(response, "embeddings", None)
            if embeddings:
                return [float(value) for value in embeddings[0]]
        except Exception as exc:
            print_ts(f"Ollama client.embed() call failed (model={model}): {exc}")

    return None


def get_ollama_embeddings_batch(
    client: Client,
    texts: list[str],
    model: str = SELF_RAG_EMBED_MODEL,
) -> list[Optional[list[float]]]:
    return [get_ollama_embedding(client, text, model=model) for text in texts]


def rebuild_self_rag_collection(store: SelfRAGStore) -> SelfRAGStore:
    if not store.enabled or store.ollama_client is None:
        print_ts("Self-RAG is not enabled; nothing to rebuild.")
        return store

    try:
        import chromadb
    except Exception as exc:
        print_ts(f"Cannot rebuild Self-RAG collection; chromadb import failed: {exc}")
        return store

    try:
        chroma_client = chromadb.PersistentClient(path=SELF_RAG_DB_DIR)
        try:
            chroma_client.delete_collection(name=SELF_RAG_COLLECTION)
            print_ts(f"Deleted existing Self-RAG collection '{SELF_RAG_COLLECTION}'.")
        except Exception as delete_exc:
            print_ts(f"No existing collection to delete (or delete failed): {delete_exc}")

        collection = chroma_client.get_or_create_collection(
            name=SELF_RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        new_store = SelfRAGStore(
            enabled=True,
            collection=collection,
            ollama_client=store.ollama_client,
            embed_model=store.embed_model,
        )

        if resolve_scrape_script_path():
            print_ts("Rebuilding Self-RAG index via scrape.py...")
            if not run_rrlab_scraper():
                print_ts("scrape.py failed; falling back to local knowledge_base indexing.")
                index_self_rag_knowledge(new_store)
        else:
            index_self_rag_knowledge(new_store)

        print_ts(f"Self-RAG collection rebuilt. chunks={collection.count()}.")
        return new_store

    except Exception as exc:
        print_ts(f"Self-RAG rebuild failed: {exc}")
        return store


def init_self_rag_store(client: Client) -> SelfRAGStore:
    if not SELF_RAG_ENABLED:
        print_ts("Self-RAG disabled by SELF_RAG_ENABLED=0.")
        return SelfRAGStore(enabled=False, error="Self-RAG disabled.")

    try:
        import chromadb
    except Exception as exc:
        print_ts(
            "Self-RAG dependencies missing. Install with: "
            "pip install chromadb pypdf"
        )
        return SelfRAGStore(enabled=False, error=str(exc))

    try:
        os.makedirs(SELF_RAG_DB_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=SELF_RAG_DB_DIR)
        collection = chroma_client.get_or_create_collection(
            name=SELF_RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        probe_embedding = get_ollama_embedding(client, "self-rag startup check", model=SELF_RAG_EMBED_MODEL)
        if probe_embedding is None:
            error_msg = (
                f"Could not get a test embedding from Ollama model '{SELF_RAG_EMBED_MODEL}'. "
                f"Make sure it is pulled, e.g.: ollama pull {SELF_RAG_EMBED_MODEL}"
            )
            print_ts(f"Self-RAG initialization failed: {error_msg}")
            return SelfRAGStore(enabled=False, error=error_msg)

        existing_count = collection.count()
        needs_reindex = SELF_RAG_REINDEX_ON_START

        if existing_count > 0:
            try:
                collection.query(query_embeddings=[probe_embedding], n_results=1)
            except Exception as dim_exc:
                dim_exc_text = str(dim_exc)
                if "dimension" in dim_exc_text.lower():
                    print_ts(
                        f"Self-RAG collection '{SELF_RAG_COLLECTION}' was built with a "
                        f"different embedding dimension than '{SELF_RAG_EMBED_MODEL}' "
                        f"currently produces ({dim_exc_text}). This would otherwise fail "
                        f"on every single retrieval this session. Recreating the "
                        f"collection from scratch."
                    )
                    try:
                        chroma_client.delete_collection(name=SELF_RAG_COLLECTION)
                    except Exception as delete_exc:
                        print_ts(f"Could not delete the stale Self-RAG collection: {delete_exc}")
                    collection = chroma_client.get_or_create_collection(
                        name=SELF_RAG_COLLECTION,
                        metadata={"hnsw:space": "cosine"},
                    )
                    existing_count = 0
                    needs_reindex = True
                    print_ts(
                        "Stale Self-RAG collection recreated empty at the correct "
                        f"dimension for '{SELF_RAG_EMBED_MODEL}'. It will be reindexed "
                        "now (scrape.py if available, otherwise the local "
                        f"'{SELF_RAG_KB_DIR}' folder)."
                    )
                else:
                    print_ts(
                        f"Self-RAG startup sanity query failed for an unrelated reason "
                        f"(collection left as-is): {dim_exc_text}"
                    )

        store = SelfRAGStore(
            enabled=True,
            collection=collection,
            ollama_client=client,
            embed_model=SELF_RAG_EMBED_MODEL,
        )

        if needs_reindex or existing_count == 0:
            if resolve_scrape_script_path():
                print_ts("Self-RAG collection needs (re)indexing; attempting scrape.py first...")
                if not run_rrlab_scraper():
                    print_ts("scrape.py failed or unavailable; falling back to local knowledge_base indexing.")
                    index_self_rag_knowledge(store)
            else:
                index_self_rag_knowledge(store)

        count = collection.count()

        if count == 0 and SELF_RAG_AUTO_SCRAPE_ON_EMPTY:
            print_ts("Self-RAG collection is empty. Running scrape.py to build the RRLab website index...")
            run_rrlab_scraper()
            count = collection.count()

        if count == 0:
            print_ts(
                "Self-RAG collection is empty. Run 'python scrape.py' first, "
                "or type '/rrlab crawl' while the app is running."
            )

        print_ts(f"Self-RAG ready. Collection='{SELF_RAG_COLLECTION}', chunks={count}.")
        return store
    except Exception as exc:
        print_ts(f"Self-RAG initialization failed: {exc}")
        return SelfRAGStore(enabled=False, error=str(exc))


def index_self_rag_knowledge(store: SelfRAGStore) -> None:
    if not store.enabled or store.collection is None or store.ollama_client is None:
        return

    paths = iter_knowledge_files(SELF_RAG_KB_DIR)
    if not paths:
        print_ts(
            f"Self-RAG knowledge folder '{SELF_RAG_KB_DIR}' has no supported files. "
            "Create the folder and add .txt, .md, .pdf, .json, .csv, .py, or .html files."
        )
        return

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for path in paths:
        text = read_knowledge_file(path)
        if len(text) < SELF_RAG_MIN_CONTEXT_CHARS:
            continue

        rel_path = os.path.relpath(path, SELF_RAG_KB_DIR)
        for index, chunk in enumerate(chunk_text(text)):
            if len(chunk) < SELF_RAG_MIN_CONTEXT_CHARS:
                continue
            ids.append(stable_chunk_id(rel_path, chunk, index))
            docs.append(chunk)
            metas.append({
                "source": rel_path,
                "chunk_index": index,
                "source_path": path,
                "indexed_at": now_ts(),
            })

    if not docs:
        print_ts("Self-RAG found knowledge files, but no usable text chunks were extracted.")
        return

    raw_embeddings = get_ollama_embeddings_batch(store.ollama_client, docs, model=store.embed_model)

    kept_ids: list[str] = []
    kept_docs: list[str] = []
    kept_metas: list[dict] = []
    kept_embeddings: list[list[float]] = []
    failed_count = 0

    for chunk_id, doc, meta, embedding in zip(ids, docs, metas, raw_embeddings):
        if embedding is None:
            failed_count += 1
            continue
        kept_ids.append(chunk_id)
        kept_docs.append(doc)
        kept_metas.append(meta)
        kept_embeddings.append(embedding)

    if failed_count:
        print_ts(f"Self-RAG: {failed_count} chunk(s) could not be embedded via Ollama and were skipped.")

    if not kept_docs:
        print_ts("Self-RAG: no chunks could be embedded; nothing was indexed.")
        return

    store.collection.upsert(ids=kept_ids, documents=kept_docs, metadatas=kept_metas, embeddings=kept_embeddings)
    print_ts(f"Self-RAG indexed/updated {len(kept_docs)} chunks from {len(paths)} files.")


def normalize_self_rag_query_text(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "ashita ashuk": "ashita ashok",
        "ashita ashook": "ashita ashok",
        "ashuk": "ashok",
        "ashook": "ashok",
        "robots amaker": "robot ameca",
        "robot amaker": "robot ameca",
        "amaker": "ameca",
        "emeka": "ameca",
        "robotic lab": "robotics research lab",
        "robotics lab": "robotics research lab",
    }
    for wrong, right in replacements.items():
        lowered = lowered.replace(wrong, right)
    return lowered


def extract_person_lookup_name(text: str) -> Optional[str]:
    cleaned = normalize_self_rag_query_text(text)
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    patterns = [
        r"\bwho is ([a-z]+(?:\s+[a-z]+){0,2})\b",
        r"\bdo you know who is ([a-z]+(?:\s+[a-z]+){0,2})\b",
        r"\bdo you know about ([a-z]+(?:\s+[a-z]+){0,2})\b",
        r"\babout ([a-z]+(?:\s+[a-z]+){0,2}) in the robotics research lab\b",
    ]

    stop_words = {
        "the", "a", "an", "robot", "robots", "robotic", "robotics",
        "lab", "laboratory", "research", "group", "people", "person",
    }

    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        name = match.group(1).strip()
        words = [w for w in name.split() if w not in stop_words]
        if words:
            return " ".join(words)

    return None


def candidate_contains_person(candidate: dict[str, Any], person_name: str) -> bool:
    if not person_name:
        return True

    normalized_person = normalize_self_rag_query_text(person_name).strip()
    tokens = [t for t in normalized_person.split() if len(t) > 2]
    haystack = " ".join([
        str(candidate.get("text") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("source") or ""),
    ]).lower()

    if normalized_person and normalized_person in haystack:
        return True
    if normalized_person and normalized_person.replace(" ", "-") in haystack:
        return True
    if len(tokens) == 1:
        return bool(re.search(rf"\b{re.escape(tokens[0])}\b", haystack))
    return all(re.search(rf"\b{re.escape(token)}\b", haystack) for token in tokens)


def infer_self_rag_category(query: str) -> Optional[str]:
    q = normalize_self_rag_query_text(query)
    if any(token in q for token in ["ashita", "ashita ashok", "professor", "head", "leader", "leads", "staff", "research associate", "who is"]):
        return "staff"
    if any(token in q for token in ["project", "current project", "sembai", "senna", "casrew", "zukunftbau", "znt"]):
        return "project"
    if any(token in q for token in ["robot", "robots", "ameca", "emah", "ravon", "robin", "unimog", "carl"]):
        return "robot"
    if any(token in q for token in ["publication", "paper", "textbook", "dissertation"]):
        return "publication"
    if any(token in q for token in ["research area", "what does rrlab research", "rrlab research", "researches"]):
        return "research_area"
    return None


def rewrite_self_rag_query(query: str) -> str:
    q = normalize_self_rag_query_text(query).strip()
    if "who leads" in q or ("head" in q and "laboratory" in q):
        return "head of laboratory professor robotics research lab"
    if "current projects" in q:
        return "current RRLab projects SEmbAI SENNA CASREW ZukunftBau ZNT project"
    if "what is ameca" in q or "what is emah" in q:
        return "Ameca Emah humanoid robot RRLab student companion"
    if "what robots" in q:
        return "RRLab robots Ameca RAVON Robin Unimog CARL robot platforms"
    if "ashita" in q:
        return "M. Sc. Ashita Ashok Ameca Robothespian human robot interaction trust expectation alignment"
    if "what does rrlab research" in q or "rrlab research" in q:
        return "RRLab research areas control architectures outdoor robots indoor robots humanoid robots simulation projects"
    return query


def self_rag_hybrid_score(candidate: dict[str, Any], inferred_category: Optional[str], query: str) -> float:
    distance = float(candidate.get("distance", 1.0))
    score = 1.0 - distance
    category = candidate.get("category")
    title = str(candidate.get("title") or "").lower()
    source = str(candidate.get("source") or "").lower()
    q = query.lower()

    try:
        priority = int(candidate.get("priority", 0) or 0)
    except Exception:
        priority = 0

    if inferred_category and category == inferred_category:
        score += 0.12
    if priority >= 100:
        score += 0.12
    elif priority >= 90:
        score += 0.08
    elif priority >= 80:
        score += 0.04
    if "/former-staff-members/" in source:
        score -= 0.14
    if category == "conference":
        score -= 0.15
    if inferred_category == "robot" and category == "staff":
        score -= 0.18
    if inferred_category == "robot" and category == "conference":
        score -= 0.25
    if inferred_category == "project" and category not in {"project", "general"}:
        score -= 0.12
    if inferred_category == "staff" and category not in {"staff", "general"}:
        score -= 0.12
    if "head of laboratory" in q or "who leads" in q:
        if "head of the laboratory" in title or "head-of-the-laboratory" in source:
            score += 0.25
        if "technical staff" in title:
            score -= 0.25
        if "research associates" in title:
            score -= 0.12
    if "ashita" in q and ("ashita ashok" in title or "ashita-ashok" in source):
        score += 0.25
    if ("ameca" in q or "emah" in q) and ("/robots/ameca" in source or title == "emah"):
        score += 0.25
    if "project" in q:
        if "/research/projects/" in source and "/finished-projects/" not in source:
            score += 0.10
        if "/finished-projects/" in source and "current" in q:
            score -= 0.18
    return score


def retrieve_self_rag_candidates(store: SelfRAGStore, query: str, top_k: int = SELF_RAG_TOP_K) -> list[dict[str, Any]]:
    
    if not store.enabled or store.collection is None or store.ollama_client is None:
        return []
    if not query.strip():
        return []

    normalized_query = normalize_self_rag_query_text(query)
    rewritten_query = rewrite_self_rag_query(normalized_query)
    inferred_category = infer_self_rag_category(rewritten_query)
    person_lookup_name = extract_person_lookup_name(normalized_query)

    try:
        query_embedding = get_ollama_embedding(store.ollama_client, rewritten_query, model=store.embed_model)
        if query_embedding is None:
            print_ts(f"Self-RAG: could not embed query via Ollama model '{store.embed_model}'; skipping retrieval.")
            return []

        def run_query(where_filter: Optional[dict] = None) -> list[dict[str, Any]]:
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": max(1, top_k),
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter

            result = store.collection.query(**kwargs)
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            rows: list[dict[str, Any]] = []
            for doc, meta, distance in zip(docs, metas, distances):
                meta = meta or {}
                rows.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "title": meta.get("title"),
                    "kind": meta.get("kind"),
                    "category": meta.get("category"),
                    "priority": meta.get("priority"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance": float(distance),
                })
            return rows

        rows: list[dict[str, Any]] = []
        if inferred_category:
            rows.extend(run_query({"category": inferred_category}))
        rows.extend(run_query(None))

        rows = [
            row for row in rows
            if float(row.get("distance", 1.0)) <= SELF_RAG_MAX_DISTANCE
        ]

        if SELF_RAG_PERSON_LOOKUP_STRICT and person_lookup_name:
            rows = [
                row for row in rows
                if candidate_contains_person(row, person_lookup_name)
            ]

        best_by_source: dict[str, dict[str, Any]] = {}
        for row in rows:
            row["hybrid_score"] = self_rag_hybrid_score(row, inferred_category, rewritten_query)

            if row["hybrid_score"] < SELF_RAG_MIN_HYBRID_SCORE:
                continue

            source = str(row.get("source") or "")
            if source not in best_by_source or row["hybrid_score"] > best_by_source[source]["hybrid_score"]:
                best_by_source[source] = row

        reranked = sorted(
            best_by_source.values(),
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )
        return reranked[:SELF_RAG_FINAL_TOP_K]

    except Exception as exc:
        print_ts(f"Self-RAG retrieval failed: {exc}")
        return []


def get_source_page_text(
    store: SelfRAGStore,
    source: str,
    center_chunk_index: Optional[int] = None,
    max_chars: int = 3500,
) -> str:
    if not source or not store.enabled or store.collection is None:
        return ""

    try:
        result = store.collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        print_ts(f"Self-RAG source expansion failed for {source}: {exc}")
        return ""

    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    rows: list[tuple[int, str]] = []

    for doc, meta in zip(docs, metas):
        meta = meta or {}
        try:
            chunk_index = int(meta.get("chunk_index", 0) or 0)
        except Exception:
            chunk_index = 0
        cleaned = clean_knowledge_text(str(doc or ""))
        if cleaned:
            rows.append((chunk_index, cleaned))

    if not rows:
        return ""

    rows.sort(key=lambda item: item[0])

    if center_chunk_index is not None:
        try:
            center = int(center_chunk_index)
            rows.sort(key=lambda item: (abs(item[0] - center), item[0]))
        except Exception:
            pass

    selected: list[tuple[int, str]] = []
    used = 0
    for chunk_index, chunk in rows:
        if used >= max_chars:
            break
        clipped = chunk[: max(0, max_chars - used)]
        if clipped:
            selected.append((chunk_index, clipped))
            used += len(clipped) + 2

    selected.sort(key=lambda item: item[0])
    return "\n".join(f"[chunk {idx}] {chunk}" for idx, chunk in selected)


def context_has_placeholder_risk(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ["[name]", "professor [", "dr. [", "dr [", "unknown name"])


def generate_grounded_self_rag_answer(
    client: Client,
    user_text: str,
    self_rag_context: SelfRAGContext,
    lesson_state: Optional["LessonState"] = None,
    emotion: str = "neutral",
    confidence: float = 1.0,
) -> Optional[GeneratedReply]:
    if not self_rag_context or not self_rag_context.used or not self_rag_context.context_text.strip():
        return None

    lesson_block = lesson_state.to_prompt_block() if lesson_state else ""

    prompt = f"""
        You are Ameca, answering a factual question using only the retrieved local lab knowledge below.

        {lesson_block}

        Important disambiguation:
        - "Ameca" is your own name and physical identity (the robot).
        - "EMAH" refers to a research software pipeline that runs on you, NOT your name.

        User question:
        {user_text}

        Retrieved local lab knowledge:
        {self_rag_context.context_text}

        Instructions:
        - Answer using the retrieved knowledge while keeping the ongoing lesson context in mind.
        - If the exact name, role, or project fact is missing, state that you could not verify it from lab knowledge.
        - Keep the answer to 1-2 short sentences.
        - "emoji" is your own facial expression for THIS answer (e.g. \U0001F60A \U0001F622 \U0001F620 \U0001F628 \U0001F62E \U0001F922 \U0001F642).

        Return JSON only:
        {{
          "reply": "answer without emoji",
          "emoji": "exactly one emoji from the allowed set"
        }}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only and never invent facts."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "num_predict": 200,
                "num_ctx": 8192,
                "repeat_penalty": 1.15,
            },
            stream=False,
        )
        raw = response.get("message", {}).get("content", "")
        data = safe_json_extract(raw)
        if not isinstance(data, dict):
            return None
        reply = str(data.get("reply", "")).strip()
        model_emoji = str(data.get("emoji", "")).strip()
        response_emotion = emoji_to_emotion(model_emoji) or emotion
        if not reply or context_has_placeholder_risk(reply):
            return None
        final_text = normalize_reply(reply, response_emotion, confidence)
        resolved_emotion = resolve_expressive_emotion(response_emotion, confidence)
        return GeneratedReply(text=final_text, response_emotion=resolved_emotion)
    except Exception as exc:
        print_ts(f"Grounded Self-RAG answer generation failed: {exc}")
        return None


def grade_self_rag_context(client: Client, user_text: str, candidates: list[dict[str, Any]]) -> tuple[bool, str]:
    if not candidates:
        return False, "No retrieved knowledge chunks were available."

    compact_context = "\n\n".join(
        f"[{idx + 1}] source={item['source']}\n{limit_text_length(item['text'], 700)}"
        for idx, item in enumerate(candidates[:SELF_RAG_FINAL_TOP_K])
    )

    prompt = f"""
        You are the retrieval judge in a Self-RAG pipeline for a humanoid robot assistant.

        Decide whether the retrieved local knowledge is useful for answering the user's latest message.

        User message:
        {user_text}

        Retrieved local knowledge:
        {compact_context}

        Return JSON only:
        {{
        "use_context": true,
        "reason": "brief reason"
        }}

        Rules:
        - use_context must be true only when the retrieved knowledge directly and specifically answers the message.
        - use_context must be false if the retrieved text is too weak, unrelated, generic, only keyword-matched, or does not contain the named person/entity asked about.
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 200, "num_ctx": 3072},
            stream=False,
        )
        data = safe_json_extract(response.get("message", {}).get("content", ""))
        if not isinstance(data, dict):
            return False, "Retrieval judge returned unparseable output."
        return bool(data.get("use_context", False)), str(data.get("reason", "")).strip()
    except Exception as exc:
        print_ts(f"Self-RAG relevance grading failed: {exc}")
        return False, f"Retrieval judge failed: {exc}"


def build_self_rag_context(
    client: Client,
    store: SelfRAGStore,
    user_text: str,
) -> SelfRAGContext:

    if not store.enabled:
        return self_rag_disabled_context(user_text, "Self-RAG store is not enabled.", store.error)

    if not mentions_self_rag_trigger_keyword(user_text):
        return SelfRAGContext(
            available=True, used=False, query=user_text, context_text="", sources=[],
            reason="No trigger phrase ('robotics research lab' / 'RRLab') mentioned.",
            trigger="none",
        )

    trigger = "keyword"
    candidates = retrieve_self_rag_candidates(store, user_text)
    if not candidates:
        person_lookup_name = extract_person_lookup_name(user_text)
        if person_lookup_name:
            return SelfRAGContext(
                available=True,
                used=False,
                query=user_text,
                context_text="",
                sources=[],
                reason=(
                    f"No direct local knowledge was found for person lookup: {person_lookup_name}. "
                    "Generic RRLab pages were not used because they do not directly answer the question."
                ),
                trigger=trigger,
            )
        result = self_rag_disabled_context(user_text, "No sufficiently relevant local knowledge was retrieved.")
        result.trigger = trigger
        return result


    # When the user explicitly named the lab (keyword trigger, not the LLM
    # gate) and the top retrieved candidate already clears a stricter bar
    # than the normal minimum, skip the extra grading LLM call entirely --
    # intent and relevance are both already strongly confirmed, so grading
    # would just be a redundant round-trip on the critical path.
    top_hybrid_score = candidates[0].get("hybrid_score", 0.0)
    skip_grading = top_hybrid_score >= SELF_RAG_SKIP_GRADE_KEYWORD_MIN_SCORE

    if skip_grading:
        should_use, reason = True, (
            f"Grading skipped: keyword trigger fired with a strong top hybrid "
            f"score ({top_hybrid_score:.2f} >= {SELF_RAG_SKIP_GRADE_KEYWORD_MIN_SCORE:.2f})."
        )
    else:
        should_use, reason = grade_self_rag_context(client, user_text, candidates)
    if not should_use:
        return SelfRAGContext(
            available=True,
            used=False,
            query=user_text,
            context_text="",
            sources=[{k: v for k, v in item.items() if k != "text"} for item in candidates],
            reason=reason or "Retrieved context was judged not useful.",
            trigger=trigger,
        )

    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []
    remaining = SELF_RAG_MAX_CONTEXT_CHARS

    for idx, item in enumerate(candidates, start=1):
        source = str(item.get("source") or "unknown")
        expanded_text = get_source_page_text(
            store=store,
            source=source,
            center_chunk_index=item.get("chunk_index"),
            max_chars=min(3500, max(1200, remaining)),
        )
        text = expanded_text or clean_knowledge_text(item["text"])
        if not text:
            continue
        clipped = text[:remaining]
        if not clipped:
            break
        context_parts.append(f"[Source {idx}: {source}]\n{clipped}")
        source_info = {k: v for k, v in item.items() if k != "text"}
        source_info["expanded_source_context"] = bool(expanded_text)
        sources.append(source_info)
        remaining -= len(clipped)
        if remaining <= 0:
            break

    return SelfRAGContext(
        available=True,
        used=bool(context_parts),
        query=user_text,
        context_text="\n\n".join(context_parts),
        sources=sources,
        reason=reason or "Retrieved context was judged useful.",
        trigger=trigger,
    )


def judge_response_usefulness(
    client: Client,
    user_text: str,
    reply: str,
    self_rag_context: Optional[SelfRAGContext],
) -> dict[str, Any]:
    """
    Prompted analogue of Self-RAG's "ISUSE" token. LOGGING ONLY -- never
    alters or regenerates the reply, so it can't destabilize the turn loop.
    Exists purely so the transcript captures a per-turn usefulness signal
    for thesis analysis.
    """
    if not ENABLE_ISUSE_CHECK:
        return {"enabled": False}

    context_note = (
        "Local lab knowledge was used to ground this reply."
        if self_rag_context and self_rag_context.used
        else "No local lab knowledge was used for this reply."
    )
    prompt = f"""
        Judge whether the ASSISTANT REPLY is a useful answer to the USER MESSAGE.
        {context_note}
        User message: {user_text}
        Assistant reply: {reply}
        Return JSON only: {{"is_useful": true, "reason": "short reason"}}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME, format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 100, "num_ctx": 1024},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return {"enabled": True, "is_useful": None, "reason": "unparseable"}
        return {
            "enabled": True,
            "is_useful": data.get("is_useful"),
            "reason": str(data.get("reason", "")).strip(),
        }
    except Exception as exc:
        return {"enabled": True, "is_useful": None, "reason": f"judge call failed: {exc}"}


def build_self_rag_prompt_block(self_rag_context: Optional[SelfRAGContext]) -> str:
    if not self_rag_context or not self_rag_context.used:
        return "SELF-RAG CONTEXT\nNo local knowledge was used for this turn."

    return f"""
        SELF-RAG CONTEXT
        The following local knowledge was retrieved and judged relevant. Use it as grounding evidence.
        If the knowledge is insufficient, say what is missing instead of inventing details.
        Do not expose raw source metadata unless the user asks.

        {self_rag_context.context_text}
        """.strip()


def build_session_context_prompt_block(session_context: Optional["SessionContext"]) -> str:
    
    if session_context is None:
        return ""

    lines = [
        "SESSION CONTEXT",
        f"This is experiment session {session_context.session_number} with "
        f"{session_context.spelt_name}. The explanation level for this session "
        f"is: {session_context.explanation_level}.",
        "Use this level by default; do not ask the participant to choose a level "
        "themselves.",
    ]

    if session_context.ask_comprehension_check:
        lines.append(
            "Before moving on, end your reply with exactly ONE short "
            "comprehension-check question about what was just discussed, to "
            "confirm the participant is following."
        )

    if session_context.scaffold_mode and session_context.scaffold_stage == "ask_leading":
        lines.append(
            f"The participant originally asked (in an earlier turn): "
            f"\"{session_context.scaffold_original_question}\". That question is at "
            f"{session_context.scaffold_target_level} level, above this session's "
            f"{session_context.explanation_level} level. Do NOT answer that original "
            "question yet. Instead, briefly acknowledge their most recent answer if "
            f"this is a follow-up, then ask exactly ONE leading question at "
            f"{session_context.scaffold_current_level} level that builds toward being "
            "able to answer the original question. Make it clear you are building up "
            "to it step by step."
        )
    elif session_context.scaffold_mode and session_context.scaffold_stage == "final_answer":
        lines.append(
            f"The participant has now been guided through leading questions toward "
            f"their original question: \"{session_context.scaffold_original_question}\" "
            f"({session_context.scaffold_target_level} level). Briefly acknowledge their "
            "last answer, then now directly and fully answer their original question "
            "at its proper level."
        )

    return "\n".join(lines)


# =========================
# Text-only emotion resolution
# =========================
#
# Text is the sole emotion-detection modality in this pipeline. Prosody
# (RMS/energy-based heuristics over the utterance's audio) and vision
# were both removed entirely: prosody, in particular, was found in
# practice to reliably fire "surprise" on nearly every turn (its
# RMS/energy thresholds were not well matched to this microphone/room's
# typical speaking volume), adding noise rather than signal.

def expand_emotion_to_score_distribution(emotion: str, confidence: float) -> dict[str, float]:
    
    if emotion not in EKMAN_EMOTIONS:
        emotion = "neutral"
    confidence = max(0.0, min(1.0, float(confidence)))
    all_emotions = list(EKMAN_EMOTIONS.keys())
    remaining = max(0.0, 1.0 - confidence)
    other = remaining / max(1, len(all_emotions) - 1)
    return {emo: confidence if emo == emotion else other for emo in all_emotions}


def assess_text_emotion(
    text_emotion: EmotionResult,
    user_text: str = "",
    response_times: Optional[dict[str, Optional[float]]] = None,
) -> TextEmotionAssessment:

    scores = expand_emotion_to_score_distribution(text_emotion.emotion, text_emotion.confidence)

    reason = (
        f"Text emotion classifier selected {text_emotion.emotion} "
        f"(confidence={text_emotion.confidence:.2f})."
    )

    return TextEmotionAssessment(
        emotion=text_emotion.emotion,
        confidence=text_emotion.confidence,
        reason=reason,
        scores=scores,
        raw_text_emotion={
            "emotion": text_emotion.emotion,
            "confidence": text_emotion.confidence,
            "reason": text_emotion.reason,
        },
        response_times=response_times or {},
    )


# =========================
# Temporal emotion smoothing (across turns)
# =========================
#

def apply_temporal_emotion_smoothing(
    current_scores: dict[str, float],
    previous_smoothed_scores: Optional[dict[str, float]],
    alpha: float = EMOTION_SMOOTHING_ALPHA,
) -> dict[str, float]:
    """
    Exponential moving average (EMA) over the per-emotion score
    distribution, applied ACROSS TURNS within a session:
        smoothed = alpha * current + (1 - alpha) * previous_smoothed
    `alpha` close to 1.0 tracks the current turn almost exactly (little
    smoothing, more reactive to change); closer to 0.0 changes slowly
    (heavy smoothing, more resistant to a single outlier turn).
    """
    if not previous_smoothed_scores:
        return dict(current_scores)

    all_emotions = set(current_scores) | set(previous_smoothed_scores)
    alpha = max(0.0, min(1.0, float(alpha)))

    return {
        emo: alpha * current_scores.get(emo, 0.0) + (1.0 - alpha) * previous_smoothed_scores.get(emo, 0.0)
        for emo in all_emotions
    }


def dominant_from_scores(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "neutral", 0.0
    dominant, value = max(scores.items(), key=lambda item: item[1])
    return dominant, max(0.0, min(1.0, value))


# =========================
# Emotion detection (text-only classifier)
# =========================

def build_emotion_prompt(transcribed_text: str) -> str:
    emotions = ", ".join(EKMAN_EMOTIONS.keys())

    return f"""
        You are an emotion classification system for a human-robot interaction warm-up.

        Classify the user's dominant emotional state from the text into EXACTLY ONE of
        these {len(EKMAN_EMOTIONS)} emotion classes:

        {emotions}

        Guidelines:
        - Use only the user's words as evidence.
        - Do not infer emotions that are not clearly supported.
        - "neutral" is a full, equally valid class, not a fallback of last resort: choose
          it whenever the message is primarily factual, informational, or a question
          without explicit emotional language.
        - Positive excitement, appreciation, compliments, or enthusiasm \u2192 happiness.
        - Surprise requires genuine shock or being caught off guard, not simply saying "wow" or "interesting".
        - Anger requires clear hostility, insults, or explicit frustration. Repeating a question or correcting the assistant is not anger.
        - If multiple emotions appear, choose the strongest single emotion.

        Confidence:
        - 0.85\u20131.00: explicit emotional language.
        - 0.55\u20130.80: emotion is present but somewhat inferred.
        - 0.30\u20130.50: mostly neutral or factual with only weak emotional evidence.

        Return ONLY valid JSON in exactly this format:

        {{
        "emotion": "{'|'.join(EKMAN_EMOTIONS.keys())}",
        "confidence": 0.0,
        "reason": "brief explanation based on the text"
        }}

        Text:
        {transcribed_text}
        """.strip()


def simple_emotion_fallback(transcribed_text: str) -> Optional[EmotionResult]:
    text = transcribed_text.strip().lower()

    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
    farewells = {"bye", "goodbye", "see you", "see you later", "talk later", "have a good day", "have a nice day"}

    if text.rstrip(".!?") in greetings:
        return EmotionResult(
            emotion="neutral",
            confidence=0.6,
            reason="The user is opening a friendly social interaction.",
        )

    if any(phrase in text for phrase in farewells):
        return EmotionResult(
            emotion="neutral",
            confidence=0.7,
            reason="The user is closing the conversation politely.",
        )

    if "today's date" in text or "todays date" in text or "what is the date" in text:
        return EmotionResult(
            emotion="neutral",
            confidence=0.5,
            reason="The user is asking for current date information.",
        )

    return None


def detect_emotion(
    client: Client,
    transcribed_text: str,
) -> EmotionResult:
    fallback = simple_emotion_fallback(transcribed_text)
    if fallback:
        return fallback

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {
                    "role": "user",
                    "content": build_emotion_prompt(transcribed_text=transcribed_text),
                },
            ],
            stream=False,
            options={
                "temperature": 0.1,
                "num_predict": 200,
                "num_ctx": 2048,
            },
        )
    except Exception as exc:
        print_ts(f"Emotion detection LLM call failed ({exc}); using neutral fallback.")
        return EmotionResult(
            emotion="neutral",
            confidence=0.3,
            reason=f"Emotion model call failed ({exc}); neutral fallback used.",
        )

    raw = response["message"]["content"]
    data = safe_json_extract(raw)

    if not data:
        return EmotionResult(
            emotion="neutral",
            confidence=0.3,
            reason="Could not parse model output, so a neutral fallback was used.",
        )

    emotion = str(data.get("emotion", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if emotion not in EKMAN_EMOTIONS:
        emotion = "neutral"
        confidence = min(confidence, 0.3)
        reason = "Invalid emotion returned, so fallback emotion was used."

    return EmotionResult(
        emotion=emotion,
        confidence=confidence,
        reason=reason or "Emotion inferred from the transcribed message.",
    )


_QUESTION_LEAD_WORDS_RE = re.compile(
    r"^(?:what|why|how|when|where|which|who|can|could|does|do|is|are|explain|tell me|describe)\b",
    re.IGNORECASE,
)


def looks_like_question(text: str) -> bool:
    """
    Cheap pre-filter so classify_question_level() (an extra LLM call) only
    runs on messages that plausibly ask about something, rather than on
    every single turn.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    return bool(_QUESTION_LEAD_WORDS_RE.match(stripped))


def classify_question_level(client: Client, user_text: str) -> Optional[str]:
    prompt = f"""
        You classify which curriculum level an AI/Robotics question belongs to,
        for a robot tutor with three levels: beginner, intermediate, advanced.

        beginner: everyday-language concepts, no jargon required
        intermediate: correct terminology, basic mechanisms/workflows
        advanced: architectures, trade-offs, mathematical/technical depth

        Message:
        {user_text}

        If the message is not really an AI/Robotics topic question (small talk,
        a greeting, an off-topic remark, a question about Ameca itself, etc.),
        return "not_applicable".

        Return JSON only: {{"level": "beginner|intermediate|advanced|not_applicable"}}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 60, "num_ctx": 1024},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return None
        level = str(data.get("level", "")).strip().lower()
        return level if level in QUESTION_LEVEL_RANK else None
    except Exception as exc:
        print_ts(f"[WARN] Question-level classification failed: {exc}")
        return None


# =========================
# Emoji enforcement
# =========================

def remove_all_emojis_except_allowed_faces(text: str) -> str:
    result = []

    for char in text:
        if char in ALLOWED_FACE_EMOJIS:
            result.append(char)
            continue

        code = ord(char)

        is_emoji_or_symbol = (
            0x1F300 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or code in {0x200D, 0xFE0F}
        )

        if is_emoji_or_symbol:
            continue

        result.append(char)

    return "".join(result)


def remove_allowed_face_emojis(text: str) -> str:
    return "".join(char for char in text if char not in ALLOWED_FACE_EMOJIS)


_COMPARISON_RE = re.compile(r"\bdifference between\b|\bcompare\b|\bversus\b|\bvs\.?\b", re.IGNORECASE)
_EXAMPLE_REQUEST_RE = re.compile(r"\bexamples?\b", re.IGNORECASE)
_MULTI_PART_QUESTION_RE = re.compile(
    r"\b(?:and|then|also)\s+(?:then\s+)?(?:what|who|how|why|where|which|when|do|does|can|could|"
    r"is|are|please|give|explain|tell|describe|provide|show|compare|list)\b",
    re.IGNORECASE,
)


_ABBREVIATION_PERIOD_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|no|fig|eq)\.$", re.IGNORECASE)
_LIST_MARKER_PERIOD_RE = re.compile(r"(?:^|\s)\d{1,2}\.$")


def _is_real_sentence_boundary(text_up_to_period: str) -> bool:
    """
    True only if the trailing '.' actually ends a sentence, rather than
    being part of an abbreviation (e.g., i.e., etc.) or a bare numbered-
    list marker (e.g. '4.'). Without this check, truncate_to_max_words()
    would treat those periods as valid sentence ends and cut the reply
    off mid-parenthetical or right after a bare list number.
    """
    if _ABBREVIATION_PERIOD_RE.search(text_up_to_period):
        return False
    if _LIST_MARKER_PERIOD_RE.search(text_up_to_period):
        return False
    return True


def truncate_to_max_words(text: str, max_words: int = MAX_REPLY_WORDS) -> str:
    text = text.strip()
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])

    last_sentence_end = -1
    for idx in range(len(truncated) - 1, -1, -1):
        if truncated[idx] in ".!?":
            if truncated[idx] != "." or _is_real_sentence_boundary(truncated[: idx + 1]):
                last_sentence_end = idx
                break

    if last_sentence_end > int(len(truncated) * 0.4):
        return truncated[: last_sentence_end + 1].strip()

    truncated = truncated.rstrip()
    # No usable sentence boundary was found -- strip a dangling bare list
    # marker/number or an open "(e.g." left at the very end by word-count
    # truncation, rather than leaving it hanging in front of the emoji.
    truncated = re.sub(r"(?:^|\s)\(?\d{1,2}\.?\s*$", "", truncated).rstrip()
    truncated = re.sub(r"\(\s*e\.?g\.?\s*$", "", truncated, flags=re.IGNORECASE).rstrip()
    if truncated and truncated[-1] not in ".!?":
        truncated += "."
    return truncated


def normalize_reply(
    raw_reply: str,
    emotion: str,
    confidence: float = 1.0,
    max_words: int = MAX_REPLY_WORDS,
) -> str:
    resolved_emotion = resolve_expressive_emotion(emotion, confidence)
    required_emoji = EKMAN_EMOTIONS.get(resolved_emotion, EKMAN_EMOTIONS["neutral"])

    cleaned = remove_all_emojis_except_allowed_faces(raw_reply)
    cleaned = remove_allowed_face_emojis(cleaned)

    cleaned = re.sub(r"[:;=8][\-^]?[)(DPp/\\|]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = truncate_to_max_words(cleaned, max_words=max_words)

    if not cleaned:
        cleaned = "I'm here with you."

    return f"{cleaned} {required_emoji}"


# =========================
# Date / time helpers
# =========================

def runtime_context() -> str:
    return f"""
        RUNTIME CONTEXT
        Current local date and time: {now_ts()}.
        Use this date/time when the user asks about today, now, or the current date.
        """.strip()


FAREWELL_TERMINATION_PHRASES = {
    "bye", "goodbye", "good bye", "see you", "see you later",
    "talk later", "have a nice day", "have a good day",
}


def is_farewell_utterance(text: str) -> bool:
    lowered = text.strip().lower().rstrip(".!?")
    return any(phrase in lowered for phrase in FAREWELL_TERMINATION_PHRASES)


TOPIC_CHANGE_REQUEST_RE = re.compile(
    r"\b(?:other topic|different topic|another topic|change (?:the )?topic|"
    r"something else|what else can we|what other topic|what other topics|"
    r"switch topic|new topic|talk about (?:something|anything) else)\b",
    re.IGNORECASE,
)


def looks_like_topic_change_request(text: str) -> bool:
    """
    Detects meta-conversational "let's move to a different topic" requests
    (e.g. "which other topic can we talk about?"). These have no natural
    fit in the teaching-answer JSON schema and were observed causing the
    small local model to repeatedly fail to produce parseable output (two
    failed attempts in a row, falling through to the generic "didn't quite
    catch that" apology). Handled deterministically instead, the same way
    date/time questions are, for both reliability and one fewer LLM call.
    """
    return bool(TOPIC_CHANGE_REQUEST_RE.search(text.strip()))


# Mirrors the three options offered in TOPIC_PROMPT_QUESTION at session
# start, plus a couple of natural follow-on topics, so a mid-session
# "what else can we talk about" gets concrete, curriculum-consistent
# suggestions rather than an open-ended question.
AVAILABLE_LESSON_TOPICS: list[tuple[str, str]] = [
    ("machine learning", "the basics of machine learning"),
    ("robot perception", "how robots sense the world"),
    ("large language models", "how large language models work"),
    ("neural networks", "how neural networks are built and trained"),
    ("human-robot interaction", "how robots like me interact with people"),
]


def build_topic_change_reply(lesson_state: Optional["LessonState"]) -> str:
    covered: set[str] = set()
    if lesson_state is not None:
        if lesson_state.current_topic:
            covered.add(lesson_state.current_topic.strip().lower())
        for concept in lesson_state.covered_concepts:
            covered.add(concept.strip().lower())

    def _already_covered(name: str) -> bool:
        name_lower = name.lower()
        return any(name_lower in c or c in name_lower for c in covered)

    remaining = [(name, desc) for name, desc in AVAILABLE_LESSON_TOPICS if not _already_covered(name)]
    if not remaining:
        # Everything on the list has come up already; offer to go deeper
        # on the existing set rather than claiming there's nothing left.
        remaining = AVAILABLE_LESSON_TOPICS

    options = remaining[:3]
    descriptions = [desc for _, desc in options]
    if len(descriptions) == 1:
        topic_text = descriptions[0]
    else:
        topic_text = ", ".join(descriptions[:-1]) + f", or {descriptions[-1]}"

    return f"We could talk about {topic_text}. Which would you like?"


def deterministic_reply_if_applicable(
    user_text: str,
    emotion: str,
    lesson_state: Optional["LessonState"] = None,
) -> Optional[str]:
    text = user_text.strip().lower()
    emoji = EKMAN_EMOTIONS.get(emotion, "\U0001F642")

    if "today's date" in text or "todays date" in text or "what is the date" in text:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. {emoji}"

    if "what is the time" in text or "what time is it" in text or "current time" in text:
        return f"The current time is {datetime.now().strftime('%H:%M')}. {emoji}"

    if looks_like_topic_change_request(text):
        return f"{build_topic_change_reply(lesson_state)} {emoji}"

    return None


# =========================
# Response generation
# =========================

def emotion_confidence_label(confidence: float) -> str:
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


def build_clean_emotion_summary(emotion_result: EmotionResult) -> dict[str, Any]:
    return {
        "emotion": emotion_result.emotion,
        "confidence_label": emotion_confidence_label(emotion_result.confidence),
    }

def extra_reponse_propmt_guideline(clean_emotion_summary):
    return {
        "emotion_context": {
            "detected_emotion_summary": json.dumps(clean_emotion_summary),
            "rule": (
                "This is the emotion detected in the USER's message. Use it only to "
                "adjust the empathy, warmth, or care in your WORDING -- it is not "
                "necessarily the emotion your own reply should express."
            ),
        },
        "output_format": {
            "instruction": "Return JSON only.",
            "schema": {
                "reply": "assistant response without emoji",
                "emoji": "exactly one facial emoji expressing the emotional tone of YOUR OWN reply",
                "tone": "short description of communication style",
            },
        },
        "emoji_rules": [
            "The 'emoji' field is Ameca's own congruent facial expression for THIS "
            "reply -- the affect your response itself carries -- not a copy of the "
            "user's detected emotion.",
            f"Choose exactly one emoji, treating all {len(EKMAN_EMOTIONS)} classes as "
            "equally valid options: " + " ".join(EKMAN_EMOTIONS.values()),
            "Do not use any other emoji, emoticon, or symbol.",
            "Neutral (\U0001F642) is a genuine, first-class choice for ordinary informative or "
            "teaching answers with no strong affect -- it is not a fallback for "
            "uncertainty; choose whichever of the seven classes actually matches the "
            "tone of your reply.",
        ],
    }

# Fixed tail: schema guidance + hard output instructions. Both MUST always
# survive truncation intact, regardless of how large the variable-length
# background context (memory summary, Self-RAG retrieved knowledge, etc.)
# grows. See build_response_system_prompt() below -- only the variable
# background is truncated to MAX_SYSTEM_PROMPT_CHARS; this fixed tail is
# appended afterward unconditionally.
RESPONSE_OUTPUT_INSTRUCTIONS = """
OUTPUT INSTRUCTIONS -- these override anything above about output format:
- Your entire response must be exactly ONE JSON object and nothing else: no
  preamble, no markdown fences, no restating or copying any part of the
  background context above.
- That JSON object must have EXACTLY these three keys and no others:
  "reply", "emoji", "tone".
- "reply" is your actual spoken response as a plain string, with NO emoji or
  emoticon inside it. If the user asked for examples, a list, or specific
  details, include the real content -- not just a lead-in sentence.
- "emoji" is YOUR OWN facial expression for this reply, chosen from exactly
  this set: \U0001F60A \U0001F622 \U0001F620 \U0001F628 \U0001F62E \U0001F922 \U0001F642 -- congruent with what your reply is
  expressing, not a copy of the user's detected emotion.
- Correct output shape: {"reply": "AI differs from normal programming because...", "emoji": "\U0001F60A", "tone": "curious"}
- Incorrect (never do this): {"role": "Ameca, a humanoid social robot..."}
""".strip()

MAX_SYSTEM_PROMPT_CHARS = int(os.environ.get("MAX_SYSTEM_PROMPT_CHARS", "12000"))


def build_response_system_prompt(
    emotion_result: EmotionResult,
    user_profile: Optional[dict] = None,
    self_rag_context: Optional[SelfRAGContext] = None,
    session_context: Optional["SessionContext"] = None,
    lesson_state: Optional["LessonState"] = None,
    target_word_count: Optional[int] = None,
) -> str:
    memory_context = build_user_memory_context(user_profile)

    clean_emotion_summary = build_clean_emotion_summary(emotion_result)
    additional_guidelines_text = json.dumps(
        extra_reponse_propmt_guideline(clean_emotion_summary), indent=2
    )
    ameca_system_prompt_text = json.dumps(AMECA_SYSTEM_PROMPT, indent=2)
    session_context_text = build_session_context_prompt_block(session_context)

    fixed_tail_parts = [additional_guidelines_text]
    if lesson_state is not None:
        fixed_tail_parts.append(lesson_state.to_prompt_block())
    if session_context_text:
        fixed_tail_parts.append(session_context_text)
    if target_word_count:
        fixed_tail_parts.append(
            f"REPLY LENGTH TARGET\nAim for roughly {target_word_count} words in "
            "\"reply\" -- enough to fully cover what's needed, without padding."
        )
    fixed_tail_parts.append(RESPONSE_OUTPUT_INSTRUCTIONS)
    fixed_tail = "\n\n".join(fixed_tail_parts)

    # VARIABLE BACKGROUND -- only this part gets truncated.
    background_text = f"""
    BEGIN BACKGROUND CONTEXT -- for your own reference only. Never repeat, quote, or
    output any of this verbatim. None of the JSON keys below (such as "role",
    "identity", "capability_boundaries", "task", "possible_topics", etc.) are your
    output format -- they only describe you.

    {ameca_system_prompt_text}

    {runtime_context()}

    {memory_context}

    {build_self_rag_prompt_block(self_rag_context)}

    END BACKGROUND CONTEXT
    """.strip()

    max_background_chars = max(1000, MAX_SYSTEM_PROMPT_CHARS - len(fixed_tail) - 20)
    background_text = background_text[:max_background_chars]

    return f"{background_text}\n\n{fixed_tail}"


def limit_text_length(text: str, max_chars: int = 1500) -> str:
    return text[:max_chars]


def trim_history(history: list[dict]) -> list[dict]:
    return history[-MAX_HISTORY_MESSAGES:]


def prompt_ready_history(history: list[dict]) -> list[dict]:
    return [{"role": item["role"], "content": item["content"]} for item in history]


def _is_degenerate_reply_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    return stripped.lower() in {"{}", "[]", "null", "none", "{ }", "[ ]"}


class _LLMCallFailed(Exception):
    pass


def _looks_like_unparsed_json_schema(text: str) -> bool:

    stripped = str(text or "").strip()
    return stripped.startswith("{")

def _attempt_llm_response(
    client: Client,
    messages: list[dict],
    emotion_result: EmotionResult,
    self_rag_context: Optional[SelfRAGContext],
    repeat_penalty: float,
    max_words: int = MAX_REPLY_WORDS,
    temperature_override: Optional[float] = None,
    debug_log: Optional[list[str]] = None,
    response_format: Optional[str] = "json",
) -> Optional[GeneratedReply]:

    temperature = (
        temperature_override
        if temperature_override is not None
        else (0.25 if self_rag_context and self_rag_context.used else 0.4)
    )

    try:
        response = client.chat(
            model=MODEL_NAME,
            format=response_format,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": min(500, int(max_words * 1.6) + 60),
                "repeat_penalty": repeat_penalty,
                "num_ctx": 8192,
            },
            stream=False,
        )
    except Exception as exc:
        print_ts(f"Response generation LLM call failed ({exc}).")
        if debug_log is not None:
            debug_log.append(f"[call_failed] {exc}")
        raise _LLMCallFailed(str(exc)) from exc

    raw_reply = response["message"]["content"]

    if DEBUG_LOG_RAW_LLM_REPLIES:
        print_ts(f"[DEBUG] Raw LLM reply (pre-parse, response generation): {raw_reply!r}")

    data = safe_json_extract(raw_reply)

    if data is not None and isinstance(data, dict):
        reply_text = str(data.get("reply", "")).strip()
        model_emoji = str(data.get("emoji", "")).strip()
        # The model's own congruent-expression choice, mapped back to a
        # label. Falls back to the detected user emotion only if the
        # model omitted the field or returned something outside the
        # allowed set.
        response_emotion = emoji_to_emotion(model_emoji) or emotion_result.emotion

        if reply_text and not _is_degenerate_reply_text(reply_text):
            final_text = normalize_reply(
                reply_text,
                response_emotion,
                emotion_result.confidence,
                max_words=max_words,
            )
            resolved_emotion = resolve_expressive_emotion(response_emotion, emotion_result.confidence)
            return GeneratedReply(text=final_text, response_emotion=resolved_emotion)
        print_ts(
            f"[DEBUG] Rejecting response: parsed JSON but 'reply' field was empty/degenerate. "
            f"Raw LLM reply: {raw_reply!r}"
        )
        if debug_log is not None:
            debug_log.append(f"[empty_or_degenerate_reply_field] {raw_reply!r}")
        return None

    if _is_degenerate_reply_text(raw_reply) or _looks_like_unparsed_json_schema(raw_reply):
        print_ts(
            f"[DEBUG] Rejecting response: could not parse JSON and raw text looked degenerate/"
            f"unparsed-schema. Raw LLM reply: {raw_reply!r}"
        )
        if debug_log is not None:
            debug_log.append(f"[unparseable_or_degenerate] {raw_reply!r}")
        return None

    # Unparsed but non-degenerate raw text (rare): no model-chosen emoji is
    # available at all, so fall back to the detected user emotion.
    final_text = normalize_reply(
        raw_reply, emotion_result.emotion, emotion_result.confidence, max_words=140
    )
    resolved_emotion = resolve_expressive_emotion(emotion_result.emotion, emotion_result.confidence)
    if self_rag_context and self_rag_context.used and context_has_placeholder_risk(final_text):
        apology_text = normalize_reply(
            "I found a relevant local lab page, but I could not verify the exact name from the retrieved text, so I should not invent it.",
            "neutral",
            1.0,
        )
        return GeneratedReply(text=apology_text, response_emotion="neutral")
    return GeneratedReply(text=final_text, response_emotion=resolved_emotion)


# =========================
# Post-generation verification agent
# =========================
#


def looks_like_multi_part_question(user_text: str) -> bool:
    text = user_text.strip()
    if not text:
        return False
    question_marks = text.count("?")
    joiners = len(_MULTI_PART_QUESTION_RE.findall(text))
    parts = max(1, question_marks, 1 + joiners)
    if parts > 1:
        return True
    return bool(_COMPARISON_RE.search(text) and _EXAMPLE_REQUEST_RE.search(text))


def check_response_completeness(client: Client, user_text: str, reply: str) -> Optional[str]:
    """
    Verification agent: checks whether `reply` actually addressed EVERY
    distinct question/request in `user_text`. Returns a short description
    of what's missing if the reply is incomplete, or None if it's
    complete (or the check itself failed -- fails open, since this only
    ever gates a single repair attempt and must not block the turn loop
    on its own errors).
    """
    prompt = f"""
        The user asked (possibly more than one thing in the same message):
        {user_text}

        The assistant replied:
        {reply}

        Does the reply address EVERY distinct question or request in the user's
        message? A sub-point that was skipped or only vaguely gestured at
        counts as incomplete.

        Return JSON only:
        {{"complete": true, "missing": "short description of what was not addressed, or empty string if complete"}}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.0, "num_predict": 120, "num_ctx": 2048},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return None
        complete = bool(data.get("complete", True))
        missing = str(data.get("missing", "")).strip()
        if complete or not missing:
            return None
        return missing
    except Exception as exc:
        print_ts(f"[WARN] Response-completeness check failed: {exc}")
        return None


def repair_incomplete_reply(
    client: Client,
    user_text: str,
    previous_reply_text: str,
    missing_description: str,
    emotion_result: EmotionResult,
) -> Optional[GeneratedReply]:
    """
    One extra LLM call, used only after check_response_completeness()
    flags a gap: asks for a single REVISED reply that keeps what was
    already useful and also covers what was missing. Returns None on any
    failure so the caller falls back to the original (still usable, just
    partial) reply rather than losing the turn entirely.
    """
    prompt = f"""
        You are Ameca. You previously replied to the user, but your reply did not
        fully cover everything they asked.

        User's message:
        {user_text}

        Your previous reply:
        {previous_reply_text}

        What was missing:
        {missing_description}

        Write ONE revised reply that keeps the useful part of your previous
        answer and ALSO covers what was missing. Keep it natural and concise
        (aim for 3-6 sentences total, not a bare list).

        Return JSON only:
        {{"reply": "revised response without emoji", "emoji": "one of \U0001F60A \U0001F622 \U0001F620 \U0001F628 \U0001F62E \U0001F922 \U0001F642"}}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 260, "num_ctx": 4096},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return None
        reply_text = str(data.get("reply", "")).strip()
        if not reply_text or _is_degenerate_reply_text(reply_text):
            return None
        model_emoji = str(data.get("emoji", "")).strip()
        response_emotion = emoji_to_emotion(model_emoji) or emotion_result.emotion
        final_text = normalize_reply(reply_text, response_emotion, emotion_result.confidence, max_words=140)
        resolved_emotion = resolve_expressive_emotion(response_emotion, emotion_result.confidence)
        return GeneratedReply(text=final_text, response_emotion=resolved_emotion)
    except Exception as exc:
        print_ts(f"[WARN] Reply-completeness repair call failed: {exc}")
        return None


def reply_ends_with_question(reply_text: str) -> bool:
    """
    Deterministic check (no LLM call) for whether a reply is a single
    question -- used to verify scaffold "ask_leading" turns, where the
    model is instructed to ask exactly one leading question and NOT
    answer yet. Strips the trailing required face emoji (added by
    normalize_reply()) before checking.
    """
    stripped = str(reply_text or "").strip()
    for emoji in EKMAN_EMOTIONS.values():
        if stripped.endswith(emoji):
            stripped = stripped[: -len(emoji)].strip()
            break
    return stripped.endswith("?")


def repair_scaffold_leading_question(
    client: Client,
    original_question: str,
    leading_level: str,
    previous_reply_text: str,
    emotion_result: EmotionResult,
) -> Optional[GeneratedReply]:

    prompt = f"""
        You are Ameca, mid-way through building up to answering this question
        from the participant, one step at a time: "{original_question}"

        Your last attempt did not ask a question -- it answered instead:
        "{previous_reply_text}"

        Write ONLY a single short leading question at {leading_level} level that
        moves toward answering the original question. Do NOT answer anything.
        Do not explain. It must end with a question mark.

        Return JSON only:
        {{"reply": "the single leading question, no emoji", "emoji": "one of \U0001F60A \U0001F622 \U0001F620 \U0001F628 \U0001F62E \U0001F922 \U0001F642"}}
        """.strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {"role": "system", "content": "You return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3, "num_predict": 120, "num_ctx": 2048},
            stream=False,
        )
        data = safe_json_extract(response["message"]["content"])
        if not isinstance(data, dict):
            return None
        reply_text = str(data.get("reply", "")).strip()
        if not reply_text or _is_degenerate_reply_text(reply_text):
            return None
        if "?" not in reply_text:
            reply_text = reply_text.rstrip(".! ") + "?"
        model_emoji = str(data.get("emoji", "")).strip()
        response_emotion = emoji_to_emotion(model_emoji) or emotion_result.emotion
        final_text = normalize_reply(reply_text, response_emotion, emotion_result.confidence, max_words=25)
        resolved_emotion = resolve_expressive_emotion(response_emotion, emotion_result.confidence)
        return GeneratedReply(text=final_text, response_emotion=resolved_emotion)
    except Exception as exc:
        print_ts(f"[WARN] Scaffold leading-question repair call failed: {exc}")
        return None


def verify_and_repair_reply(
    client: Client,
    user_text: str,
    generated_reply: GeneratedReply,
    session_context: Optional["SessionContext"],
    emotion_result: EmotionResult,
    force_multi_part: bool = False,
) -> tuple[GeneratedReply, dict]:

    verification_log: dict[str, Any] = {"checked": None, "repaired": False, "reason": None}

    if session_context is not None and session_context.scaffold_mode and session_context.scaffold_stage == "ask_leading":
        verification_log["checked"] = "scaffold_leading_question"
        if not reply_ends_with_question(generated_reply.text):
            print_ts(
                "[VERIFY] Scaffold leading-question turn did not end in a question; "
                "attempting one repair call."
            )
            repaired = repair_scaffold_leading_question(
                client=client,
                original_question=session_context.scaffold_original_question or "",
                leading_level=session_context.scaffold_current_level or session_context.explanation_level,
                previous_reply_text=generated_reply.text,
                emotion_result=emotion_result,
            )
            if repaired is not None:
                verification_log["repaired"] = True
                verification_log["reason"] = "reply did not end in a question"
                return repaired, verification_log
            verification_log["reason"] = "reply did not end in a question; repair call failed, kept original"
        return generated_reply, verification_log

    if looks_like_multi_part_question(user_text) or force_multi_part:
        verification_log["checked"] = "multi_part_completeness"
        missing = check_response_completeness(client, user_text, generated_reply.text)
        if missing:
            print_ts(f"[VERIFY] Reply looked incomplete ({missing}); attempting one repair call.")
            repaired = repair_incomplete_reply(
                client=client,
                user_text=user_text,
                previous_reply_text=generated_reply.text,
                missing_description=missing,
                emotion_result=emotion_result,
            )
            if repaired is not None:
                verification_log["repaired"] = True
                verification_log["reason"] = missing
                return repaired, verification_log
            verification_log["reason"] = f"{missing} (repair call failed, kept original)"

    return generated_reply, verification_log


# Deliberately tiny system prompt used only as a last resort (see
# generate_response()'s third attempt below) when the full-context prompt
# has already failed to produce parseable output twice in a row. Session
# transcripts have shown occasional back-to-back generation failures on
# otherwise-ordinary questions; stripping away the large background
# context/lesson-state/self-RAG blocks for one final attempt trades away
# lesson continuity in exchange for a real answer instead of a second
# "sorry, could you repeat that" in the same conversation.
MINIMAL_FALLBACK_SYSTEM_PROMPT = """
You are Ameca, a friendly humanoid robot tutor for AI and Robotics topics,
speaking with a beginner-level student.

Answer the user's message directly and simply, in 2-4 sentences.

Return JSON only, exactly this shape, nothing else:
{"reply": "your answer, no emoji inside it", "emoji": "one of \U0001F60A \U0001F622 \U0001F620 \U0001F628 \U0001F62E \U0001F922 \U0001F642", "tone": "short description"}
""".strip()


def generate_response(
    client: Client,
    user_text: str,
    emotion_result: EmotionResult,
    history: list[dict],
    user_profile: Optional[dict] = None,
    self_rag_context: Optional[SelfRAGContext] = None,
    llm_call_samples: Optional[list[dict]] = None,
    session_context: Optional["SessionContext"] = None,
    lesson_state: Optional["LessonState"] = None,
    force_multi_part: bool = False,
    awaiting_retention_check: bool = False,
    generation_debug: Optional[dict] = None,
) -> GeneratedReply:

    deterministic = deterministic_reply_if_applicable(
        user_text=user_text,
        emotion=emotion_result.emotion,
        lesson_state=lesson_state,
    )

    if deterministic:
        resolved_emotion = resolve_expressive_emotion(emotion_result.emotion, emotion_result.confidence)
        return GeneratedReply(text=deterministic, response_emotion=resolved_emotion)

    safe_user_text = limit_text_length(user_text)

    reply_max_words = resolve_reply_word_budget(
        intent=lesson_state.last_intent if lesson_state else "continue",
        is_multi_part=looks_like_multi_part_question(safe_user_text) or force_multi_part,
        awaiting_retention_check=awaiting_retention_check,
    )

    system_prompt = build_response_system_prompt(
        emotion_result=emotion_result,
        user_profile=user_profile,
        self_rag_context=self_rag_context,
        session_context=session_context,
        lesson_state=lesson_state,
        target_word_count=reply_max_words,
    )
    

    messages = [
        {"role": "system", "content": system_prompt},
        *prompt_ready_history(trim_history(history)),
        {"role": "user", "content": safe_user_text},
    ]

    def _record_sample(reply_text: str) -> None:
        if llm_call_samples is not None and len(llm_call_samples) < 3:
            llm_call_samples.append({
                "turn_index": len(llm_call_samples) + 1,
                "messages": messages,
                "user_text": safe_user_text,
                "reply": reply_text,
                "timestamp": now_ts(),
            })

    if self_rag_context and self_rag_context.used:
        grounded = generate_grounded_self_rag_answer(
            client=client,
            user_text=safe_user_text,
            self_rag_context=self_rag_context,
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
        )
        if grounded:
            _record_sample(grounded.text)
            return grounded

    call_failed = False
    generated: Optional[GeneratedReply] = None
    raw_attempts: list[str] = []

    try:
        generated = _attempt_llm_response(
            client=client,
            messages=messages,
            emotion_result=emotion_result,
            self_rag_context=self_rag_context,
            repeat_penalty=1.1,
            max_words=reply_max_words,
            debug_log=raw_attempts,
        )
    except _LLMCallFailed:
        call_failed = True

    if generated is not None:
        _record_sample(generated.text)
        return generated

    print_ts("Response generation produced no usable content on the first attempt; retrying once.")
    try:
        # Bump temperature on retry rather than repeating an identical
        # call -- a retry with the same sampling settings on the same
        # messages is prone to correlated failure (same malformed-JSON
        # pattern recurring), giving little real diversity of outcome.
        generated = _attempt_llm_response(
            client=client,
            messages=messages,
            emotion_result=emotion_result,
            self_rag_context=self_rag_context,
            repeat_penalty=1.1,
            max_words=reply_max_words,
            temperature_override=0.65,
            debug_log=raw_attempts,
        )
        call_failed = False
    except _LLMCallFailed:
        call_failed = True

    if generated is not None:
        _record_sample(generated.text)
        return generated

    if not call_failed:
        # Both real attempts produced unparseable/degenerate output but the
        # connection itself is fine -- try once more with a deliberately
        # tiny system prompt (no background context, lesson state, or
        # Self-RAG block) before giving up. This trades lesson continuity
        # for an actual answer, which is better than a second consecutive
        # "sorry, could you repeat that" in the same conversation.
        print_ts("Both full-context attempts failed; trying a minimal-prompt last resort.")
        minimal_messages = [
            {"role": "system", "content": MINIMAL_FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": safe_user_text},
        ]
        try:
            generated = _attempt_llm_response(
                client=client,
                messages=minimal_messages,
                emotion_result=emotion_result,
                self_rag_context=None,
                repeat_penalty=1.1,
                max_words=min(reply_max_words, 90),
                temperature_override=0.5,
                debug_log=raw_attempts,
                # Session logs have shown the root cause of these
                # back-to-back failures is literally an empty '{}' from
                # Ollama's JSON-mode constrained decoding, not malformed
                # text -- a known small-model failure pattern under
                # grammar-constrained decoding when it's uncertain. Drop
                # the hard format constraint for this last attempt; the
                # prompt still asks for JSON, but a plain-text reply is
                # also handled by the unparsed-non-degenerate path below.
                response_format=None,
            )
        except _LLMCallFailed:
            call_failed = True

        if generated is not None:
            if generation_debug is not None:
                generation_debug["raw_attempts"] = raw_attempts
                generation_debug["recovered_via"] = "minimal_prompt_last_resort"
            _record_sample(generated.text)
            return generated

    if generation_debug is not None:
        generation_debug["raw_attempts"] = raw_attempts
        generation_debug["call_failed"] = call_failed

    if call_failed:
        print_ts("Response generation LLM call failed on both attempts; using connectivity fallback reply.")
        fallback_text = normalize_reply(
            "I'm having trouble reaching my language model right now, so I can't respond properly to that.",
            "neutral",
            1.0,
        )
        _record_sample(fallback_text)
        return GeneratedReply(text=fallback_text, response_emotion="neutral")

    print_ts("Response generation produced no usable content after all attempts; using fallback reply.")
    fallback_text = normalize_reply(
        "Sorry, could you say that again? I didn't quite catch a clear response that time.",
        "neutral",
        1.0,
    )
    _record_sample(fallback_text)
    return GeneratedReply(text=fallback_text, response_emotion="neutral")


# =========================
# CLI args (robot-specific)
# =========================

def parse_robot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ameca demo: Silero VAD + faster-whisper + text-only emotion detection "
        "(7-class taxonomy: 6 Ekman-derived classes + neutral as a coequal class) "
        "+ Self-RAG, with Tritium TTS output, Tritium facial expression, and ZED-camera session-video "
        "recording. No visual/facial-recognition modality and no vocal-prosody modality are used for "
        "emotion detection -- text is the sole signal; the camera here is for recording only."
    )

    parser.add_argument(
        "--participant_id",
        default=os.environ.get("PARTICIPANT_ID", ""),
        help="Fixed participant identifier for the user study. If omitted, it will be "
        "requested interactively (typed on the keyboard) at startup instead. This "
        "becomes the storage key for users.json/session transcripts (stable across "
        "sessions and immune to ASR name-transcription drift), while Ameca still "
        "addresses the person by whatever name they actually say.",
    )

    parser.add_argument(
        "--explanation_level",
        choices=["beginner", "intermediate", "advanced"],
        default=os.environ.get("EXPLANATION_LEVEL", ""),
        help="Fixed explanation level provided by the experimenter via keyboard input "
        "(beginner/intermediate/advanced). If omitted, you'll be prompted for it "
        "interactively at startup; leave that prompt blank to fall back to the "
        "automatic per-session level instead.",
    )

    parser.add_argument(
        "--chat_model",
        default=MODEL_NAME,
        help=f"Ollama model used for text chat, emotion detection, Self-RAG, and session summaries (default: {MODEL_NAME}).",
    )

    parser.add_argument("--tts_url", default=os.environ.get("TTS_URL", "http://emah/tritium/text_to_speech/say?voice=Lucy"))
    parser.add_argument("--speaking_cooldown", type=float, default=0.3, help="Seconds of echo-guard cooldown after TTS finishes speaking.")
    parser.add_argument(
        "--tts_token",
        default=os.environ.get("TTS_TOKEN", "ZWNFuNQVIPyztWCfPPM5VLPslpj8rR"),
        help="X-Tritium-Auth-Token used for both the TTS 'say' endpoint and the sequence_player expression endpoint.",
    )
    parser.add_argument(
        "--tts_activity_debounce",
        type=float,
        default=TTS_ACTIVITY_DEBOUNCE_SECONDS,
        help="Seconds of confirmed quiet (via the live TTS-activity EMA) required before Ameca is "
        "considered done speaking. Integrated from ameca_warm_up.py so the turn-end nod fires at the "
        "right time instead of relying on a rough word-count estimate.",
    )
    parser.add_argument(
        "--expression_host",
        default=os.environ.get("EXPRESSION_HOST", "http://emah"),
        help="Base host for the Tritium sequence_player facial-expression endpoint (default: http://emah, same host as tts_url).",
    )
    parser.add_argument(
        "--disable_expression",
        action="store_true",
        help="Disable driving Ameca's physical facial expression from the resolved text emotion result.",
    )
    parser.add_argument(
        "--nod_sequence",
        default=NOD_SEQUENCE_NAME,
        help=f"Tritium sequence name played as the turn-end nod cue (default: {NOD_SEQUENCE_NAME}, matching ameca_warm_up.py).",
    )
    parser.add_argument(
        "--expression_timing",
        choices=["before", "during", "after"],
        default=EXPRESSION_TIMING,
        help=(
            "When Ameca's facial expression is driven relative to speaking: "
            "'before' (default) sets the expression and waits for that animation "
            "to finish before speaking; 'during' starts speaking and sets the "
            "expression at the same time; 'after' speaks first, waits for speech "
            f"to finish, then sets the expression (default: {EXPRESSION_TIMING})."
        ),
    )

    parser.add_argument(
        "--disable_emotion_smoothing",
        action="store_true",
        help="Disable temporal (cross-turn) smoothing of the resolved emotion result; use each turn's raw text emotion directly.",
    )
    parser.add_argument(
        "--emotion_smoothing_alpha",
        type=float,
        default=EMOTION_SMOOTHING_ALPHA,
        help=f"EMA weight given to the current turn's scores when temporal smoothing is enabled (default: {EMOTION_SMOOTHING_ALPHA}). Lower = smoother/slower to change.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=int(os.environ.get("CAMERA_DEVICE", "0")),
        help="ZED/OpenCV camera device index used ONLY for session video recording (default: 0).",
    )
    parser.add_argument(
        "--resolution",
        choices=list(RESOLUTION_MAP),
        default=_DEFAULT_RESOLUTION,
        help=f"ZED camera resolution preset (SBS). Default: {_DEFAULT_RESOLUTION}.",
    )
    parser.add_argument("--video_fps", type=float, default=VIDEO_RECORD_FPS, help="Frames per second for the recorded session video.")
    parser.add_argument(
        "--video_fourcc",
        default=VIDEO_FOURCC,
        help="FourCC codec for the recorded session video (default: mp4v). Try 'XVID' with a .avi path if mp4v isn't available.",
    )
    parser.add_argument(
        "--disable_video_recording",
        action="store_true",
        help="Disable recording a video of the session via the ZED camera.",
    )

    return parser.parse_args()


# =========================
# Main loop
# =========================

def main() -> None:
    global MODEL_NAME, NOD_SEQUENCE_NAME, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, EXPRESSION_TIMING

    args = parse_robot_args()

    experimenter_level_override = str(args.explanation_level or "").strip().lower()
    if not experimenter_level_override:
        print()
        print_ts("No --explanation_level was provided.")
        try:
            typed_level = input(
                "Enter explanation level for this session (beginner/intermediate/advanced), "
                "or press Enter to use the automatic per-session level: "
            ).strip().lower()
        except EOFError:
            typed_level = ""
        if typed_level in QUESTION_LEVEL_RANK:
            experimenter_level_override = typed_level
        elif typed_level:
            print_ts(f"'{typed_level}' is not a recognized level; using the automatic per-session level instead.")

    MODEL_NAME = args.chat_model
    NOD_SEQUENCE_NAME = args.nod_sequence
    EXPRESSION_TIMING = args.expression_timing

    preset_width, preset_height, preset_fps = RESOLUTION_MAP[args.resolution]
    if "CAMERA_WIDTH" not in os.environ:
        CAMERA_WIDTH = preset_width
    if "CAMERA_HEIGHT" not in os.environ:
        CAMERA_HEIGHT = preset_height
    if "CAMERA_FPS" not in os.environ:
        CAMERA_FPS = preset_fps

    emotion_smoothing_enabled = EMOTION_SMOOTHING_ENABLED and not args.disable_emotion_smoothing
    emotion_smoothing_alpha = args.emotion_smoothing_alpha

    print()

    ensure_data_dirs()

    check_ollama_available()
    ensure_model_available(MODEL_NAME)

    if SELF_RAG_ENABLED:
        ensure_model_available(SELF_RAG_EMBED_MODEL)

    client = Client(host=OLLAMA_HOST)

    self_rag_store = init_self_rag_store(client)

    list_input_devices()

    print_ts("Loading Silero VAD...")
    try:
        silero_model = load_silero_vad()
        print_ts("Silero VAD ready.")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Silero VAD. Install it with:\n"
            "pip install silero-vad torch\n\n"
            f"Original error: {exc}"
        )

    print_ts(f"Name spelling enabled: {ENABLE_NAME_SPELLING}")
    print_ts(f"Returning-user memory greeting enabled: {ENABLE_RETURNING_USER_MEMORY_GREETING}")

    print_ts("Loading faster-whisper...")
    try:
        whisper_model = WhisperModel(
            FAST_WHISPER_CONFIG["model"],
            device=FAST_WHISPER_CONFIG["device"],
            compute_type=FAST_WHISPER_CONFIG["compute_type"],
        )
        print_ts("faster-whisper ready.")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load faster-whisper with active config:\n"
            f"{json.dumps(FAST_WHISPER_CONFIG, indent=2)}\n\n"
            "Install it with:\n"
            "pip install faster-whisper\n\n"
            f"Original error: {exc}"
        )

    # ---- Robot output: Tritium TTS (with EMA-based activity detection, integrated from ameca_warm_up.py) ----
    robot_speaker = RobotSpeaker(
        tts_url=args.tts_url,
        tts_token=args.tts_token,
        speaking_cooldown_s=args.speaking_cooldown,
        activity_debounce_seconds=args.tts_activity_debounce,
    )

    # ---- Robot output: Tritium facial expression (sequence_player) ----
    # Every response gets a facial expression via set_emotion(); negative
    # emotions are always remapped to a neutral/attentive sequence (see
    # EMOTION_SEQUENCE_MAP), so the physical face never shows sadness,
    # anger, fear, or disgust. speak_with_turn_end_cue() now also waits
    # for the sequence's reported "expected_duration" to elapse before
    # Ameca starts talking (when EXPRESSION_TIMING == "before").
    robot_expression = RobotExpression(
        host=args.expression_host,
        tts_token=args.tts_token,
    )

    # Optional TTS-activity monitor (avoids the robot hearing its own voice,
    # and is also what powers the more accurate "has Ameca finished
    # speaking?" signal used for turn-end nod timing -- see RobotSpeaker).
    if HAS_TTS_ACTIVITY_MONITOR:
        try:
            import asyncio as _asyncio
            dev_id, name, scale = find_target_device()
            if dev_id:
                tts_monitor_thread = threading.Thread(
                    target=lambda: _asyncio.run(listen_levels_for_device(dev_id, name, scale)),
                    daemon=True,
                )
                tts_monitor_thread.start()
                print_ts("[TTS] TTS activity monitor started.")
            else:
                print_ts("[WARN] Acapela/Tritium output device not found; TTS activity monitor disabled.")
        except Exception as exc:
            print_ts(f"[WARN] Could not start TTS activity monitor: {exc}")

  
    background_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    background_futures: list[concurrent.futures.Future] = []

    session_log: list[dict] = []

    # ---- Participant ID: requested up front, before name capture, so
    # session storage is keyed by it from the very start of the run ----
    participant_id = resolve_participant_id(
        args.participant_id,
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        session_log=session_log,
    )
    participant_folder = slugify_name(participant_id)

    # ---- ZED camera: session video recording, and per-turn face-crop
    # capture (both use the same shared camera; not used for emotion) ----
    camera: Optional[Camera] = None
    video_recorder: Optional[SessionVideoRecorder] = None
    video_path: Optional[str] = None
    if not args.disable_video_recording:
        try:
            camera = Camera(args.camera)
            video_filename = f"{participant_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = os.path.join(VIDEOS_DIR, video_filename)
            video_recorder = SessionVideoRecorder(
                camera=camera,
                output_path=video_path,
                fps=args.video_fps,
                fourcc=args.video_fourcc,
            )
            video_recorder.start()
            print_ts(f"[ZED] Session video recording started: {video_path}")
        except Exception as exc:
            print_ts(f"[WARN] Could not start ZED camera session recording: {exc}")
            camera = None
            video_recorder = None
            video_path = None

    if camera is None:
        print_ts(
            "[INFO] No camera available (video recording disabled or camera init "
            "failed); per-turn face images will not be captured this session."
        )

    session_info = prompt_for_user_name(
        client=client,
        whisper_model=whisper_model,
        silero_model=silero_model,
        input_device=INPUT_DEVICE,
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        participant_id=participant_id,
        session_log=session_log,
    )
    user_key = session_info.user_key
    user_profile = session_info.user_profile
    intro_reply = session_info.intro_reply
    session_number = session_info.session_number
    explanation_level = session_info.explanation_level
    if experimenter_level_override:
        explanation_level = experimenter_level_override
        print_ts(f"Explanation level set by experimenter keyboard input: {explanation_level}")
    lesson_state = session_info.lesson_state

    print_ts(
        f"Experiment session number: {session_number} "
        f"(explanation level: {explanation_level}, "
        f"first session after warm-up: {session_info.is_first_after_warmup})"
    )

    speak_with_turn_end_cue(
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        text=intro_reply,
        emotion="neutral",
        confidence=1.0,
        disable_expression=args.disable_expression,
        force_expression=True,
    )

    print()

    if user_profile.get("conversation_summary"):
        preview = compact_previous_summary_for_greeting(
            str(user_profile.get("conversation_summary", "")),
            max_chars=180,
        )
        print_ts("Loaded previous conversation summary for continuity.")
        print_ts(f"Memory preview: {preview}")
        print()

    print("Automatic listening mode is active.")
    print("Say '/exit', or say a farewell such as 'goodbye', to save the transcript and quit.")
    print()

    history: list[dict] = []

    smoothed_emotion_scores: Optional[dict[str, float]] = None

    llm_call_samples: list[dict] = []

    session_log.append({
        "role": "assistant",
        "content": intro_reply,
        "timestamp": now_ts(),
        "intent": "self_introduction",
    })
    history.append({"role": "assistant", "content": intro_reply})

    # ---- Session recap + two review questions, for session 2+ only ----
    # (see ParticipantSessionInfo.needs_recap / run_session_recap_qa()).
    if session_info.needs_recap:
        wants_recap = ask_recap_consent(
            whisper_model=whisper_model,
            silero_model=silero_model,
            input_device=INPUT_DEVICE,
            robot_speaker=robot_speaker,
            robot_expression=robot_expression,
            disable_expression=args.disable_expression,
            session_log=session_log,
            history=history,
        )
        if wants_recap:
            run_session_recap_qa(
                client=client,
                whisper_model=whisper_model,
                silero_model=silero_model,
                input_device=INPUT_DEVICE,
                robot_speaker=robot_speaker,
                robot_expression=robot_expression,
                disable_expression=args.disable_expression,
                session_log=session_log,
                history=history,
                user_profile=user_profile,
                explanation_level=explanation_level,
            )

    ask_topic_choice_question(
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        disable_expression=args.disable_expression,
        session_log=session_log,
        history=history,
    )

    turn_index = 0
    turns_since_comprehension_check = 0
    pending_scaffold: Optional[dict] = None
    pending_asr_correction: Optional[dict] = None

    try:
        while True:
            wav_path, turn_frames = listen_for_utterance_with_silero_vad(
                input_device=INPUT_DEVICE,
                silero_model=silero_model,
                prompt_label="utterance",
                robot_speaker=robot_speaker,
                camera=camera,
            )

            if not wav_path:
                continue

            turn_index += 1

            try:
                user_text = transcribe_with_faster_whisper(wav_path, whisper_model)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            if not user_text:
                print_ts("No speech detected after transcription (faster-whisper returned an empty transcript).")
                continue

            print_ts(f"Transcript [faster-whisper]: {user_text}")

            # ---------- Spoken farewell: terminate the session ----------
            if is_farewell_utterance(user_text):
                farewell_reply = "Thank you, and take care. \U0001F642"
                print_ts(f"Assistant: {farewell_reply}")
                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text=farewell_reply,
                    emotion="neutral",
                    confidence=1.0,
                    disable_expression=args.disable_expression,
                )
                print()

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": farewell_reply})

                session_log.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "input_mode": "silero_vad_faster-whisper",
                    "intent": "spoken_farewell_termination",
                })
                session_log.append({
                    "role": "assistant",
                    "content": farewell_reply,
                    "timestamp": now_ts(),
                    "intent": "spoken_farewell_termination",
                })

                print_ts("Spoken farewell detected. Ending session.")
                break

            command = normalize_command(user_text)

            if command in {"exit", "quit"}:
                print_ts("Goodbye.")
                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text="Goodbye, and thank you for talking with me.",
                    disable_expression=args.disable_expression,
                )
                break

            if command == "clear":
                history.clear()
                print_ts("Conversation history cleared.")
                continue

            if command in {"rrlab crawl", "crawl rrlab", "scrape rrlab", "rrlab scrape"}:
                ok = run_rrlab_scraper()
                if ok:
                    print_ts("RRLab website knowledge base rebuilt. Self-RAG will use the updated ChromaDB index.")
                else:
                    print_ts("RRLab website knowledge base was not rebuilt.")
                continue

            if command in {"rag reindex", "reindex rag", "selfrag reindex", "self-rag reindex"}:
                if resolve_scrape_script_path():
                    ok = run_rrlab_scraper()
                    if ok:
                        print_ts("Self-RAG RRLab website index rebuilt from scrape.py.")
                    else:
                        print_ts("scrape.py failed; falling back to local knowledge folder indexing.")
                        index_self_rag_knowledge(self_rag_store)
                else:
                    index_self_rag_knowledge(self_rag_store)
                    print_ts("Self-RAG local knowledge base reindexed.")
                continue

            if command in {"rag rebuild", "rebuild rag", "selfrag rebuild", "self-rag rebuild"}:
                self_rag_store = rebuild_self_rag_collection(self_rag_store)
                continue

            # ---------- Resolve a pending ASR-mishearing confirmation ----------
            # If the previous turn ended by asking "did you mean X, not Y?",
            # this turn's utterance is the answer to that question, not a
            # new topic in its own right -- substitute the corrected (or,
            # if declined, original) text and fall through to normal
            # processing using it.
            if pending_asr_correction is not None:
                correction = pending_asr_correction
                pending_asr_correction = None
                raw_confirmation_reply = user_text

                wants_correction = parse_yes_no(raw_confirmation_reply, default=True)
                user_text = (
                    apply_asr_correction(
                        correction["original_text"],
                        correction["heard_phrase"],
                        correction["corrected_phrase"],
                    )
                    if wants_correction
                    else correction["original_text"]
                )
                print_ts(
                    f"[ASR] Correction {'confirmed' if wants_correction else 'declined'}; "
                    f"proceeding with: {user_text!r}"
                )

                session_log.append({
                    "role": "user",
                    "content": raw_confirmation_reply,
                    "timestamp": now_ts(),
                    "intent": "asr_correction_confirmation_response",
                })
                history.append({"role": "user", "content": raw_confirmation_reply})
                # user_text now holds the effective text for this turn;
                # falls through to the normal processing try-block below.

            # ---------- Detect a NEW likely ASR mishearing ----------
            # Only checked when we're not already resolving one above (a
            # confirmation reply like "yes"/"no" should never itself be
            # re-scanned for mishearings).
            elif (misrecognition := find_likely_asr_misrecognition(user_text)) is not None:
                heard_phrase, corrected_phrase = misrecognition
                confirmation_text = normalize_reply(
                    build_asr_correction_confirmation(heard_phrase, corrected_phrase),
                    "neutral",
                    1.0,
                )
                print_ts(f"[ASR] Possible mishearing detected: '{heard_phrase}' -> '{corrected_phrase}'")

                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text=confirmation_text,
                    emotion="neutral",
                    disable_expression=args.disable_expression,
                )

                session_log.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "intent": "possible_asr_misrecognition",
                })
                session_log.append({
                    "role": "assistant",
                    "content": confirmation_text,
                    "timestamp": now_ts(),
                    "intent": "asr_correction_confirmation_request",
                })
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": confirmation_text})

                pending_asr_correction = {
                    "original_text": user_text,
                    "heard_phrase": heard_phrase,
                    "corrected_phrase": corrected_phrase,
                }
                continue

            try:
                parallel_start = time.time()
                should_classify_question_level = pending_scaffold is None and looks_like_question(user_text)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as concurrent_executor:
                    text_emotion_future = concurrent_executor.submit(
                        detect_emotion,
                        client=client,
                        transcribed_text=user_text,
                    )
                    self_rag_future = concurrent_executor.submit(
                        build_self_rag_context,
                        client=client,
                        store=self_rag_store,
                        user_text=user_text,
                    )
                    question_level_future = (
                        concurrent_executor.submit(classify_question_level, client, user_text)
                        if should_classify_question_level
                        else None
                    )

                    planner_future = concurrent_executor.submit(
                        classify_dialogue_turn,
                        client=client,
                        user_text=user_text,
                        lesson_state=lesson_state,
                        history=history,
                        safe_json_extract=safe_json_extract,
                        model_name=MODEL_NAME,
                        print_ts=print_ts,
                    )
                    text_emotion_result = text_emotion_future.result()
                    text_response_seconds = time.time() - parallel_start

                    self_rag_context = self_rag_future.result()
                    self_rag_response_seconds = time.time() - parallel_start

                    question_level = question_level_future.result() if question_level_future else None
                    planner_output = planner_future.result()

                    # Capture BEFORE apply_planner_output() clears the flag
                    # (it consumes/resets awaiting_retention_check as part
                    # of applying this turn's planner decision) -- needed
                    # further down to give the retention-check reply a
                    # tighter word budget than a normal "continue" answer.
                    was_awaiting_retention_check = (
                        lesson_state.awaiting_retention_check if lesson_state else False
                    )

                    apply_planner_output(lesson_state, planner_output)

                    is_multi_part_turn = (
                        looks_like_multi_part_question(user_text)
                        or bool(planner_output.get("new_pending_questions"))
                    )

                text_emotion_assessment = assess_text_emotion(
                    text_emotion=text_emotion_result,
                    user_text=user_text,
                    response_times={
                        "text_seconds": text_response_seconds,
                    },
                )

                # ---- Temporal smoothing across turns (keeps Ameca adaptive
                # to genuine emotional change without flickering on noise) ----
                if emotion_smoothing_enabled:
                    smoothed_emotion_scores = apply_temporal_emotion_smoothing(
                        current_scores=text_emotion_assessment.scores,
                        previous_smoothed_scores=smoothed_emotion_scores,
                        alpha=emotion_smoothing_alpha,
                    )
                    smoothed_dominant, smoothed_confidence = dominant_from_scores(smoothed_emotion_scores)
                    emotion_result = EmotionResult(
                        emotion=smoothed_dominant,
                        confidence=smoothed_confidence,
                        reason=text_emotion_assessment.reason,
                    )
                else:
                    emotion_result = text_emotion_assessment.to_emotion_result()

                text_emotion_json = {
                    "emotion": text_emotion_result.emotion,
                    "confidence": text_emotion_result.confidence,
                    "reason": text_emotion_result.reason,
                }

                emotion_json = text_emotion_assessment.as_json
                emotion_json["temporal_smoothing"] = {
                    "enabled": emotion_smoothing_enabled,
                    "alpha": emotion_smoothing_alpha,
                    "smoothed_scores": smoothed_emotion_scores,
                    "smoothed_emotion": emotion_result.emotion,
                    "smoothed_confidence": emotion_result.confidence,
                }
                emotion_json["is_negative"] = emotion_result.emotion in NEGATIVE_EMOTIONS

                print_ts("Text-only emotion resolution JSON (raw, pre-smoothing):")
                print(json.dumps(text_emotion_assessment.as_json, indent=2))
                print()

                print_ts(
                    f"Smoothed emotion used for tone/expression: {emotion_result.emotion} "
                    f"(confidence={emotion_result.confidence:.2f}, alpha={emotion_smoothing_alpha}, "
                    f"negative={emotion_result.emotion in NEGATIVE_EMOTIONS})"
                )
                print()

                print_ts(
                    f"Self-RAG JSON (computed concurrently with emotion detection, {self_rag_response_seconds:.2f}s, "
                    f"trigger={self_rag_context.trigger}):"
                )
                print(json.dumps(self_rag_context.as_json, indent=2))
                print()

                # ---- Comprehension-check scheduling (sessions 2+ only) ----
                turns_since_comprehension_check += 1
                ask_comprehension_check = (
                    explanation_level != "beginner"
                    and turns_since_comprehension_check >= COMPREHENSION_CHECK_INTERVAL
                )
                if ask_comprehension_check:
                    turns_since_comprehension_check = 0

                
                scaffold_stage: Optional[str] = None
                if pending_scaffold is not None:
                    
                    current_rank = QUESTION_LEVEL_RANK.get(pending_scaffold["current_level"], 1)
                    target_rank = QUESTION_LEVEL_RANK.get(pending_scaffold["target_level"], current_rank)
                    if current_rank >= target_rank:
                        scaffold_stage = "final_answer"
                    else:
                        new_rank = current_rank + 1
                        pending_scaffold["current_level"] = level_name_for_rank(new_rank)

                        scaffold_stage = "final_answer" if new_rank >= target_rank else "ask_leading"
                elif (
                    ENABLE_SCAFFOLD_MODE
                    and question_level
                    and QUESTION_LEVEL_RANK.get(question_level, 0) > QUESTION_LEVEL_RANK.get(explanation_level, 0)
                ):
                    
                    pending_scaffold = {
                        "original_question": user_text,
                        "target_level": question_level,
                        "current_level": explanation_level,
                    }
                    scaffold_stage = "ask_leading"
                    print_ts(
                        f"Question classified as '{question_level}' level, above this "
                        f"session's '{explanation_level}' level; starting scaffolded Q&A."
                    )

                session_context = SessionContext(
                    session_number=session_number,
                    explanation_level=explanation_level,
                    spelt_name=str(user_profile.get("name", "the participant")),
                    ask_comprehension_check=ask_comprehension_check,
                    scaffold_mode=pending_scaffold is not None,
                    scaffold_stage=scaffold_stage,
                    scaffold_target_level=(pending_scaffold or {}).get("target_level"),
                    scaffold_current_level=(pending_scaffold or {}).get("current_level"),
                    scaffold_original_question=(pending_scaffold or {}).get("original_question"),
                )

                # Capture a plain-dict snapshot for session_log BEFORE
                # possibly clearing pending_scaffold below, so the logged
                # turn still shows what scaffold state was active for it.
                scaffold_log = {
                    "active": pending_scaffold is not None,
                    "stage": scaffold_stage,
                    "target_level": (pending_scaffold or {}).get("target_level"),
                    "current_level": (pending_scaffold or {}).get("current_level"),
                    "original_question": (pending_scaffold or {}).get("original_question"),
                }

                generation_debug: dict = {}
                generated_reply = generate_response(
                    client=client,
                    user_text=user_text,
                    emotion_result=emotion_result,
                    history=history,
                    user_profile=user_profile,
                    self_rag_context=self_rag_context,
                    llm_call_samples=llm_call_samples,
                    session_context=session_context,
                    lesson_state=lesson_state,
                    force_multi_part=is_multi_part_turn,
                    awaiting_retention_check=was_awaiting_retention_check,
                    generation_debug=generation_debug,
                )


                generated_reply, verification_log = verify_and_repair_reply(
                    client=client,
                    user_text=user_text,
                    generated_reply=generated_reply,
                    session_context=session_context,
                    emotion_result=emotion_result,
                    force_multi_part=is_multi_part_turn,
                )

                finalize_lesson_state_after_reply(lesson_state, planner_output)

                reply = generated_reply.text
                response_emotion = generated_reply.response_emotion

                # The scaffold completes once this turn's reply has fully
                # answered the original (above-level) question.
                if scaffold_stage == "final_answer":
                    pending_scaffold = None

                print_ts(f"Assistant: {reply}")
                print_ts(
                    f"Congruent response expression: {response_emotion} "
                    f"(detected user emotion: {emotion_result.emotion}, "
                    f"confidence={emotion_result.confidence:.2f})"
                )

                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text=reply,
                    emotion=response_emotion,
                    confidence=emotion_result.confidence,
                    disable_expression=args.disable_expression,
                )
                print()
                user_message = {
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "emotion": emotion_json,
                    "text_emotion": text_emotion_json,
                    "response_emotion": response_emotion,
                    "self_rag": self_rag_context.as_json,
                    "isuse_check": (
                        {"enabled": True, "is_useful": None, "reason": "pending"}
                        if ENABLE_ISUSE_CHECK
                        else {"enabled": False}
                    ),
                    "input_mode": "silero_vad_faster-whisper_text_only_emotion_ekman_temporal_smoothing_self_rag",
                    "face_images": [],
                    "session_number": session_number,
                    "explanation_level": explanation_level,
                    "comprehension_check_asked": ask_comprehension_check,
                    "scaffold": scaffold_log,
                    "response_verification": verification_log,
                    "generation_debug": generation_debug or None,
                }

                assistant_message = {
                    "role": "assistant",
                    "content": reply,
                    "timestamp": now_ts(),
                }

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": reply})
                history = trim_history(history)

                session_log.append(user_message)
                session_log.append(assistant_message)

                # Bind current-iteration values as default args so these
                # closures don't fall victim to Python's late-binding
                # (user_text/reply/self_rag_context/turn_frames/turn_index
                # are all reassigned on the next loop iteration).
                def _run_isuse_check(
                    _user_text=user_text,
                    _reply=reply,
                    _self_rag_context=self_rag_context,
                    _message=user_message,
                ) -> None:
                    result = judge_response_usefulness(
                        client=client,
                        user_text=_user_text,
                        reply=_reply,
                        self_rag_context=_self_rag_context,
                    )
                    _message["isuse_check"] = result

                def _run_face_crop_save(
                    _frames=turn_frames,
                    _folder=participant_folder,
                    _turn_index=turn_index,
                    _message=user_message,
                ) -> None:
                    paths = save_turn_face_crops(
                        frames=_frames,
                        participant_folder=_folder,
                        turn_index=_turn_index,
                    )
                    _message["face_images"] = paths

                background_futures.append(background_executor.submit(_run_isuse_check))
                background_futures.append(background_executor.submit(_run_face_crop_save))

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print_ts(f"[ERROR] Unexpected error while processing this turn: {exc!r}")
                import traceback as _traceback
                _traceback.print_exc()

                apology = "I'm sorry, something went wrong on my end with that. Could you try again?"
                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text=apology,
                    disable_expression=args.disable_expression,
                )

                session_log.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "input_mode": "silero_vad_faster-whisper",
                    "intent": "turn_processing_error",
                    "error": repr(exc),
                })
                session_log.append({
                    "role": "assistant",
                    "content": apology,
                    "timestamp": now_ts(),
                    "intent": "turn_processing_error_fallback",
                })

    except KeyboardInterrupt:
        print()
        print_ts("Goodbye.")

    finally:
        if video_recorder is not None:
            saved_video_path = video_recorder.stop()
            if saved_video_path:
                video_path = saved_video_path
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass

        if background_futures:
            print_ts(
                f"Waiting briefly for {len(background_futures)} background task(s) "
                "(usefulness checks, face-crop saves) before saving the transcript..."
            )
            concurrent.futures.wait(background_futures, timeout=15)
        background_executor.shutdown(wait=False)

        if session_log:
            session_path = save_session_transcript(
                user_key=user_key,
                user_profile=user_profile,
                session_log=session_log,
                participant_id=participant_id,
                video_path=video_path,
                llm_call_samples=llm_call_samples,
                session_number=session_number,
                explanation_level=explanation_level,
                is_first_after_warmup=session_info.is_first_after_warmup,
            )

            update_user_after_session(
                client=client,
                user_key=user_key,
                session_path=session_path,
                session_log=session_log,
                lesson_state=lesson_state,
            )

            record_session_completion(
                user_key=user_key,
                session_number=session_number,
                explanation_level=explanation_level,
                session_log=session_log,
            )

            print_ts(f"Conversation transcript saved to: {session_path}")
        else:
            print_ts("No conversation messages to save.")


if __name__ == "__main__":
    main()
    