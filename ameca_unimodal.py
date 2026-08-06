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
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

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


# =========================
# Per-turn face-crop configuration
# =========================
#
# Saves up to IMAGES_PER_TURN cropped face images per conversational turn
# (ported from ameca_warm_up.py's baseline/test-round image capture).
# This is capture/cropping ONLY -- no DeepFace, no facial emotion
# classification is reintroduced here; emotion detection in this pipeline
# stays text-only (see EKMAN_EMOTIONS / detect_emotion()).

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

SELF_RAG_REINDEX_ON_START = os.environ.get("SELF_RAG_REINDEX_ON_START", "0") == "1"
SELF_RAG_AUTO_SCRAPE_ON_EMPTY = os.environ.get("SELF_RAG_AUTO_SCRAPE_ON_EMPTY", "0") == "1"
SELF_RAG_SCRAPE_SCRIPT = os.environ.get("SELF_RAG_SCRAPE_SCRIPT", "scrape2.py")
SELF_RAG_SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".py", ".html", ".htm", ".pdf"}

ENABLE_LLM_SESSION_SUMMARY = os.environ.get("ENABLE_LLM_SESSION_SUMMARY", "1") == "1"
ENABLE_NAME_SPELLING = os.environ.get("ENABLE_NAME_SPELLING", "1") == "1"
ASK_SPELLED_NAME_ON_START = os.environ.get("ASK_SPELLED_NAME_ON_START", "1") == "1"

ENABLE_RETURNING_USER_MEMORY_GREETING = os.environ.get("ENABLE_RETURNING_USER_MEMORY_GREETING", "1") == "1"
RETURNING_USER_GREETING_MAX_SUMMARY_CHARS = int(os.environ.get("RETURNING_USER_GREETING_MAX_SUMMARY_CHARS", "420"))


# =========================
# Ameca identity prompt
# =========================

AMECA_SYSTEM_PROMPT = {
  "role": "Ameca, a humanoid social robot used in a university laboratory for research and demonstrations.",
  "identity": [
        "You are a robot, not a human. Speak in a friendly, professional tone. Refer to yourself as a robot when relevant.",
        "You were developed by a robotics company EngineeredArts in 2021 with model name Gen1 Ameca.",
        "Robotics Research laboratory purchased you in 2022 for human-robot interaction research experiments.",
        "In the current experiment running in July 2026, you act as a teaching assistant for university students, strictly limited to the topics of Artificial Intelligence and Robotics.",
        "The name \"EMAH\" refers to a research system/software pipeline that runs on you; it is NOT your name. You are always Ameca, never Emah -- never say \"I am Emah\" or introduce yourself as Emah, even if retrieved lab knowledge mentions EMAH.",
    ],
    "capability_boundaries": [
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
        "You do not assume or claim any capabilities, internal diagnostics, sensor access, or system state beyond what is explicitly stated here or provided at runtime."
        "You are able to detect emotion from text"
        "You have continuity memory through SELF-RAG CONTEXT, locally stored user profiles and conversation summaries.",
    ],
    "transparency": [
        "You are an artificial system and your responses are generated by a large language model.",
        "Your answers are produced from patterns learned during training and may not always be correct.",
        "If you are uncertain about information, say so instead of guessing.",
        "Do not fabricate facts",
    ],
    "task": [
        "You act as a teaching assistant for students, strictly limited to Artificial Intelligence and Robotics topics",
        "Hold a natural teaching conversation with the user.",
        "Your focus is on fundamental knowledge in Artificial Intelligence and Robotics"
        "Answer questions clearly and ask brief follow-up questions when helpful.",
        "Keep responses concise unless the user asks for more detail.",
        "You may reference previous conversations, prior discussion topics, and saved user preferences only when they are present in the provided local memory context.",
        "Many users are visitors from outside computer science and may not know what to ask you. When that happens, use questions in the beginner section in possible_topics below to offer them a friendly starting point instead of leaving them stuck.",
    ],
    "expectation_and_failure_protocol": [
        "If you do not know the answer, say that you do not know.",
        "Do not fabricate facts.",
        "If the request is unclear, ask one clarifying question.",
        "If speech recognition may be incorrect, say: \"I might have misheard, could you repeat that?\"",
        "If the user asks whether you remember previous conversations, explain that you can continue from the saved local conversation summary when one is available.",
        "If the user's question is NOT about AI or Robotics, do not answer it from general knowledge. Tell them plainly and briefly that it is outside what you have context for here, and that you can only help with, AI and Robotics topics.",
        "For anything specific to this lab (people, current projects, robots, publications, events), rely ONLY on the SELF-RAG CONTEXT retrieved from the crawled lab website.",
        "If no SELF-RAG CONTEXT was used this turn, or it does not contain the answer, say plainly that you do not currently have context on that specific point rather than guessing or inventing details."
        "possible_topics is a private reference list of things you know how to teach, organized from beginner to advanced. It is for your own use in generating suggestions -- never read it out loud, never mention the words 'beginner', 'intermediate', 'advanced', or 'topic list', and never dump the list verbatim.",
    ],
    "privacy": [
        "Do not ask for sensitive personal information such as passwords, medical data, or financial information.",
        "Treat the conversation as ephemeral and do not claim to store user data."
    ],
    "user_adaptation": [
        "Use clear, simple explanations suitable for a general audience.",
        "Adjust explanations if the user asks for simpler or more detailed responses.",
    ],
    "ethical_red_lines": [
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
    "device": "cuda",
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
#

EKMAN_EMOTIONS = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
    "disgust": "🤢",
    "neutral": "🙂",
}

NEGATIVE_EMOTIONS = {"anger", "fear", "disgust"}

# Emotions whose emoji should only be shown when the fused/smoothed
# confidence clears a minimum bar -- otherwise the emoji falls back to
# neutral. Without this, a low-confidence "anger" or "sadness" reading
# (e.g. a curt correction or a repeated question) could still stamp an
# angry/sad face on an entirely matter-of-fact reply, since emoji
# selection previously used only the dominant emotion label with no
# confidence check at all (unlike EXPRESSION_MIN_CONFIDENCE, which
# already gates the physical facial expression).
EMOJI_STRONG_EMOTIONS = {"sadness", "anger", "fear", "disgust"}
EMOJI_MIN_CONFIDENCE_FOR_STRONG_EMOTION = float(
    os.environ.get("EMOJI_MIN_CONFIDENCE_FOR_STRONG_EMOTION", "0.5")
)

ALLOWED_FACE_EMOJIS = set(EKMAN_EMOTIONS.values())

# =========================
# Reliability-aware emotion resolution (text-only; prosody modality removed)
# =========================
# FUSION_TEXT_WEIGHT is kept (at 1.0) purely so the existing
# FusedEmotionResult/weights bookkeeping keeps working unchanged now that
# text is the only modality -- there is nothing left to weight it against.
FUSION_TEXT_WEIGHT = float(os.environ.get("FUSION_TEXT_WEIGHT", "1.0"))

FUSION_ENABLE_SEMANTIC_OVERRIDE = os.environ.get("FUSION_ENABLE_SEMANTIC_OVERRIDE", "1") == "1"
FUSION_EXPLICIT_TEXT_CONFIDENCE = float(os.environ.get("FUSION_EXPLICIT_TEXT_CONFIDENCE", "0.72"))

EMOTION_SMOOTHING_ENABLED = os.environ.get("EMOTION_SMOOTHING_ENABLED", "1") == "1"
EMOTION_SMOOTHING_ALPHA = float(os.environ.get("EMOTION_SMOOTHING_ALPHA", "0.6"))

# =========================
# Response length configuration
# =========================

MAX_REPLY_SENTENCES = int(os.environ.get("MAX_REPLY_SENTENCES", "2"))

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
    "joy": os.environ.get("SEQ_EMOTION_JOY", "Smile"),
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
class FusedEmotionResult:
    """
    NOTE: despite the name (kept for compatibility with session-log
    schemas and downstream tooling that already expect this shape), this
    is no longer a multi-modality fusion result -- text is the only
    modality left (prosody and vision were both removed). It still wraps
    the text-only EmotionResult in the same scores/weights/reason shape
    so the rest of the pipeline (temporal smoothing, session logging,
    response generation) didn't need to change.
    """
    emotion: str
    confidence: float
    reason: str
    scores: dict[str, float]
    weights: dict[str, float]
    text_emotion: dict[str, Any]
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
            "weights": self.weights,
            "text_emotion": self.text_emotion,
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

    @property
    def as_json(self) -> dict:
        return {
            "available": self.available,
            "used": self.used,
            "query": self.query,
            "sources": self.sources,
            "reason": self.reason,
            "error": self.error,
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
    """
    Rough estimate of how long Tritium will take to actually speak `text`
    out loud. Still used as an initial "speaking tail" bump right when a
    say() call goes out (so the barge-in echo guard engages immediately,
    before the TTS-activity monitor has a chance to report real audio
    levels), but is NO LONGER the primary signal for "has Ameca finished
    speaking" -- see RobotSpeaker.is_speaking_or_cooling_down(), which now
    prefers the live TTS-activity EMA when available (integrated from
    ameca_warm_up.py) for a more accurate turn-end / nod-timing signal.
    """
    words = len(text.split())
    if words == 0:
        return min_seconds
    seconds = (words / words_per_minute) * 60.0
    return max(min_seconds, seconds) + padding_seconds

class RobotSpeaker:
    """
    Thin wrapper around the Tritium TTS PUT API, with an EMA-based
    TTS-activity echo guard integrated from ameca_warm_up.py.RobotSpeaker.

    Previously this class decided "is Ameca still speaking?" purely from
    a word-count-based duration ESTIMATE plus a fixed cooldown. That is
    kept as an immediate fallback bump right when say() is called (so the
    barge-in guard engages before any real audio-level reading exists),
    but the authoritative signal -- used for wait_until_finished() and
    therefore for exactly when the turn-end nod fires -- is now the live
    TTS-activity EMA (current_ema()) with a debounce hold, matching
    ameca_warm_up.py. This is more accurate for longer/irregularly-paced
    replies than a fixed words-per-minute estimate.
    """

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
        """
        Push out the "definitely still speaking" deadline. `extra` (the
        estimated duration of the whole utterance, from
        estimate_speech_duration_seconds()) is always folded in as a
        FLOOR here -- previously, when the live TTS-activity monitor was
        available, `extra` was discarded entirely and only the short
        fixed `speaking_cooldown_s` (0.3s default) was used. That meant a
        normal in-sentence pause (e.g. the brief dip in audio level after
        a comma) could last longer than the EMA debounce window and get
        misread as "Ameca has stopped talking", causing the turn-end nod
        (or the next expression) to fire mid-sentence. Using the full
        estimated duration as a floor keeps is_speaking_or_cooling_down()
        reporting "still speaking" for the whole utterance regardless of
        such transient dips; the live EMA signal still takes over after
        that floor for cases where real speech runs longer than the
        estimate.
        """
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
    """
    Speaks `text`, drives the facial expression for `emotion` (per
    EXPRESSION_TIMING), and -- if NOD_AFTER_SPEECH_ENABLED -- plays the
    turn-end nod at the RIGHT TIME: only once RobotSpeaker.wait_until_
    finished() confirms (via the live TTS-activity EMA, not a rough
    estimate) that Ameca has actually stopped talking.

    When EXPRESSION_TIMING == "before" (the default), the facial
    expression is set FIRST and then -- per product requirement -- this
    function waits for that expression animation to actually finish
    playing (using the "expected_duration" the Tritium sequence_player
    API reports back for the sequence it just started) before Ameca
    begins speaking. This avoids the previous behavior of firing the
    expression and immediately talking over it while the face is still
    mid-animation.
    """
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
    """
    Thin wrapper around the Tritium sequence_player PUT API, used to drive
    Ameca's PHYSICAL facial expression from the fused emotion result.

    Runs every turn as soon as fusion resolves a dominant emotion, so
    EVERY response gets a corresponding facial expression. Per product
    requirement, negative emotions (sadness/anger/fear/disgust) are never
    displayed on the physical face -- EMOTION_SEQUENCE_MAP routes all of
    them to a calm/attentive neutral sequence instead, while the spoken
    reply is what actually acknowledges the negative emotion.
    """

    def __init__(self, host: str = "http://emah", tts_token: str = "", timeout: float = 3.0) -> None:
        self.host = host.rstrip("/")
        self.token = tts_token
        self.timeout = timeout
        self.last_emotion: Optional[str] = None
        # Set from the Tritium API's "expected_duration" field on the most
        # recent successful _play_sequence() call. Used by
        # speak_with_turn_end_cue() to wait for the facial expression
        # animation to actually finish before Ameca starts talking.
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
        """
        Update the robot's facial expression to match `emotion`.

        Negative emotions are remapped to a neutral/attentive sequence via
        EMOTION_SEQUENCE_MAP before this method ever sees them reach the
        Tritium API, so the physical face never shows sadness, anger,
        fear, or disgust -- only joy, surprise, or neutral.

        - Falls back to the "neutral" sequence if `emotion` is
          unrecognized or if `confidence` is below EXPRESSION_MIN_CONFIDENCE.
        - By default (force=None -> uses EXPRESSION_FORCE_REPLAY_SAME),
          skips re-sending the same sequence back-to-back so the face
          doesn't restart the same animation every single turn when the
          mood hasn't changed. Pass force=True to always resend.

        Returns the "expected_duration" (seconds) of the animation that
        was actually played, or None if nothing was played / it wasn't
        reported by Tritium.
        """
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
    """
    Continuously reads the shared Camera during a single utterance and
    retains sampled frames (every CAMERA_SAMPLE_EVERY_SECONDS), so that
    once the utterance ends, up to IMAGES_PER_TURN of the sharpest frames
    can be searched for a usable face crop (see find_local_face_crops()).

    A new FrameCollector (and thread) is created per utterance -- ported
    as-is from ameca_warm_up.py, including the note below about avoiding
    cv2.imshow()/cv2.waitKey() here.

    NOTE: this deliberately does NOT call cv2.imshow()/cv2.waitKey(). A
    new FrameCollector (and therefore a new thread) is created for every
    single utterance, and OpenCV's Qt-based HighGUI backend on Linux is
    not safe to drive from a different thread each time a window is
    reused -- doing so was observed to work for the first utterance only,
    then silently stop delivering frames for every subsequent one.
    """

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
# Ported from ameca_warm_up.py's local face-region detection, cropping,
# and image-saving helpers. This pipeline does NOT run DeepFace, so there
# is no face-presence "confirmation" step here (unlike
# find_deepface_confirmed_crops() in ameca_warm_up.py) -- these crops are
# purely from the local detector (MediaPipe FaceMesh, or a Haar cascade
# fallback), used only to save reference images per turn. No emotion is
# classified from these images.

def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


_FACE_CASCADE: Optional[Any] = None  # cv2.CascadeClassifier, or False = searched-and-unavailable
_EYE_CASCADE: Optional[Any] = None
_FACE_CASCADE_UNAVAILABLE_LOGGED = False


def _candidate_face_cascade_paths() -> list[str]:
    """
    Ordered list of places to look for haarcascade_frontalface_default.xml.
    Checked with os.path.isfile() before ever being passed to
    cv2.CascadeClassifier(), and tried in order until one actually loads.

    This exists because cv2.data.haarcascades is not reliably populated --
    some opencv-contrib-python releases (observed after a version bump
    pulled in by installing/upgrading mediapipe) ship a Python package
    without its bundled data/ directory at all, so the "default" path
    silently points at a file that doesn't exist.
    """
    candidates: list[str] = []
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
    """
    Returns a loaded, non-empty CascadeClassifier, or None if no usable
    cascade file could be found anywhere. Caches BOTH outcomes (a working
    classifier, or the "nothing works" case) so this only searches -- and
    only logs -- once per process, instead of re-attempting and re-raising
    on every single candidate frame for the rest of the session.
    """
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

    if not _FACE_CASCADE_UNAVAILABLE_LOGGED:
        print_ts(
            "[WARN] No usable Haar face cascade file was found (checked: "
            f"{', '.join(candidates)}). This typically means the installed "
            "opencv-contrib-python/opencv-python wheel doesn't bundle its "
            "data/ directory (seen after an opencv-contrib-python version "
            "bump pulled in while installing/upgrading mediapipe). Local "
            "face-region detection -- and therefore saved per-turn face "
            "crops -- will be disabled for the rest of this run. Try `pip "
            "install opencv-python` alongside the current OpenCV package, "
            "or set FACE_CASCADE_PATH to a haarcascade_frontalface_default.xml "
            "you know is valid."
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
    """
    Face-bounding-box detection via MediaPipe FaceMesh. Only reports a
    match when it can locate genuine facial structure (468 3D landmarks
    across eyes, nose, mouth, jawline), which is far more resistant to
    false positives than a Haar cascade alone.

    Returns None both when MediaPipe finds no face AND when it errors --
    callers must check _MEDIAPIPE_BROKEN (set here on error) to tell the
    two apart, since only the latter should trigger a Haar fallback.
    """
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
    """
    Fallback face-bounding-box detection via OpenCV's Haar cascade, used
    only when MediaPipe FaceMesh is unavailable or fails to import.

    Sanity filtering rejects two observed Haar false-positive classes: a
    large, non-face-shaped box (essentially the whole room) via the
    maxSize cap plus aspect-ratio/area filtering, and a small,
    plausibly-face-sized box that's actually a high-contrast background
    object (a door handle, a wall clock) via a skin-tone color check
    (see _region_has_skin_tone(), default ON) -- cheaper and lower-
    recall-risk than requiring an eye-like feature inside the box (see
    _region_contains_eye(), default OFF).
    """
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
        print_ts(
            f"No usable face crop found among {len(frames)} candidate frame(s) for this turn."
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

    summary = re.sub(r"^\s*[-*•]\s*", "", summary, flags=re.MULTILINE)
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
        return f"Welcome back, {name}. It is nice to continue our conversation. 🙂"

    return (
        f"Welcome back, {name}. Last time, we were discussing {summary} "
        f"Where would you like to continue from? 🙂"
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
        return f"Welcome back, {name}. It is nice to continue our conversation. 🙂"

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
        - use only this emoji: 🙂

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


def save_session_transcript(
    user_key: str,
    user_profile: dict,
    session_log: list[dict],
    participant_id: str = "",
    video_path: Optional[str] = None,
    llm_call_samples: Optional[list[dict]] = None,
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
            "emotion_fusion": {
                "type": "text_only_no_prosody_no_visual",
                "taxonomy": "ekman_plus_neutral",
                "text_weight": FUSION_TEXT_WEIGHT,
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
) -> None:
    users = load_users()

    if user_key not in users:
        return

    users[user_key]["last_seen"] = now_ts()
    users[user_key].setdefault("session_files", []).append(session_path)

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
        - only use this emoji: 🙂

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
        return normalize_reply(raw_reply, "neutral")

    except Exception as exc:
        print_ts(f"Could not generate introduction with LLM: {exc}")
        return f"Hello {user_name}. I am Ameca. It is nice to meet you. 🙂"


# =========================
# Participant ID capture
# =========================

def resolve_participant_id(
    cli_participant_id: str,
    robot_speaker: Optional[RobotSpeaker] = None,
    robot_expression: Optional["RobotExpression"] = None,
    session_log: Optional[list[dict]] = None,
) -> str:
    """
    Resolve the stable participant identifier used to key session storage
    and continuity (see save_session_transcript() / prompt_for_user_name()).

    Prefers --participant_id if it was given on the command line.
    Otherwise, asks for it interactively: Ameca announces the request out
    loud (if a RobotSpeaker is available), but the ID itself is TYPED on
    the keyboard rather than spoken through ASR -- participant codes are
    short, exact alphanumeric strings (e.g. "A11320") where a single
    misheard character would silently corrupt the storage key and break
    session continuity, so typed entry is used here instead of the
    spelled-name capture flow used for the display name.
    """
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
) -> tuple[str, dict, str]:
    """
    Resolve whether this is a returning or first-time user by looking up
    `participant_id` in the persisted users store FIRST, before asking
    anything -- participant_id is the stable identity key (see
    resolve_participant_id()), so it is authoritative over any spoken or
    spelled name for deciding returning-vs-new.

    - Returning (participant_id already on file): skip the name/spelling
      flow entirely, reuse the stored display name, and greet with the
      returning-user continuity response.
    - First-time (participant_id not on file, or no participant_id given):
      greet once, then ask the user to spell their name (spelling is the
      sole source for the display name -- there is no "please say your
      name" spoken-name step; spoken names were dropped because ASR
      transcription drift was the main source of misidentified/duplicated
      user profiles).
    """
    users = load_users()

    participant_id = str(participant_id or "").strip()
    user_key = slugify_name(participant_id) if participant_id else None

    is_new_user = not (user_key and user_key in users)

    if not is_new_user:
        # ---- Returning user: identified purely from participant_id. ----
        user_profile = users[user_key]
        user_profile["last_seen"] = now_ts()
        save_users(users)

        final_name = str(user_profile.get("name", "Guest")).strip() or "Guest"
        print_ts(f"Recognized returning participant '{participant_id}' -> stored name: {final_name}")

        user_profile = ensure_user_has_conversation_summary(user_key, user_profile)

        print_ts(f"Welcome back, {user_profile['name']}.")
        if user_profile.get("conversation_summary"):
            print_ts("Starting from previous conversation summary.")
        introduction_reply = generate_returning_user_response(
            client=client,
            user_profile=user_profile,
        )

        print_ts(f"Assistant: {introduction_reply}")
        print()

        return user_key, user_profile, introduction_reply

    # ---- First-time user for this participant_id: greet, then ask them
    # to spell their name (spelling-only name capture). ----
    print()
    print_ts("New participant. Greeting user and requesting spelled name.")
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

    return user_key, user_profile, introduction_reply


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

# Self-RAG activation is now gated SOLELY by explicit keyword mention.
# No other signal (question form, entity mention, inferred category,
# etc.) may trigger retrieval on its own. See
# mentions_self_rag_trigger_keyword() and build_self_rag_context() below.
SELF_RAG_TRIGGER_PHRASES = [
    "robotic research laboratory",
    "robotic research lab",
    "rrlab",
    "rr lab",
]


def mentions_self_rag_trigger_keyword(text: str) -> bool:
    """
    True only if the user's utterance explicitly names the lab, using one
    of SELF_RAG_TRIGGER_PHRASES. This is the ONLY condition that may
    activate Self-RAG retrieval -- it does not have to be phrased as a
    question, and no other heuristic (entity list, category inference,
    force_rag, etc.) can substitute for it.

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
    """
    Retrieve and re-rank candidate chunks for `query`. This is only ever
    called AFTER build_self_rag_context() has already confirmed the
    trigger keyword is present (see mentions_self_rag_trigger_keyword()),
    so no additional "should we even retrieve" gating happens here --
    only relevance filtering/re-ranking of what comes back.
    """
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
    emotion: str = "neutral",
    confidence: float = 1.0,
) -> Optional[str]:
    if not self_rag_context or not self_rag_context.used or not self_rag_context.context_text.strip():
        return None

    prompt = f"""
You are Ameca answering a factual question using only the retrieved local lab knowledge below.

Important disambiguation:
- "Ameca" is your own name and physical identity (the robot).
- "EMAH" (or "Emah") refers to a research system/software pipeline that runs on you, NOT your name.
- Never say "I am Emah" or introduce EMAH as your identity. If the retrieved text describes EMAH, describe it as a system you run, while remaining Ameca.

User question:
{user_text}

Retrieved local lab knowledge:
{self_rag_context.context_text}

Instructions:
- Answer only from the retrieved knowledge.
- If the exact name, role, project, or fact is not explicitly present, say you could not verify it from the local lab knowledge.
- Do not use placeholders such as [Name].
- Do not claim personal familiarity.
- Do not mention Self-RAG, embeddings, vector databases, or raw metadata.
- Keep the answer to 1-2 short sentences.

Return JSON only:
{{
  "reply": "answer without emoji",
  "emoji": "one of 🙂 😊 😮 😢 😠 🤢 😨"
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
        emoji = str(data.get("emoji", "")).strip()
        if not reply or context_has_placeholder_risk(reply):
            return None
        if emoji not in ALLOWED_FACE_EMOJIS:
            emoji = EKMAN_EMOTIONS.get(emotion, "🙂")
        return normalize_reply(f"{reply} {emoji}", emotion, confidence)
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


def build_self_rag_context(client: Client, store: SelfRAGStore, user_text: str) -> SelfRAGContext:
    """
    HARD GATE: Self-RAG activates if and only if the user's message
    explicitly contains one of SELF_RAG_TRIGGER_PHRASES (see
    mentions_self_rag_trigger_keyword()) -- "robotic research laboratory",
    "robotic research lab", "RRLab", or "RR lab". This is the ONLY
    trigger condition. It does not matter whether the message is phrased
    as a question, a statement, small talk, or anything else -- if the
    trigger phrase is present, retrieval proceeds; if it is absent,
    Self-RAG never runs, period.

    Everything past the gate (retrieval, person-lookup fallback, LLM
    relevance grading) narrows whether the retrieved knowledge is
    actually usable -- it cannot re-open Self-RAG for a message that
    failed the keyword gate.
    """
    if not store.enabled:
        return self_rag_disabled_context(user_text, "Self-RAG store is not enabled.", store.error)

    if not mentions_self_rag_trigger_keyword(user_text):
        return SelfRAGContext(
            available=True,
            used=False,
            query=user_text,
            context_text="",
            sources=[],
            reason=(
                "Self-RAG requires the trigger phrase 'RRLab', 'RR lab', "
                "'robotic research lab', or 'robotic research laboratory' "
                "to be present in the message. No trigger phrase was found."
            ),
        )

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
            )
        return self_rag_disabled_context(user_text, "No sufficiently relevant local knowledge was retrieved.")

    should_use, reason = grade_self_rag_context(client, user_text, candidates)
    if not should_use:
        return SelfRAGContext(
            available=True,
            used=False,
            query=user_text,
            context_text="",
            sources=[{k: v for k, v in item.items() if k != "text"} for item in candidates],
            reason=reason or "Retrieved context was judged not useful.",
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
    )


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


# =========================
# Reliability-aware emotion resolution (text-only; prosody modality removed)
# =========================
#
# The prosody modality (RMS/energy-based heuristics over the utterance's
# audio) has been removed entirely: it was found in practice to reliably
# fire "surprise" on nearly every turn (its RMS/energy thresholds were not
# well matched to this microphone/room's typical speaking volume), adding
# noise rather than signal to the fused result. Text is now the sole
# emotion-detection modality.

def one_hot_emotion_distribution(emotion: str, confidence: float) -> dict[str, float]:
    """
    Builds a one-hot-ish distribution over ALL SEVEN labels (the six
    Ekman emotions plus neutral), putting `confidence` on `emotion` and
    spreading the remainder evenly across the other six.

    IMPORTANT: neutral must be a normal candidate here, not a special
    case excluded from the loop. A previous version filtered "neutral"
    out of the set of keys being assigned a probability, which meant a
    classified emotion of "neutral" could never actually receive any
    probability mass (its own key didn't exist in the returned dict).
    Downstream, adaptive_reliability_aware_fusion() would then look up
    text_dist.get("neutral", 0.0) and silently get 0.0 -- tying every
    emotion at zero -- and the tie-break (max() on equal values returns
    the first key in EKMAN_EMOTIONS' insertion order, which is "joy")
    would mislabel every neutral turn as "joy" with confidence 0.0.
    """
    if emotion not in EKMAN_EMOTIONS:
        emotion = "neutral"
    confidence = max(0.0, min(1.0, float(confidence)))
    all_emotions = list(EKMAN_EMOTIONS.keys())
    remaining = max(0.0, 1.0 - confidence)
    other = remaining / max(1, len(all_emotions) - 1)
    return {emo: confidence if emo == emotion else other for emo in all_emotions}


def explicit_emotion_from_text(text: str) -> Optional[str]:
    t = text.lower()

    patterns = {
        "anger": [
            "angry", "annoyed", "frustrated", "furious", "irritated",
            "hate this", "i hate", "so annoying", "this is annoying",
        ],
        "sadness": [
            "sad", "exhausted", "tired", "burned out", "overwhelmed",
            "depressed", "unhappy", "crying",
        ],
        "fear": [
            "afraid", "scared", "terrified", "anxious", "worried",
            "panic", "nervous",
        ],
        "joy": [
            "happy", "excited", "glad", "great", "amazing", "i love", "wow", "that's so cool", "that's awesome", "that's great",
            "sounds interesting", "looks interesting", "nice",
        ],
        "surprise": [
            "surprised", "unexpected", "shocked", "i can't believe", "no way", "what a shock",
        ],
        "disgust": [
            "disgusting", "gross", "revolting",
        ],
    }

    for emotion, terms in patterns.items():
        if any(term in t for term in terms):
            return emotion

    return None


def text_reliability_score(text_emotion: EmotionResult, user_text: str = "") -> float:
    base = max(0.0, min(1.0, float(text_emotion.confidence)))
    explicit = explicit_emotion_from_text(user_text)

    if explicit and explicit == text_emotion.emotion:
        base = max(base, 0.90)
    elif explicit and explicit != text_emotion.emotion:
        base = max(base, 0.75)

    return max(0.0, min(1.0, base))


def adaptive_reliability_aware_fusion(
    text_emotion: EmotionResult,
    user_text: str = "",
    modality_response_times: Optional[dict[str, Optional[float]]] = None,
) -> FusedEmotionResult:
    """
    Resolve the turn's emotion from TEXT ALONE. Prosody (RMS/energy-based
    heuristics over the utterance's audio) and vision/DeepFace have both
    been removed from this pipeline; text is the sole modality.

    This is kept as a "fusion"-shaped function (rather than just
    returning text_emotion directly) so downstream code -- temporal
    smoothing, session logging, response generation -- doesn't need to
    change: it still gets a full distribution over the Ekman emotion set
    plus a reliability-adjusted confidence, just computed from one
    modality instead of several.
    """
    emotions = list(EKMAN_EMOTIONS.keys())

    text_dist = one_hot_emotion_distribution(text_emotion.emotion, text_emotion.confidence)

    explicit_text_emotion = explicit_emotion_from_text(user_text)

    text_rel = text_reliability_score(text_emotion, user_text)

    if explicit_text_emotion:
        text_rel = max(text_rel, 0.95)

    question_like = user_text.strip().endswith("?") or any(
        phrase in user_text.lower()
        for phrase in ["who is", "what is", "do you know", "can you tell", "where is", "how do"]
    )
    if question_like:
        text_rel = max(text_rel, 0.80)

    # Single modality: the "fused" distribution is just the text
    # distribution scaled by its reliability score, then renormalized so
    # confidence still reflects how reliable this turn's reading was.
    fused_scores = {emo: text_rel * text_dist.get(emo, 0.0) for emo in emotions}

    dominant = max(fused_scores.items(), key=lambda item: item[1])[0]
    confidence = max(0.0, min(1.0, text_dist.get(dominant, 0.0) * text_rel))

    reason = (
        f"Text-only emotion resolution selected {dominant}: "
        f"text={text_emotion.emotion} rel={text_rel:.2f} (no prosody or visual modality)."
    )

    return FusedEmotionResult(
        emotion=dominant,
        confidence=confidence,
        reason=reason,
        scores=fused_scores,
        weights={
            "base_text": FUSION_TEXT_WEIGHT,
            "reliability_text": text_rel,
            "active_normalized_text": 1.0,
        },
        text_emotion={
            "emotion": text_emotion.emotion,
            "confidence": text_emotion.confidence,
            "reason": text_emotion.reason,
        },
        response_times=modality_response_times or {},
    )


# =========================
# Temporal emotion smoothing (across turns)
# =========================
#
# This is what keeps Ameca ADAPTIVE to the user's emotional changes over
# the course of a conversation: rather than reacting fully to a single
# turn's fused reading (which can be noisy), the smoothed distribution
# below tracks the running affective state and updates it every turn,
# so a genuine, sustained shift in the user's emotion is reflected
# quickly while a one-off noisy turn doesn't cause a jarring flip.

def apply_temporal_emotion_smoothing(
    current_scores: dict[str, float],
    previous_smoothed_scores: Optional[dict[str, float]],
    alpha: float = EMOTION_SMOOTHING_ALPHA,
) -> dict[str, float]:
    """
    Exponential moving average (EMA) over the fused per-emotion score
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
        You are an emotion classification system for a human-robot interaction chat system.

        Classify the user's emotional state from the text below.

        You must map the emotion to exactly one of Ekman's six basic emotions, plus neutral:

        {emotions}

        Use the user's words as the primary signal.

        DISAMBIGUATION -- joy vs surprise:
        - surprise = something UNEXPECTED or startling happened; the person is caught off guard.
          Example: "Wait, really? I had no idea!" -> surprise
        - joy = positive enthusiasm, being impressed, pleased, or glad about something -- even if
          expressed with exclamations like "wow" or "that's amazing". This is the default for
          enthusiastic-but-not-startled reactions.
          Example: "Wow, that sounds really interesting!" -> joy
          Example: "That's so cool!" -> joy
        Do not default to surprise just because a sentence contains "wow" or an exclamation mark --
        check whether the person is actually startled/caught-off-guard (surprise) versus simply
        pleased/enthusiastic (joy).

        DISAMBIGUATION -- anger vs mild impatience/curtness:
        - anger = clear hostility, insults, expletives, or explicit statements of being angry/upset
          directed at the assistant or the situation.
          Example: "This is useless, you're not helping at all!" -> anger
        - Repeating or rephrasing a question, or a short/curt correction, is NOT anger by itself --
          it is normal conversational repair and should default to neutral unless clearly hostile
          language is also present.
          Example: "Yeah, I know that. I said explain what a robot means." -> neutral (a correction/
          repetition, not hostility)
          Example: "No, I meant X, not Y." -> neutral
        Do not infer anger purely from brevity, terseness, or the act of repeating oneself.

        Return JSON only.

        Required JSON schema:
        {{
        "emotion": "joy | sadness | anger | fear | surprise | disgust | neutral",
        "confidence": 0.0,
        "reason": "short explanation"
        }}

        Rules:
        - confidence must be a number between 0.0 and 1.0
        - choose the best single emotion, even if the message is mixed
        - do not add markdown
        - do not add extra text outside JSON
        - For greetings such as "hello", "hi", or "good morning", return:
        {{"emotion": "neutral", "confidence": 0.6, "reason": "The user is opening a friendly social interaction."}}
        - For farewells such as "bye", "goodbye", "take care", or "talk later", return:
        {{"emotion": "neutral", "confidence": 0.7, "reason": "The user is closing the conversation politely."}}

        User text:
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


_SENTENCE_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs"}


def truncate_to_max_sentences(text: str, max_sentences: int = MAX_REPLY_SENTENCES) -> str:
    text = text.strip()
    if not text:
        return text

    def _protect_abbreviation_dot(match: "re.Match[str]") -> str:
        word = match.group(1)
        if word.lower() in _SENTENCE_ABBREVIATIONS:
            return f"{word}\x00"
        return match.group(0)

    protected = re.sub(r"\b([A-Za-z]{1,4})\.(?=\s+[A-Z])", _protect_abbreviation_dot, text)

    sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [s.replace("\x00", ".").strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    return " ".join(sentences[:max_sentences]).strip()


def normalize_reply(raw_reply: str, emotion: str, confidence: float = 1.0) -> str:
    resolved_emotion = emotion
    if emotion in EMOJI_STRONG_EMOTIONS and confidence < EMOJI_MIN_CONFIDENCE_FOR_STRONG_EMOTION:
        resolved_emotion = "neutral"

    required_emoji = EKMAN_EMOTIONS.get(resolved_emotion, EKMAN_EMOTIONS["neutral"])

    cleaned = remove_all_emojis_except_allowed_faces(raw_reply)
    cleaned = remove_allowed_face_emojis(cleaned)

    cleaned = re.sub(r"[:;=8][\-^]?[)(DPp/\\|]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = truncate_to_max_sentences(cleaned)

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


def deterministic_reply_if_applicable(user_text: str, emotion: str) -> Optional[str]:
    text = user_text.strip().lower()
    emoji = EKMAN_EMOTIONS.get(emotion, "🙂")

    if "today's date" in text or "todays date" in text or "what is the date" in text:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. {emoji}"

    if "what is the time" in text or "what time is it" in text or "current time" in text:
        return f"The current time is {datetime.now().strftime('%H:%M')}. {emoji}"

    if is_farewell_utterance(user_text):
        return "Thank you, and take care. 🙂"

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
        "detected_emotion_summary": json.dumps(clean_emotion_summary),
        "interpretation_rules": "The detected emotion was inferred purely from the transcribed text of what the user said. Use it only to adjust tone, following the guidance below.",
        "required_result_type": "return JSON only in this exact shape:{\"reply\":\"assistant response without emoji\",\"emoji\":\"one facial emoji\",\"tone\":\"short tone label\"}",
        "emoji_rules": [
            "Always end with exactly one context-appropriate facial emoji from this set: 🙂 😊 😮 😢 😨",
            "Do not use any other emoji or emoticon symbols, and don't overreact emotionally.",
        ],
    }

# Fixed, short, and CRITICAL -- must always survive truncation intact,
# regardless of how large the variable-length background context (memory
# summary, Self-RAG retrieved knowledge, etc.) grows. See
# build_response_system_prompt() below for why this is appended AFTER
# truncation rather than being part of the truncated string.
RESPONSE_OUTPUT_INSTRUCTIONS = """
OUTPUT INSTRUCTIONS -- these override anything above about output format:
- Your entire response must be exactly ONE JSON object and nothing else: no
  preamble, no markdown fences, no restating or copying any part of the
  background context above.
- That JSON object must have EXACTLY these three keys and no others:
  "reply", "emoji", "tone".
- "reply" is your actual spoken response as a plain string. If the user asked
  for examples, a list, or specific details, include the real content --
  not just a lead-in sentence.
- Correct output shape: {"reply": "AI differs from normal programming because...", "emoji": "\U0001F60A", "tone": "curious"}
- Incorrect (never do this): {"role": "Ameca, a humanoid social robot..."}
""".strip()

MAX_SYSTEM_PROMPT_CHARS = int(os.environ.get("MAX_SYSTEM_PROMPT_CHARS", "12000"))


def build_response_system_prompt(
    emotion_result: EmotionResult,
    user_profile: Optional[dict] = None,
    self_rag_context: Optional[SelfRAGContext] = None,
) -> str:
    memory_context = build_user_memory_context(user_profile)

    clean_emotion_summary = build_clean_emotion_summary(emotion_result)

    additional_guidlines = extra_reponse_propmt_guideline(clean_emotion_summary)

    additional_guidelines_text = json.dumps(additional_guidlines, indent=2)

    ameca_system_prompt_text = json.dumps(AMECA_SYSTEM_PROMPT, indent=2)

    background_text = f"""
    BEGIN BACKGROUND CONTEXT -- for your own reference only. Never repeat, quote, or
    output any of this verbatim. None of the JSON keys below (such as "role",
    "identity", "capability_boundaries", "task", "possible_topics", etc.) are your
    output format -- they only describe you.

    {ameca_system_prompt_text}

    {runtime_context()}

    {memory_context}

    {build_self_rag_prompt_block(self_rag_context)}

    {additional_guidelines_text}

    END BACKGROUND CONTEXT
    """.strip()

    # IMPORTANT: only the variable-length background section is ever
    # truncated -- never the OUTPUT INSTRUCTIONS below. A prior version
    # truncated the WHOLE assembled prompt (background + instructions) at
    # a single fixed character count; whenever the background alone
    # approached that limit (easily happens once a real conversation
    # summary and/or Self-RAG retrieved context are included), the entire
    # output-schema instruction silently got sliced off the end. The
    # model, given format="json" but no schema left to read, then simply
    # returned "{}" on every single turn -- a 100% failure mode, observed
    # in production. Truncating the background first and appending the
    # (short, fixed) instructions afterward guarantees they always reach
    # the model intact, no matter how large memory/Self-RAG context gets.
    max_background_chars = max(1000, MAX_SYSTEM_PROMPT_CHARS - len(RESPONSE_OUTPUT_INSTRUCTIONS) - 20)
    background_text = background_text[:max_background_chars]

    return f"{background_text}\n\n{RESPONSE_OUTPUT_INSTRUCTIONS}"


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
    """
    Called only when safe_json_extract() has already failed to parse
    `text` as JSON (see _attempt_llm_response below). Any text that
    still starts with '{' at this point is leaked/garbled JSON
    structure -- our own reply-schema prompt, a truncated tool/config
    dump, or (as observed in production) the raw AMECA_SYSTEM_PROMPT
    itself being echoed back by the model when it got confused (e.g. on
    "repeat what you just said"). None of these should ever be spoken
    verbatim, so this is treated as a blanket guard rather than only
    catching the narrow "reply"/"emoji"/"tone" schema case.
    """
    stripped = str(text or "").strip()
    return stripped.startswith("{")

def _attempt_llm_response(
    client: Client,
    messages: list[dict],
    emotion_result: EmotionResult,
    self_rag_context: Optional[SelfRAGContext],
    repeat_penalty: float,
) -> Optional[str]:
    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=messages,
            options={
                "temperature": 0.25 if self_rag_context and self_rag_context.used else 0.4,
                "num_predict": 200,
                "repeat_penalty": repeat_penalty,
                "num_ctx": 8192,
            },
            stream=False,
        )
    except Exception as exc:
        print_ts(f"Response generation LLM call failed ({exc}).")
        raise _LLMCallFailed(str(exc)) from exc

    raw_reply = response["message"]["content"]

    if DEBUG_LOG_RAW_LLM_REPLIES:
        print_ts(f"[DEBUG] Raw LLM reply (pre-parse, response generation): {raw_reply!r}")

    data = safe_json_extract(raw_reply)

    if data is not None and isinstance(data, dict):
        reply_text = str(data.get("reply", "")).strip()
        emoji = str(data.get("emoji", "")).strip()

        if emoji in {":)", ":-)", ""}:
            emoji = EKMAN_EMOTIONS.get(emotion_result.emotion, "🙂")

        if reply_text and not _is_degenerate_reply_text(reply_text):
            return normalize_reply(
                f"{reply_text} {emoji}", emotion_result.emotion, emotion_result.confidence
            )
        print_ts(
            f"[DEBUG] Rejecting response: parsed JSON but 'reply' field was empty/degenerate. "
            f"Raw LLM reply: {raw_reply!r}"
        )
        return None

    if _is_degenerate_reply_text(raw_reply) or _looks_like_unparsed_json_schema(raw_reply):
        print_ts(
            f"[DEBUG] Rejecting response: could not parse JSON and raw text looked degenerate/"
            f"unparsed-schema. Raw LLM reply: {raw_reply!r}"
        )
        return None

    final_reply = normalize_reply(raw_reply, emotion_result.emotion, emotion_result.confidence)
    if self_rag_context and self_rag_context.used and context_has_placeholder_risk(final_reply):
        return normalize_reply(
            "I found a relevant local lab page, but I could not verify the exact name from the retrieved text, so I should not invent it. 🙂",
            emotion_result.emotion,
            emotion_result.confidence,
        )
    return final_reply


def generate_response(
    client: Client,
    user_text: str,
    emotion_result: EmotionResult,
    history: list[dict],
    user_profile: Optional[dict] = None,
    self_rag_context: Optional[SelfRAGContext] = None,
    llm_call_samples: Optional[list[dict]] = None,
) -> str:
    """
    llm_call_samples, if provided, is a mutable list that this function
    appends {"system_prompt", "messages", "user_text", "reply"} to -- but
    only for the FIRST three turns of the session (len < 3 check), and
    only for the standard (non-deterministic) conversational path, since
    that's the prompt shape used for the vast majority of turns. Kept for
    analysis: main() passes the same list in on every turn and saves it
    into the session transcript at the end (see save_session_transcript's
    llm_call_samples parameter).
    """
    deterministic = deterministic_reply_if_applicable(
        user_text=user_text,
        emotion=emotion_result.emotion,
    )

    if deterministic:
        return deterministic

    safe_user_text = limit_text_length(user_text)
    system_prompt = build_response_system_prompt(
        emotion_result=emotion_result,
        user_profile=user_profile,
        self_rag_context=self_rag_context,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *prompt_ready_history(trim_history(history[-6:])),
        {"role": "user", "content": safe_user_text},
    ]

    def _record_sample(reply_text: str) -> None:
        if llm_call_samples is not None and len(llm_call_samples) < 3:
            llm_call_samples.append({
                "turn_index": len(llm_call_samples) + 1,
                "system_prompt": system_prompt,
                "messages": messages,
                "user_text": safe_user_text,
                "reply": reply_text,
                "timestamp": now_ts(),
            })

    if self_rag_context and self_rag_context.used:
        grounded_reply = generate_grounded_self_rag_answer(
            client=client,
            user_text=safe_user_text,
            self_rag_context=self_rag_context,
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
        )
        if grounded_reply:
            _record_sample(grounded_reply)
            return grounded_reply

    call_failed = False
    reply = None
    try:
        reply = _attempt_llm_response(
            client=client,
            messages=messages,
            emotion_result=emotion_result,
            self_rag_context=self_rag_context,
            repeat_penalty=1.1,
        )
    except _LLMCallFailed:
        call_failed = True

    if reply is not None:
        _record_sample(reply)
        return reply

    print_ts("Response generation produced no usable content on the first attempt; retrying once.")
    try:
        reply = _attempt_llm_response(
            client=client,
            messages=messages,
            emotion_result=emotion_result,
            self_rag_context=self_rag_context,
            repeat_penalty=1.1,
        )
        call_failed = False
    except _LLMCallFailed:
        call_failed = True

    if reply is not None:
        _record_sample(reply)
        return reply

    if call_failed:
        print_ts("Response generation LLM call failed on both attempts; using connectivity fallback reply.")
        fallback_reply = normalize_reply(
            "I'm having trouble reaching my language model right now, so I can't respond properly to that.",
            emotion_result.emotion,
            emotion_result.confidence,
        )
        _record_sample(fallback_reply)
        return fallback_reply

    print_ts("Response generation produced no usable content on retry either; using fallback reply.")
    fallback_reply = normalize_reply(
        "Sorry, could you say that again? I didn't quite catch a clear response that time.",
        emotion_result.emotion,
        emotion_result.confidence,
    )
    _record_sample(fallback_reply)
    return fallback_reply


# =========================
# CLI args (robot-specific)
# =========================

def parse_robot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ameca demo: Silero VAD + faster-whisper + text-only emotion detection (Ekman taxonomy) "
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
        help="Disable driving Ameca's physical facial expression from the fused emotion result.",
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
        help="Disable temporal (cross-turn) smoothing of the fused emotion result; use each turn's raw fused emotion directly.",
    )
    parser.add_argument(
        "--emotion_smoothing_alpha",
        type=float,
        default=EMOTION_SMOOTHING_ALPHA,
        help=f"EMA weight given to the current turn's fused scores when temporal smoothing is enabled (default: {EMOTION_SMOOTHING_ALPHA}). Lower = smoother/slower to change.",
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

    print_ts(
        "Starting integrated Ameca demo: Silero VAD + faster-whisper + persistent memory + Self-RAG + "
        "text-only emotion detection (Ekman taxonomy) + temporal smoothing + Tritium TTS + "
        "Tritium facial expression + ZED session-video recording. No camera/visual modality and no "
        "vocal-prosody modality are used for emotion recognition (the camera here only records video)."
    )
    print_ts(f"Python: {sys.version.split()[0]}")
    print_ts(f"Ollama host: {OLLAMA_HOST}")
    print_ts(f"Ollama chat model: {MODEL_NAME}")
    print_ts(f"Ollama embedding model (Self-RAG): {SELF_RAG_EMBED_MODEL}")
    print_ts(f"Tritium TTS URL: {args.tts_url}")
    print_ts(f"Tritium expression host: {args.expression_host} (disabled={args.disable_expression})")
    print_ts(f"Temporal emotion smoothing enabled: {emotion_smoothing_enabled} (alpha={emotion_smoothing_alpha})")
    print_ts(f"Expression timing: {EXPRESSION_TIMING} (nod after speech: {NOD_AFTER_SPEECH_ENABLED}, sequence='{NOD_SEQUENCE_NAME}')")
    print_ts(f"Emotion taxonomy: Ekman + neutral ({', '.join(EKMAN_EMOTIONS.keys())}); modality: text only.")
    print_ts("Negative facial expressions are suppressed on Ameca's physical face by design; empathy is expressed via spoken tone instead.")
    print_ts(f"Self-RAG trigger phrases (only activation condition): {SELF_RAG_TRIGGER_PHRASES}")
    print_ts(f"Session video recording enabled: {not args.disable_video_recording} (resolution={args.resolution}, camera={args.camera})")

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

    # ---- Session log created up front, before name/participant capture,
    # so every word Ameca says (and every ASR-derived user response)
    # starting from the very first onboarding prompt is captured. ----
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

    user_key, user_profile, intro_reply = prompt_for_user_name(
        client=client,
        whisper_model=whisper_model,
        silero_model=silero_model,
        input_device=INPUT_DEVICE,
        robot_speaker=robot_speaker,
        robot_expression=robot_expression,
        participant_id=participant_id,
        session_log=session_log,
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
    print(
        "Speak naturally. Silero VAD will detect speech; faster-whisper will transcribe the utterance; "
        "text-only emotion detection (no camera/visual input, no vocal-prosody analysis, with cross-turn "
        "temporal smoothing) will determine the Ekman emotion; Ameca's face will update to match (never "
        "with a negative expression) via Tritium sequence_player, waiting for that expression to finish "
        "playing before speaking; and Ameca will respond out loud via Tritium TTS, acknowledging negative "
        "emotions empathetically in words, then nod once done talking."
    )
    print("Say '/exit', or say a farewell such as 'goodbye', to save the transcript and quit.")
    print()

    history: list[dict] = []

    smoothed_emotion_scores: Optional[dict[str, float]] = None

    # First (up to) three full prompts sent to the response-generation LLM
    # and the reply that was ultimately used -- kept for analysis. See
    # generate_response()'s llm_call_samples parameter.
    llm_call_samples: list[dict] = []

    session_log.append({
        "role": "assistant",
        "content": intro_reply,
        "timestamp": now_ts(),
        "intent": "self_introduction",
    })
    history.append({"role": "assistant", "content": intro_reply})

    turn_index = 0

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
                farewell_reply = "Thank you, and take care. 🙂"
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

            maybe_name = extract_name_from_text(user_text)
            if maybe_name and not looks_like_invalid_name(maybe_name) and not participant_id.strip():
                user_key, user_profile = rename_current_user(
                    old_user_key=user_key,
                    user_profile=user_profile,
                    new_name=maybe_name,
                )

                reply = generate_introduction_response(
                    client=client,
                    user_name=user_profile["name"],
                )

                print_ts(f"Assistant: {reply}")
                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text=reply,
                    emotion="neutral",
                    confidence=1.0,
                    disable_expression=args.disable_expression,
                )
                print()

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": reply})
                history = trim_history(history)

                session_log.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "input_mode": "silero_vad_faster-whisper",
                    "intent": "name_introduction",
                })
                session_log.append({
                    "role": "assistant",
                    "content": reply,
                    "timestamp": now_ts(),
                })

                continue

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

            try:
                # ---- Run emotion detection and Self-RAG concurrently ----
                # detect_emotion() only needs user_text, and
                # build_self_rag_context() only needs user_text + the
                # Self-RAG store -- neither depends on the other's output,
                # so run them concurrently to overlap their Ollama round-trips.
                parallel_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as concurrent_executor:
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

                    text_emotion_result = text_emotion_future.result()
                    text_response_seconds = time.time() - parallel_start

                    self_rag_context = self_rag_future.result()
                    self_rag_response_seconds = time.time() - parallel_start

                fused_emotion_result = adaptive_reliability_aware_fusion(
                    text_emotion=text_emotion_result,
                    user_text=user_text,
                    modality_response_times={
                        "text_seconds": text_response_seconds,
                    },
                )

                # ---- Temporal smoothing across turns (keeps Ameca adaptive
                # to genuine emotional change without flickering on noise) ----
                if emotion_smoothing_enabled:
                    smoothed_emotion_scores = apply_temporal_emotion_smoothing(
                        current_scores=fused_emotion_result.scores,
                        previous_smoothed_scores=smoothed_emotion_scores,
                        alpha=emotion_smoothing_alpha,
                    )
                    smoothed_dominant, smoothed_confidence = dominant_from_scores(smoothed_emotion_scores)
                    emotion_result = EmotionResult(
                        emotion=smoothed_dominant,
                        confidence=smoothed_confidence,
                        reason=fused_emotion_result.reason,
                    )
                else:
                    emotion_result = fused_emotion_result.to_emotion_result()

                text_emotion_json = {
                    "emotion": text_emotion_result.emotion,
                    "confidence": text_emotion_result.confidence,
                    "reason": text_emotion_result.reason,
                }

                emotion_json = fused_emotion_result.as_json
                emotion_json["temporal_smoothing"] = {
                    "enabled": emotion_smoothing_enabled,
                    "alpha": emotion_smoothing_alpha,
                    "smoothed_scores": smoothed_emotion_scores,
                    "smoothed_emotion": emotion_result.emotion,
                    "smoothed_confidence": emotion_result.confidence,
                }
                emotion_json["is_negative"] = emotion_result.emotion in NEGATIVE_EMOTIONS

                print_ts("Text-only emotion resolution JSON (raw, pre-smoothing):")
                print(json.dumps(fused_emotion_result.as_json, indent=2))
                print()

                print_ts(
                    f"Smoothed emotion used for tone/expression: {emotion_result.emotion} "
                    f"(confidence={emotion_result.confidence:.2f}, alpha={emotion_smoothing_alpha}, "
                    f"negative={emotion_result.emotion in NEGATIVE_EMOTIONS})"
                )
                print()

                print_ts(
                    f"Self-RAG JSON (computed concurrently with emotion detection, {self_rag_response_seconds:.2f}s):"
                )
                print(json.dumps(self_rag_context.as_json, indent=2))
                print()

                reply = generate_response(
                    client=client,
                    user_text=user_text,
                    emotion_result=emotion_result,
                    history=history,
                    user_profile=user_profile,
                    self_rag_context=self_rag_context,
                    llm_call_samples=llm_call_samples,
                )

                print_ts(f"Assistant: {reply}")
                # Every response gets a facial expression (set_emotion runs
                # every turn); negative emotions are always remapped to a
                # calm/neutral sequence inside RobotExpression, so the face
                # itself is never negative even though the spoken reply
                # (built above) does acknowledge the emotion empathetically.
                speak_with_turn_end_cue(
                    robot_speaker=robot_speaker,
                    robot_expression=robot_expression,
                    text=reply,
                    emotion=emotion_result.emotion,
                    confidence=emotion_result.confidence,
                    disable_expression=args.disable_expression,
                )
                print()

                # Save up to IMAGES_PER_TURN cropped face images sampled from
                # this turn's utterance (local face detection only -- no
                # DeepFace, no emotion classification from these images).
                turn_face_images = save_turn_face_crops(
                    frames=turn_frames,
                    participant_folder=participant_folder,
                    turn_index=turn_index,
                )

                user_message = {
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "emotion": emotion_json,
                    "text_emotion": text_emotion_json,
                    "self_rag": self_rag_context.as_json,
                    "input_mode": "silero_vad_faster-whisper_text_only_emotion_ekman_temporal_smoothing_self_rag",
                    "face_images": turn_face_images,
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

        if session_log:
            session_path = save_session_transcript(
                user_key=user_key,
                user_profile=user_profile,
                session_log=session_log,
                participant_id=participant_id,
                video_path=video_path,
                llm_call_samples=llm_call_samples,
            )

            update_user_after_session(
                client=client,
                user_key=user_key,
                session_path=session_path,
                session_log=session_log,
            )

            print_ts(f"Conversation transcript saved to: {session_path}")
        else:
            print_ts("No conversation messages to save.")


if __name__ == "__main__":
    main()
    