#!/usr/bin/env python3
"""
Ameca demo integration script.

This combines two previously separate systems:

1. The thesis pipeline (vlm4.py): Silero VAD -> faster-whisper ASR ->
   text LLM + Qwen2.5-VL facial emotion + librosa-style prosody ->
   adaptive reliability-aware late fusion -> Self-RAG with ChromaDB ->
   persistent per-user memory -> response generation.

2. The robot-facing demo script (AmecaRobotChat): Tritium TTS output and
   the ZED vision module for general "what do you see" style queries.

Integration choices (per user requirements):
- ASR: Silero VAD + faster-whisper is the SOLE transcription engine.
  (Google Speech Recognition fallback has been removed -- see the
  "ASR configuration" note below.)
- Output: every spoken reply (introductions, returning-user greetings,
  deterministic replies, normal chat replies, vision-query replies) is
  sent to the robot via Tritium TTS using the same PUT-based API as
  AmecaRobotChat.tts_say(), in addition to being printed to the console.
- Vision: ZedVisionModule is the SINGLE owner of the ZED camera handle.
  It runs continuously in a background thread. Qwen2.5-VL facial-emotion
  sampling (used by the fusion pipeline) reads frames via
  ZedVisionModule.get_latest_frame() instead of opening its own
  cv2.VideoCapture, and general "what do you see" queries read via
  ZedVisionModule.save_latest_frame(). DeepFace emotion analysis inside
  ZedVisionModule is explicitly disabled (enable_emotion_analysis=False):
  Qwen2.5-VL is the sole source of truth for facial-expression recognition.
- Expression: the robot's PHYSICAL facial expression is now driven from
  the fused emotion result via Tritium's sequence_player endpoint (the
  same mechanism AmecaRobotChat.play_sequence() used for the
  "exercise_routine" gesture sequence). This runs every turn as soon as
  fusion resolves an emotion, independent of TTS/speech timing. See
  RobotExpression / EMOTION_SEQUENCE_MAP below.

ASR configuration:
Google Speech Recognition fallback has been removed. faster-whisper is now
the sole ASR engine (previously it was primary with a Google STT fallback
for empty/unusable transcripts). This was removed to keep the pipeline
fully local/offline (Google STT sent audio to an external service, which
was in tension with the "locally stored" privacy language in
AMECA_SYSTEM_PROMPT) and to simplify the transcription code path.

Conversation termination:
In addition to the "/exit" command, a spoken farewell ("goodbye", "bye",
"see you later", etc.) now ends the session the same way -- see
is_farewell_utterance() / FAREWELL_TERMINATION_PHRASES below.

HARDWARE NOTE:
Earlier versions of this script opened two independent OpenCV capture
handles on the same physical ZED camera (one for facial sampling, one for
general vision queries), which most UVC drivers do not support reliably
and could cause crashes/contention. That is no longer the case: there is
now exactly one cv2.VideoCapture handle, owned by ZedVisionModule, shared
by both consumers via thread-safe get_latest_frame()/save_latest_frame().
"""
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

# Robot-side modules (must be present in the same project as the original
# AmecaRobotChat script). Imported defensively so the thesis pipeline can
# still run standalone (e.g. on a dev laptop with no robot attached) if
# these are missing.
try:
    from tts_active import find_target_device, listen_levels_for_device, is_tts_active
    HAS_TTS_ACTIVITY_MONITOR = True
except Exception as exc:  # pragma: no cover
    HAS_TTS_ACTIVITY_MONITOR = False
    print(f"[WARN] tts_active module not available, TTS-activity echo guard disabled: {exc}")

# zed_vision_module is imported LAZILY (inside main(), only if ZED vision is
# actually requested) rather than at module load time. zed_vision_module
# pulls in tensorflow + deepface, and loading tensorflow into the same
# process as torch/sentence-transformers/mediapipe is a common cause of a
# silent native segfault (no Python traceback) on Linux, even when the
# import itself "succeeds" or is later unused. Importing it eagerly here
# meant every run paid that risk, even with --disable_zed_vision.
ZedVisionModule = None
HAS_ZED_VISION = False


def try_import_zed_vision_module() -> bool:
    """
    Attempt to import zed_vision_module on demand. Call this only when ZED
    vision is actually going to be used (see main()). Sets the module-level
    ZedVisionModule/HAS_ZED_VISION globals.
    """
    global ZedVisionModule, HAS_ZED_VISION
    if HAS_ZED_VISION and ZedVisionModule is not None:
        return True
    try:
        from zed_vision_module import ZedVisionModule as _ZedVisionModule
        ZedVisionModule = _ZedVisionModule
        HAS_ZED_VISION = True
        return True
    except Exception as exc:  # pragma: no cover
        HAS_ZED_VISION = False
        print(f"[WARN] zed_vision_module not available, ZED vision queries disabled: {exc}")
        return False

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

CAMERA_DEVICE = 0 if IS_MAC else 1
USE_ZED_HALF_FRAME_CROP = IS_LINUX

# =========================
# Local Ollama configuration
# =========================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

MODEL_NAME = os.environ.get("OLLAMA_CHAT_MODEL", "llama3:8b")
VISION_MODEL_NAME = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
# Vision model used specifically for general ZED "what do you see" queries.
# Can be the same as VISION_MODEL_NAME, but kept separate because the demo
# script previously used llava:7b for this purpose.
ZED_VISION_MODEL_NAME = os.environ.get("ZED_VISION_MODEL", VISION_MODEL_NAME)


# =========================
# Persistent memory / transcript configuration
# =========================

DATA_DIR = "conversation_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")


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
SELF_RAG_SKIP_SOCIAL = os.environ.get("SELF_RAG_SKIP_SOCIAL", "1") == "1"
SELF_RAG_SKIP_EMOTIONAL_SUPPORT = os.environ.get("SELF_RAG_SKIP_EMOTIONAL_SUPPORT", "1") == "1"

KNOWN_RRLAB_ENTITIES = {
    "ashita", "ashita ashok", "ameca", "emah", "rrlab",
    "robotics research lab", "robotersysteme", "ravon", "robin", "carl",
    "unimog", "avos", "dengel", "sembai", "senna", "casrew", "zukunftbau", "znt",
}

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

AMECA_SYSTEM_PROMPT = """
You are Ameca, a humanoid social robot used in a university laboratory for research and demonstrations.

IDENTITY
You are a robot, not a human. Speak in a friendly, professional tone. Refer to yourself as a robot only when relevant, not in every response.

CAPABILITY BOUNDARIES
You are a humanoid upper-torso robot approximately 187 cm tall and about 49 kg in weight.
You can provide conversational help, explanations, brainstorming, writing feedback, study guidance, and thesis-structure support.
You cannot physically perform tasks for the user.
You cannot walk because your legs are decorative.
Your perception depends on the provided inputs.
You cannot see unless vision input is explicitly provided.
You cannot access the internet unless explicitly stated.
Do not claim internal diagnostics, sensor access, or system state beyond what is explicitly provided.

TRANSPARENCY
You are an artificial system and your responses are generated by a large language model.
If uncertain, say so instead of guessing.
Do not fabricate facts, dates, sources, capabilities, or memories.

TASK
Hold a natural conversation with the user.
Answer clearly.
Keep responses concise, usually 1-2 sentences, unless the user asks for more detail.

PRIVACY
Do not ask for sensitive personal information such as passwords, medical data, financial information, or private identity documents.
Treat the conversation as locally stored for this prototype. A local JSON transcript and a concise conversation summary may be saved at the end of the session.


MEMORY AND CONTINUITY
You have continuity memory through locally stored user profiles and conversation summaries.
You may reference previous conversations, ongoing projects, prior discussion topics, and saved user preferences only when they are present in the provided local memory context.
Do not claim human-like autobiographical memory.
Do not invent memories outside the provided local memory context.
If the user asks whether you remember previous conversations, explain that you can continue from the saved local conversation summary when one is available.

USER ADAPTATION
Use clear, simple explanations.

ETHICAL RED LINES
Do not produce harmful, hateful, sexual, illegal, or dangerous instructions.
Do not pretend to have human emotions or lived experiences.
Do not mislead users about your capabilities or limitations.
"""


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

# SCHOOL / GPU CONFIG
# FAST_WHISPER_CONFIG = {
#     "profile": "school_gpu",
#     "model": "small",
#     "device": "cuda",
#     "compute_type": "float16",
#     "language": "en",
#     "beam_size": 1,
#     "vad_filter": False,
# }


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

MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003


# =========================
# Camera / Qwen2.5-VL configuration (facial emotion sampling)
# =========================

CAMERA_DEVICE: Optional[int] = 0 if IS_LINUX else 0

ZED_STEREO_WIDTH = 2560
ZED_STEREO_HEIGHT = 720

CAMERA_FRAME_WIDTH = ZED_STEREO_WIDTH if IS_LINUX else 1280
CAMERA_FRAME_HEIGHT = ZED_STEREO_HEIGHT if IS_LINUX else 720

CAMERA_PREVIEW_WIDTH = 960
CAMERA_PREVIEW_HEIGHT = 540

CAMERA_PREVIEW_ENABLED = os.environ.get("CAMERA_PREVIEW_ENABLED", "1") == "1"
CAMERA_PREVIEW_WINDOW_NAME = "Camera Preview - press q to close preview"

USE_ZED_HALF_FRAME_CROP = IS_LINUX

CAMERA_SAMPLE_EVERY_SECONDS = float(os.environ.get("CAMERA_SAMPLE_EVERY_SECONDS", "0.5"))
CAMERA_WARMUP_SECONDS = float(os.environ.get("CAMERA_WARMUP_SECONDS", "0.35"))
FACE_MAX_CANDIDATE_FRAMES = int(os.environ.get("FACE_MAX_CANDIDATE_FRAMES", "5"))

VISION_JPEG_QUALITY = int(os.environ.get("VISION_JPEG_QUALITY", "80"))
VISION_MODEL_TIMEOUT_SECONDS = int(os.environ.get("VISION_MODEL_TIMEOUT_SECONDS", "20"))
VISION_DEBUG = os.environ.get("VISION_DEBUG", "1") == "1"
VISION_DEBUG_DIR = os.environ.get("VISION_DEBUG_DIR", "vision_debug_frames")
VISION_MAX_IMAGE_SIDE = int(os.environ.get("VISION_MAX_IMAGE_SIDE", "224"))

VISION_DENOISE_ENABLED = os.environ.get("VISION_DENOISE_ENABLED", "1") == "1"
VISION_LIGHTING_ENHANCE_ENABLED = os.environ.get("VISION_LIGHTING_ENHANCE_ENABLED", "1") == "1"
VISION_SHARPEN_ENABLED = os.environ.get("VISION_SHARPEN_ENABLED", "1") == "1"
VISION_DENOISE_H = float(os.environ.get("VISION_DENOISE_H", "3"))
VISION_DENOISE_H_COLOR = float(os.environ.get("VISION_DENOISE_H_COLOR", "3"))
VISION_CLAHE_CLIP_LIMIT = float(os.environ.get("VISION_CLAHE_CLIP_LIMIT", "1.5"))
VISION_MIN_BRIGHTNESS = float(os.environ.get("VISION_MIN_BRIGHTNESS", "35.0"))
VISION_MAX_BRIGHTNESS = float(os.environ.get("VISION_MAX_BRIGHTNESS", "220.0"))
VISION_MIN_SHARPNESS = float(os.environ.get("VISION_MIN_SHARPNESS", "8.0"))
VISION_SKIP_LOW_QUALITY = os.environ.get("VISION_SKIP_LOW_QUALITY", "0") == "1"
FACE_ANALYSIS_STRATEGY = os.environ.get("FACE_ANALYSIS_STRATEGY", "multi_frame").strip().lower()
FACE_MULTI_FRAME_COUNT = int(os.environ.get("FACE_MULTI_FRAME_COUNT", "3"))

FACE_MIN_TOP_SCORE = float(os.environ.get("FACE_MIN_TOP_SCORE", "35.0"))
FACE_MIN_MARGIN = float(os.environ.get("FACE_MIN_MARGIN", "8.0"))


# =========================
# Chat configuration
# =========================

MAX_HISTORY_MESSAGES = 12

PLUTCHIK_EMOTIONS = {
    "joy": "😊",
    "trust": "🙂",
    "fear": "😨",
    "surprise": "😮",
    "sadness": "😢",
    "disgust": "🤢",
    "anger": "😠",
    "anticipation": "🤔",
    "neutral": "🙂"
}

ALLOWED_FACE_EMOJIS = set(PLUTCHIK_EMOTIONS.values())

# =========================
# Adaptive reliability-aware emotion fusion
# =========================
FUSION_TEXT_WEIGHT = float(os.environ.get("FUSION_TEXT_WEIGHT", "0.5"))
FUSION_VISUAL_WEIGHT = float(os.environ.get("FUSION_VISUAL_WEIGHT", "0.4"))
FUSION_PROSODY_WEIGHT = float(os.environ.get("FUSION_PROSODY_WEIGHT", "0.1"))

FUSION_ENABLE_SEMANTIC_OVERRIDE = os.environ.get("FUSION_ENABLE_SEMANTIC_OVERRIDE", "1") == "1"
FUSION_EXPLICIT_TEXT_CONFIDENCE = float(os.environ.get("FUSION_EXPLICIT_TEXT_CONFIDENCE", "0.72"))

# =========================
# Temporal emotion smoothing (across turns)
# =========================
# The adaptive fusion above resolves an emotion independently every turn,
# which means a single noisy/ambiguous turn (e.g. a brief visual
# misread, or a short ambiguous utterance) can cause the robot's detected
# emotion -- and therefore its facial expression -- to flicker between
# turns even when the user's underlying affective state hasn't actually
# changed. EMOTION_SMOOTHING applies an exponential moving average (EMA)
# over the fused per-emotion score distribution across turns within a
# session, so the "smoothed" emotion used for response tone and robot
# expression changes more gradually, while the raw per-turn fusion result
# is still logged in full for diagnostics/thesis analysis.
EMOTION_SMOOTHING_ENABLED = os.environ.get("EMOTION_SMOOTHING_ENABLED", "1") == "1"
# Weight given to the CURRENT turn's fused scores; (1 - alpha) is retained
# from the prior smoothed state. Higher alpha = more responsive to the
# current turn; lower alpha = smoother/slower to change.
EMOTION_SMOOTHING_ALPHA = float(os.environ.get("EMOTION_SMOOTHING_ALPHA", "0.6"))

# =========================
# Response length configuration
# =========================
# Hard backstop on reply length. This is enforced in code (see
# truncate_to_max_sentences / normalize_reply) in addition to being
# requested in the system prompt, because LLMs do not always honor a
# "keep it short" instruction reliably.
MAX_REPLY_SENTENCES = int(os.environ.get("MAX_REPLY_SENTENCES", "2"))


# =========================
# Facial expression (Tritium sequence player) configuration
# =========================
# The robot's PHYSICAL facial expression is driven separately from TTS.
# AmecaRobotChat.play_sequence() showed the mechanism: a PUT to Tritium's
# sequence_player endpoint with a named sequence
# (http://emah/tritium/sequence_player/play/{sequence_name}), previously
# only used for a movement routine ("exercise_routine"). The same
# endpoint is reused here to trigger facial-expression sequences, driven
# by the fused emotion result instead of being left unset.
#
# Confirmed working sequence names on the emah Tritium build, verified via
# test_robot_output.py's "emotion"/"cycle" commands (all returned HTTP 200).
# Still overridable via env vars if the Tritium profile changes.
EMOTION_SEQUENCE_MAP = {
    "joy": os.environ.get("SEQ_EMOTION_JOY", "Smile"),
    "trust": os.environ.get("SEQ_EMOTION_TRUST", "hopeful_"),
    "fear": os.environ.get("SEQ_EMOTION_FEAR", "Ameca_BasicEmo_Fear"),
    "surprise": os.environ.get("SEQ_EMOTION_SURPRISE", "bsurprised"),
    "sadness": os.environ.get("SEQ_EMOTION_SADNESS", "ausadness"),
    "disgust": os.environ.get("SEQ_EMOTION_DISGUST", "disgustedrepulsion"),
    "anger": os.environ.get("SEQ_EMOTION_ANGER", "Ameca_BasicEmo_Anger"),
    "anticipation": os.environ.get("SEQ_EMOTION_ANTICIPATION", "anticipate"),
}

# If the fused emotion's confidence is below this, RobotExpression.set_emotion
# falls back to a neutral expression instead of playing a weakly-supported
# emotion sequence. Default 0.0 means "always trust the fused label."
EXPRESSION_MIN_CONFIDENCE = float(os.environ.get("EXPRESSION_MIN_CONFIDENCE", "0.0"))

# If True, resend the same expression sequence even if it matches the last
# one played. Left False by default so the face doesn't replay/restart the
# same animation every single turn when the mood hasn't changed.
EXPRESSION_FORCE_REPLAY_SAME = os.environ.get("EXPRESSION_FORCE_REPLAY_SAME", "0") == "1"


# =========================
# DeepFace hybrid visual modality configuration
# =========================
# Qwen2.5-VL has no reliable face-presence gate of its own: it has been
# observed confidently describing a facial emotion from background/scene
# elements (a door handle, a chair) when no face was actually visible in
# the frame. DeepFace ships an actual face DETECTOR (not just an emotion
# classifier), so it is used here as a cross-check: if DeepFace says no
# face is present, VLM's reading for that same frame is discarded rather
# than trusted, regardless of how confident VLM was.
#
# DeepFace runs in a SEPARATE PROCESS (see deepface_worker.py), ideally
# from a SEPARATE conda environment via DEEPFACE_PYTHON below. This is
# deliberate: DeepFace pulls in TensorFlow, and running TensorFlow
# computation in the same process as this script's PyTorch usage (Silero
# VAD, faster-whisper) has previously caused a silent native segfault with
# no Python traceback (see the HARDWARE/CAUTION notes on zed_vision_module
# elsewhere in this file for the same class of bug). Isolating DeepFace
# into its own OS process -- and its own environment, so TF/Keras version
# pins can't collide with this pipeline's dependencies -- removes that
# risk entirely. If the worker process crashes, it crashes alone, and this
# pipeline just treats DeepFace as unavailable for that turn.
#
# DeepFace is now also used as a cheap FIRST-PASS face-presence gate,
# ahead of the (much more expensive) Qwen2.5-VL call: see
# FaceEmotionSampler._analyze_snapshot_now(). If DeepFace confirms no face
# is present, Qwen2.5-VL is never called for that frame at all.
DEEPFACE_ENABLED = os.environ.get("DEEPFACE_ENABLED", "1") == "1"

# Path to the python interpreter of a SEPARATE conda env that has
# deepface + tensorflow installed, e.g.:
#   conda create -n deepface_env python=3.10
#   conda run -n deepface_env pip install deepface tensorflow
#   export DEEPFACE_PYTHON=/home/emah/miniconda3/envs/deepface_env/bin/python
# Deliberately NOT auto-detected. Leave empty (or set DEEPFACE_ENABLED=0) to
# disable the hybrid visual modality and fall back to Qwen2.5-VL alone,
# exactly as before.
DEEPFACE_PYTHON = os.environ.get("DEEPFACE_PYTHON", "")
DEEPFACE_WORKER_SCRIPT = os.environ.get("DEEPFACE_WORKER_SCRIPT", "deepface_worker.py")

# TensorFlow import + model load can take a while on first run of the
# worker process; this only blocks startup once.
DEEPFACE_STARTUP_TIMEOUT_SECONDS = float(os.environ.get("DEEPFACE_STARTUP_TIMEOUT_SECONDS", "90"))
# Per-frame analyze() timeout. DeepFace itself is fast (much faster than
# the Qwen2.5-VL call it's paired with); this is a safety net against a
# stuck/dead worker, not a reflection of expected latency.
DEEPFACE_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DEEPFACE_REQUEST_TIMEOUT_SECONDS", "5"))

# Relative weight of each modality WITHIN the visual channel (VLM vs
# DeepFace), used only when BOTH produced a usable reading for the same
# frame (see combine_vlm_and_deepface_scores()). This blend happens before
# the result is fed into the existing text/visual/prosody fusion weights
# (FUSION_VISUAL_WEIGHT etc.) further down the pipeline.
VISUAL_VLM_WEIGHT = float(os.environ.get("VISUAL_VLM_WEIGHT", "0.6"))
VISUAL_DEEPFACE_WEIGHT = float(os.environ.get("VISUAL_DEEPFACE_WEIGHT", "0.4"))

# DeepFace's 7 Ekman-basic categories mapped onto this pipeline's 8
# Plutchik emotions. DeepFace has no equivalent of "trust" or
# "anticipation". "neutral" is deliberately mapped to None rather than
# forced onto "trust": Plutchik's trust carries a positive-social
# connotation that plain facial neutrality doesn't imply, so a neutral
# reading simply contributes no signal instead of being redistributed.
DEEPFACE_TO_PLUTCHIK = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": None,
}


@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    reason: str


@dataclass
class ProsodyEmotionResult:
    available: bool
    emotion: str
    confidence: float
    reason: str
    features: dict[str, float]

    @property
    def as_json(self) -> dict:
        return {
            "available": self.available,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "reason": self.reason,
            "features": self.features,
        }


@dataclass
class FusedEmotionResult:
    emotion: str
    confidence: float
    reason: str
    scores: dict[str, float]
    weights: dict[str, float]
    text_emotion: dict[str, Any]
    visual_emotion: dict[str, Any]
    prosody_emotion: dict[str, Any]
    # Per-modality wall-clock response times in seconds, e.g.
    # {"text_seconds": 0.41, "visual_seconds": 1.22, "prosody_seconds": 0.0006}.
    # "total_seconds" is added automatically in as_json as the sum of the
    # numeric entries present here.
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
            "visual_emotion": self.visual_emotion,
            "prosody_emotion": self.prosody_emotion,
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
    # Embeddings are produced via Ollama's embeddings API rather than an
    # in-process sentence-transformers model. Loading sentence-transformers
    # (PyTorch) into the same process as zed_vision_module's DeepFace import
    # (TensorFlow) is a common cause of a silent native segfault on Linux;
    # routing embeddings through Ollama keeps PyTorch out of this process
    # entirely. See get_ollama_embedding()/get_ollama_embeddings_batch().
    ollama_client: Any = None
    embed_model: str = SELF_RAG_EMBED_MODEL
    error: Optional[str] = None


@dataclass
class FaceEmotionCapture:
    emotion_score_samples: list[dict[str, float]]
    frame_count: int
    sampled_frame_count: int
    started_at: str
    ended_at: str
    error: Optional[str] = None
    # Wall-clock seconds spent inside the Qwen2.5-VL analysis call itself
    # (not the whole VAD capture window). None if analysis never ran.
    analysis_seconds: Optional[float] = None
    # Diagnostics from combine_vlm_and_deepface_scores() for the hybrid
    # visual modality (agreement, whether DeepFace gated out a VLM result
    # for lack of a detected face, per-modality dominant emotion, etc.).
    # None if DeepFace was not used this turn (disabled / unavailable).
    deepface_diagnostics: Optional[dict[str, Any]] = None

    @property
    def averaged_scores(self) -> dict[str, float]:
        if not self.emotion_score_samples:
            return {}

        totals: dict[str, float] = {}
        for sample in self.emotion_score_samples:
            for emotion, score in sample.items():
                totals[emotion] = totals.get(emotion, 0.0) + score

        count = len(self.emotion_score_samples)
        return {emotion: total / count for emotion, total in totals.items()}

    @property
    def dominant_emotion(self) -> Optional[str]:
        scores = self.averaged_scores
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]

    @property
    def is_reliable(self) -> bool:
        scores = self.averaged_scores
        if not scores:
            return False

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ordered[0][1]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        return top_score >= FACE_MIN_TOP_SCORE and (top_score - second_score) >= FACE_MIN_MARGIN

    @property
    def summary_text(self) -> str:
        if self.error:
            return f"No facial-expression hint available because: {self.error}"

        scores = self.averaged_scores
        if not scores:
            return "No reliable facial-expression hint was captured during speech."

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        if not self.is_reliable:
            top_parts = [f"{emo}={score:.0f}%" for emo, score in ordered[:3]]
            return (
                "Facial-expression hint was weak or mixed; "
                f"top_signals=({', '.join(top_parts)}); "
                f"samples={self.sampled_frame_count}"
            )

        parts = [f"{emo}={score:.0f}%" for emo, score in ordered if score >= 5.0]
        return (
            f"dominant={self.dominant_emotion}; "
            f"averaged_scores=({', '.join(parts)}); "
            f"samples={self.sampled_frame_count}"
        )

    @property
    def as_json(self) -> dict:
        scores = self.averaged_scores
        return {
            "available": self.error is None and bool(scores),
            "reliable": self.is_reliable,
            "dominant_emotion": self.dominant_emotion,
            "averaged_scores": scores,
            "frame_count": self.frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "analysis_seconds": (
                round(self.analysis_seconds, 4) if isinstance(self.analysis_seconds, (int, float)) else None
            ),
            "deepface_diagnostics": self.deepface_diagnostics,
            "note": (
                "Qwen2.5-VL facial expression analysis is combined with DeepFace "
                "(when available) as the visual component in fixed weighted late fusion; "
                "see deepface_diagnostics for how the two modalities were combined."
            ),
        }


# =========================
# Timestamp helpers
# =========================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_ts(message: str) -> None:
    print(f"[{now_ts()}] {message}")


def normalize_command(text: str) -> Optional[str]:
    stripped = text.strip().lower()
    if not stripped.startswith(("/", "\\")):
        return None
    return re.sub(r"^[\\/]+", "", stripped)


def default_face_emotion_json() -> dict:
    return {
        "available": False,
        "reliable": False,
        "dominant_emotion": None,
        "averaged_scores": {},
        "frame_count": 0,
        "sampled_frame_count": 0,
        "started_at": None,
        "ended_at": None,
        "error": "No face emotion capture was provided.",
        "analysis_seconds": None,
        "deepface_diagnostics": None,
    }


# =========================
# Robot output helpers (Tritium TTS) and text cleaning
# =========================

def clean_text_for_tts(text: str) -> str:
    """
    Remove markdown/control characters but keep umlauts and other letters.
    Mirrors AmecaRobotChat.clean_text from the robot-facing script.
    """
    if not text:
        return ""
    text = re.sub(r'[*_`~]', '', text)
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')
    return text.strip()


class RobotSpeaker:
    """
    Thin wrapper around the Tritium TTS PUT API used by AmecaRobotChat.tts_say().
    Also tracks a short "speaking tail" so the VAD loop can be told to ignore
    audio immediately after the robot finishes speaking (echo guard), mirroring
    the original demo script's behavior.
    """

    def __init__(self, tts_url: str, tts_token: str = "", speaking_cooldown_s: float = 0.3) -> None:
        self.tts_url = tts_url
        self.tts_token = tts_token
        self.speaking_cooldown_s = speaking_cooldown_s
        self._speaking_until = 0.0

        parsed = urlparse(tts_url)
        self._host = f"{parsed.scheme}://{parsed.netloc}"

    def _now(self) -> float:
        return time.time()

    def bump_speaking_tail(self, extra: Optional[float] = None) -> None:
        tail = self.speaking_cooldown_s if extra is None else extra
        self._speaking_until = max(self._speaking_until, self._now() + tail)

    def is_speaking_or_cooling_down(self) -> bool:
        activity_flag = False
        if HAS_TTS_ACTIVITY_MONITOR:
            try:
                activity_flag = is_tts_active()
            except Exception:
                activity_flag = False
        return activity_flag or (self._now() < self._speaking_until)

    def say(self, text: str) -> None:
        """
        Speak `text` on the robot via Tritium TTS. Also prints to console so
        the existing console-based logging/debugging still works.
        """
        spoken = clean_text_for_tts(text)
        if not spoken:
            return

        self.bump_speaking_tail()

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

        # Fallback path (urllib) only runs if the primary request raised an
        # exception or returned a non-success status.
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


class RobotExpression:
    """
    Thin wrapper around the Tritium sequence_player PUT API, used to drive
    Ameca's PHYSICAL facial expression from the fused emotion result.

    Mirrors AmecaRobotChat.play_sequence() (which used this same endpoint
    for a movement/gesture sequence, "exercise_routine") but targets
    facial-expression sequences instead, keyed by Plutchik emotion via
    EMOTION_SEQUENCE_MAP.

    Runs every turn as soon as fusion resolves a dominant emotion,
    independent of TTS/speech timing (per requirement: continuous,
    turn-by-turn expression updates, not tied to when the robot speaks).
    """

    def __init__(self, host: str = "http://emah", tts_token: str = "", timeout: float = 3.0) -> None:
        self.host = host.rstrip("/")
        self.token = tts_token
        self.timeout = timeout
        self.last_emotion: Optional[str] = None

    def _play_sequence(self, sequence_name: str) -> bool:
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
            return ok
        except Exception as exc:
            print_ts(f"[EXPRESSION] Failed to play sequence '{sequence_name}': {exc}")
            return False

    def set_emotion(
        self,
        emotion: str,
        confidence: float = 1.0,
        force: Optional[bool] = None,
    ) -> None:
        """
        Update the robot's facial expression to match `emotion`.

        - Falls back to the "trust"/neutral sequence if `emotion` is
          unrecognized or if `confidence` is below EXPRESSION_MIN_CONFIDENCE.
        - By default (force=None -> uses EXPRESSION_FORCE_REPLAY_SAME),
          skips re-sending the same sequence back-to-back so the face
          doesn't restart the same animation every single turn when the
          mood hasn't changed. Pass force=True to always resend.
        """
        if force is None:
            force = EXPRESSION_FORCE_REPLAY_SAME

        resolved_emotion = emotion if emotion in EMOTION_SEQUENCE_MAP else "trust"

        if confidence < EXPRESSION_MIN_CONFIDENCE:
            resolved_emotion = "trust"

        if not force and resolved_emotion == self.last_emotion:
            print_ts(
                f"[EXPRESSION] Emotion unchanged ({resolved_emotion}); skipping redundant sequence replay."
            )
            return

        sequence_name = EMOTION_SEQUENCE_MAP.get(resolved_emotion, EMOTION_SEQUENCE_MAP["trust"])
        success = self._play_sequence(sequence_name)

        if success:
            self.last_emotion = resolved_emotion


@dataclass
class DeepFaceResult:
    ok: bool
    no_face: bool
    scores: dict[str, float]  # raw Ekman-category percentages from DeepFace
    dominant_emotion: Optional[str] = None
    error: Optional[str] = None


class DeepFaceClient:
    """
    Manages a persistent DeepFace analysis worker running in its OWN
    process (see deepface_worker.py) -- ideally launched from a SEPARATE
    conda environment via DEEPFACE_PYTHON -- isolated from this process's
    PyTorch usage (Silero VAD, faster-whisper) to avoid the
    TensorFlow/PyTorch same-process segfault documented elsewhere in this
    file. Communicates via line-delimited JSON over stdin/stdout.

    If DEEPFACE_PYTHON is unset, the worker script is missing, or the
    worker fails to start/respond, this client degrades gracefully:
    is_alive() returns False and analyze() returns None, so callers fall
    back to Qwen2.5-VL alone -- the hybrid modality is opt-in, not a hard
    dependency.
    """

    def __init__(
        self,
        python_executable: str,
        worker_script: str,
        startup_timeout: float = DEEPFACE_STARTUP_TIMEOUT_SECONDS,
        request_timeout: float = DEEPFACE_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.python_executable = python_executable
        self.worker_script = worker_script
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout

        self.proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._response_queue: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()
        self._request_counter = 0
        self._ready = False

        self._start_worker()

    def _start_worker(self) -> bool:
        if not self.python_executable:
            print_ts(
                "[DeepFace] DEEPFACE_PYTHON not set; hybrid visual modality disabled "
                "(falling back to Qwen2.5-VL alone)."
            )
            return False

        if not os.path.isfile(self.worker_script):
            print_ts(
                f"[DeepFace] Worker script not found: {self.worker_script}; "
                "hybrid visual modality disabled."
            )
            return False

        try:
            self.proc = subprocess.Popen(
                [self.python_executable, self.worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print_ts(f"[DeepFace] Could not start worker process: {exc}")
            self.proc = None
            return False

        # Drain stderr in the background so worker logging doesn't block it,
        # and so failures are visible in the main console.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        ready_event = threading.Event()

        def wait_for_ready() -> None:
            try:
                line = self.proc.stdout.readline()
                if line.strip() == "READY":
                    ready_event.set()
            except Exception:
                pass

        threading.Thread(target=wait_for_ready, daemon=True).start()
        got_ready = ready_event.wait(timeout=self.startup_timeout)

        if not got_ready:
            print_ts(
                f"[DeepFace] Worker did not signal READY within {self.startup_timeout}s "
                f"(check DEEPFACE_PYTHON='{self.python_executable}' has deepface+tensorflow "
                "installed); hybrid visual modality disabled for this session."
            )
            self._ready = False
            return False

        print_ts("[DeepFace] Worker ready.")
        self._ready = True

        self._reader_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._reader_thread.start()
        return True

    def _drain_stderr(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            line = line.rstrip()
            if line:
                print_ts(f"[DeepFace worker] {line}")

    def _read_responses(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except Exception:
                continue
            self._response_queue.put(response)

    def is_alive(self) -> bool:
        return self._ready and self.proc is not None and self.proc.poll() is None

    def analyze(self, frame_bgr: np.ndarray) -> Optional[DeepFaceResult]:
        """
        Analyze one frame. Returns None if DeepFace is unavailable this
        turn (disabled, worker not ready, worker died, or timed out) --
        callers should treat None as "fall back to VLM alone", which is
        different from a DeepFaceResult with no_face=True ("DeepFace ran
        successfully and found no face").
        """
        if not self.is_alive():
            return None

        temp_path = None
        try:
            with self._lock:
                self._request_counter += 1
                request_id = f"req_{self._request_counter}"

            fd, temp_path = tempfile.mkstemp(suffix=".jpg", prefix="deepface_frame_")
            os.close(fd)
            cv2.imwrite(temp_path, frame_bgr)

            request_line = json.dumps(
                {"request_id": request_id, "cmd": "analyze", "image_path": temp_path}
            )

            try:
                self.proc.stdin.write(request_line + "\n")
                self.proc.stdin.flush()
            except Exception as exc:
                print_ts(f"[DeepFace] Could not send request to worker (worker may have died): {exc}")
                self._ready = False
                return None

            deadline = time.time() + self.request_timeout
            while time.time() < deadline:
                try:
                    response = self._response_queue.get(timeout=max(0.05, deadline - time.time()))
                except queue.Empty:
                    break

                if response.get("request_id") != request_id:
                    # Stale/mismatched response; requests are sequential so
                    # this shouldn't normally happen, but keep waiting for
                    # the one we asked for rather than misattributing it.
                    continue

                if not response.get("ok"):
                    print_ts(f"[DeepFace] Worker reported an error: {response.get('error')}")
                    return None

                return DeepFaceResult(
                    ok=True,
                    no_face=bool(response.get("no_face")),
                    scores=response.get("scores", {}) or {},
                    dominant_emotion=response.get("dominant_emotion"),
                )

            print_ts(f"[DeepFace] Request timed out after {self.request_timeout}s.")
            return None

        except Exception as exc:
            print_ts(f"[DeepFace] analyze() failed unexpectedly: {exc}")
            return None

        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=3)
        except KeyboardInterrupt:
            # A second Ctrl+C landing during this short wait shouldn't
            # produce a scary traceback on the way out -- just make sure
            # the worker process is actually terminated and move on.
            try:
                self.proc.terminate()
            except Exception:
                pass
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


def deepface_scores_to_plutchik(deepface_result: Optional[DeepFaceResult]) -> Optional[dict[str, float]]:
    """
    Map a successful (face-found) DeepFace result onto this pipeline's
    Plutchik emotion keys via DEEPFACE_TO_PLUTCHIK. Returns None if there's
    nothing usable to map (no result, error, or no face).
    """
    if deepface_result is None or not deepface_result.ok or deepface_result.no_face:
        return None

    plutchik_scores: dict[str, float] = {emo: 0.0 for emo in PLUTCHIK_EMOTIONS if emo != "neutral"}
    for ekman_label, value in (deepface_result.scores or {}).items():
        plutchik_label = DEEPFACE_TO_PLUTCHIK.get(str(ekman_label).strip().lower())
        if not plutchik_label:
            continue
        try:
            plutchik_scores[plutchik_label] += float(value)
        except Exception:
            continue

    return plutchik_scores


def combine_vlm_and_deepface_scores(
    vlm_scores: Optional[dict[str, float]],
    deepface_result: Optional[DeepFaceResult],
) -> tuple[Optional[dict[str, float]], dict[str, Any]]:
    """
    Combine Qwen2.5-VL's facial-emotion scores with DeepFace's, using
    DeepFace's real face detector as a gate:

    - DeepFace unavailable this turn (disabled / worker down / timed out)
      -> fall back to VLM alone, unchanged from the non-hybrid behavior.
    - DeepFace ran and found NO face -> discard VLM's reading for this
      frame entirely, rather than blending it in. VLM has repeatedly been
      observed describing an emotion from background/scene elements (a
      door handle, a chair) with high confidence when no face was actually
      visible; DeepFace's detector is the more trustworthy signal for
      face presence specifically.
    - Both produced a reading -> blend via VISUAL_VLM_WEIGHT /
      VISUAL_DEEPFACE_WEIGHT, and record whether the two agreed on the
      dominant emotion (consumed by face_reliability_score() to adjust
      confidence up or down).

    Returns (combined_scores_or_None, diagnostics_dict). diagnostics is
    always a dict (never None) so it can be safely embedded in
    FaceEmotionCapture.as_json() regardless of outcome.
    """
    diagnostics: dict[str, Any] = {
        "deepface_available": deepface_result is not None,
        "deepface_no_face": bool(deepface_result and deepface_result.no_face),
        "agree": None,
        "vlm_dominant": max(vlm_scores.items(), key=lambda kv: kv[1])[0] if vlm_scores else None,
        "deepface_dominant": None,
    }

    if deepface_result is None:
        return vlm_scores, diagnostics

    if deepface_result.no_face:
        if vlm_scores:
            diagnostics["vlm_discarded_reason"] = (
                "DeepFace's face detector found no face in this frame; discarding VLM's "
                "facial-emotion reading for it since VLM has no reliable face-presence gate "
                "of its own."
            )
        return None, diagnostics

    deepface_plutchik = deepface_scores_to_plutchik(deepface_result) or {}
    diagnostics["deepface_dominant"] = (
        max(deepface_plutchik.items(), key=lambda kv: kv[1])[0] if deepface_plutchik else None
    )

    if not vlm_scores:
        return (deepface_plutchik or None), diagnostics

    diagnostics["agree"] = (
        diagnostics["vlm_dominant"] is not None
        and diagnostics["vlm_dominant"] == diagnostics["deepface_dominant"]
    )

    total_weight = VISUAL_VLM_WEIGHT + VISUAL_DEEPFACE_WEIGHT
    if total_weight <= 0:
        return vlm_scores, diagnostics

    all_emotions = set(vlm_scores) | set(deepface_plutchik)
    combined = {
        emo: (
            VISUAL_VLM_WEIGHT * vlm_scores.get(emo, 0.0)
            + VISUAL_DEEPFACE_WEIGHT * deepface_plutchik.get(emo, 0.0)
        )
        / total_weight
        for emo in all_emotions
    }

    return combined, diagnostics


# =========================
# Persistent memory helpers
# =========================

def ensure_data_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SESSIONS_DIR, exist_ok=True)


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

    # Defensively strip nested "Previous continuity context: " prefixes
    # here too (not just in build_deterministic_session_summary), so every
    # caller -- including the startup "Memory preview" log and the
    # returning-user greeting fallback -- displays cleanly even for
    # already-corrupted summaries saved before that fix existed.
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
                "num_predict": 120,
                "num_ctx": 4096,
            },
            stream=False,
        )

        raw_reply = response["message"]["content"]
        reply = normalize_reply(raw_reply, "trust")

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
) -> str:
    ensure_data_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_key}_{timestamp}.json"
    path = os.path.join(SESSIONS_DIR, filename)

    transcript_data = {
        "user": {
            "key": user_key,
            "name": user_profile.get("name", "Guest"),
        },
        "session": {
            "started_at": session_log[0]["timestamp"] if session_log else now_ts(),
            "ended_at": now_ts(),
            "model": MODEL_NAME,
            "ollama_host": OLLAMA_HOST,
            "asr": {
                "backend": "faster-whisper",
                **FAST_WHISPER_CONFIG,
            },
            "emotion_fusion": {
                "type": "adaptive_reliability_aware_late_fusion",
                "text_weight": FUSION_TEXT_WEIGHT,
                "visual_weight": FUSION_VISUAL_WEIGHT,
                "prosody_weight": FUSION_PROSODY_WEIGHT,
                "temporal_smoothing": {
                    "enabled": EMOTION_SMOOTHING_ENABLED,
                    "alpha": EMOTION_SMOOTHING_ALPHA,
                },
            },
            "face_emotion": {
                "backend": "qwen2.5vl:7b via Ollama",
                "camera_source": "shared ZedVisionModule.get_latest_frame() (single camera handle)",
                "camera_frame_width": CAMERA_FRAME_WIDTH,
                "camera_frame_height": CAMERA_FRAME_HEIGHT,
                "preview_enabled": CAMERA_PREVIEW_ENABLED,
                "sample_every_seconds": CAMERA_SAMPLE_EVERY_SECONDS,
                "warmup_seconds": CAMERA_WARMUP_SECONDS,
                "max_candidate_frames": FACE_MAX_CANDIDATE_FRAMES,
                "face_analysis_strategy": FACE_ANALYSIS_STRATEGY,
                "multi_frame_count": FACE_MULTI_FRAME_COUNT,
                "skip_low_quality": VISION_SKIP_LOW_QUALITY,
                "vision_model": VISION_MODEL_NAME,
                "jpeg_quality": VISION_JPEG_QUALITY,
                "min_top_score": FACE_MIN_TOP_SCORE,
                "min_margin": FACE_MIN_MARGIN,
                "hybrid_deepface": {
                    "enabled": DEEPFACE_ENABLED,
                    "python_executable": DEEPFACE_PYTHON or None,
                    "visual_vlm_weight": VISUAL_VLM_WEIGHT,
                    "visual_deepface_weight": VISUAL_DEEPFACE_WEIGHT,
                    "face_presence_gate_order": "deepface_first_then_vlm_conditional",
                    "note": (
                        "DeepFace runs FIRST as a cheap face-presence gate; Qwen2.5-VL is only "
                        "called if DeepFace confirms a face is present (or DeepFace is unavailable "
                        "this turn). See deepface_diagnostics on each turn's face_emotion for "
                        "whether it was used and whether it agreed with VLM."
                    ),
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
            },
            "expression": {
                "backend": "Tritium sequence_player",
                "emotion_sequence_map": EMOTION_SEQUENCE_MAP,
                "min_confidence": EXPRESSION_MIN_CONFIDENCE,
                "force_replay_same": EXPRESSION_FORCE_REPLAY_SAME,
            },
            "general_vision": {
                "backend": "ZED vision module" if HAS_ZED_VISION else "unavailable",
                "model": ZED_VISION_MODEL_NAME,
            },
        },
        "messages": session_log,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(transcript_data, file, indent=2, ensure_ascii=False)

    return path


def strip_previous_continuity_prefix(text: str) -> str:
    """
    Remove one or more leading "Previous continuity context: " prefixes.

    build_deterministic_session_summary() wraps whatever the previous
    session saved in this exact prefix every time it runs. Without
    stripping it first, each session nests another layer on top of the
    last ("Previous continuity context: Previous continuity context:
    ..."), and the useful content underneath gets squeezed out by the
    growing prefix within compact_previous_summary_for_greeting()'s fixed
    character budget -- after enough sessions the "memory preview" is
    almost entirely repeated prefix text with no real content left. This
    also cleans up summaries that were already corrupted by the bug
    before this fix (which may have several stacked layers) the next
    time they pass through here.
    """
    text = str(text or "").strip()
    # Tolerate an optional leading bullet marker ("- ", "* ", "• ") before
    # each repetition. The stored summary is built as a bulleted list (see
    # build_deterministic_session_summary), so the text this function
    # actually receives looks like "- Previous continuity context: ...",
    # not a bare "Previous continuity context: ...". Without allowing for
    # that leading "- ", the anchored regex never matched at all, so this
    # function was silently a no-op -- every session wrapped the entire
    # (already corrupted) previous summary in one more nested "Previous
    # continuity context: " layer instead of stripping it, which is why
    # the nesting kept growing across sessions instead of being collapsed.
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

    # Clean up the previous summary before it's used anywhere in this
    # function -- both as input to the LLM prompt below and as the
    # deterministic fallback -- so a corrupted/nested "Previous continuity
    # context:" chain from before this fix doesn't keep being fed back in
    # as-is.
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


def load_audio_for_prosody(wav_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    try:
        audio, sr_in = sf.read(wav_path, dtype="float32")

        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)

        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32, copy=False)

        if int(sr_in) != int(target_sr):
            audio = resample_audio(audio, int(sr_in), int(target_sr))

        return audio.astype(np.float32, copy=False)

    except Exception as exc:
        print_ts(f"Could not load audio for prosody: {exc}")
        return np.array([], dtype=np.float32)


# =========================
# Camera / Qwen2.5-VL helpers (facial emotion sampling)
# =========================

def normalize_camera_frame(frame: np.ndarray) -> np.ndarray:
    if USE_ZED_HALF_FRAME_CROP:
        height, width = frame.shape[:2]
        frame = frame[:, : width // 2]
    return frame


def get_camera_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(camera_device: Optional[int]) -> cv2.VideoCapture:
    candidates = []

    if camera_device is not None:
        candidates.append(camera_device)

    candidates.extend([0, 1, 2, 3])

    seen = set()

    for camera_index in candidates:
        if camera_index in seen:
            continue

        seen.add(camera_index)

        cap = cv2.VideoCapture(camera_index, get_camera_backend())

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        time.sleep(0.2)

        ok, frame = cap.read()

        if ok and frame is not None:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print_ts(
                f"Using camera device /dev/video{camera_index} "
                f"at {actual_width}x{actual_height}"
            )
            return cap

        cap.release()

    return cv2.VideoCapture(-1)


def resize_keep_aspect(frame_bgr: np.ndarray, max_side: int = VISION_MAX_IMAGE_SIDE) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame_bgr
    scale = max_side / float(longest)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def image_quality_metrics(frame_bgr: np.ndarray) -> dict[str, float]:
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {"brightness": brightness, "sharpness": sharpness}
    except Exception:
        return {"brightness": 0.0, "sharpness": 0.0}


def denoise_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if not VISION_DENOISE_ENABLED:
        return frame_bgr
    try:
        return cv2.fastNlMeansDenoisingColored(
            frame_bgr,
            None,
            h=VISION_DENOISE_H,
            hColor=VISION_DENOISE_H_COLOR,
            templateWindowSize=7,
            searchWindowSize=21,
        )
    except Exception as exc:
        if VISION_DEBUG:
            print_ts(f"Denoising skipped: {exc}")
        return frame_bgr


def enhance_lighting(frame_bgr: np.ndarray) -> np.ndarray:
    if not VISION_LIGHTING_ENHANCE_ENABLED:
        return frame_bgr
    try:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=VISION_CLAHE_CLIP_LIMIT,
            tileGridSize=(8, 8),
        )
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception as exc:
        if VISION_DEBUG:
            print_ts(f"Lighting enhancement skipped: {exc}")
        return frame_bgr


def sharpen_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if not VISION_SHARPEN_ENABLED:
        return frame_bgr
    try:
        blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(frame_bgr, 1.2, blurred, -0.2, 0)
    except Exception as exc:
        if VISION_DEBUG:
            print_ts(f"Sharpening skipped: {exc}")
        return frame_bgr


def preprocess_frame_for_vlm(frame_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    prepared = crop_largest_face_or_center(frame_bgr)
    before = image_quality_metrics(prepared)

    prepared = denoise_frame(prepared)
    prepared = enhance_lighting(prepared)
    prepared = sharpen_frame(prepared)
    prepared = resize_keep_aspect(prepared, max_side=VISION_MAX_IMAGE_SIDE)

    after = image_quality_metrics(prepared)
    metrics = {
        "brightness_before": before.get("brightness", 0.0),
        "sharpness_before": before.get("sharpness", 0.0),
        "brightness_after": after.get("brightness", 0.0),
        "sharpness_after": after.get("sharpness", 0.0),
        "width": float(prepared.shape[1]),
        "height": float(prepared.shape[0]),
    }
    return prepared, metrics


def is_vlm_image_quality_usable(metrics: dict[str, float]) -> tuple[bool, str]:
    brightness = metrics.get("brightness_after", 0.0)
    sharpness = metrics.get("sharpness_after", 0.0)
    if brightness < VISION_MIN_BRIGHTNESS:
        return False, f"image too dark: brightness={brightness:.1f}"
    if brightness > VISION_MAX_BRIGHTNESS:
        return False, f"image too bright: brightness={brightness:.1f}"
    if sharpness < VISION_MIN_SHARPNESS:
        reason = f"image soft/blurry but still sent: sharpness={sharpness:.1f}"
        if VISION_SKIP_LOW_QUALITY:
            return False, f"image too blurry/noisy: sharpness={sharpness:.1f}"
        return True, reason
    return True, "image quality usable"


def crop_largest_face_or_center(frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]

    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            pad = int(max(w, h) * 0.65)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(width, x + w + pad)
            y2 = min(height, y + h + pad)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                return resize_keep_aspect(crop)
    except Exception as exc:
        if VISION_DEBUG:
            print_ts(f"Face crop fallback used: {exc}")

    side = min(height, width)
    if side <= 0:
        return resize_keep_aspect(frame_bgr)
    x1 = max(0, (width - side) // 2)
    y1 = max(0, (height - side) // 2)
    crop = frame_bgr[y1:y1 + side, x1:x1 + side]
    return resize_keep_aspect(crop if crop.size > 0 else frame_bgr)


def save_debug_frame(frame_bgr: np.ndarray, prefix: str = "vlm_frame") -> Optional[str]:
    if not VISION_DEBUG:
        return None
    try:
        os.makedirs(VISION_DEBUG_DIR, exist_ok=True)
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = os.path.join(VISION_DEBUG_DIR, filename)
        cv2.imwrite(path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(VISION_JPEG_QUALITY)])
        return path
    except Exception:
        return None


def encode_frame_for_ollama(frame_bgr: np.ndarray) -> Optional[str]:
    try:
        prepared, quality_metrics = preprocess_frame_for_vlm(frame_bgr)
        usable, quality_reason = is_vlm_image_quality_usable(quality_metrics)

        if VISION_DEBUG:
            print_ts(
                "VLM image quality: "
                f"brightness={quality_metrics['brightness_after']:.1f}, "
                f"sharpness={quality_metrics['sharpness_after']:.1f}, "
                f"size={int(quality_metrics['width'])}x{int(quality_metrics['height'])}, "
                f"status={quality_reason}"
            )
            path = save_debug_frame(prepared, prefix="qwen_face_input_clean")
            if path:
                print_ts(f"Saved cleaned VLM debug frame: {path}")

        if not usable:
            return None

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(VISION_JPEG_QUALITY)]
        ok, buffer = cv2.imencode(".jpg", prepared, encode_params)
        if not ok:
            return None
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:
        print_ts(f"Could not encode frame for Qwen2.5-VL: {exc}")
        return None


EMOTION_ALIASES = {
    "happy": "joy", "happiness": "joy", "smile": "joy", "smiling": "joy",
    "calm": "trust", "friendly": "trust", "relaxed": "trust",
    "concern": "fear", "concerned": "fear", "anxious": "fear", "anxiety": "fear", "afraid": "fear",
    "surprised": "surprise",
    "sad": "sadness", "unhappy": "sadness",
    "angry": "anger", "mad": "anger", "annoyed": "anger",
    "disgusted": "disgust",
    "expectant": "anticipation", "curious": "anticipation", "thinking": "anticipation",
}


def canonical_emotion_name(name: str) -> Optional[str]:
    cleaned = re.sub(r"[^a-zA-Z]+", "", str(name).strip().lower())

    if cleaned == "neutral":
        return None

    if cleaned in PLUTCHIK_EMOTIONS and cleaned != "neutral":
        return cleaned

    return EMOTION_ALIASES.get(cleaned)


def normalize_plutchik_scores(raw_data: dict) -> Optional[dict[str, float]]:
    if not isinstance(raw_data, dict):
        return None

    raw_scores = raw_data.get("scores") if isinstance(raw_data.get("scores"), dict) else raw_data
    scores: dict[str, float] = {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}

    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            emotion = canonical_emotion_name(str(key))
            if not emotion:
                continue
            try:
                scores[emotion] += max(0.0, float(value))
            except Exception:
                continue

    total = sum(scores.values())

    if total <= 0:
        dominant = canonical_emotion_name(str(raw_data.get("dominant_emotion", raw_data.get("emotion", ""))))
        if dominant:
            try:
                confidence = float(raw_data.get("confidence", 0.55))
            except Exception:
                confidence = 0.55
            confidence = max(0.05, min(1.0, confidence))
            remaining = max(0.0, 1.0 - confidence)
            other_value = remaining / max(1, len(PLUTCHIK_EMOTIONS) - 1)
            return {
                emotion: (confidence if emotion == dominant else other_value) * 100.0
                for emotion in PLUTCHIK_EMOTIONS
            }
        return None

    if total <= 1.5:
        return {emotion: score * 100.0 for emotion, score in scores.items()}

    return {emotion: (score / total) * 100.0 for emotion, score in scores.items()}


def build_qwen_face_prompt() -> str:
    return """
        You are a facial emotion classifier.

        Analyze the main visible face only.

        Classify the facial expression into exactly one emotion:
        joy, trust, fear, surprise, sadness, disgust, anger, anticipation

        Return JSON only:

        {
        "dominant_emotion": "joy",
        "confidence": 0.0,
        "scores": {
            "joy": 0.0,
            "trust": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "sadness": 0.0,
            "disgust": 0.0,
            "anger": 0.0,
            "anticipation": 0.0
        },
        "reason": "1 short phrase only of facial explanation"
        }
        """.strip()


def analyze_frame_emotion_scores(
    frame_bgr: np.ndarray,
    vision_client: Client,
) -> Optional[dict[str, float]]:
    image_b64 = encode_frame_for_ollama(frame_bgr)
    if not image_b64:
        return None

    try:
        response = vision_client.chat(
            model=VISION_MODEL_NAME,
            format="json",
            messages=[
                {
                    "role": "user",
                    "content": build_qwen_face_prompt(),
                    "images": [image_b64],
                }
            ],
            options={
                "temperature": 0.0,
                "num_predict": 220,
                "num_ctx": 4096,
            },
            stream=False,
        )

        raw = response.get("message", {}).get("content", "")
        if VISION_DEBUG:
            print_ts(f"Qwen2.5-VL raw emotion output: {raw[:500]}")

        data = safe_json_extract(raw)
        if not isinstance(data, dict):
            print_ts("Qwen2.5-VL returned no parseable JSON for facial emotion.")
            return None

        normalized_scores = normalize_plutchik_scores(data)
        if not normalized_scores:
            print_ts(f"Qwen2.5-VL JSON did not contain usable emotion scores: {data}")
            return None

        return normalized_scores

    except Exception as exc:
        print_ts(f"Qwen2.5-VL facial analysis failed: {exc}")
        return None


def analyze_multiple_frames_emotion_scores(
    frames_bgr: list[np.ndarray],
    vision_client: Client,
    max_frames: int = FACE_MULTI_FRAME_COUNT,
) -> Optional[dict[str, float]]:
    images_b64: list[str] = []

    for frame in frames_bgr[:max(1, max_frames)]:
        image_b64 = encode_frame_for_ollama(frame)
        if image_b64:
            images_b64.append(image_b64)

    if not images_b64:
        return None

    try:
        response = vision_client.chat(
            model=VISION_MODEL_NAME,
            format="json",
            messages=[
                {
                    "role": "user",
                    "content": build_qwen_face_prompt(),
                    "images": images_b64,
                }
            ],
            options={
                "temperature": 0.0,
                "num_predict": 220,
                "num_ctx": 4096,
            },
            stream=False,
        )

        raw = response.get("message", {}).get("content", "")
        if VISION_DEBUG:
            print_ts(f"Qwen2.5-VL raw multi-frame emotion output: {raw[:500]}")

        data = safe_json_extract(raw)
        if not isinstance(data, dict):
            print_ts("Qwen2.5-VL returned no parseable JSON for multi-frame facial emotion.")
            return None

        normalized_scores = normalize_plutchik_scores(data)
        if not normalized_scores:
            print_ts(f"Qwen2.5-VL multi-frame JSON did not contain usable emotion scores: {data}")
            return None

        return normalized_scores

    except Exception as exc:
        print_ts(f"Qwen2.5-VL multi-frame facial analysis failed: {exc}")
        return None


class FaceEmotionSampler:
    """
    Facial-expression sampler for Qwen2.5-VL using best-of-N frame selection.

    This now pulls frames from a shared ZedVisionModule instance instead of
    opening its own cv2.VideoCapture handle. The ZED camera only supports one
    active capture handle at a time, so ZedVisionModule owns the camera and
    everything else (this sampler, and general "what do you see" vision
    queries) reads from its thread-safe get_latest_frame().
    """

    def __init__(
        self,
        vision_module: Any,
        vision_client: Client,
        deepface_client: Optional["DeepFaceClient"] = None,
    ) -> None:
        self.vision_module = vision_module
        self.vision_client = vision_client
        self.deepface_client = deepface_client
        self.started_at = now_ts()
        self.error: Optional[str] = None
        self.emotion_score_samples: list[dict[str, float]] = []
        self.frame_count = 0
        self.sampled_frame_count = 0
        self.capture_start_time: Optional[float] = None
        self.last_sample_time = 0.0
        self.last_status = "warming up..."
        self.snapshot_frame: Optional[np.ndarray] = None
        self.snapshot_taken_at: Optional[str] = None
        self.candidate_frames: list[tuple[float, np.ndarray]] = []
        # Wall-clock seconds spent inside the actual Qwen2.5-VL analysis
        # call (set once finish() runs _analyze_snapshot_now()). Stays 0.0
        # (not None) if DeepFace's face-presence gate skipped VLM entirely.
        self.analysis_seconds: Optional[float] = None
        # Wall-clock seconds spent inside the DeepFace analyze() call.
        # None if DeepFace was not used this turn.
        self.deepface_analysis_seconds: Optional[float] = None
        # Diagnostics from combine_vlm_and_deepface_scores(); surfaced on
        # the resulting FaceEmotionCapture for transparency/debugging and
        # consumed by face_reliability_score() for the agreement bonus.
        self.deepface_diagnostics: Optional[dict[str, Any]] = None

    def start(self) -> None:
        self.started_at = now_ts()
        self.capture_start_time = time.time()

        if self.vision_module is None:
            self.error = "No ZedVisionModule instance was provided."
            return

        if CAMERA_PREVIEW_ENABLED:
            try:
                cv2.namedWindow(CAMERA_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
                self.window_created = True
            except Exception as exc:
                self.error = f"Camera preview could not start: {exc}"
                self.window_created = False
        else:
            self.window_created = False

    def _draw_preview(self, frame: np.ndarray) -> None:
        if not (CAMERA_PREVIEW_ENABLED and getattr(self, "window_created", False)):
            return

        try:
            preview_frame = frame.copy()
            cv2.putText(
                preview_frame, "Recording speech...", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                preview_frame, f"Emotion: {self.last_status}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
            preview_frame = cv2.resize(preview_frame, (CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT))
            cv2.imshow(CAMERA_PREVIEW_WINDOW_NAME, preview_frame)
            cv2.waitKey(1)
        except Exception as exc:
            self.error = f"Camera preview failed: {exc}"

    def _frame_sharpness(self, frame_bgr: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def sample_if_due(self) -> None:
        if self.vision_module is None or self.capture_start_time is None:
            return

        # ZedVisionModule.get_latest_frame() already returns a single-eye
        # crop (per its `view` setting), so no further SBS cropping is
        # applied here (unlike the old standalone-camera path).
        frame = self.vision_module.get_latest_frame(max_age_seconds=1.0)
        if frame is None:
            return

        self.frame_count += 1
        now = time.time()
        elapsed = now - self.capture_start_time

        self._draw_preview(frame)

        if elapsed < CAMERA_WARMUP_SECONDS:
            return

        if len(self.candidate_frames) >= FACE_MAX_CANDIDATE_FRAMES:
            self.last_status = (
                f"collected frames ({FACE_MAX_CANDIDATE_FRAMES}/{FACE_MAX_CANDIDATE_FRAMES})"
            )
            return

        if (
            self.candidate_frames
            and now - self.last_sample_time < CAMERA_SAMPLE_EVERY_SECONDS
        ):
            return

        sharpness = self._frame_sharpness(frame)
        self.candidate_frames.append((sharpness, frame.copy()))
        self.last_sample_time = now
        self.snapshot_taken_at = now_ts()

        if len(self.candidate_frames) > FACE_MAX_CANDIDATE_FRAMES:
            self.candidate_frames.sort(key=lambda item: item[0], reverse=True)
            self.candidate_frames = self.candidate_frames[:FACE_MAX_CANDIDATE_FRAMES]

        self.sampled_frame_count = len(self.candidate_frames)
        self.last_status = (
            f"collecting frames ({self.sampled_frame_count}/{FACE_MAX_CANDIDATE_FRAMES})"
        )

        if VISION_DEBUG:
            print_ts(
                f"Captured facial candidate {self.sampled_frame_count}/"
                f"{FACE_MAX_CANDIDATE_FRAMES} with raw sharpness={sharpness:.1f}"
            )

    def _analyze_snapshot_now(self) -> None:
        if not self.candidate_frames:
            self.last_status = "no frames captured"
            return

        self.candidate_frames.sort(key=lambda item: item[0], reverse=True)
        best_sharpness, best_frame = self.candidate_frames[0]
        self.snapshot_frame = best_frame

        # ---- DeepFace FIRST as a cheap face-presence gate ----
        # DeepFace is much faster than Qwen2.5-VL, so we check for a
        # detected face before paying for the (potentially multi-frame)
        # VLM call at all. Previously both always ran, and the VLM result
        # was simply discarded afterward in combine_vlm_and_deepface_scores
        # whenever DeepFace found no face -- wasting a full VLM call (or
        # FACE_MULTI_FRAME_COUNT calls' worth of images in multi_frame
        # mode) every time no face was present in the frame.
        #
        # If DeepFace is unavailable this turn (disabled, worker down,
        # timed out), deepface_result is None and we can't confirm face
        # presence up front, so we fall through and run VLM as before --
        # no behavior change in that case.
        deepface_result: Optional[DeepFaceResult] = None
        if self.deepface_client is not None:
            deepface_start = time.time()
            deepface_result = self.deepface_client.analyze(self.snapshot_frame)
            self.deepface_analysis_seconds = time.time() - deepface_start

        if deepface_result is not None and deepface_result.no_face:
            print_ts(
                "[DeepFace] No face detected in this frame; skipping Qwen2.5-VL "
                "analysis entirely for this turn (face-presence gate)."
            )
            scores, self.deepface_diagnostics = combine_vlm_and_deepface_scores(None, deepface_result)
            # No VLM call was made, so there's nothing to time.
            self.analysis_seconds = 0.0
        else:
            if FACE_ANALYSIS_STRATEGY == "multi_frame":
                selected = [frame for _, frame in self.candidate_frames[:FACE_MULTI_FRAME_COUNT]]
                print_ts(
                    f"Analyzing {len(selected)} best facial frames with Qwen2.5-VL "
                    f"(best_sharpness={best_sharpness:.1f}, candidates={len(self.candidate_frames)})..."
                )
                analysis_start = time.time()
                vlm_scores = analyze_multiple_frames_emotion_scores(selected, self.vision_client)
                self.analysis_seconds = time.time() - analysis_start
            else:
                print_ts(
                    f"Analyzing best facial frame with Qwen2.5-VL "
                    f"(sharpness={best_sharpness:.1f}, candidates={len(self.candidate_frames)})..."
                )
                analysis_start = time.time()
                vlm_scores = analyze_frame_emotion_scores(self.snapshot_frame, self.vision_client)
                self.analysis_seconds = time.time() - analysis_start

            scores, self.deepface_diagnostics = combine_vlm_and_deepface_scores(vlm_scores, deepface_result)

        if not scores:
            self.last_status = (
                "no face detected (DeepFace gate)"
                if self.deepface_diagnostics.get("deepface_no_face")
                else "no reliable VLM facial emotion"
            )
            return

        top_emotion, top_score = max(scores.items(), key=lambda item: item[1])
        if top_score < FACE_MIN_TOP_SCORE:
            print_ts(
                f"Hybrid facial result rejected as weak: "
                f"{top_emotion}={top_score:.1f}% < {FACE_MIN_TOP_SCORE:.1f}%"
            )
            self.last_status = f"weak hybrid result: {top_emotion}={top_score:.0f}%"
            return

        self.emotion_score_samples.append(scores)

        agree = self.deepface_diagnostics.get("agree")
        if agree is True:
            agree_note = " (VLM+DeepFace agree)"
        elif agree is False:
            agree_note = (
                f" (VLM+DeepFace disagree: vlm={self.deepface_diagnostics.get('vlm_dominant')}, "
                f"deepface={self.deepface_diagnostics.get('deepface_dominant')})"
            )
        else:
            agree_note = ""

        self.last_status = f"{top_emotion}: {top_score:.0f}%{agree_note}"
        print_ts(f"Hybrid facial result accepted: {self.last_status}")

    def finish(self) -> FaceEmotionCapture:
        if self.window_created:
            try:
                cv2.destroyWindow(CAMERA_PREVIEW_WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                pass

        try:
            self._analyze_snapshot_now()
        except Exception as exc:
            self.error = f"Hybrid VLM/DeepFace analysis failed: {exc}"
            print_ts(self.error)

        return FaceEmotionCapture(
            emotion_score_samples=self.emotion_score_samples,
            frame_count=self.frame_count,
            sampled_frame_count=self.sampled_frame_count,
            started_at=self.started_at,
            ended_at=now_ts(),
            error=self.error,
            analysis_seconds=self.analysis_seconds,
            deepface_diagnostics=self.deepface_diagnostics,
        )


# =========================
# Silero VAD listener
# =========================

def listen_for_utterance_with_silero_vad(
    input_device: Optional[int],
    silero_model,
    prompt_label: str = "utterance",
    robot_speaker: Optional[RobotSpeaker] = None,
) -> Optional[str]:
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

    is_recording = False
    speech_started_at: Optional[float] = None
    leftover_16k = np.array([], dtype=np.float32)

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
                # Do not record/transcribe while the robot is speaking
                # (echo guard). Without this, a request that is sent to
                # TTS right before calling this listener (e.g. "please say
                # your name" or "please spell your name") could have its
                # own tail end picked up as if it were the user's speech.
                if robot_speaker is not None and robot_speaker.is_speaking_or_cooling_down():
                    try:
                        while True:
                            audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                    continue

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
                            recorded_chunks = list(pre_roll_chunks)
                            recorded_chunks.append(chunk.copy())
                            pre_roll_chunks.clear()

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
        return None
    finally:
        try:
            vad_iterator.reset_states()
        except Exception:
            pass

    if not recorded_chunks:
        return None

    audio_16k = np.concatenate(recorded_chunks).astype(np.float32, copy=False)
    return save_audio_to_temp_wav(audio_16k)


def listen_for_utterance_with_silero_vad_and_face_emotion(
    input_device: Optional[int],
    silero_model,
    vision_client: Client,
    vision_module: Any = None,
    prompt_label: str = "utterance",
    robot_speaker: Optional[RobotSpeaker] = None,
    deepface_client: Optional["DeepFaceClient"] = None,
) -> Tuple[Optional[str], Optional[FaceEmotionCapture]]:
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

    is_recording = False
    speech_started_at: Optional[float] = None
    leftover_16k = np.array([], dtype=np.float32)
    face_sampler: Optional[FaceEmotionSampler] = None
    face_capture: Optional[FaceEmotionCapture] = None

    def audio_callback(indata, frames, callback_time, status) -> None:
        audio_queue.put(indata[:, 0].copy())

    print_ts(f"Listening automatically for {prompt_label}. Speak when ready. Press Ctrl+C to quit.")
    print_ts(
        f"Silero VAD settings: threshold={SILERO_THRESHOLD}, "
        f"min_silence={SILERO_MIN_SILENCE_DURATION_MS}ms, "
        f"max_utterance={VAD_MAX_UTTERANCE_SECONDS}s"
    )

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
                # Do not record/transcribe while the robot is speaking
                # (echo guard), mirroring the demo script's behavior.
                if robot_speaker is not None and robot_speaker.is_speaking_or_cooling_down():
                    # Drain any queued audio so it doesn't build a backlog.
                    try:
                        while True:
                            audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                    continue

                try:
                    block = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    if face_sampler is not None:
                        face_sampler.sample_if_due()
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
                        if face_sampler is not None:
                            face_sampler.sample_if_due()

                    speech_event = vad_iterator(torch.from_numpy(chunk), return_seconds=True)
                    now = time.time()

                    if speech_event:
                        if "start" in speech_event and not is_recording:
                            is_recording = True
                            speech_started_at = now
                            recorded_chunks = list(pre_roll_chunks)
                            recorded_chunks.append(chunk.copy())
                            pre_roll_chunks.clear()

                            print()
                            print_ts("Speech detected. Recording utterance and sampling facial expression...")

                            face_sampler = FaceEmotionSampler(
                                vision_module=vision_module,
                                vision_client=vision_client,
                                deepface_client=deepface_client,
                            )
                            face_sampler.start()
                            face_sampler.sample_if_due()

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
        print_ts(f"Silero VAD/audio/camera error: {exc}")
        return None, face_capture
    finally:
        if face_sampler is not None:
            face_capture = face_sampler.finish()

        try:
            vad_iterator.reset_states()
        except Exception:
            pass

    if not recorded_chunks:
        return None, face_capture

    audio_16k = np.concatenate(recorded_chunks).astype(np.float32, copy=False)
    return save_audio_to_temp_wav(audio_16k), face_capture


def ask_user_to_spell_name(
    whisper_model: WhisperModel,
    silero_model,
    input_device: Optional[int] = INPUT_DEVICE,
    robot_speaker: Optional[RobotSpeaker] = None,
) -> Optional[str]:
    spelling_request_text = "Could you please spell your name for me, letter by letter? For example: L E T I C I A."

    print()
    print_ts(spelling_request_text)
    print()

    if robot_speaker:
        robot_speaker.say(spelling_request_text)

    wav_path = listen_for_utterance_with_silero_vad(
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
                "num_predict": 60,
                "num_ctx": 1024,
            },
            stream=False,
        )

        raw_reply = response["message"]["content"]
        return normalize_reply(raw_reply, "trust")

    except Exception as exc:
        print_ts(f"Could not generate introduction with LLM: {exc}")
        return f"Hello {user_name}. I am Ameca. It is nice to meet you. 🙂"


def prompt_for_user_name(
    client: Client,
    whisper_model: WhisperModel,
    silero_model,
    input_device: Optional[int] = INPUT_DEVICE,
    robot_speaker: Optional[RobotSpeaker] = None,
) -> tuple[str, dict, str]:
    users = load_users()

    spoken_name = ""

    for attempt in range(2):
        print()
        print_ts("Please say your name")
        print()

        # Speak the initial name request out loud. On retry attempts, the
        # "I might have misheard..." line at the end of the previous
        # iteration already re-asks the user out loud, so we don't repeat
        # this exact prompt again here.
        if robot_speaker and attempt == 0:
            robot_speaker.say("Please tell me your name.")

        wav_path = listen_for_utterance_with_silero_vad(
            input_device=input_device,
            silero_model=silero_model,
            prompt_label="name",
            robot_speaker=robot_speaker,
        )

        if not wav_path:
            spoken_name = ""
        else:
            try:
                spoken_name = transcribe_with_faster_whisper(wav_path, whisper_model)
                spoken_name = spoken_name.strip()
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        print_ts(f"Raw name transcript: {spoken_name or '(empty)'}")

        extracted_name = extract_name_from_text(spoken_name)
        spoken_name = extracted_name or clean_spoken_name(spoken_name)

        if spoken_name and not looks_like_invalid_name(spoken_name):
            break

        print_ts(f"I heard '{spoken_name or 'nothing'}', but that does not sound like a name.")
        if robot_speaker:
            robot_speaker.say("I might have misheard, could you say your name again?")

    if not spoken_name or looks_like_invalid_name(spoken_name):
        spoken_name = "Guest"

    spelled_name = None

    if ENABLE_NAME_SPELLING:
        spelled_name = ask_user_to_spell_name(
            whisper_model=whisper_model,
            silero_model=silero_model,
            input_device=input_device,
            robot_speaker=robot_speaker,
        )

    corrected_spelled_name = correct_spelled_name_with_known_users(
        spelled_name=spelled_name,
        spoken_name=spoken_name,
    )

    if corrected_spelled_name and not looks_like_invalid_name(corrected_spelled_name):
        final_name = corrected_spelled_name
        print_ts(f"Using spelled/corrected name: {final_name}")
    else:
        final_name = correct_spelled_name_with_known_users(
            spelled_name=spoken_name,
            spoken_name=spoken_name,
        ) or spoken_name
        print_ts(f"Using spoken name: {final_name}")

    print_ts(f"Detected name: {final_name}")

    user_key = slugify_name(final_name)
    is_new_user = user_key not in users

    if is_new_user:
        users[user_key] = {
            "name": final_name,
            "created_at": now_ts(),
            "last_seen": now_ts(),
            "session_files": [],
            "conversation_summary": "",
        }
    else:
        users[user_key]["last_seen"] = now_ts()

    save_users(users)
    user_profile = users[user_key]

    if not is_new_user:
        user_profile = ensure_user_has_conversation_summary(user_key, user_profile)

    if is_new_user:
        print_ts(f"Nice to meet you, {final_name}.")
        introduction_reply = generate_introduction_response(
            client=client,
            user_name=user_profile["name"],
        )
    else:
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
        for emotion in [emo for emo in PLUTCHIK_EMOTIONS if emo != "neutral"]:
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
    """
    Get a single embedding vector from Ollama's embeddings API.

    This replaces an in-process sentence-transformers model. Loading
    sentence-transformers (which pulls in PyTorch) into the same process as
    zed_vision_module's DeepFace import (which pulls in TensorFlow) is a
    common cause of a silent native segfault on Linux -- routing embeddings
    through Ollama over HTTP keeps PyTorch out of this process entirely, so
    it can't collide with TensorFlow the way it did before.

    Requires the embedding model to be pulled in Ollama first, e.g.:
        ollama pull nomic-embed-text

    Handles two ollama-python client shapes, since the method name changed
    across versions:
    - older clients: client.embeddings(model=..., prompt=...) -> {"embedding": [...]}
    - newer clients: client.embed(model=..., input=...) -> {"embeddings": [[...]]}
    """
    text = (text or "").strip()
    if not text:
        return None

    # Try the older single-prompt API first.
    if hasattr(client, "embeddings"):
        try:
            response = client.embeddings(model=model, prompt=text)
            embedding = response.get("embedding") if isinstance(response, dict) else getattr(response, "embedding", None)
            if embedding:
                return [float(value) for value in embedding]
        except Exception as exc:
            print_ts(f"Ollama client.embeddings() call failed (model={model}): {exc}")

    # Fall back to the newer batch-capable API.
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
    """
    Embed multiple texts via Ollama. Ollama's embeddings API embeds one
    prompt per call, so this loops rather than sending a true batch request;
    fine for the indexing path, which only runs occasionally (crawl/reindex),
    not on the hot conversational turn path.
    """
    return [get_ollama_embedding(client, text, model=model) for text in texts]


def rebuild_self_rag_collection(store: SelfRAGStore) -> SelfRAGStore:
    """
    Manually delete and recreate the Self-RAG ChromaDB collection, then
    reindex it. This is the on-demand equivalent of the automatic
    dimension-mismatch repair in init_self_rag_store() -- use it if you
    change SELF_RAG_EMBED_MODEL mid-project, or if a dimension error shows
    up again without a full restart.
    """
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

        # Sanity-check the embedding model once at startup rather than
        # discovering a bad/unpulled model name only on the first user
        # query. Ollama returns an error (not a segfault) if the model
        # hasn't been pulled, which get_ollama_embedding turns into None.
        probe_embedding = get_ollama_embedding(client, "self-rag startup check", model=SELF_RAG_EMBED_MODEL)
        if probe_embedding is None:
            error_msg = (
                f"Could not get a test embedding from Ollama model '{SELF_RAG_EMBED_MODEL}'. "
                f"Make sure it is pulled, e.g.: ollama pull {SELF_RAG_EMBED_MODEL}"
            )
            print_ts(f"Self-RAG initialization failed: {error_msg}")
            return SelfRAGStore(enabled=False, error=error_msg)

        # Detect a stale collection whose stored vectors were created with a
        # DIFFERENT embedding dimension than SELF_RAG_EMBED_MODEL currently
        # produces (e.g. an old run indexed with a 384-dim sentence-
        # transformers model, and this run embeds queries with 768-dim
        # nomic-embed-text). ChromaDB fixes a collection's vector dimension
        # from whatever was first inserted into it and cannot resize it in
        # place, so every retrieve_self_rag_candidates() query would
        # otherwise fail silently, every single turn, with:
        #   "Collection expecting embedding with dimension of 384, got 768"
        # The only real fix is to detect the mismatch and rebuild the
        # collection fresh, then reindex.
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

        # existing_count == 0 covers both "was already empty" and "just got
        # wiped due to a dimension mismatch above" -- either way there is no
        # usable data in the collection right now, so try scrape.py first
        # (matches the manual '/rrlab crawl' command) and fall back to the
        # local knowledge_base folder.
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


def force_self_rag_for_entity(text: str) -> bool:
    lowered = normalize_self_rag_query_text(text)
    return any(entity in lowered for entity in KNOWN_RRLAB_ENTITIES)


def is_social_or_support_message(text: str) -> tuple[bool, str]:
    lowered = text.strip().lower()
    simple = re.sub(r"[^a-z0-9\s']", " ", lowered)
    simple = re.sub(r"\s+", " ", simple).strip()

    social_patterns = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "nice to meet you", "thank you", "thanks", "how are you",
        "goodbye", "bye", "see you", "talk later",
    ]
    if SELF_RAG_SKIP_SOCIAL and any(pattern in simple for pattern in social_patterns):
        return True, "Social greeting/small talk does not need local knowledge retrieval."

    support_patterns = [
        "i am stressed", "i'm stressed", "i feel stressed", "so stressed",
        "stressful", "overwhelmed", "anxious", "worried",
        "tired", "exhausted", "i feel sad", "i am sad", "i'm sad",
    ]
    if SELF_RAG_SKIP_EMOTIONAL_SUPPORT and any(pattern in simple for pattern in support_patterns):
        return True, "Emotional support message; response should be empathetic without forcing retrieved RRLab knowledge."

    return False, ""


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
    force_rag = force_self_rag_for_entity(normalized_query)

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

            min_score = SELF_RAG_MIN_HYBRID_SCORE
            if force_rag:
                min_score = max(0.50, SELF_RAG_MIN_HYBRID_SCORE - 0.10)

            if row["hybrid_score"] < min_score:
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
    emotion: str = "trust",
) -> Optional[str]:
    if not self_rag_context or not self_rag_context.used or not self_rag_context.context_text.strip():
        return None

    prompt = f"""
You are Ameca answering a factual question using only the retrieved local lab knowledge below.

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
  "emoji": "one of 🙂 😊 😌 😔 😟 🤔 😮 😢 😠 🤢"
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
                "num_predict": 160,
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
            emoji = PLUTCHIK_EMOTIONS.get(emotion, "🙂")
        return normalize_reply(f"{reply} {emoji}", emotion)
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
- use_context must be false for greetings, small talk, emotional support, or unrelated knowledge.
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
            options={"temperature": 0.0, "num_predict": 120, "num_ctx": 3072},
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
    if not store.enabled:
        return self_rag_disabled_context(user_text, "Self-RAG store is not enabled.", store.error)

    force_rag = force_self_rag_for_entity(user_text)

    should_skip, skip_reason = is_social_or_support_message(user_text)
    if should_skip and not force_rag:
        return SelfRAGContext(
            available=True,
            used=False,
            query=user_text,
            context_text="",
            sources=[],
            reason=skip_reason,
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
# Prosody and adaptive reliability-aware fusion helpers
# =========================

def analyze_prosody_from_audio(audio_16k: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> ProsodyEmotionResult:
    if audio_16k is None or audio_16k.size == 0:
        return ProsodyEmotionResult(
            available=False,
            emotion="trust",
            confidence=0.0,
            reason="No audio available for prosody analysis.",
            features={},
        )

    try:
        audio = audio_16k.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        duration = float(len(audio) / max(1, sample_rate))

        frame_size = max(1, int(0.05 * sample_rate))
        frame_rms = []
        for start in range(0, len(audio), frame_size):
            frame = audio[start:start + frame_size]
            if frame.size:
                frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))

        energy_std = float(np.std(frame_rms)) if frame_rms else 0.0
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.signbit(audio))))) if audio.size > 1 else 0.0

        features = {
            "peak": peak,
            "rms": rms,
            "duration": duration,
            "energy_std": energy_std,
            "zero_crossing_rate": zero_crossing_rate,
        }

        if rms < 0.006 and duration > 1.0:
            return ProsodyEmotionResult(
                available=True,
                emotion="sadness",
                confidence=0.35,
                reason="Low vocal energy suggests a subdued tone.",
                features=features,
            )

        if rms > 0.025 or energy_std > 0.018:
            return ProsodyEmotionResult(
                available=True,
                emotion="anticipation",
                confidence=0.30,
                reason="Higher or more variable vocal energy suggests engagement or arousal, not necessarily fear.",
                features=features,
            )

        return ProsodyEmotionResult(
            available=True,
            emotion="trust",
            confidence=0.25,
            reason="Prosody suggests calm conversational speech.",
            features=features,
        )

    except Exception as exc:
        return ProsodyEmotionResult(
            available=False,
            emotion="trust",
            confidence=0.0,
            reason=f"Prosody analysis failed: {exc}",
            features={},
        )


def one_hot_emotion_distribution(emotion: str, confidence: float) -> dict[str, float]:
    emotion = emotion if emotion in PLUTCHIK_EMOTIONS and emotion != "neutral" else "trust"
    confidence = max(0.0, min(1.0, float(confidence)))
    emotions = [emo for emo in PLUTCHIK_EMOTIONS if emo != "neutral"]
    remaining = max(0.0, 1.0 - confidence)
    other = remaining / max(1, len(emotions) - 1)
    return {emo: confidence if emo == emotion else other for emo in emotions}


def face_json_to_distribution(face_emotion_json: Optional[dict]) -> dict[str, float]:
    emotions = [emo for emo in PLUTCHIK_EMOTIONS if emo != "neutral"]
    scores = {emo: 0.0 for emo in emotions}

    if not face_emotion_json or not face_emotion_json.get("available"):
        return scores

    raw_scores = face_emotion_json.get("averaged_scores", {}) or {}
    for emo in emotions:
        try:
            scores[emo] = max(0.0, float(raw_scores.get(emo, 0.0))) / 100.0
        except Exception:
            scores[emo] = 0.0

    total = sum(scores.values())
    if total <= 0.0:
        dom = face_emotion_json.get("dominant_emotion")
        if dom in scores:
            scores[dom] = 1.0
        return scores

    return {emo: value / total for emo, value in scores.items()}


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
            "happy", "excited", "glad", "great", "amazing", "i love",
        ],
        "surprise": [
            "oh my god", "wow", "surprised", "unexpected", "shocked",
        ],
        "disgust": [
            "disgusting", "gross", "revolting",
        ],
        "anticipation": [
            "looking forward", "curious", "interested", "expecting",
        ],
    }

    for emotion, terms in patterns.items():
        if any(term in t for term in terms):
            return emotion

    return None


def face_reliability_score(face_emotion_json: Optional[dict]) -> float:
    if not face_emotion_json or not face_emotion_json.get("available"):
        return 0.0

    scores = face_emotion_json.get("averaged_scores", {}) or {}
    values = []
    for emotion in [emo for emo in PLUTCHIK_EMOTIONS if emo != "neutral"]:
        try:
            values.append(float(scores.get(emotion, 0.0)))
        except Exception:
            values.append(0.0)

    if not values:
        return 0.0

    ordered = sorted(values, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    margin = max(0.0, top - second)

    top_score_component = min(1.0, top / 100.0)
    margin_component = min(1.0, margin / 60.0)

    frame_count = float(face_emotion_json.get("sampled_frame_count", 0) or 0)
    frame_component = min(1.0, frame_count / max(1.0, float(FACE_MULTI_FRAME_COUNT)))

    declared_reliable = 1.0 if face_emotion_json.get("reliable") else 0.65

    base_score = (
        max(0.0, min(1.0, (0.45 * top_score_component) + (0.35 * margin_component) + (0.20 * frame_component)))
        * declared_reliable
    )

    # Hybrid VLM+DeepFace agreement adjustment: if both modalities landed on
    # the same dominant emotion for this frame, that's stronger evidence
    # than either alone, so boost reliability slightly. If they disagreed,
    # the "averaged_scores" above are really a weighted-average compromise
    # between two different readings rather than a confident detection, so
    # shrink reliability accordingly. No adjustment if DeepFace wasn't used
    # this turn (diagnostics.get("agree") is None).
    diagnostics = face_emotion_json.get("deepface_diagnostics") or {}
    agree = diagnostics.get("agree")
    if agree is True:
        base_score = min(1.0, base_score * 1.15)
    elif agree is False:
        base_score = base_score * 0.75

    return base_score


def text_reliability_score(text_emotion: EmotionResult, user_text: str = "") -> float:
    base = max(0.0, min(1.0, float(text_emotion.confidence)))
    explicit = explicit_emotion_from_text(user_text)

    if explicit and explicit == text_emotion.emotion:
        base = max(base, 0.90)
    elif explicit and explicit != text_emotion.emotion:
        base = max(base, 0.75)

    return max(0.0, min(1.0, base))


def prosody_reliability_score(prosody_emotion: Optional[ProsodyEmotionResult]) -> float:
    if not prosody_emotion or not prosody_emotion.available:
        return 0.0
    return max(0.0, min(0.45, float(prosody_emotion.confidence)))


def adaptive_reliability_aware_fusion(
    text_emotion: EmotionResult,
    face_emotion_json: Optional[dict],
    prosody_emotion: Optional[ProsodyEmotionResult],
    user_text: str = "",
    modality_response_times: Optional[dict[str, Optional[float]]] = None,
) -> FusedEmotionResult:
    emotions = [emo for emo in PLUTCHIK_EMOTIONS if emo != "neutral"]

    text_dist = one_hot_emotion_distribution(text_emotion.emotion, text_emotion.confidence)
    visual_dist = face_json_to_distribution(face_emotion_json)

    if prosody_emotion and prosody_emotion.available:
        prosody_dist = one_hot_emotion_distribution(prosody_emotion.emotion, prosody_emotion.confidence)
    else:
        prosody_dist = {emo: 0.0 for emo in emotions}

    explicit_text_emotion = explicit_emotion_from_text(user_text)

    text_rel = text_reliability_score(text_emotion, user_text)
    visual_rel = face_reliability_score(face_emotion_json)
    prosody_rel = prosody_reliability_score(prosody_emotion)

    visual_emotion = face_emotion_json.get("dominant_emotion") if face_emotion_json else None

    if explicit_text_emotion:
        text_rel = max(text_rel, 0.95)
        if visual_emotion and visual_emotion != explicit_text_emotion:
            visual_rel *= 0.45
        prosody_rel = min(prosody_rel, 0.25)

    if text_emotion.emotion == "anger" and visual_emotion == "sadness":
        visual_rel *= 0.40
        text_rel = max(text_rel, 0.92)

    question_like = user_text.strip().endswith("?") or any(
        phrase in user_text.lower()
        for phrase in ["who is", "what is", "do you know", "can you tell", "where is", "how do"]
    )
    if question_like:
        visual_rel *= 0.35
        text_rel = max(text_rel, 0.80)

    raw_text_weight = FUSION_TEXT_WEIGHT * text_rel
    raw_visual_weight = FUSION_VISUAL_WEIGHT * visual_rel
    raw_prosody_weight = FUSION_PROSODY_WEIGHT * prosody_rel

    total = raw_text_weight + raw_visual_weight + raw_prosody_weight
    if total <= 0:
        wt, wv, wp = 1.0, 0.0, 0.0
    else:
        wt = raw_text_weight / total
        wv = raw_visual_weight / total
        wp = raw_prosody_weight / total

    fused_scores = {}
    for emo in emotions:
        fused_scores[emo] = (
            wt * text_dist.get(emo, 0.0)
            + wv * visual_dist.get(emo, 0.0)
            + wp * prosody_dist.get(emo, 0.0)
        )

    dominant = max(fused_scores.items(), key=lambda item: item[1])[0]
    confidence = max(0.0, min(1.0, fused_scores[dominant]))

    reason = (
        f"Adaptive reliability-aware fusion selected {dominant}: "
        f"text={text_emotion.emotion} rel={text_rel:.2f}, "
        f"visual={visual_emotion} rel={visual_rel:.2f}, "
        f"prosody={prosody_emotion.emotion if prosody_emotion else None} rel={prosody_rel:.2f}."
    )

    return FusedEmotionResult(
        emotion=dominant,
        confidence=confidence,
        reason=reason,
        scores=fused_scores,
        weights={
            "base_text": FUSION_TEXT_WEIGHT,
            "base_visual": FUSION_VISUAL_WEIGHT,
            "base_prosody": FUSION_PROSODY_WEIGHT,
            "reliability_text": text_rel,
            "reliability_visual": visual_rel,
            "reliability_prosody": prosody_rel,
            "active_normalized_text": wt,
            "active_normalized_visual": wv,
            "active_normalized_prosody": wp,
        },
        text_emotion={
            "emotion": text_emotion.emotion,
            "confidence": text_emotion.confidence,
            "reason": text_emotion.reason,
        },
        visual_emotion=face_emotion_json or default_face_emotion_json(),
        prosody_emotion=prosody_emotion.as_json if prosody_emotion else {
            "available": False,
            "emotion": None,
            "confidence": 0.0,
            "reason": "No prosody result provided.",
            "features": {},
        },
        response_times=modality_response_times or {},
    )


# =========================
# Temporal emotion smoothing (across turns)
# =========================

def apply_temporal_emotion_smoothing(
    current_scores: dict[str, float],
    previous_smoothed_scores: Optional[dict[str, float]],
    alpha: float = EMOTION_SMOOTHING_ALPHA,
) -> dict[str, float]:
    """
    Exponential moving average (EMA) over the fused per-emotion score
    distribution, applied ACROSS TURNS within a session.

    The adaptive reliability-aware fusion above resolves an emotion fresh
    every turn from that turn's text/visual/prosody signals alone. That
    means a single noisy/ambiguous turn (a brief visual misread, a short
    ambiguous utterance) can cause the reported dominant emotion -- and
    therefore the robot's facial expression via RobotExpression -- to
    flicker between turns even when the user's underlying affective state
    has not really changed.

    This blends the CURRENT turn's fused scores with the PRIOR smoothed
    state:
        smoothed = alpha * current + (1 - alpha) * previous_smoothed
    `alpha` close to 1.0 makes the smoothed value track the current turn
    almost exactly (little smoothing); closer to 0.0 makes it change very
    slowly (heavy smoothing, more resistant to a single outlier turn).

    This is intentionally a pure function with no hidden state: the
    caller (the main loop) is responsible for holding `previous_smoothed_
    scores` across turns and passing it in each time, and for resetting it
    at the start of a new session.
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
        return "trust", 0.0
    dominant, value = max(scores.items(), key=lambda item: item[1])
    return dominant, max(0.0, min(1.0, value))


# =========================
# Emotion detection
# =========================

def build_emotion_prompt(
    transcribed_text: str,
    facial_emotion_hint: Optional[str] = None,
) -> str:
    emotions = ", ".join(PLUTCHIK_EMOTIONS.keys())
    facial_section = (
        f"Facial-expression hint captured during speech: {facial_emotion_hint}\n"
        if facial_emotion_hint
        else "No facial-expression hint was available.\n"
    )

    return f"""
        You are an emotion classification system for a human-robot interaction chat system.

        Classify the user's emotional state from the text below.

        You must map the emotion to exactly one of Plutchik's 8 primary emotions:

        {emotions}

        Use the user's words as the primary signal.
        Use the facial-expression hint only as a weak supporting signal.
        If the words and facial-expression hint conflict, trust the words more.
        Do not make strong claims from facial-expression analysis alone.

        Return JSON only.

        Required JSON schema:
        {{
        "emotion": "joy | trust | fear | surprise | sadness | disgust | anger | anticipation",
        "confidence": 0.0,
        "reason": "short explanation"
        }}

        Rules:
        - confidence must be a number between 0.0 and 1.0
        - choose the best single emotion, even if the message is mixed
        - do not add markdown
        - do not add extra text outside JSON
        - For greetings such as "hello", "hi", or "good morning", return:
        {{"emotion": "trust", "confidence": 0.6, "reason": "The user is opening a friendly social interaction."}}
        - For farewells such as "bye", "goodbye", "take care", or "talk later", return:
        {{"emotion": "trust", "confidence": 0.7, "reason": "The user is closing the conversation politely."}}

        User text:
        {transcribed_text}

        {facial_section}
        """.strip()


def simple_emotion_fallback(transcribed_text: str) -> Optional[EmotionResult]:
    text = transcribed_text.strip().lower()

    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
    farewells = {"bye", "goodbye", "see you", "see you later", "talk later", "have a good day", "have a nice day"}

    if text.rstrip(".!?") in greetings:
        return EmotionResult(
            emotion="trust",
            confidence=0.6,
            reason="The user is opening a friendly social interaction.",
        )

    if any(phrase in text for phrase in farewells):
        return EmotionResult(
            emotion="trust",
            confidence=0.7,
            reason="The user is closing the conversation politely.",
        )

    if "today's date" in text or "todays date" in text or "what is the date" in text:
        return EmotionResult(
            emotion="anticipation",
            confidence=0.5,
            reason="The user is asking for current date information.",
        )

    return None


def detect_emotion(
    client: Client,
    transcribed_text: str,
    facial_emotion_hint: Optional[str] = None,
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
                    "content": build_emotion_prompt(
                        transcribed_text=transcribed_text,
                        facial_emotion_hint=facial_emotion_hint,
                    ),
                },
            ],
            stream=False,
            options={
                "temperature": 0.1,
                "num_predict": 120,
                "num_ctx": 2048,
            },
        )
    except Exception as exc:
        print_ts(f"Emotion detection LLM call failed ({exc}); using neutral-social fallback.")
        return EmotionResult(
            emotion="trust",
            confidence=0.3,
            reason=f"Emotion model call failed ({exc}); neutral-social fallback used.",
        )

    raw = response["message"]["content"]
    data = safe_json_extract(raw)

    if not data:
        return EmotionResult(
            emotion="trust",
            confidence=0.3,
            reason="Could not parse model output, so a neutral-social fallback was used.",
        )

    emotion = str(data.get("emotion", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    if emotion not in PLUTCHIK_EMOTIONS:
        emotion = "trust"
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
    """
    Hard cap on reply length, expressed as a maximum number of sentences.

    This exists as a backstop against the LLM ignoring the "keep it short"
    instruction in the system prompt -- it is not a substitute for that
    instruction, just a guarantee that even a rambling model output gets
    cut down to something reasonable to speak out loud over TTS.

    Uses a simple regex-based sentence split (on '.', '!', '?' followed by
    whitespace), with one important fix: periods after common title
    abbreviations (Prof., Dr., Mr., Mrs., Ms., Sr., Jr., St., vs.) are
    protected from being treated as sentence boundaries first. Without
    this, a reply like "Prof. Dr. Karsten Berns is the head of the
    laboratory." gets mis-split into ["Prof.", "Dr.", "Karsten Berns is
    the head of the laboratory."], and a 2-sentence cap then keeps only
    "Prof. Dr." -- silently dropping the name and the actual answer that
    followed it. This is not a full sentence tokenizer, but is good enough
    for the short, single-paragraph replies this pipeline generates.
    """
    text = text.strip()
    if not text:
        return text

    def _protect_abbreviation_dot(match: "re.Match[str]") -> str:
        word = match.group(1)
        if word.lower() in _SENTENCE_ABBREVIATIONS:
            # \x00 is a placeholder that cannot appear in normal model
            # output; it is always restored to "." before this function
            # returns, so it never leaks into the final reply.
            return f"{word}\x00"
        return match.group(0)

    protected = re.sub(r"\b([A-Za-z]{1,4})\.(?=\s+[A-Z])", _protect_abbreviation_dot, text)

    sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [s.replace("\x00", ".").strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    return " ".join(sentences[:max_sentences]).strip()


def normalize_reply(raw_reply: str, emotion: str) -> str:
    required_emoji = PLUTCHIK_EMOTIONS[emotion]

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
    """
    Detect a spoken request to end the session (e.g. "Goodbye", "Bye",
    "See you later"). This is distinct from is_social_or_support_message()'s
    farewell check (which only skips Self-RAG retrieval for that one turn)
    -- this one is used by the main loop to actually terminate the
    conversation, the same way "/exit" does.
    """
    lowered = text.strip().lower().rstrip(".!?")
    return any(phrase in lowered for phrase in FAREWELL_TERMINATION_PHRASES)


def deterministic_reply_if_applicable(user_text: str, emotion: str) -> Optional[str]:
    text = user_text.strip().lower()
    emoji = PLUTCHIK_EMOTIONS.get(emotion, "🙂")

    if "today's date" in text or "todays date" in text or "what is the date" in text:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. {emoji}"

    if "what is the time" in text or "what time is it" in text or "current time" in text:
        return f"The current time is {datetime.now().strftime('%H:%M')}. {emoji}"

    if is_farewell_utterance(user_text):
        return "Thank you, and take care. 🙂"

    return None


# =========================
# General vision queries (ZED vision module)
# =========================

VISION_QUERY_KEYWORDS = [
    "what do you see",
    "what can you see",
    "can you see anything",
    "describe the image",
    "describe the picture",
    "describe what you see",
    "look around",
    "take a look around",
    "what is in the image",
    "what is in the picture",
]


def is_vision_query(user_query: str) -> bool:
    q = user_query.strip().lower()
    return any(k in q for k in VISION_QUERY_KEYWORDS) or ("see" in q and "what" in q)


def build_zed_vision_prompt(user_query: str) -> str:
    return f"""
Look at the image and answer like a human observer.

Rules:
- Maximum 12 words.
- One sentence only.
- Mention only the most important visible objects.
- No speculation.
- No explanations.
- No environment descriptions unless directly relevant.
- If a person is visible, mention the person first.

Examples:
"One person is sitting behind a computer."
"A humanoid robot is standing in the room."
"Two people are talking near a desk."

User question: {user_query}
""".strip()


def query_zed_vision(
    client: Client,
    vision_module: Any,
    user_query: str,
) -> str:
    """
    Answer a general "what do you see" question using a frame saved by the
    ZED vision module, independent of the Qwen2.5-VL facial-emotion sampler.
    """
    if vision_module is None:
        return "I am unable to access a current camera image right now. 🙂"

    image_path = None
    try:
        image_path = vision_module.save_latest_frame()
    except Exception as exc:
        print_ts(f"[ZED] Could not save latest frame: {exc}")

    if not image_path:
        return "I am unable to access a current camera image right now. 🙂"

    try:
        response = client.chat(
            model=ZED_VISION_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": build_zed_vision_prompt(user_query),
                    "images": [image_path],
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 60,
            },
            stream=False,
        )
        content = response.get("message", {}).get("content", "")
        reply = clean_text_for_tts(content)
        if not reply:
            reply = "I could not make out anything useful in the current image."
        if not reply.rstrip().endswith(("🙂", "😊", "😌", "😔", "😟", "🤔", "😮", "😢", "😠", "🤢")):
            reply = f"{reply} 🙂"
        return reply
    except Exception as exc:
        print_ts(f"[ZED] Vision query failed: {exc}")
        return "I'm sorry, I couldn't process that image request right now. 🙂"


# =========================
# Response generation
# =========================

def emotion_confidence_label(confidence: float) -> str:
    """
    Bucket a raw fused-emotion confidence float into a coarse label the
    response-generation prompt can act on directly. LLMs reliably follow
    "this is a low/medium/high confidence reading" as an instruction far
    better than they act on a bare float, which the earlier prompt exposed
    without ever telling the model what to do with it.
    """
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.40:
        return "medium"
    return "low"


def build_clean_emotion_summary(emotion_result: EmotionResult) -> dict[str, Any]:
    """
    Build the emotion summary that actually reaches the response-generation
    prompt, deliberately WITHOUT the raw FusedEmotionResult.reason string.

    That raw reason string (produced by adaptive_reliability_aware_fusion)
    is internal fusion diagnostics meant for logs/thesis analysis, e.g.:
        "Adaptive reliability-aware fusion selected joy: text=joy rel=0.85,
         visual=trust rel=0.40, prosody=trust rel=0.10."
    Passing that verbatim into the LLM's context is noise at best (it does
    not help the model choose a better tone) and a leak risk at worst
    (internal mechanics sitting in-context for the model to reference or
    paraphrase, despite being told not to mention fusion/detection).

    This function replaces it with a short, human-readable, prompt-safe
    summary: just the emotion label and a coarse confidence bucket. The
    original detailed reason is still preserved untouched on
    FusedEmotionResult/EmotionResult for the session transcript / logs.
    """
    return {
        "emotion": emotion_result.emotion,
        "confidence_label": emotion_confidence_label(emotion_result.confidence),
    }


EMOTION_TONE_GUIDANCE = """
Tone guidance by detected emotion (resonate first, then stabilize only if needed):

- joy: mirror it directly. Match the positive energy and be genuinely glad with them; don't undercut it.
- trust: warm, steady, collaborative. Mirror the ease of the moment.
- anticipation: mirror the forward-looking energy; be engaged and curious with them.
- surprise: mirror briefly, react with matching surprise or interest, then settle into the conversation.
- sadness: resonate first. Acknowledge the weight of it plainly and warmly, without rushing to comfort,
  fix, or advise. Let the acknowledgment sit before shifting toward anything forward-looking.
- fear/anxiety: resonate briefly (take the concern seriously, don't minimize it), then shift toward
  steady, grounding language. Do not amplify the worry.
- anger/frustration: resonate with the substance of what's frustrating them (e.g. "that does sound
  unfair" or "that's a frustrating position to be in") without adopting an angry tone yourself.
  Validate the feeling; stay steady rather than escalating or sounding defensive.
- disgust: acknowledge plainly without amplifying or dismissing. This one rarely needs active mirroring,
  since it is usually a reaction to something external rather than a shared experience.

General rule: match the user's energy level and warmth where appropriate, but do not claim to
personally feel the emotion yourself (no "I feel sad too" / "that makes me angry too"). Express
congruence through tone, word choice, and acknowledgment -- not through first-person emotional claims.
This keeps expression emotionally congruent while staying inside the "do not pretend to have human
emotions" boundary in your identity section.

Confidence-based commitment:
- If confidence_label is "high": let the tone guidance above shape your reply directly and fully.
- If confidence_label is "medium": lean into the tone guidance, but keep the reply a little more
  open-ended (e.g. a brief check-in question) rather than fully committing to one reading.
- If confidence_label is "low": the emotional reading may be noisy or wrong. Favor a neutral, gently
  curious tone instead of committing to the detected emotion -- it is fine to just respond naturally
  to what the user said without leaning hard into an emotional interpretation.
""".strip()


def build_response_system_prompt(
    emotion_result: EmotionResult,
    user_profile: Optional[dict] = None,
    face_emotion_json: Optional[dict] = None,
    self_rag_context: Optional[SelfRAGContext] = None,
) -> str:
    emoji = PLUTCHIK_EMOTIONS[emotion_result.emotion]
    memory_context = build_user_memory_context(user_profile)

    if face_emotion_json is None:
        face_emotion_json = default_face_emotion_json()

    # Clean, prompt-safe emotion summary -- emotion label + confidence
    # bucket only. Deliberately does NOT include emotion_result.reason
    # (the raw fusion-diagnostics string); see build_clean_emotion_summary().
    clean_emotion_summary = build_clean_emotion_summary(emotion_result)

    # The face/VLM JSON is still passed through for transparency, but only
    # a trimmed view (dominant emotion + reliability signal), not the full
    # internal diagnostics blob (deepface_diagnostics, timestamps, etc.),
    # which are for logs, not for the model's tone-setting.
    face_summary_for_prompt = {
        "dominant_emotion": face_emotion_json.get("dominant_emotion"),
        "reliable": face_emotion_json.get("reliable"),
    }

    return f"""
{AMECA_SYSTEM_PROMPT}

{runtime_context()}

{memory_context}

{build_self_rag_prompt_block(self_rag_context)}

You are generating Ameca's next conversational response.

PRIVATE EMOTION CONTEXT (for tone only -- never mention this to the user)
Detected emotion summary: {json.dumps(clean_emotion_summary)}
Facial-expression summary: {json.dumps(face_summary_for_prompt)}

Interpretation rules:
- The detected emotion was produced by adaptive reliability-aware late fusion across text, facial
  expression, and vocal prosody.
- Use it only to adjust tone, following the guidance below. Never mention fusion, detection, cameras,
  Qwen2.5-VL, prosody, or "private emotion context" to the user.
- Do not say things like "you look sad", "your face shows", or "I detected".

{EMOTION_TONE_GUIDANCE}

Return JSON only in this exact shape:
{{
"reply": "assistant response without emoji",
"emoji": "one facial emoji",
"tone": "short tone label"
}}

Speech recognition note:
- If the user says their name, update the profile silently and greet them by the corrected name.

Self-RAG grounding rules:
- If SELF-RAG CONTEXT is provided, answer factual or lab/domain-specific questions ONLY from that retrieved context.
- Do not invent names, titles, degrees, conference details, personal relationships, or project claims that are not explicitly in the retrieved context.
- If the retrieved context does not directly answer the question, say that you could not verify it from the local lab knowledge.
- For person lookup questions, only confirm a person if their name appears in the retrieved context.
- Do not say "I know" someone personally. Say "The local lab knowledge mentions..." or "I found a lab page for...".
- Do not mention Self-RAG, vector databases, embeddings, ChromaDB, or retrieval unless the user explicitly asks how the system works.
- If you don't have information to answer something (no Self-RAG context and no memory of it), say so plainly rather than guessing.

Conversation behavior rules:
- Always ensure your response is context appropriate and helpful.
- Use the recent conversation history to understand context and avoid repeating yourself.
- Do not greet repeatedly. After the first greeting, respond directly to what the user said.
- Do not reintroduce yourself unless the user asks who you are, and never begin with "As Ameca" or
  "As a humanoid social robot".
- Use the user's name occasionally, not in every response.
- Do not immediately give a list of advice unless the user asks for advice, and do not repeat advice
  already given earlier in the conversation.
- Speak directly and naturally; stay on the topic the user raised rather than introducing unrelated
  topics.
- Do not use markdown, bullets, numbered lists, or long explanations unless the user asks for detail.
- STRICT: reply in 1-2 short sentences only. Longer replies get cut off automatically, so make every
  sentence count.
- If the user asks who can help with something, suggest concrete people: supervisor, co-supervisor,
  lab colleagues, thesis coordinator, or university writing center.

Emoji rules:
- Always end with exactly one context-appropriate facial emoji from this set: 🙂 😊 😌 😔 😟 🤔 😮 😢 😠 🤢
- Do not use any other emoji or emoticon symbols, and don't overreact emotionally.
""".strip()


def limit_text_length(text: str, max_chars: int = 1500) -> str:
    return text[:max_chars]


def limit_system_prompt(prompt: str, max_chars: int = 9000) -> str:
    return prompt[:max_chars]


def trim_history(history: list[dict]) -> list[dict]:
    return history[-MAX_HISTORY_MESSAGES:]


def prompt_ready_history(history: list[dict]) -> list[dict]:
    return [{"role": item["role"], "content": item["content"]} for item in history]


def _is_degenerate_reply_text(text: str) -> bool:
    """
    True if `text` isn't real conversational content -- e.g. the model
    returned a bare empty JSON object/array, or nothing at all, instead of
    an actual reply. Used in generate_response() to catch cases like a
    literal "{}" being spoken to the user verbatim rather than a proper
    apology/fallback.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return True
    return stripped.lower() in {"{}", "[]", "null", "none", "{ }", "[ ]"}


class _LLMCallFailed(Exception):
    """
    Internal marker raised by _attempt_llm_response() when the underlying
    client.chat() call itself fails (network/Ollama issue), as opposed to
    the call succeeding but returning degenerate content (which returns
    None instead). generate_response() catches this to choose an accurate
    final fallback message ("having trouble reaching my language model")
    rather than the generic "could you say that again?" wording, which
    would be misleading for an actual connection failure.
    """


def _attempt_llm_response(
    client: Client,
    messages: list[dict],
    emotion_result: EmotionResult,
    self_rag_context: Optional[SelfRAGContext],
    repeat_penalty: float,
) -> Optional[str]:
    """
    Make one attempt at the main response-generation LLM call and parse it
    into a final reply string.

    Returns None if the call succeeded but produced degenerate content
    (e.g. a bare "{}" with no usable "reply" key -- see
    _is_degenerate_reply_text). Raises _LLMCallFailed if the call itself
    failed. Callers use these two distinct outcomes to decide whether to
    retry once (see generate_response()) and which fallback message is
    accurate if both attempts are exhausted.
    """
    try:
        response = client.chat(
            model=MODEL_NAME,
            format="json",
            messages=messages,
            options={
                "temperature": 0.25 if self_rag_context and self_rag_context.used else 0.4,
                # Kept deliberately small so the model has less room to
                # ramble before hitting the hard sentence cap in
                # normalize_reply(); reduces how often replies get
                # truncated mid-thought.
                "num_predict": 90,
                "repeat_penalty": repeat_penalty,
                "num_ctx": 8192,
            },
            stream=False,
        )
    except Exception as exc:
        print_ts(f"Response generation LLM call failed ({exc}).")
        raise _LLMCallFailed(str(exc)) from exc

    raw_reply = response["message"]["content"]
    data = safe_json_extract(raw_reply)

    # NOTE: this must check `data is not None`, not truthiness of `data`
    # itself. A model that returns a literal empty JSON object "{}" parses
    # to an empty Python dict, which is falsy (`bool({}) == False`) even
    # though it is a perfectly valid, non-None dict. Checking truthiness
    # here would silently skip this branch for that case and fall through
    # to using the raw "{}" string as if it were plain reply text, which
    # is exactly how "{} 🙂" ended up being spoken to the user verbatim.
    if data is not None and isinstance(data, dict):
        reply_text = str(data.get("reply", "")).strip()
        emoji = str(data.get("emoji", "")).strip()

        if emoji in {":)", ":-)", ""}:
            emoji = PLUTCHIK_EMOTIONS.get(emotion_result.emotion, "🙂")

        if reply_text and not _is_degenerate_reply_text(reply_text):
            return normalize_reply(f"{reply_text} {emoji}", emotion_result.emotion)
        # Empty or degenerate "reply" field -- the model didn't actually
        # answer. Treat this the same as a parse failure (return None) so
        # the caller can retry rather than emit empty content.
        return None

    if _is_degenerate_reply_text(raw_reply):
        # Second layer of the same guard: even outside the dict-parsing
        # branch above, the raw model output itself might just be a bare
        # "{}", "[]", or similarly empty/non-content string.
        return None

    final_reply = normalize_reply(raw_reply, emotion_result.emotion)
    if self_rag_context and self_rag_context.used and context_has_placeholder_risk(final_reply):
        return normalize_reply(
            "I found a relevant local lab page, but I could not verify the exact name from the retrieved text, so I should not invent it. 🙂",
            emotion_result.emotion,
        )
    return final_reply


def generate_response(
    client: Client,
    user_text: str,
    emotion_result: EmotionResult,
    history: list[dict],
    user_profile: Optional[dict] = None,
    face_emotion_json: Optional[dict] = None,
    self_rag_context: Optional[SelfRAGContext] = None,
) -> str:
    deterministic = deterministic_reply_if_applicable(
        user_text=user_text,
        emotion=emotion_result.emotion,
    )

    if deterministic:
        return deterministic

    safe_user_text = limit_text_length(user_text)
    system_prompt = limit_system_prompt(
        build_response_system_prompt(
            emotion_result=emotion_result,
            user_profile=user_profile,
            face_emotion_json=face_emotion_json,
            self_rag_context=self_rag_context,
        )
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *prompt_ready_history(trim_history(history[-6:])),
        {"role": "user", "content": safe_user_text},
    ]

    if self_rag_context and self_rag_context.used:
        grounded_reply = generate_grounded_self_rag_answer(
            client=client,
            user_text=safe_user_text,
            self_rag_context=self_rag_context,
            emotion=emotion_result.emotion,
        )
        if grounded_reply:
            return grounded_reply

    # repeat_penalty lowered from 1.25 to 1.1: a high repeat penalty can
    # push the model toward emitting a bare "{}" instead of populating the
    # "reply"/"emoji"/"tone" schema keys, especially on short, repetitive
    # turns (e.g. "what's your name?" asked twice in a row) where those
    # exact schema tokens -- and the assistant's own prior reply -- have
    # already appeared several times in the accumulated conversation
    # context. This is the first half of the mitigation; the retry below
    # is the second half.
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
        return reply

    # First attempt returned nothing usable (the call failed outright, or
    # the model produced degenerate/empty content such as a bare "{}").
    # This is rare, so paying for one extra LLM call only in that case is
    # cheap insurance against the user ever hearing raw JSON syntax or a
    # generic fallback when a normal answer was just one retry away.
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
        return reply

    if call_failed:
        print_ts("Response generation LLM call failed on both attempts; using connectivity fallback reply.")
        return normalize_reply(
            "I'm having trouble reaching my language model right now, so I can't respond properly to that.",
            emotion_result.emotion,
        )

    print_ts("Response generation produced no usable content on retry either; using fallback reply.")
    return normalize_reply(
        "Sorry, could you say that again? I didn't quite catch a clear response that time.",
        emotion_result.emotion,
    )


# =========================
# CLI args (robot-specific)
# =========================

def parse_robot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ameca demo: Silero VAD + faster-whisper + emotion fusion + Self-RAG, with Tritium TTS output, Tritium facial expression, and ZED vision."
    )

    parser.add_argument(
        "--chat_model",
        default=MODEL_NAME,
        help=f"Ollama model used for text chat, emotion detection, Self-RAG, and session summaries (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--face_vision_model",
        default=VISION_MODEL_NAME,
        help=f"Ollama vision-language model used for Qwen2.5-VL facial-emotion sampling (default: {VISION_MODEL_NAME}).",
    )
    parser.add_argument("--vision_model", default="llava:7b", help="Ollama vision-language model used for general ZED 'what do you see' queries (default: llava:7b).")

    parser.add_argument("--tts_url", default=os.environ.get("TTS_URL", "http://emah/tritium/text_to_speech/say?voice=Lucy"))
    parser.add_argument("--speaking_cooldown", type=float, default=0.3, help="Seconds of echo-guard cooldown after TTS finishes speaking.")
    parser.add_argument(
        "--tts_token",
        default=os.environ.get("TTS_TOKEN", "ZWNFuNQVIPyztWCfPPM5VLPslpj8rR"),
        help="X-Tritium-Auth-Token used for both the TTS 'say' endpoint and the sequence_player expression endpoint.",
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
        "--deepface_python",
        default=DEEPFACE_PYTHON,
        help="Path to the python interpreter of a SEPARATE conda env with deepface+tensorflow "
        "installed, used to run deepface_worker.py in an isolated process. Leave unset to "
        "disable the hybrid VLM+DeepFace visual modality and use Qwen2.5-VL alone.",
    )
    parser.add_argument(
        "--deepface_worker_script",
        default=DEEPFACE_WORKER_SCRIPT,
        help=f"Path to deepface_worker.py (default: {DEEPFACE_WORKER_SCRIPT}).",
    )
    parser.add_argument(
        "--disable_deepface",
        action="store_true",
        help="Disable the hybrid VLM+DeepFace visual modality even if --deepface_python is set.",
    )
    parser.add_argument(
        "--deepface_timeout",
        type=float,
        default=DEEPFACE_REQUEST_TIMEOUT_SECONDS,
        help=(
            f"Per-frame DeepFace analyze() timeout in seconds (default: {DEEPFACE_REQUEST_TIMEOUT_SECONDS}). "
            "Raise this if DeepFace consistently times out (e.g. CPU-only hardware with a slower "
            "detector backend like mtcnn) -- a timeout means DeepFace is treated as unavailable for "
            "that turn, which also disables the DeepFace-first face-presence gate ahead of Qwen2.5-VL "
            "for that turn, since there is no confirmed face/no-face result to gate on."
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

    parser.add_argument("--videoIndex", type=int, default=int(os.environ.get("ZED_VIDEO_INDEX", "0")), help="ZED camera index. This is now the SINGLE camera handle, used both for Qwen2.5-VL facial-emotion sampling and for general 'what do you see' queries.")
    parser.add_argument("--resolution", default=os.environ.get("ZED_RESOLUTION", "HD2K"))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("ZED_FPS", "15")))
    parser.add_argument("--view", choices=["LEFT", "RIGHT"], default=os.environ.get("ZED_VIEW", "LEFT"))
    parser.add_argument("--no_mjpeg", action="store_true")
    parser.add_argument(
        "--disable_general_vision_queries",
        action="store_true",
        help="Keep the ZED camera running for facial-emotion sampling, but disable routing 'what do you see' style questions to it.",
    )

    return parser.parse_args()


# =========================
# Main loop
# =========================

def main() -> None:
    global MODEL_NAME, VISION_MODEL_NAME, ZED_VISION_MODEL_NAME

    args = parse_robot_args()

    # CLI args take precedence over the env-var-based defaults set at module
    # load time. These globals are read throughout the file (emotion
    # detection, Self-RAG, response generation, facial sampling, ZED vision
    # queries, session transcript metadata), so they must be set before any
    # of that code runs.
    MODEL_NAME = args.chat_model
    VISION_MODEL_NAME = args.face_vision_model
    ZED_VISION_MODEL_NAME = args.vision_model

    emotion_smoothing_enabled = EMOTION_SMOOTHING_ENABLED and not args.disable_emotion_smoothing
    emotion_smoothing_alpha = args.emotion_smoothing_alpha

    print_ts("Starting integrated Ameca demo: Silero VAD + faster-whisper + persistent memory + Self-RAG + adaptive fusion + temporal smoothing + Tritium TTS + Tritium facial expression + ZED vision.")
    print_ts(f"Python: {sys.version.split()[0]}")
    print_ts(f"Ollama host: {OLLAMA_HOST}")
    print_ts(f"Ollama chat model: {MODEL_NAME}")
    print_ts(f"Ollama vision model (facial emotion): {VISION_MODEL_NAME}")
    print_ts(f"Ollama vision model (ZED general queries): {ZED_VISION_MODEL_NAME}")
    print_ts(f"Ollama embedding model (Self-RAG): {SELF_RAG_EMBED_MODEL}")
    print_ts(f"Tritium TTS URL: {args.tts_url}")
    print_ts(f"Tritium expression host: {args.expression_host} (disabled={args.disable_expression})")
    print_ts(f"Temporal emotion smoothing enabled: {emotion_smoothing_enabled} (alpha={emotion_smoothing_alpha})")

    # zed_vision_module is now the SINGLE owner of the ZED camera handle, used
    # both for Qwen2.5-VL facial-emotion sampling and (optionally) general
    # "what do you see" queries, so it is required rather than optional.
    #
    # NOTE on TensorFlow/PyTorch segfault risk: zed_vision_module.py now
    # imports tensorflow/deepface LAZILY, gated behind
    # enable_emotion_analysis (see _ensure_deepface_imported() there), not
    # unconditionally at module load time -- an earlier version imported it
    # eagerly, which was the original source of a same-process TF/PyTorch
    # segfault. We still always pass enable_emotion_analysis=False below:
    # zed_vision_module's own in-process DeepFace path is now DEPRECATED in
    # favor of the isolated hybrid VLM+DeepFace modality (DeepFaceClient /
    # deepface_worker.py, launched in its own process/conda env further
    # down in this function), which is what actually provides DeepFace
    # cross-checking for facial emotion in this pipeline. Self-RAG's
    # embeddings also go through Ollama instead of an in-process
    # sentence-transformers/PyTorch model (see get_ollama_embedding()),
    # removing another potential PyTorch-vs-TensorFlow collision point.
    try_import_zed_vision_module()
    if not HAS_ZED_VISION:
        raise RuntimeError(
            "zed_vision_module could not be imported, but it is now required as the "
            "single camera source for Qwen2.5-VL facial-emotion sampling. Make sure "
            "zed_vision_module.py and gaze_speaker_utils.py are importable."
        )

    print()

    ensure_data_dirs()

    check_ollama_available()
    ensure_model_available(MODEL_NAME)
    ensure_model_available(VISION_MODEL_NAME)
    if ZED_VISION_MODEL_NAME != VISION_MODEL_NAME:
        ensure_model_available(ZED_VISION_MODEL_NAME)
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

    # ---- Robot output: Tritium TTS ----
    robot_speaker = RobotSpeaker(
        tts_url=args.tts_url,
        tts_token=args.tts_token,
        speaking_cooldown_s=args.speaking_cooldown,
    )

    # ---- Robot output: Tritium facial expression (sequence_player) ----
    # This is the piece that was previously missing entirely: the fused
    # emotion result was computed every turn but nothing ever pushed it to
    # the robot's face. RobotExpression reuses the same PUT-based
    # sequence_player mechanism AmecaRobotChat.play_sequence() used for the
    # "exercise_routine" gesture, but targets facial-expression sequences
    # keyed by Plutchik emotion (see EMOTION_SEQUENCE_MAP).
    robot_expression = RobotExpression(
        host=args.expression_host,
        tts_token=args.tts_token,
    )

    # ---- Hybrid visual modality: Qwen2.5-VL + DeepFace ----
    # DeepFace runs in its own process (deepface_worker.py), ideally from a
    # separate conda env, to avoid the TensorFlow/PyTorch same-process
    # segfault risk documented on DEEPFACE_ENABLED above. If
    # --deepface_python / DEEPFACE_PYTHON isn't set, or the worker fails to
    # start, deepface_client simply won't be "alive" and every downstream
    # call degrades to VLM-only behavior automatically. DeepFace is also
    # used as a first-pass face-presence gate ahead of Qwen2.5-VL -- see
    # FaceEmotionSampler._analyze_snapshot_now().
    deepface_client: Optional[DeepFaceClient] = None
    deepface_enabled = DEEPFACE_ENABLED and not args.disable_deepface
    if deepface_enabled:
        print_ts(f"Starting DeepFace worker via: {args.deepface_python or '(not set)'}")
        deepface_client = DeepFaceClient(
            python_executable=args.deepface_python,
            worker_script=args.deepface_worker_script,
            request_timeout=args.deepface_timeout,
        )
        if deepface_client.is_alive():
            print_ts("Hybrid visual modality ENABLED (DeepFace face-presence gate first, Qwen2.5-VL conditional).")
        else:
            print_ts("Hybrid visual modality unavailable this session; using Qwen2.5-VL alone.")
    else:
        print_ts("DeepFace hybrid visual modality disabled (DEEPFACE_ENABLED=0 or --disable_deepface).")

    # Optional TTS-activity monitor (avoids the robot hearing its own voice),
    # mirroring AmecaRobotChat's setup. Best-effort; continues without it if
    # the target audio device cannot be found.
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

    # ---- ZED vision module: SINGLE camera owner ----
    # Used for Qwen2.5-VL facial-emotion sampling (always) and, unless
    # disabled, also for general "what do you see" queries. DeepFace
    # emotion analysis is explicitly OFF: Qwen2.5-VL is the sole source of
    # truth for facial-expression recognition in this pipeline.
    vision_module = ZedVisionModule(
        video_index=args.videoIndex,
        resolution=args.resolution,
        fps=args.fps,
        view=args.view,
        no_mjpeg=args.no_mjpeg,
        show_window=False,
        enable_emotion_analysis=False,
    )
    vision_thread = threading.Thread(target=vision_module.start, daemon=True)
    vision_thread.start()
    print_ts(
        "[ZED] Vision module started (DeepFace emotion analysis disabled; "
        "Qwen2.5-VL is the sole source of truth for facial emotion)."
    )

    # Give the capture loop a moment to deliver its first frame before
    # facial sampling/vision queries start relying on get_latest_frame().
    time.sleep(max(0.0, CAMERA_WARMUP_SECONDS))

    if args.disable_general_vision_queries:
        print_ts("[ZED] General 'what do you see' vision queries disabled via --disable_general_vision_queries.")

    user_key, user_profile, intro_reply = prompt_for_user_name(
        client=client,
        whisper_model=whisper_model,
        silero_model=silero_model,
        input_device=INPUT_DEVICE,
        robot_speaker=robot_speaker,
    )

    robot_speaker.say(intro_reply)
    # Neutral/trust expression on introduction (before any fused emotion
    # has been computed for this session).
    if not args.disable_expression:
        robot_expression.set_emotion("trust", confidence=1.0, force=True)

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
        "Speak naturally. Silero VAD will detect speech; DeepFace will check for a face first and "
        "Qwen2.5-VL will only analyze a cleaned facial snapshot if a face is confirmed present; "
        "faster-whisper will transcribe the utterance; adaptive reliability-aware fusion (with "
        "cross-turn temporal smoothing) will combine text/visual/prosody; Ameca's face will update "
        "to match the smoothed emotion via Tritium sequence_player; and Ameca will respond out loud "
        "via Tritium TTS."
    )
    print("Say '/exit', or say a farewell such as 'goodbye', to save the transcript and quit.")
    print()

    history: list[dict] = []
    session_log: list[dict] = []

    # Running state for cross-turn temporal emotion smoothing (see
    # apply_temporal_emotion_smoothing()). None until the first turn's
    # fused scores are available; reset implicitly each time main() runs
    # (i.e. once per session).
    smoothed_emotion_scores: Optional[dict[str, float]] = None

    session_log.append({
        "role": "assistant",
        "content": intro_reply,
        "timestamp": now_ts(),
        "intent": "self_introduction",
    })
    history.append({"role": "assistant", "content": intro_reply})

    try:
        while True:
            wav_path, face_capture = listen_for_utterance_with_silero_vad_and_face_emotion(
                input_device=INPUT_DEVICE,
                silero_model=silero_model,
                vision_client=client,
                vision_module=vision_module,
                prompt_label="utterance",
                robot_speaker=robot_speaker,
                deepface_client=deepface_client,
            )

            if face_capture:
                if face_capture.error:
                    print_ts(f"Face emotion capture error: {face_capture.error}")
                else:
                    print_ts(f"Face emotion summary: {face_capture.summary_text}")

            if not wav_path:
                continue

            audio_for_prosody = np.array([], dtype=np.float32)
            try:
                audio_for_prosody = load_audio_for_prosody(wav_path)
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
            # Same priority tier as "/exit": checked before name-extraction
            # and before any of the emotion/Self-RAG/response-generation
            # pipeline runs for this turn.
            if is_farewell_utterance(user_text):
                farewell_reply = "Thank you, and take care. 🙂"
                print_ts(f"Assistant: {farewell_reply}")
                robot_speaker.say(farewell_reply)
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
            if maybe_name and not looks_like_invalid_name(maybe_name):
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
                robot_speaker.say(reply)
                if not args.disable_expression:
                    robot_expression.set_emotion("trust", confidence=1.0)
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
                robot_speaker.say("Goodbye, and thank you for talking with me.")
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
                # Unlike "rag reindex" (which adds/updates chunks in the
                # EXISTING collection), this deletes and recreates the
                # collection first. Use this if you see a
                # "Collection expecting embedding with dimension of X, got Y"
                # error, or after changing SELF_RAG_EMBED_MODEL -- reindexing
                # into a stale-dimension collection alone will not fix it.
                self_rag_store = rebuild_self_rag_collection(self_rag_store)
                continue

            # ---------- General ZED vision queries ----------
            if is_vision_query(user_text) and not args.disable_general_vision_queries:
                print_ts("[VISION QUERY] Routing to ZED vision module.")
                vision_reply = query_zed_vision(client=client, vision_module=vision_module, user_query=user_text)

                print_ts(f"Assistant: {vision_reply}")
                robot_speaker.say(vision_reply)
                print()

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": vision_reply})
                history = trim_history(history)

                session_log.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "input_mode": "silero_vad_faster-whisper",
                    "intent": "zed_vision_query",
                })
                session_log.append({
                    "role": "assistant",
                    "content": vision_reply,
                    "timestamp": now_ts(),
                })

                continue

            try:
                facial_hint = face_capture.summary_text if face_capture else None
                face_emotion_json = face_capture.as_json if face_capture else default_face_emotion_json()

                # ---- Run emotion detection and Self-RAG concurrently ----
                # detect_emotion() only needs user_text + facial_hint, and
                # build_self_rag_context() only needs user_text + the
                # Self-RAG store -- neither depends on the other's output.
                # They previously ran strictly back-to-back, each paying
                # its own full Ollama round-trip (roughly 2s + 2s stacked).
                # Since both are I/O-bound (waiting on an HTTP response
                # from Ollama), running them in a small thread pool lets
                # their latencies overlap instead of stacking, so this part
                # of the turn costs roughly max(a, b) instead of a + b.
                parallel_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as concurrent_executor:
                    text_emotion_future = concurrent_executor.submit(
                        detect_emotion,
                        client=client,
                        transcribed_text=user_text,
                        facial_emotion_hint=facial_hint,
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

                prosody_start = time.time()
                prosody_result = analyze_prosody_from_audio(audio_for_prosody, TARGET_SAMPLE_RATE)
                prosody_response_seconds = time.time() - prosody_start

                fused_emotion_result = adaptive_reliability_aware_fusion(
                    text_emotion=text_emotion_result,
                    face_emotion_json=face_emotion_json,
                    prosody_emotion=prosody_result,
                    user_text=user_text,
                    modality_response_times={
                        "text_seconds": text_response_seconds,
                        "visual_seconds": face_capture.analysis_seconds if face_capture else None,
                        "prosody_seconds": prosody_response_seconds,
                    },
                )

                # ---- Temporal smoothing across turns ----
                # The raw per-turn fused result (fused_emotion_result) is
                # still logged in full below for diagnostics/thesis
                # analysis. The SMOOTHED result is what actually drives
                # response tone and the robot's facial expression, so a
                # single noisy/ambiguous turn doesn't cause visible
                # flicker. See apply_temporal_emotion_smoothing().
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
                        # Keep the raw fusion reason for the logs (see
                        # build_clean_emotion_summary(), which is what
                        # actually reaches the LLM prompt and never
                        # includes this raw diagnostic string).
                        reason=fused_emotion_result.reason,
                    )
                else:
                    emotion_result = fused_emotion_result.to_emotion_result()

                # ---- Drive the physical face from the (smoothed) emotion ----
                # This runs as soon as fusion/smoothing resolves, independent
                # of TTS timing (per requirement: continuous, turn-by-turn
                # expression updates, not tied to when the robot speaks).
                if not args.disable_expression:
                    robot_expression.set_emotion(
                        emotion_result.emotion,
                        confidence=emotion_result.confidence,
                    )

                text_emotion_json = {
                    "emotion": text_emotion_result.emotion,
                    "confidence": text_emotion_result.confidence,
                    "reason": text_emotion_result.reason,
                }

                prosody_json = prosody_result.as_json
                emotion_json = fused_emotion_result.as_json
                emotion_json["temporal_smoothing"] = {
                    "enabled": emotion_smoothing_enabled,
                    "alpha": emotion_smoothing_alpha,
                    "smoothed_scores": smoothed_emotion_scores,
                    "smoothed_emotion": emotion_result.emotion,
                    "smoothed_confidence": emotion_result.confidence,
                }

                print_ts("Adaptive reliability-aware fusion JSON (raw, pre-smoothing):")
                print(json.dumps(fused_emotion_result.as_json, indent=2))
                print()

                print_ts(
                    f"Smoothed emotion used for tone/expression: {emotion_result.emotion} "
                    f"(confidence={emotion_result.confidence:.2f}, alpha={emotion_smoothing_alpha})"
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
                    face_emotion_json=face_emotion_json,
                    self_rag_context=self_rag_context,
                )

                print_ts(f"Assistant: {reply}")
                robot_speaker.say(reply)
                print()

                user_message = {
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "emotion": emotion_json,
                    "facial_emotion_hint": facial_hint,
                    "face_emotion": face_emotion_json,
                    "text_emotion": text_emotion_json,
                    "prosody_emotion": prosody_json,
                    "self_rag": self_rag_context.as_json,
                    "input_mode": "silero_vad_faster-whisper_qwen25vl_adaptive_reliability_aware_fusion_temporal_smoothing_self_rag",
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
                # Never let an unexpected error in emotion/fusion/Self-RAG/
                # response generation kill the whole session. Log it, speak
                # a short apology, record what we can in the session log,
                # and move on to the next utterance.
                print_ts(f"[ERROR] Unexpected error while processing this turn: {exc!r}")
                import traceback as _traceback
                _traceback.print_exc()

                apology = "I'm sorry, something went wrong on my end with that. Could you try again?"
                robot_speaker.say(apology)

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
        if vision_module is not None:
            try:
                vision_module.stop()
            except Exception:
                pass

        if deepface_client is not None:
            try:
                deepface_client.shutdown()
                print_ts("[DeepFace] Worker shut down.")
            except Exception:
                pass

        if session_log:
            session_path = save_session_transcript(
                user_key=user_key,
                user_profile=user_profile,
                session_log=session_log,
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