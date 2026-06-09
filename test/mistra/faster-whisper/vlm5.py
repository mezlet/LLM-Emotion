#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import sys
import tempfile
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple

# Linux-only Qt fix. Do not force this on macOS.
if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

# Harmless on non-Windows; prevents some OpenCV backend priority issues on Windows.
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from ollama import Client
from silero_vad import load_silero_vad, VADIterator
import platform

IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

CAMERA_DEVICE = 0 if IS_MAC else 1
USE_ZED_HALF_FRAME_CROP = IS_LINUX

# =========================
# Local Ollama configuration
# =========================

# OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_HOST = "https://consideration-consultancy-blocked-awareness.trycloudflare.com"

MODEL_NAME = "llama3:8b"
# MODEL_NAME = "mistral:7b"
VISION_MODEL_NAME = "qwen2.5vl:7b"


# =========================
# Persistent memory / transcript configuration
# =========================

DATA_DIR = "conversation_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")


# =========================
# Self-RAG / local knowledge configuration
# =========================

# Self-RAG lets Ameca retrieve local knowledge, judge whether it is useful,
# and answer with that knowledge only when it is relevant.
SELF_RAG_ENABLED = os.environ.get("SELF_RAG_ENABLED", "1") == "1"
SELF_RAG_KB_DIR = os.environ.get("SELF_RAG_KB_DIR", "knowledge_base")

# IMPORTANT:
# scrape.py writes the RRLab website index to CHROMA_PERSIST_DIR="chroma_db"
# and CHROMA_COLLECTION="emah_knowledge". These defaults must match scrape.py.
SELF_RAG_DB_DIR = os.environ.get("SELF_RAG_DB_DIR", "chroma_db")
SELF_RAG_COLLECTION = os.environ.get("SELF_RAG_COLLECTION", "emah_knowledge")
SELF_RAG_EMBED_MODEL = os.environ.get("SELF_RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
SELF_RAG_TOP_K = int(os.environ.get("SELF_RAG_TOP_K", "12"))
SELF_RAG_CHUNK_SIZE = int(os.environ.get("SELF_RAG_CHUNK_SIZE", "900"))
SELF_RAG_CHUNK_OVERLAP = int(os.environ.get("SELF_RAG_CHUNK_OVERLAP", "150"))
SELF_RAG_MIN_CONTEXT_CHARS = int(os.environ.get("SELF_RAG_MIN_CONTEXT_CHARS", "80"))
SELF_RAG_MAX_CONTEXT_CHARS = int(os.environ.get("SELF_RAG_MAX_CONTEXT_CHARS", "2500"))

SELF_RAG_MAX_DISTANCE = float(os.environ.get("SELF_RAG_MAX_DISTANCE", "0.52"))
SELF_RAG_FINAL_TOP_K = int(os.environ.get("SELF_RAG_FINAL_TOP_K", "5"))
SELF_RAG_MIN_HYBRID_SCORE = float(os.environ.get("SELF_RAG_MIN_HYBRID_SCORE", "0.62"))
SELF_RAG_PERSON_LOOKUP_STRICT = os.environ.get("SELF_RAG_PERSON_LOOKUP_STRICT", "1") == "1"
SELF_RAG_SKIP_SOCIAL = os.environ.get("SELF_RAG_SKIP_SOCIAL", "1") == "1"
SELF_RAG_SKIP_EMOTIONAL_SUPPORT = os.environ.get("SELF_RAG_SKIP_EMOTIONAL_SUPPORT", "1") == "1"

# Entity triggers force RAG even if the message looks social.
# This prevents failures like "Do you know Ashita?" being treated as small talk.
KNOWN_RRLAB_ENTITIES = {
    "ashita",
    "ashita ashok",
    "ameca",
    "emah",
    "rrlab",
    "robotics research lab",
    "robotersysteme",
    "ravon",
    "robin",
    "carl",
    "unimog",
    "avos",
    "dengel",
    "sembai",
    "senna",
    "casrew",
    "zukunftbau",
    "znt",
}

# Keep this off by default because the RRLab scraper already rebuilds the index.
# Use /rrlab crawl or /self-rag reindex inside the app to refresh the website KB.
SELF_RAG_REINDEX_ON_START = os.environ.get("SELF_RAG_REINDEX_ON_START", "0") == "1"
SELF_RAG_AUTO_SCRAPE_ON_EMPTY = os.environ.get("SELF_RAG_AUTO_SCRAPE_ON_EMPTY", "0") == "1"
SELF_RAG_SCRAPE_SCRIPT = os.environ.get("SELF_RAG_SCRAPE_SCRIPT", "scrape.py")
SELF_RAG_SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".py", ".html", ".htm", ".pdf"}

# Keep this False for fast, clean shutdown.
# If True, the app will call Ollama at shutdown to summarize the session.
ENABLE_LLM_SESSION_SUMMARY = False

# System requirement: ask the user to spell their name after the spoken-name step.
ENABLE_NAME_SPELLING = os.environ.get("ENABLE_NAME_SPELLING", "1") == "1"

# Letter-by-letter spelling is unreliable with ASR, so keep it off by default.
ASK_SPELLED_NAME_ON_START = os.environ.get("ASK_SPELLED_NAME_ON_START", "0") == "1"


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
Treat the conversation as ephemeral in speech, but save only the local JSON transcript for this prototype when the session ends.

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
    "device": "cpu",
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

# Silero VAD works best with 16kHz audio and 512-sample chunks.
SILERO_SAMPLE_RATE = 16000
SILERO_CHUNK_SIZE = 512

# Sensitivity threshold. Lower = more sensitive. Higher = less false triggering.
SILERO_THRESHOLD = 0.55

# Silero internal timing. These are in milliseconds.
SILERO_MIN_SILENCE_DURATION_MS = 700
SILERO_SPEECH_PAD_MS = 250

# Extra utterance controls.
VAD_MAX_UTTERANCE_SECONDS = 15.0
VAD_MIN_UTTERANCE_SECONDS = 0.60
VAD_PRE_ROLL_SECONDS = 0.35

MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003


# =========================
# Camera / Qwen2.5-VL configuration
# =========================

# =========================
# Camera / Qwen2.5-VL configuration
# =========================

CAMERA_DEVICE: Optional[int] = 0 if IS_LINUX else 0

# ZED stereo frame is side-by-side.
# 2560x720 means each eye becomes 1280x720 after cropping.
ZED_STEREO_WIDTH = 2560
ZED_STEREO_HEIGHT = 720

CAMERA_FRAME_WIDTH = ZED_STEREO_WIDTH if IS_LINUX else 1280
CAMERA_FRAME_HEIGHT = ZED_STEREO_HEIGHT if IS_LINUX else 720

CAMERA_PREVIEW_WIDTH = 960
CAMERA_PREVIEW_HEIGHT = 540

CAMERA_PREVIEW_ENABLED = True
CAMERA_PREVIEW_WINDOW_NAME = "Camera Preview - press q to close preview"

USE_ZED_HALF_FRAME_CROP = IS_LINUX

# Qwen2.5-VL captures multiple candidate frames during speech, then analyzes
# only the sharpest candidate after speech ends. This keeps VLM latency close to
# one-frame inference while avoiding the single-snapshot timing gamble.
CAMERA_SAMPLE_EVERY_SECONDS = float(os.environ.get("CAMERA_SAMPLE_EVERY_SECONDS", "0.5"))
CAMERA_WARMUP_SECONDS = float(os.environ.get("CAMERA_WARMUP_SECONDS", "0.35"))
FACE_MAX_CANDIDATE_FRAMES = int(os.environ.get("FACE_MAX_CANDIDATE_FRAMES", "5"))

VISION_JPEG_QUALITY = int(os.environ.get("VISION_JPEG_QUALITY", "80"))
VISION_MODEL_TIMEOUT_SECONDS = int(os.environ.get("VISION_MODEL_TIMEOUT_SECONDS", "20"))
VISION_DEBUG = os.environ.get("VISION_DEBUG", "1") == "1"
VISION_DEBUG_DIR = os.environ.get("VISION_DEBUG_DIR", "vision_debug_frames")
VISION_MAX_IMAGE_SIDE = int(os.environ.get("VISION_MAX_IMAGE_SIDE", "224"))

# Image preprocessing for Qwen2.5-VL. These reduce ZED/camera noise before VLM inference.
VISION_DENOISE_ENABLED = os.environ.get("VISION_DENOISE_ENABLED", "1") == "1"
VISION_LIGHTING_ENHANCE_ENABLED = os.environ.get("VISION_LIGHTING_ENHANCE_ENABLED", "1") == "1"
VISION_SHARPEN_ENABLED = os.environ.get("VISION_SHARPEN_ENABLED", "1") == "1"
VISION_DENOISE_H = float(os.environ.get("VISION_DENOISE_H", "3"))
VISION_DENOISE_H_COLOR = float(os.environ.get("VISION_DENOISE_H_COLOR", "3"))
VISION_CLAHE_CLIP_LIMIT = float(os.environ.get("VISION_CLAHE_CLIP_LIMIT", "1.5"))
VISION_MIN_BRIGHTNESS = float(os.environ.get("VISION_MIN_BRIGHTNESS", "35.0"))
VISION_MAX_BRIGHTNESS = float(os.environ.get("VISION_MAX_BRIGHTNESS", "220.0"))
# Sharpness is useful for choosing the best candidate frame, but webcam crops can
# still have low Laplacian scores even when the face is usable for a VLM.
# Keep this threshold soft, and do not reject on sharpness unless explicitly enabled.
VISION_MIN_SHARPNESS = float(os.environ.get("VISION_MIN_SHARPNESS", "8.0"))
VISION_SKIP_LOW_QUALITY = os.environ.get("VISION_SKIP_LOW_QUALITY", "0") == "1"
FACE_ANALYSIS_STRATEGY = os.environ.get("FACE_ANALYSIS_STRATEGY", "multi_frame").strip().lower()
FACE_MULTI_FRAME_COUNT = int(os.environ.get("FACE_MULTI_FRAME_COUNT", "3"))

# Reliability filter. Qwen2.5-VL emotion scores are percentages.
# VLM scores are usually less calibrated than DeepFace scores, so these defaults are softer.
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
# Base weights before reliability adjustment.
FUSION_TEXT_WEIGHT = float(os.environ.get("FUSION_TEXT_WEIGHT", "0.5"))
FUSION_VISUAL_WEIGHT = float(os.environ.get("FUSION_VISUAL_WEIGHT", "0.4"))
FUSION_PROSODY_WEIGHT = float(os.environ.get("FUSION_PROSODY_WEIGHT", "0.1"))

# Semantic override keeps explicit emotional language from being overruled by FER.
FUSION_ENABLE_SEMANTIC_OVERRIDE = os.environ.get("FUSION_ENABLE_SEMANTIC_OVERRIDE", "1") == "1"
FUSION_EXPLICIT_TEXT_CONFIDENCE = float(os.environ.get("FUSION_EXPLICIT_TEXT_CONFIDENCE", "0.72"))



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

    @property
    def as_json(self) -> dict:
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "reason": self.reason,
            "scores": self.scores,
            "weights": self.weights,
            "text_emotion": self.text_emotion,
            "visual_emotion": self.visual_emotion,
            "prosody_emotion": self.prosody_emotion,
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
    embedder: Any = None
    error: Optional[str] = None


@dataclass
class FaceEmotionCapture:
    emotion_score_samples: list[dict[str, float]]
    frame_count: int
    sampled_frame_count: int
    started_at: str
    ended_at: str
    error: Optional[str] = None

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
            "note": (
                "Qwen2.5-VL facial expression analysis is used as the visual component "
                "in fixed weighted late fusion."
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
    }


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
    """
    Convert a spelling transcript like:
      "L E T I C I A"
    into:
      "Leticia"

    faster-whisper may output punctuation or words around the spelling;
    this function keeps only standalone letters.
    """
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


def extract_name_from_text(text: str) -> Optional[str]:
    """
    Extract names only from explicit name-introduction phrases.

    Important:
    We intentionally do NOT support "I am ..." or "I'm ..." here because
    normal phrases like "I'm happy" could be incorrectly treated as a name.
    """
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
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "yes",
        "no",
        "okay",
        "ok",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
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
        - Do not claim to remember anything beyond this saved local JSON context.
        - Do not reveal the raw JSON file unless the user asks.
        """.strip()


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
            },
            "face_emotion": {
                "backend": "qwen2.5vl:7b via Ollama",
                "camera_device": CAMERA_DEVICE,
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
        },
        "messages": session_log,
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(transcript_data, file, indent=2, ensure_ascii=False)

    return path


def summarize_session_with_llm(
    client: Client,
    session_log: list[dict],
    previous_summary: str = "",
) -> str:
    if not session_log:
        return previous_summary

    compact_messages = []

    for item in session_log[-20:]:
        role = item.get("role", "")
        content = item.get("content", "")
        compact_messages.append(f"{role}: {content}")

    prompt = f"""
Summarize this human-robot conversation for future continuity.

Previous saved summary:
{previous_summary or "None"}

Recent session:
{chr(10).join(compact_messages)}

Return a concise memory summary in 3-6 bullet points.

Include only:
- user's name if known
- main topics discussed
- useful preferences or ongoing tasks
- emotional context if relevant

Do not invent facts.
""".strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You summarize conversations accurately and concisely.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.2,
                "num_predict": 180,
                "num_ctx": 4096,
            },
            stream=False,
        )

        return response["message"]["content"].strip()

    except KeyboardInterrupt:
        print_ts("Summary generation interrupted. Transcript was still saved.")
        return previous_summary

    except Exception as exc:
        print_ts(f"Could not summarize session: {exc}")
        return previous_summary


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

    if ENABLE_LLM_SESSION_SUMMARY:
        previous_summary = users[user_key].get("conversation_summary", "")
        users[user_key]["conversation_summary"] = summarize_session_with_llm(
            client=client,
            session_log=session_log,
            previous_summary=previous_summary,
        )
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
    """
    Transcribe one utterance using faster-whisper.

    Notes:
    - Silero VAD has already cut the microphone stream into utterances, so
      FAST_WHISPER_CONFIG["vad_filter"] is False by default.
    - beam_size=1 is usually faster for real-time conversation.
    """
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


# Backwards-compatible alias so the rest of the script stays readable.
transcribe_audio = transcribe_with_faster_whisper


def load_audio_for_prosody(wav_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Load mono float32 audio for lightweight prosody analysis.

    This is used after the utterance WAV is saved and before it is deleted.
    """
    try:
        audio, sr = sf.read(wav_path, dtype="float32")

        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)

        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32, copy=False)

        if int(sr) != int(target_sr):
            audio = resample_audio(audio, int(sr), int(target_sr))

        return audio.astype(np.float32, copy=False)

    except Exception as exc:
        print_ts(f"Could not load audio for prosody: {exc}")
        return np.array([], dtype=np.float32)




# =========================
# Camera / Qwen2.5-VL helpers
# =========================

def normalize_camera_frame(frame: np.ndarray) -> np.ndarray:
    if USE_ZED_HALF_FRAME_CROP:
        height, width = frame.shape[:2]

        # ZED gives left + right images side by side.
        # Keep only the left camera image.
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

        # MJPG is important for high-resolution ZED over USB.
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


def list_camera_devices(max_indices: int = 6) -> None:
    print("\nAvailable camera devices:")
    found = False

    for idx in range(max_indices):
        cap = open_camera(idx)
        try:
            if not cap.isOpened():
                continue

            ok, frame = cap.read()
            if ok and frame is not None:
                frame = normalize_camera_frame(frame)

            found = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[cam {idx}] available | resolution={width}x{height}")
        finally:
            cap.release()

    if not found:
        print("No camera devices found.")

    print()


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
    """Return simple no-reference image quality metrics for debugging and reliability."""
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return {"brightness": brightness, "sharpness": sharpness}
    except Exception:
        return {"brightness": 0.0, "sharpness": 0.0}


def denoise_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Reduce color/camera grain while keeping facial features reasonably intact."""
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
    """Apply CLAHE on luminance to improve dim/uneven lighting without changing colors too much."""
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
    """Add a mild unsharp mask after denoising/lighting correction."""
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
    """Crop, denoise, improve lighting, mildly sharpen, and resize before sending to Qwen."""
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
    """
    Decide whether a cleaned frame should be sent to the VLM.

    Brightness failures are hard failures because very dark or overexposed
    images usually contain little usable facial evidence. Sharpness is treated
    as a warning by default because the best-of-N sampler already uses
    sharpness to select the cleanest frame. Set VISION_SKIP_LOW_QUALITY=1 if
    you want to restore hard rejection for blurry frames during experiments.
    """
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
    """
    Give the VLM a face-focused image instead of the full camera frame.
    This matters for ZED / wide camera frames where the user's face may be small.
    """
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

    # Fallback: center crop to reduce background before sending to VLM.
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
    """
    Convert one OpenCV BGR frame to base64 JPEG for Ollama vision models.
    """
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

        # Do not waste a slow VLM call on unusable frames.
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
    "happy": "joy",
    "happiness": "joy",
    "smile": "joy",
    "smiling": "joy",
    "calm": "trust",
    "friendly": "trust",
    "relaxed": "trust",
    "concern": "fear",
    "concerned": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "afraid": "fear",
    "surprised": "surprise",
    "sad": "sadness",
    "unhappy": "sadness",
    "angry": "anger",
    "mad": "anger",
    "annoyed": "anger",
    "disgusted": "disgust",
    "expectant": "anticipation",
    "curious": "anticipation",
    "thinking": "anticipation",
}


def canonical_emotion_name(name: str) -> Optional[str]:
    cleaned = re.sub(r"[^a-zA-Z]+", "", str(name).strip().lower())

    # Do not map neutral to trust. Neutral is not a Plutchik emotion, and
    # forcing it into trust artificially inflates the trust score.
    if cleaned == "neutral":
        return None

    if cleaned in PLUTCHIK_EMOTIONS and cleaned != "neutral":
        return cleaned

    return EMOTION_ALIASES.get(cleaned)


def normalize_plutchik_scores(raw_data: dict) -> Optional[dict[str, float]]:
    """
    Normalize model-produced scores to percentages over Plutchik's 8 emotions.
    Accepts common VLM variants:
    - {"scores": {"joy": 0.5, ...}}
    - {"dominant_emotion": "neutral", "confidence": 0.7}
    - flat emotion-score dictionaries
    """
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

    # If the model returned probabilities around 0..1, convert to percentages.
    if total <= 1.5:
        return {emotion: score * 100.0 for emotion, score in scores.items()}

    # If the model returned percentages already and roughly sum to 100, keep the distribution normalized.
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
    """
    Analyze one camera frame with qwen2.5vl:7b through Ollama.

    Returns:
        dict mapping Plutchik emotion -> percentage score, or None on failure.
    """
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
    """
    Analyze several candidate frames in one Qwen2.5-VL call.

    This is optional and controlled by FACE_ANALYSIS_STRATEGY=multi_frame.
    Default remains best_of_n because it keeps latency close to one-image
    inference while still avoiding the single-snapshot gamble.
    """
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
    Camera sampler for facial-expression hints using best-of-N frame selection.

    Design choice:
    - During speech, keep camera preview/audio loop fast.
    - Collect a small set of candidate frames after camera warmup.
    - Score candidates with a cheap sharpness metric.
    - Do NOT call Qwen inside the live preview/audio loop.
    - When speech ends, send only the sharpest candidate to Qwen2.5-VL.

    This improves frame quality without adding multi-image VLM latency. The
    FaceEmotionCapture data model still stores emotion_score_samples as a list,
    so the system can later be upgraded to true multi-frame VLM inference.
    """

    def __init__(self, camera_device: Optional[int], vision_client: Client) -> None:
        self.camera_device = camera_device
        self.vision_client = vision_client
        self.cap: Optional[cv2.VideoCapture] = None
        self.window_created = False
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

    def start(self) -> None:
        self.started_at = now_ts()
        self.capture_start_time = time.time()
        self.cap = open_camera(self.camera_device)

        if not self.cap.isOpened():
            self.error = "Could not open camera."
            self.cap.release()
            self.cap = None
            return

        if CAMERA_PREVIEW_ENABLED:
            try:
                cv2.namedWindow(CAMERA_PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
                self.window_created = True
            except Exception as exc:
                self.error = f"Camera preview could not start: {exc}"

    def _draw_preview(self, frame: np.ndarray) -> None:
        if not (CAMERA_PREVIEW_ENABLED and self.window_created):
            return

        try:
            preview_frame = frame.copy()
            cv2.putText(
                preview_frame,
                "Recording speech...",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview_frame,
                f"Emotion: {self.last_status}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            preview_frame = cv2.resize(
                preview_frame,
                (CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT),
            )
            cv2.imshow(CAMERA_PREVIEW_WINDOW_NAME, preview_frame)
            cv2.waitKey(1)
        except Exception as exc:
            self.error = f"Camera preview failed: {exc}"

    def _frame_sharpness(self, frame_bgr: np.ndarray) -> float:
        """Cheap no-reference sharpness score used only for best-frame selection."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def sample_if_due(self) -> None:
        if self.cap is None or self.capture_start_time is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        frame = normalize_camera_frame(frame)
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.capture_start_time

        self._draw_preview(frame)

        if elapsed < CAMERA_WARMUP_SECONDS:
            return

        # Stop collecting new candidates once the target count is reached.
        # Preview still updates above, but we avoid repeated frame copies and
        # sharpness calculations during long utterances.
        if len(self.candidate_frames) >= FACE_MAX_CANDIDATE_FRAMES:
            self.last_status = (
                f"collected frames ({FACE_MAX_CANDIDATE_FRAMES}/"
                f"{FACE_MAX_CANDIDATE_FRAMES})"
            )
            return

        # Collect candidate frames throughout speech, spaced by
        # CAMERA_SAMPLE_EVERY_SECONDS. Qwen analysis happens later in finish(),
        # not here, to avoid freezing preview/audio capture.
        if (
            self.candidate_frames
            and now - self.last_sample_time < CAMERA_SAMPLE_EVERY_SECONDS
        ):
            return

        sharpness = self._frame_sharpness(frame)
        self.candidate_frames.append((sharpness, frame.copy()))
        self.last_sample_time = now
        self.snapshot_taken_at = now_ts()

        # Keep only the sharpest N candidates to bound memory and processing.
        if len(self.candidate_frames) > FACE_MAX_CANDIDATE_FRAMES:
            self.candidate_frames.sort(key=lambda item: item[0], reverse=True)
            self.candidate_frames = self.candidate_frames[:FACE_MAX_CANDIDATE_FRAMES]

        self.sampled_frame_count = len(self.candidate_frames)
        self.last_status = (
            f"collecting frames ({self.sampled_frame_count}/"
            f"{FACE_MAX_CANDIDATE_FRAMES})"
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

        # Use the sharpest frame collected during the utterance.
        self.candidate_frames.sort(key=lambda item: item[0], reverse=True)
        best_sharpness, best_frame = self.candidate_frames[0]
        self.snapshot_frame = best_frame

        if FACE_ANALYSIS_STRATEGY == "multi_frame":
            selected = [frame for _, frame in self.candidate_frames[:FACE_MULTI_FRAME_COUNT]]
            print_ts(
                f"Analyzing {len(selected)} best facial frames with Qwen2.5-VL "
                f"(best_sharpness={best_sharpness:.1f}, "
                f"candidates={len(self.candidate_frames)})..."
            )
            scores = analyze_multiple_frames_emotion_scores(selected, self.vision_client)
        else:
            print_ts(
                f"Analyzing best facial frame with Qwen2.5-VL "
                f"(sharpness={best_sharpness:.1f}, "
                f"candidates={len(self.candidate_frames)})..."
            )
            scores = analyze_frame_emotion_scores(self.snapshot_frame, self.vision_client)

        if not scores:
            self.last_status = "no reliable VLM facial emotion"
            return

        # Reject weak neutral/trust outputs such as trust=5% or trust=10%.
        top_emotion, top_score = max(scores.items(), key=lambda item: item[1])
        if top_score < FACE_MIN_TOP_SCORE:
            print_ts(
                f"Qwen2.5-VL facial result rejected as weak: "
                f"{top_emotion}={top_score:.1f}% < {FACE_MIN_TOP_SCORE:.1f}%"
            )
            self.last_status = f"weak VLM result: {top_emotion}={top_score:.0f}%"
            return

        self.emotion_score_samples.append(scores)
        self.last_status = f"{top_emotion}: {top_score:.0f}%"
        print_ts(f"Qwen2.5-VL facial result accepted: {self.last_status}")

    def finish(self) -> FaceEmotionCapture:
        # Stop camera first so the device is released quickly; candidate frames
        # are already stored in memory.
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.window_created:
            try:
                cv2.destroyWindow(CAMERA_PREVIEW_WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                pass

        try:
            self._analyze_snapshot_now()
        except Exception as exc:
            self.error = f"Qwen2.5-VL best-frame analysis failed: {exc}"
            print_ts(self.error)

        return FaceEmotionCapture(
            emotion_score_samples=self.emotion_score_samples,
            frame_count=self.frame_count,
            sampled_frame_count=self.sampled_frame_count,
            started_at=self.started_at,
            ended_at=now_ts(),
            error=self.error,
        )


# =========================
# Silero VAD listener
# =========================

def listen_for_utterance_with_silero_vad(
    input_device: Optional[int],
    silero_model,
    prompt_label: str = "utterance",
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
    #print_ts(
    #    f"Silero VAD settings: threshold={SILERO_THRESHOLD}, "
    #    f"min_silence={SILERO_MIN_SILENCE_DURATION_MS}ms, "
    #    f"max_utterance={VAD_MAX_UTTERANCE_SECONDS}s"
    #)

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
    camera_device: Optional[int] = CAMERA_DEVICE,
    prompt_label: str = "utterance",
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

                            face_sampler = FaceEmotionSampler(camera_device=camera_device, vision_client=vision_client)
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
) -> Optional[str]:
    print()
    print_ts("Please spell your name letter by letter. For example: L E T I C I A.")
    print()

    wav_path = listen_for_utterance_with_silero_vad(
        input_device=input_device,
        silero_model=silero_model,
        prompt_label="spelled name",
    )

    if not wav_path:
        return None

    try:
        transcript = transcribe_audio(wav_path, whisper_model)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    print_ts(f"Raw spelling transcript: {transcript}")

    return clean_spelled_name(transcript)


def generate_introduction_response(
    client: Client,
    user_name: str,
) -> str:
    """
    Let the LLM introduce Ameca dynamically after the user gives their name.
    """
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
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"My name is {user_name}",
                },
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
) -> tuple[str, dict, str]:
    """
    Academic protocol:
    1. Ask the user to say their name naturally.
    2. Ask the user to spell the name letter by letter.
    3. Prefer the spelled form when it is usable.
    4. Fall back to the spoken form if spelling fails.

    This preserves the system requirement while avoiding crashes when the
    spelling ASR is noisy.
    """
    users = load_users()

    spoken_name = ""

    for attempt in range(2):
        print()
        print_ts("Please say your name")
        print()

        wav_path = listen_for_utterance_with_silero_vad(
            input_device=input_device,
            silero_model=silero_model,
            prompt_label="name",
        )

        if not wav_path:
            spoken_name = ""
        else:
            try:
                spoken_name = transcribe_audio(
                    wav_path,
                    whisper_model,
                ).strip()
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

    if not spoken_name or looks_like_invalid_name(spoken_name):
        spoken_name = "Guest"

    spelled_name = None

    if ENABLE_NAME_SPELLING:
        spelled_name = ask_user_to_spell_name(
            whisper_model=whisper_model,
            silero_model=silero_model,
            input_device=input_device,
        )

    if spelled_name and not looks_like_invalid_name(spelled_name):
        # Prefer spelling because it is part of the academic/system protocol.
        spoken_name = spelled_name
        print_ts(f"Using spelled name: {spoken_name}")
    else:
        print_ts(f"Using spoken name: {spoken_name}")

    print_ts(f"Detected name: {spoken_name}")

    user_key = slugify_name(spoken_name)
    is_new_user = user_key not in users

    if is_new_user:
        users[user_key] = {
            "name": spoken_name,
            "created_at": now_ts(),
            "last_seen": now_ts(),
            "session_files": [],
            "conversation_summary": "",
        }
    else:
        users[user_key]["last_seen"] = now_ts()

    save_users(users)

    if is_new_user:
        print_ts(f"Nice to meet you, {spoken_name}.")
    else:
        print_ts(f"Welcome back, {users[user_key]['name']}.")

    introduction_reply = generate_introduction_response(
        client=client,
        user_name=users[user_key]["name"],
    )

    print_ts(f"Assistant: {introduction_reply}")
    print()

    return user_key, users[user_key], introduction_reply


# =========================
# JSON helpers
# =========================

def safe_json_extract(raw: str):
    """
    Robust JSON extraction for Ollama/Qwen outputs.

    Why this is needed:
    - Vision models sometimes return valid JSON.
    - Sometimes they return JSON inside markdown fences.
    - Sometimes generation stops inside the final reason string.

    This function first tries strict parsing, then repairs the most common
    truncated JSON shape used by the facial-emotion classifier.
    """
    if not raw:
        return None

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract from the first opening brace to the last visible closing brace,
    # or to the end if the model was truncated before closing the object.
    start = raw.find("{")
    if start < 0:
        return None

    candidate = raw[start:].strip()

    # First try a balanced substring ending at the last closing brace.
    last_close = candidate.rfind("}")
    if last_close > 0:
        maybe = candidate[: last_close + 1]
        try:
            return json.loads(maybe)
        except Exception:
            pass

    # Repair common case: output was cut inside the final reason string.
    # Remove an incomplete reason field if necessary; the scores are what we need.
    repaired = candidate
    reason_pos = repaired.rfind('"reason"')
    if reason_pos != -1:
        before_reason = repaired[:reason_pos].rstrip().rstrip(",")
        repaired = before_reason + ', "reason": "truncated"}'

    # Balance braces and brackets conservatively.
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    if close_braces < open_braces:
        repaired += "}" * (open_braces - close_braces)

    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if close_brackets < open_brackets:
        repaired += "]" * (open_brackets - close_brackets)

    # If a quote is still unbalanced, close it at the end.
    if repaired.count('"') % 2 != 0:
        repaired += '"'

    try:
        return json.loads(repaired)
    except Exception:
        pass

    # Final targeted extraction for facial emotion JSON. This salvages scores
    # even if the reason field is badly truncated.
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

        # Prefer not to cut in the middle of a sentence when possible.
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
    """Find scrape.py so the main Ameca app can rebuild the RRLab KB on demand."""
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
    """
    Rebuild the RRLab website ChromaDB index by executing scrape.py.

    scrape.py is responsible for crawling https://rrlab.cs.rptu.de/en and writing
    to the same ChromaDB path/collection configured above:
      SELF_RAG_DB_DIR='chroma_db'
      SELF_RAG_COLLECTION='emah_knowledge'
    """
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


def init_self_rag_store() -> SelfRAGStore:
    if not SELF_RAG_ENABLED:
        print_ts("Self-RAG disabled by SELF_RAG_ENABLED=0.")
        return SelfRAGStore(enabled=False, error="Self-RAG disabled.")

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        print_ts(
            "Self-RAG dependencies missing. Install with: "
            "pip install chromadb sentence-transformers pypdf"
        )
        return SelfRAGStore(enabled=False, error=str(exc))

    try:
        os.makedirs(SELF_RAG_DB_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=SELF_RAG_DB_DIR)
        collection = chroma_client.get_or_create_collection(
            name=SELF_RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        embedder = SentenceTransformer(SELF_RAG_EMBED_MODEL)
        store = SelfRAGStore(enabled=True, collection=collection, embedder=embedder)

        if SELF_RAG_REINDEX_ON_START:
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
    if not store.enabled or store.collection is None or store.embedder is None:
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

    embeddings = store.embedder.encode(docs, normalize_embeddings=True).tolist()
    store.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    print_ts(f"Self-RAG indexed/updated {len(docs)} chunks from {len(paths)} files.")




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
    if not store.enabled or store.collection is None or store.embedder is None:
        return []
    if not query.strip():
        return []

    normalized_query = normalize_self_rag_query_text(query)
    rewritten_query = rewrite_self_rag_query(normalized_query)
    inferred_category = infer_self_rag_category(rewritten_query)
    person_lookup_name = extract_person_lookup_name(normalized_query)
    force_rag = force_self_rag_for_entity(normalized_query)

    try:
        query_embedding = store.embedder.encode([rewritten_query], normalize_embeddings=True).tolist()[0]

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
        text = clean_knowledge_text(item["text"])
        if not text:
            continue
        clipped = text[:remaining]
        if not clipped:
            break
        context_parts.append(f"[Source {idx}: {item['source']}]\n{clipped}")
        sources.append({k: v for k, v in item.items() if k != "text"})
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
    """
    Lightweight acoustic/prosody heuristic.

    This is intentionally weak. It estimates broad arousal patterns and should
    never dominate text or reliable facial emotion.
    """
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
    """
    Detect explicit emotional language. This is used as a semantic override so
    clear statements like "I am angry" are not overruled by a sad-looking face.
    """
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

    return max(0.0, min(1.0, (0.45 * top_score_component) + (0.35 * margin_component) + (0.20 * frame_component))) * declared_reliable


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
) -> FusedEmotionResult:
    """
    Adaptive reliability-aware late fusion.

    Base weights are text=0.5, visual=0.4, prosody=0.1, but each weight is
    multiplied by a modality reliability score and then normalized.

    Explicit emotional language receives semantic priority because user speech
    is usually more reliable than facial expression for direct emotion claims.
    """
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

    # If the user explicitly states anger/frustration/fear/etc., do not let a
    # generic sad-looking facial output dominate.
    if explicit_text_emotion:
        text_rel = max(text_rel, 0.95)
        if visual_emotion and visual_emotion != explicit_text_emotion:
            visual_rel *= 0.45
        prosody_rel = min(prosody_rel, 0.25)

    # Common conflict: user says anger/frustration, face appears sad/tired.
    # Keep the mixed signal but make language primary.
    if text_emotion.emotion == "anger" and visual_emotion == "sadness":
        visual_rel *= 0.40
        text_rel = max(text_rel, 0.92)

    # For factual questions, visual sadness should not hijack the response tone.
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
    )


# Backwards-compatible alias for older main-loop code paths.
adaptive_reliability_aware_fusion = adaptive_reliability_aware_fusion


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

    response = client.chat(
        model=MODEL_NAME,
        format="json",
        messages=[
            {
                "role": "system",
                "content": "You return valid JSON only.",
            },
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


def normalize_reply(raw_reply: str, emotion: str) -> str:
    required_emoji = PLUTCHIK_EMOTIONS[emotion]

    cleaned = remove_all_emojis_except_allowed_faces(raw_reply)
    cleaned = remove_allowed_face_emojis(cleaned)

    cleaned = re.sub(r"[:;=8][\-^]?[)(DPp/\\|]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)

    if not cleaned:
        cleaned = "I’m here with you."

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


def deterministic_reply_if_applicable(user_text: str, emotion: str) -> Optional[str]:
    text = user_text.strip().lower()
    emoji = PLUTCHIK_EMOTIONS.get(emotion, "🙂")

    if "today's date" in text or "todays date" in text or "what is the date" in text:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. {emoji}"

    if "what is the time" in text or "what time is it" in text or "current time" in text:
        return f"The current time is {datetime.now().strftime('%H:%M')}. {emoji}"

    farewell_words = {"bye", "goodbye", "see you", "see you later", "talk later", "have a nice day", "have a good day"}
    if any(phrase in text for phrase in farewell_words):
        return "Thank you, and take care. 🙂"

    return None


# =========================
# Response generation
# =========================

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

    text_emotion_json = {
        "emotion": emotion_result.emotion,
        "confidence": emotion_result.confidence,
        "reason": emotion_result.reason,
    }

    return f"""
        {AMECA_SYSTEM_PROMPT}

        {runtime_context()}

        {memory_context}

        {build_self_rag_prompt_block(self_rag_context)}

        You are generating Ameca's next conversational response.

        PRIVATE EMOTION CONTEXT
        Use this context only for tone control only. Do not mention it directly.

        Fused emotion JSON:
        {json.dumps(text_emotion_json, indent=2)}

        Face/Qwen2.5-VL emotion JSON:
        {json.dumps(face_emotion_json, indent=2)}

        Interpretation rules:
        - The active emotion was produced by adaptive reliability-aware late fusion.
        - The base fusion weights are text=0.5, visual=0.4, prosody=0.1 and are adjusted by modality reliability.
        - Use the fused emotion only to adjust tone.
        - Do not mention camera, Qwen2.5-VL, prosody, fusion, detected emotion, or private emotion context to the user.
        - Do not say things like "you look sad", "your face shows", or "I detected".

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

        Conversation behavior rules:
        - Always ensure your response is context appropriate.
        - Always respond like a helpful assistant
        - Use the private emotion context only to adjust tone.
        - Use the recent conversation history to understand context and avoid repeating yourself.
        - Do not greet repeatedly. After the first greeting, respond directly to what the user said.
        - Do not say phrases like "your detected emotion", "you seem sad", "trust is present", or "you are anticipating".
        - Do not reintroduce yourself unless the user asks who you are.
        - Never begin with "As Ameca" or "As a humanoid social robot".
        - Use the user's name occasionally, not in every response.
        - Do not immediately give a list of advice unless the user asks for advice.
        - Speak directly and naturally.
        - Do not introduce unrelated topics such as Lego, Legoland, legal advice, or robotics toys unless the user explicitly mentions them.
        - Do not mention emotion labels unless the user explicitly asks.
        - Do not use markdown, bullets, numbered lists, or long advice unless the user asks for detail.
        - Do not repeat advice already given in the recent conversation.
        - Keep the response to 1-2 short sentences.
        - Do not list any steps unless the user asks for detailed guidance.
        - If the user asks who can help, suggest concrete people: supervisor, co-supervisor, lab colleagues, thesis coordinator, or university writing center.

        Emoji rules:
        - Always end sentences with exactly one context appriopriate facial emoji.
        - Use gentle emojis for negative emotions.
        - Do not use emoji symbols
        - Do not overreact emotionally.
        - Use only one of these emojis: 🙂 😊 😌 😔 😟 🤔 😮 😢 😠 🤢
        - Emoji must appear only at the end of the sentences.

        Tone examples:
        - User: "I'm so tired and exhausted." -> reply should validate softly; emoji could be 😔 or 😌.
        - User: "I am very happy today." -> reply should celebrate the positive mood; emoji could be 😊.
        - User: "Well, I don't know." -> reply should stay with the uncertainty and ask gently; emoji could be 😟 or 🙂.
        """.strip()


def limit_text_length(text: str, max_chars: int = 1500) -> str:
    return text[:max_chars]


def limit_system_prompt(prompt: str, max_chars: int = 9000) -> str:
    return prompt[:max_chars]


def trim_history(history: list[dict]) -> list[dict]:
    return history[-MAX_HISTORY_MESSAGES:]


def prompt_ready_history(history: list[dict]) -> list[dict]:
    return [{"role": item["role"], "content": item["content"]} for item in history]


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
        {
            "role": "system",
            "content": system_prompt,
        },
        *prompt_ready_history(trim_history(history[-6:])),
        {
            "role": "user",
            "content": safe_user_text,
        },
    ]

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.4,
            "num_predict": 120,
            "repeat_penalty": 1.25,
            "num_ctx": 2048,
        },
        stream=False,
    )

    raw_reply = response["message"]["content"]
    data = safe_json_extract(raw_reply)

    if data and isinstance(data, dict):
        reply_text = str(data.get("reply", "")).strip()
        emoji = str(data.get("emoji", "")).strip()

        if emoji in {":)", ":-)", ""}:
            emoji = PLUTCHIK_EMOTIONS.get(emotion_result.emotion, "🙂")

        return normalize_reply(f"{reply_text} {emoji}", emotion_result.emotion)

    return normalize_reply(raw_reply, emotion_result.emotion)


# =========================
# Main loop
# =========================

def main() -> None:
    print_ts("Starting Silero VAD + faster-whisper Plutchik chat prototype with local memory.")
    print_ts(f"Python: {sys.version.split()[0]}")
    print_ts(f"Ollama host: {OLLAMA_HOST}")
    print_ts(f"Ollama chat model: {MODEL_NAME}")
    print_ts(f"Ollama vision model: {VISION_MODEL_NAME}")
    print()

    ensure_data_dirs()

    check_ollama_available()
    ensure_model_available(MODEL_NAME)
    ensure_model_available(VISION_MODEL_NAME)

    client = Client(host=OLLAMA_HOST)

    self_rag_store = init_self_rag_store()

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

    user_key, user_profile, intro_reply = prompt_for_user_name(
        client=client,
        whisper_model=whisper_model,
        silero_model=silero_model,
        input_device=INPUT_DEVICE,
    )

    print()

    if user_profile.get("conversation_summary"):
        print_ts("Loaded previous conversation memory.")
        print()

    print("Automatic listening mode is active.")
    print("Speak naturally. Silero VAD will detect speech, Qwen2.5-VL will analyze one cleaned facial snapshot after speech ends, faster-whisper will transcribe it, adaptive reliability-aware fusion will combine text/visual/prosody, and Ameca will respond.")
    print("Say '/exit' or press Ctrl+C to save the transcript and quit.")
    print()

    history: list[dict] = []
    session_log: list[dict] = []

    # Store the generated introduction in the transcript so the session is complete.
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
                camera_device=CAMERA_DEVICE,
                prompt_label="utterance",
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
                user_text = transcribe_audio(wav_path, whisper_model)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            if not user_text:
                print_ts("No speech detected after transcription.")
                continue

            print_ts(f"faster-whisper transcript: {user_text}")

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
                print()

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": reply})
                history = trim_history(history)

                session_log.append({
                    "role": "user",
                    "content": user_text,
                    "timestamp": now_ts(),
                    "input_mode": "silero_vad_faster_whisper",
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
                # Prefer the RRLab website scraper when scrape.py is available.
                # Fall back to the old local-folder indexer for custom documents.
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

            facial_hint = face_capture.summary_text if face_capture else None
            face_emotion_json = face_capture.as_json if face_capture else default_face_emotion_json()

            text_emotion_result = detect_emotion(
                client=client,
                transcribed_text=user_text,
                facial_emotion_hint=facial_hint,
            )

            prosody_result = analyze_prosody_from_audio(audio_for_prosody, TARGET_SAMPLE_RATE)

            fused_emotion_result = adaptive_reliability_aware_fusion(
                text_emotion=text_emotion_result,
                face_emotion_json=face_emotion_json,
                prosody_emotion=prosody_result,
                user_text=user_text,
            )

            emotion_result = fused_emotion_result.to_emotion_result()

            text_emotion_json = {
                "emotion": text_emotion_result.emotion,
                "confidence": text_emotion_result.confidence,
                "reason": text_emotion_result.reason,
            }

            prosody_json = prosody_result.as_json
            emotion_json = fused_emotion_result.as_json

            print_ts("Text emotion JSON:")
            print(json.dumps(text_emotion_json, indent=2))
            print()

            print_ts("Face emotion JSON:")
            print(json.dumps(face_emotion_json, indent=2))
            print()

            print_ts("Prosody emotion JSON:")
            print(json.dumps(prosody_json, indent=2))
            print()

            print_ts("Adaptive reliability-aware fusion JSON:")
            print(json.dumps(emotion_json, indent=2))
            print()

            self_rag_context = build_self_rag_context(
                client=client,
                store=self_rag_store,
                user_text=user_text,
            )
            print_ts("Self-RAG JSON:")
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
                "input_mode": "silero_vad_faster_whisper_qwen25vl_adaptive_reliability_aware_fusion_self_rag",
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
        print()
        print_ts("Goodbye.")

    finally:
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

