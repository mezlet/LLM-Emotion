"""
config.py
---------
All tunable constants for the HRI chat application.
Change values here; do not scatter magic numbers elsewhere.
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------

OLLAMA_HOST: str = "http://127.0.0.1:11434"

# Choose the model served by your local Ollama instance.
# MODEL_NAME = "llama3.2:3b"
MODEL_NAME: str = "llama3:8b"
# MODEL_NAME = "mistral:7b"

# ---------------------------------------------------------------------------
# WhisperX speech-to-text
# ---------------------------------------------------------------------------

WHISPERX_MODEL: str = "small"
WHISPERX_DEVICE: str = "cpu"
WHISPERX_COMPUTE_TYPE: str = "int8"
WHISPERX_LANGUAGE: str = "en"

# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE: int = 16_000
MAX_RECORD_SECONDS: int = 10

# Set to a specific integer to pin a microphone; None uses the system default.
INPUT_DEVICE: Optional[int] = None

# Frames below these thresholds are treated as silence.
MIN_PEAK_THRESHOLD: float = 0.01
MIN_RMS_THRESHOLD: float = 0.003

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# Set to a specific integer to pin a camera; None uses device index 0.
CAMERA_DEVICE: Optional[int] = None
CAMERA_FRAME_WIDTH: int = 640
CAMERA_FRAME_HEIGHT: int = 480

# How often (in seconds) to sample a frame for emotion analysis.
CAMERA_SAMPLE_EVERY_SECONDS: float = 1.0

# Ignore the first short period after the camera opens (auto-exposure warm-up).
CAMERA_WARMUP_SECONDS: float = 1.5

# ---------------------------------------------------------------------------
# DeepFace
# ---------------------------------------------------------------------------

# "retinaface" is more accurate; use "mediapipe" if it is too slow.
DEEPFACE_DETECTOR_BACKEND: str = "mediapipe"
DEEPFACE_ACTIONS: list[str] = ["emotion"]
DEEPFACE_ALIGN: bool = True

# ---------------------------------------------------------------------------
# Facial-hint reliability gates
# ---------------------------------------------------------------------------

# Top averaged emotion score must exceed this percentage.
FACE_MIN_TOP_SCORE: float = 45.0

# Gap between the top two scores must exceed this to count as reliable.
FACE_MIN_MARGIN: float = 15.0

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

# Maximum number of past messages sent to the LLM to prevent prompt bloat.
MAX_HISTORY_MESSAGES: int = 12

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------

DEBUG: bool = False