from __future__ import annotations

from typing import Optional

OLLAMA_HOST = "https://continue-change-leu-hoping.trycloudflare.com"
MODEL_NAME = "llama3:8b"

# Speech-to-text backend. Supported values: "faster-whisper" or "whisperx".
TRANSCRIPTION_BACKEND = "faster-whisper"

# faster-whisper settings
FASTER_WHISPER_MODEL = "small"
FASTER_WHISPER_DEVICE = "cpu"
FASTER_WHISPER_COMPUTE_TYPE = "int8"
FASTER_WHISPER_LANGUAGE = "en"

# WhisperX settings
WHISPERX_MODEL = "small"
WHISPERX_DEVICE = "cpu"
WHISPERX_COMPUTE_TYPE = "int8"
WHISPERX_LANGUAGE = "en"
WHISPERX_BATCH_SIZE = 8

TARGET_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10
INPUT_DEVICE: Optional[int] = None
MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003

CAMERA_DEVICE: Optional[int] = None
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_SAMPLE_EVERY_SECONDS = 1.0
CAMERA_WARMUP_SECONDS = 1.5

DEEPFACE_DETECTOR_BACKEND = "retinaface"
DEEPFACE_ACTIONS = ["emotion"]
DEEPFACE_ALIGN = True
FACE_MIN_TOP_SCORE = 45.0
FACE_MIN_MARGIN = 15.0

# Camera emotion-recognition backend. Supported values: "deepface" or "vlm".
CAMERA_EMOTION_BACKEND = "deepface"

# VLM emotion recognition settings. Requires: ollama pull llama3.2-vision:11b
VLM_EMOTION_MODEL = "llama3.2-vision:11b"
VLM_EMOTION_TEMPERATURE = 0.0
VLM_EMOTION_IMAGE_JPEG_QUALITY = 85

MAX_HISTORY_MESSAGES = 12
DEBUG = False
