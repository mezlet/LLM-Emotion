from typing import Optional

# Ollama settings
OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.2:1b"

# WhisperX settings
WHISPERX_MODEL = "small"
WHISPERX_DEVICE = "cpu"
WHISPERX_COMPUTE_TYPE = "int8"
WHISPERX_LANGUAGE = "en"  # Set to None for auto-detect

# Audio recording settings
TARGET_SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 10
INPUT_DEVICE: Optional[int] = None
MIN_PEAK_THRESHOLD = 0.01
MIN_RMS_THRESHOLD = 0.003
