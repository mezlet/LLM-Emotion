from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from deepface import DeepFace
from faster_whisper import WhisperModel
from ollama import Client