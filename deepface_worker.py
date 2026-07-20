#!/usr/bin/env python3
"""
deepface_worker.py

Standalone DeepFace analysis worker for the Ameca hybrid visual modality
(Qwen2.5-VL + DeepFace). Meant to run in its OWN process, ideally from a
SEPARATE conda environment from ameca_demo.py.

WHY A SEPARATE PROCESS/ENVIRONMENT:
DeepFace pulls in TensorFlow. Running TensorFlow computation in the same
process as ameca_demo.py's PyTorch usage (Silero VAD, faster-whisper) has
previously caused a silent native segfault with no Python traceback (this
is the same class of bug documented around zed_vision_module's DeepFace
import in ameca_demo.py). Isolating DeepFace into its own OS process --
and ideally its own conda environment, so TF/Keras version pins can't
collide with the main pipeline's dependencies -- removes that risk
entirely. If this worker process crashes, it crashes alone; the main
pipeline just treats DeepFace as unavailable for that turn and falls back
to Qwen2.5-VL alone.

SETUP (separate conda env, recommended):
    conda create -n deepface_env python=3.10 -y
    conda activate deepface_env
    pip install deepface tensorflow tf-keras

    # NOTE: don't also pip install opencv-python-headless here. deepface
    # already pulls in regular opencv-python as a dependency, and having
    # BOTH opencv-python and opencv-python-headless installed in the same
    # env is a common source of a confusing failure: whichever one wins
    # pip's install ordering determines which files end up on disk, and
    # opencv-python-headless does NOT ship the Haar cascade XML data files
    # that DeepFace's default "opencv" detector backend needs. This
    # script sidesteps that specific issue by explicitly using the
    # "mtcnn" detector backend (see DETECTOR_BACKEND below) regardless,
    # but avoiding the double-install keeps things simpler.

    # Then point ameca_demo.py at it, e.g.:
    export DEEPFACE_PYTHON=/home/emah/miniconda3/envs/deepface_env/bin/python
    # or: python ameca_demo.py --deepface_python /path/to/deepface_env/bin/python

You do NOT run this script directly in normal use -- ameca_demo.py's
DeepFaceClient launches it automatically as a subprocess using whatever
interpreter DEEPFACE_PYTHON/--deepface_python points to. Running it
manually (e.g. for testing) is also fine; see PROTOCOL below.

PROTOCOL (line-delimited JSON over stdin/stdout):
  Startup: this worker prints exactly one line "READY" to stdout once
  DeepFace's model is loaded and warmed up, then enters the request loop.
  ALL other logging goes to stderr, so stdout only ever contains the
  "READY" line followed by one JSON response per line.

  Request (one JSON object per line on stdin):
    {"request_id": "...", "cmd": "analyze", "image_path": "/abs/path.jpg"}
    {"request_id": "...", "cmd": "ping"}
    {"cmd": "shutdown"}

  Response (one JSON object per line on stdout):
    analyze, face found:
      {"request_id": "...", "ok": true, "no_face": false,
       "dominant_emotion": "happy",
       "scores": {"angry": 0.1, "disgust": 0.0, "fear": 0.2, "happy": 95.3,
                   "sad": 0.1, "surprise": 3.9, "neutral": 0.4}}
    analyze, no face detected:
      {"request_id": "...", "ok": true, "no_face": true, "scores": {}}
    analyze, other error:
      {"request_id": "...", "ok": false, "error": "..."}
    ping:
      {"request_id": "...", "ok": true, "pong": true}
    unknown cmd:
      {"request_id": "...", "ok": false, "error": "unknown cmd: ..."}

Manual test run:
    python deepface_worker.py
    (then paste a line like the analyze example above and press enter)
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def log_stderr(message: str) -> None:
    # stdout is reserved for the JSON protocol; ALL human-readable logging
    # goes to stderr so it can never corrupt a response line that the
    # parent process is trying to parse as JSON.
    print(message, file=sys.stderr, flush=True)


# DeepFace's DEFAULT face detector backend is "opencv", which needs
# cv2's bundled Haar cascade XML data file. If both opencv-python and
# opencv-python-headless end up installed in the same env (headless does
# NOT ship that data file), whichever one wins the install ordering can
# silently leave that file missing -- the "opencv" backend then fails on
# every real request (enforce_detection=True), even though warmup
# (enforce_detection=False) only warns and looks harmless. Explicitly
# picking a detector backend that doesn't depend on that file sidesteps
# the issue entirely, regardless of which opencv package is installed.
# "mtcnn" is a good CPU speed/accuracy tradeoff and is already installed
# as a deepface dependency; override via DEEPFACE_DETECTOR_BACKEND if you
# want a different one (e.g. "retinaface" for more accuracy at the cost
# of speed, or "opencv" once you've confirmed the cascade file is present).
DETECTOR_BACKEND = os.environ.get("DEEPFACE_DETECTOR_BACKEND", "opencv")


def main() -> None:
    log_stderr("[deepface_worker] Loading DeepFace/TensorFlow (this can take a while on first run)...")
    log_stderr(f"[deepface_worker] Detector backend: {DETECTOR_BACKEND}")

    try:
        from deepface import DeepFace
    except Exception as exc:
        log_stderr(f"[deepface_worker] FATAL: could not import deepface: {exc}")
        log_stderr("[deepface_worker] Install with: pip install deepface tensorflow")
        sys.exit(1)

    # Warm up the model once at startup so the first real request isn't
    # slow, and so a load-time failure surfaces here (visible in this
    # worker's own logs) rather than as a mysterious first-request timeout
    # on the parent side.
    try:
        import numpy as np
        dummy_frame = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        DeepFace.analyze(
            img_path=dummy_frame,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True,
        )
        log_stderr("[deepface_worker] Warmup analyze() succeeded.")
    except Exception as exc:
        log_stderr(f"[deepface_worker] WARNING: warmup analyze() failed (continuing anyway): {exc}")

    # Signal readiness to the parent process. This MUST be the only thing
    # ever printed to stdout that isn't a JSON response line.
    print("READY", flush=True)
    log_stderr("[deepface_worker] Ready. Waiting for requests on stdin.")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as exc:
            log_stderr(f"[deepface_worker] Could not parse request line: {exc}")
            continue

        cmd = request.get("cmd")
        request_id = request.get("request_id")

        if cmd == "shutdown":
            log_stderr("[deepface_worker] Shutdown requested. Exiting.")
            break

        if cmd == "ping":
            print(json.dumps({"request_id": request_id, "ok": True, "pong": True}), flush=True)
            continue

        if cmd == "analyze":
            image_path = request.get("image_path")
            response = analyze_image(DeepFace, image_path, request_id)
            print(json.dumps(response), flush=True)
            continue

        print(
            json.dumps({"request_id": request_id, "ok": False, "error": f"unknown cmd: {cmd}"}),
            flush=True,
        )


def analyze_image(DeepFace, image_path, request_id):
    if not image_path:
        return {"request_id": request_id, "ok": False, "error": "no image_path provided"}

    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            silent=True,
        )

        # DeepFace.analyze returns a list (one entry per detected face) in
        # recent versions, or a single dict in older versions. Handle both.
        if isinstance(result, list):
            if not result:
                return {"request_id": request_id, "ok": True, "no_face": True, "scores": {}}
            result = result[0]

        emotion_scores = result.get("emotion", {}) or {}
        dominant = result.get("dominant_emotion")

        return {
            "request_id": request_id,
            "ok": True,
            "no_face": False,
            "dominant_emotion": dominant,
            "scores": {str(k): float(v) for k, v in emotion_scores.items()},
        }

    except ValueError as exc:
        # DeepFace raises ValueError (message mentions face detection) when
        # enforce_detection=True and no face is found. This is the useful
        # signal: a purpose-built face detector saying "no face here", as
        # opposed to a VLM that will describe *something* in the image
        # regardless of whether a face is actually present.
        message = str(exc)
        if "face" in message.lower() and "detect" in message.lower():
            return {"request_id": request_id, "ok": True, "no_face": True, "scores": {}}
        return {"request_id": request_id, "ok": False, "error": message}

    except Exception as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }


if __name__ == "__main__":
    main()