#!/usr/bin/env python3

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


def _normalize_region(region: object) -> dict:
    """
    DeepFace.analyze() includes a "region" key (typically
    {"x": int, "y": int, "w": int, "h": int}, sometimes with extra keys
    like eye coordinates in newer versions) whenever it successfully
    detects a face. This was previously never read or forwarded by this
    worker at all -- analyze_image() only extracted "emotion" and
    "dominant_emotion" -- so every response silently omitted the bounding
    box, even on a confident detection. That forced the client side to
    fall back to its own local face detectors (MediaPipe/Haar) purely to
    get a crop region for an already-successful DeepFace detection.

    Returns a plain {"x", "y", "w", "h"} dict with int values, or {} if
    region is missing/malformed (e.g. no face was found).
    """
    if not isinstance(region, dict):
        return {}
    try:
        return {
            "x": int(region.get("x", 0)),
            "y": int(region.get("y", 0)),
            "w": int(region.get("w", 0)),
            "h": int(region.get("h", 0)),
        }
    except (TypeError, ValueError):
        return {}


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
                return {"request_id": request_id, "ok": True, "no_face": True, "scores": {}, "region": {}}
            result = result[0]

        emotion_scores = result.get("emotion", {}) or {}
        dominant = result.get("dominant_emotion")
        region = _normalize_region(result.get("region"))

        return {
            "request_id": request_id,
            "ok": True,
            "no_face": False,
            "dominant_emotion": dominant,
            "scores": {str(k): float(v) for k, v in emotion_scores.items()},
            "region": region,
        }

    except ValueError as exc:
        # DeepFace raises ValueError (message mentions face detection) when
        # enforce_detection=True and no face is found. This is the useful
        # signal: a purpose-built face detector saying "no face here", as
        # opposed to a VLM that will describe *something* in the image
        # regardless of whether a face is actually present.
        message = str(exc)
        if "face" in message.lower() and "detect" in message.lower():
            return {"request_id": request_id, "ok": True, "no_face": True, "scores": {}, "region": {}}
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