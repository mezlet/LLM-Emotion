#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
]

DEFAULT_CROPS_PER_EMOTION = 4
DEFAULT_OUTPUT_SIZE = 256
DEFAULT_NEUTRAL_MIN_CONSISTENCY = 0.80
DEFAULT_REFERENCE_MIN_CONSISTENCY = 0.55


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def round_vector(vector: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(vector).tolist()]


def cosine_similarity_nonnegative(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if (
        a.size == 0
        or b.size == 0
        or a.size != b.size
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 1e-8:
        return 0.0
    similarity = float(np.dot(a, b) / denom)
    if not np.isfinite(similarity):
        return 0.0
    return clamp01(similarity)


def rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if (
        a.size == 0
        or a.size != b.size
        or not np.all(np.isfinite(a))
        or not np.all(np.isfinite(b))
    ):
        return float("inf")
    return float(np.linalg.norm(a - b) / math.sqrt(max(1, a.size)))


def mean_pairwise_rms_distance(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return float("inf")
    distances: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(rms_distance(vectors[i], vectors[j]))
    finite = [value for value in distances if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("inf")


def mean_pairwise_cosine_similarity(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    similarities: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarities.append(cosine_similarity_nonnegative(vectors[i], vectors[j]))
    return clamp01(float(np.mean(similarities))) if similarities else 0.0


class PyFeatWorkerClient:
    """Small synchronous client for the existing crash-isolated pyfeat_worker.py."""

    def __init__(
        self,
        *,
        python_executable: str,
        worker_script: str,
        device: str,
        startup_timeout: float,
        request_timeout: float,
    ) -> None:
        self.python_executable = str(Path(python_executable).expanduser())
        self.worker_script = str(Path(worker_script).expanduser())
        self.device = device
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.proc: Optional[subprocess.Popen[str]] = None
        self.request_counter = 0
        self.au_columns: list[str] = []
        self._start()

    def _start(self) -> None:
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(f"Py-Feat Python executable not found: {self.python_executable}")
        if not Path(self.worker_script).is_file():
            raise FileNotFoundError(f"Py-Feat worker not found: {self.worker_script}")

        self.proc = subprocess.Popen(
            [self.python_executable, self.worker_script, "--device", self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=not sys.platform.startswith("win"),
        )

        def drain_stderr() -> None:
            assert self.proc is not None
            if self.proc.stderr is None:
                return
            for line in self.proc.stderr:
                line = line.rstrip()
                if line:
                    print(line, file=sys.stderr, flush=True)

        threading.Thread(target=drain_stderr, daemon=True).start()

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Py-Feat worker exited during startup with code {self.proc.returncode}")
            assert self.proc.stdout is not None
            # stdout.readline() can block, so use a helper thread for startup.
            holder: list[str] = []

            def read_one() -> None:
                assert self.proc is not None and self.proc.stdout is not None
                holder.append(self.proc.stdout.readline())

            thread = threading.Thread(target=read_one, daemon=True)
            thread.start()
            thread.join(timeout=max(0.05, deadline - time.time()))
            if thread.is_alive():
                break
            if not holder:
                continue
            line = holder[0].strip()
            if not line:
                continue
            ready = json.loads(line)
            if ready.get("type") != "ready" or not ready.get("ok"):
                raise RuntimeError(f"Py-Feat worker failed to initialize: {ready.get('error', ready)}")
            self.au_columns = [str(column) for column in (ready.get("au_columns") or [])]
            print(f"Py-Feat worker ready on device={self.device!r}.")
            return

        self.shutdown(force=True)
        raise TimeoutError(f"Py-Feat worker did not become ready within {self.startup_timeout:.1f}s")

    def extract_paths(self, image_paths: list[str], output_size: int = DEFAULT_OUTPUT_SIZE) -> list[Optional[np.ndarray]]:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Py-Feat worker is not running")
        if not image_paths:
            return []

        self.request_counter += 1
        request_id = f"calibration_{self.request_counter}"
        request = {
            "request_id": request_id,
            "cmd": "extract",
            "image_paths": [str(Path(path).resolve()) for path in image_paths],
            "output_size": int(output_size),
        }

        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()

        deadline = time.time() + self.request_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Py-Feat worker exited during extraction with code {self.proc.returncode}")

            holder: list[str] = []

            def read_one() -> None:
                assert self.proc is not None and self.proc.stdout is not None
                holder.append(self.proc.stdout.readline())

            thread = threading.Thread(target=read_one, daemon=True)
            thread.start()
            thread.join(timeout=max(0.05, deadline - time.time()))
            if thread.is_alive():
                break
            if not holder:
                continue
            line = holder[0].strip()
            if not line:
                continue
            response = json.loads(line)
            if response.get("request_id") != request_id:
                continue
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "Unknown Py-Feat extraction error")))

            columns = [str(column) for column in (response.get("au_columns") or [])]
            if self.au_columns and columns and columns != self.au_columns:
                raise RuntimeError("Py-Feat AU columns changed between requests")
            if columns:
                self.au_columns = columns

            vectors: list[Optional[np.ndarray]] = []
            for item in (response.get("vectors") or []):
                if item is None:
                    vectors.append(None)
                    continue
                vector = np.asarray(item, dtype=np.float32)
                if vector.size == 0 or not np.all(np.isfinite(vector)):
                    vectors.append(None)
                else:
                    vectors.append(vector)
            while len(vectors) < len(image_paths):
                vectors.append(None)
            return vectors[: len(image_paths)]

        raise TimeoutError(f"Py-Feat extraction timed out after {self.request_timeout:.1f}s")

    def shutdown(self, force: bool = False) -> None:
        if self.proc is None:
            return
        if not force and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                force = True
        if force and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def load_session_reference_paths(
    *,
    participant: str,
    sessions_dir: Path,
    required: int,
) -> dict[str, list[str]]:
    """Prefer the exact image paths stored by the warm-up session JSON."""
    session_path = sessions_dir / f"{participant}.json"
    if not session_path.is_file():
        return {}
    try:
        with session_path.open("r", encoding="utf-8") as file:
            session = json.load(file)
    except Exception as exc:
        print(f"[WARN] Could not read {session_path}: {exc}", file=sys.stderr)
        return {}

    rounds = session.get("baseline_emotion_rounds", {}) or {}
    result: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        paths = [str(path) for path in ((rounds.get(emotion) or {}).get("images") or [])]
        existing = [path for path in paths if Path(path).is_file()]
        if len(existing) >= required:
            result[emotion] = existing[:required]
    return result


def scan_profile_reference_paths(
    *,
    participant: str,
    profile_dir: Path,
    required: int,
) -> dict[str, list[str]]:
    """Fallback: choose the latest saved baseline images for each emotion."""
    participant_dir = profile_dir / participant
    result: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        matches = list(participant_dir.glob(f"{participant}_*_{emotion}_*.jpg"))
        matches.sort(key=lambda path: path.stat().st_mtime)
        if len(matches) >= required:
            result[emotion] = [str(path) for path in matches[-required:]]
    return result


def collect_reference_paths(
    *,
    participant: str,
    profile_dir: Path,
    sessions_dir: Path,
    required: int,
) -> dict[str, list[str]]:
    from_session = load_session_reference_paths(
        participant=participant,
        sessions_dir=sessions_dir,
        required=required,
    )
    from_scan = scan_profile_reference_paths(
        participant=participant,
        profile_dir=profile_dir,
        required=required,
    )

    references: dict[str, list[str]] = {}
    for emotion in EMOTIONS:
        references[emotion] = from_session.get(emotion) or from_scan.get(emotion) or []
    return references


def build_profile(
    *,
    participant: str,
    references: dict[str, list[str]],
    detector: PyFeatWorkerClient,
    required: int,
    neutral_min_consistency: float,
    reference_min_consistency: float,
) -> dict[str, Any]:
    raw: dict[str, dict[str, Any]] = {}

    for emotion in EMOTIONS:
        image_paths = references.get(emotion, [])[:required]
        if len(image_paths) != required:
            raise RuntimeError(
                f"{emotion}: expected {required} saved warm-up crops, found {len(image_paths)}"
            )

        print(f"[{emotion}] extracting AUs from {required} crop(s)...")
        extracted = detector.extract_paths(image_paths)
        finite_vectors: list[np.ndarray] = []
        for index, value in enumerate(extracted):
            if value is None:
                raise RuntimeError(f"{emotion}: Py-Feat returned no AU vector for crop {index + 1}")
            vector = np.asarray(value, dtype=np.float32)
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                raise RuntimeError(f"{emotion}: invalid/non-finite AU vector for crop {index + 1}")
            finite_vectors.append(vector)

        dimensions = {vector.size for vector in finite_vectors}
        if len(dimensions) != 1:
            raise RuntimeError(f"{emotion}: AU vectors have inconsistent dimensions")

        stacked = np.stack(finite_vectors)
        raw[emotion] = {
            "images": image_paths,
            "vectors": [round_vector(vector) for vector in finite_vectors],
            "mean": round_vector(np.mean(stacked, axis=0)),
            "status": "ok",
        }

    if not detector.au_columns:
        raise RuntimeError("Py-Feat returned no AU column names")

    neutral_vectors = [np.asarray(v, dtype=np.float32) for v in raw["neutral"]["vectors"]]
    neutral_mean = np.mean(np.stack(neutral_vectors), axis=0)
    neutral_distance = mean_pairwise_rms_distance(neutral_vectors)
    if not np.isfinite(neutral_distance):
        raise RuntimeError("Neutral AU reference distance is invalid")
    neutral_consistency = clamp01(1.0 - neutral_distance)

    profile: dict[str, Any] = {
        "version": 3,
        "participant_folder": participant,
        "created_at": now_iso(),
        "extractor": "py-feat Detectorv2",
        "device": detector.device,
        "au_columns": list(detector.au_columns),
        "uses_only_saved_warmup_crops": True,
        "crops_per_emotion": required,
        "reference_emotions": list(EMOTIONS),
        "parameters": {
            "neutral_min_consistency": float(neutral_min_consistency),
            "min_reference_consistency": float(reference_min_consistency),
            "note": "Four-reference AU calibration for nearest-reference live matching.",
        },
        "status": "building",
        "neutral": {
            **raw["neutral"],
            "normalized_distance": round(float(neutral_distance), 6),
            "consistency": round(float(neutral_consistency), 6),
            "mean": round_vector(neutral_mean),
            "pairwise_comparison_count": int(required * (required - 1) / 2),
            "consistency_passed": bool(neutral_consistency >= neutral_min_consistency),
        },
        "emotions": {},
        "usable_emotion_count": 0,
    }

    for emotion in EMOTIONS:
        if emotion == "neutral":
            continue
        vectors = [np.asarray(v, dtype=np.float32) for v in raw[emotion]["vectors"]]
        deltas = [vector - neutral_mean for vector in vectors]
        prototype = np.mean(np.stack(deltas), axis=0)
        consistency = mean_pairwise_cosine_similarity(deltas)
        profile["emotions"][emotion] = {
            **raw[emotion],
            "delta_vectors": [round_vector(delta) for delta in deltas],
            "delta_prototype": round_vector(prototype),
            "reference_consistency": round(float(consistency), 6),
            "pairwise_comparison_count": int(required * (required - 1) / 2),
            "consistency_passed": bool(consistency >= reference_min_consistency),
            # Direct nearest-reference matching needs the finite raw vectors;
            # prototype consistency remains a diagnostic rather than a hard gate.
            "usable": True,
        }

    profile["usable_emotion_count"] = len(profile["emotions"])
    profile["complete_reference_bank"] = True
    profile["status"] = "ready"
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and save the participant-specific four-crop AU calibration "
            "profile used by experiment_fixed.py."
        )
    )
    parser.add_argument("--participant", required=True, help="Participant folder/id, e.g. A1234")
    parser.add_argument(
        "--pyfeat_python",
        default=os.environ.get("PYFEAT_PYTHON", ""),
        help="Python executable from the Py-Feat conda environment.",
    )
    parser.add_argument(
        "--pyfeat_worker_script",
        default=os.environ.get("PYFEAT_WORKER_SCRIPT", "pyfeat_worker.py"),
        help="Path to the existing pyfeat_worker.py.",
    )
    parser.add_argument("--pyfeat_device", default=os.environ.get("PYFEAT_DEVICE", "cpu"))
    parser.add_argument("--profile_dir", default=os.environ.get("WARMUP_PROFILE_DIR", "warm_up_profile"))
    parser.add_argument("--sessions_dir", default=os.environ.get("WARMUP_SESSIONS_DIR", "warm_up_sessions"))
    parser.add_argument("--crops_per_emotion", type=int, default=DEFAULT_CROPS_PER_EMOTION)
    parser.add_argument("--startup_timeout", type=float, default=120.0)
    parser.add_argument("--request_timeout", type=float, default=30.0)
    parser.add_argument("--neutral_min_consistency", type=float, default=DEFAULT_NEUTRAL_MIN_CONSISTENCY)
    parser.add_argument("--reference_min_consistency", type=float, default=DEFAULT_REFERENCE_MIN_CONSISTENCY)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing au_calibration.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pyfeat_python:
        print(
            "ERROR: --pyfeat_python is required (or set PYFEAT_PYTHON).",
            file=sys.stderr,
        )
        return 2

    participant = str(args.participant).strip()
    required = int(args.crops_per_emotion)
    if required != 4:
        print(
            "ERROR: the current experiment expects exactly 4 warm-up AU references per emotion.",
            file=sys.stderr,
        )
        return 2

    profile_dir = Path(args.profile_dir)
    sessions_dir = Path(args.sessions_dir)
    output_path = profile_dir / participant / "au_calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        print(
            f"ERROR: {output_path} already exists. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 2

    references = collect_reference_paths(
        participant=participant,
        profile_dir=profile_dir,
        sessions_dir=sessions_dir,
        required=required,
    )

    print("Warm-up reference crops:")
    missing = False
    for emotion in EMOTIONS:
        paths = references.get(emotion, [])
        print(f"  {emotion:8s}: {len(paths)}/{required}")
        for path in paths:
            print(f"             {path}")
        if len(paths) != required:
            missing = True

    if missing:
        print(
            "ERROR: four saved baseline crops are required for every emotion before AU calibration can be built.",
            file=sys.stderr,
        )
        return 3

    detector: Optional[PyFeatWorkerClient] = None
    try:
        detector = PyFeatWorkerClient(
            python_executable=args.pyfeat_python,
            worker_script=args.pyfeat_worker_script,
            device=args.pyfeat_device,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
        )
        profile = build_profile(
            participant=participant,
            references=references,
            detector=detector,
            required=required,
            neutral_min_consistency=args.neutral_min_consistency,
            reference_min_consistency=args.reference_min_consistency,
        )

        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(profile, file, indent=2, ensure_ascii=False, allow_nan=False)
        temp_path.replace(output_path)

        print("\nAU calibration saved successfully.")
        print(f"  path: {output_path}")
        print(f"  status: {profile['status']}")
        print(f"  AU columns: {len(profile['au_columns'])}")
        print(f"  crops/emotion: {profile['crops_per_emotion']}")
        print(f"  neutral consistency: {profile['neutral']['consistency']:.6f}")
        for emotion, item in profile["emotions"].items():
            print(
                f"  {emotion:8s}: refs={len(item['vectors'])}, "
                f"reference_consistency={item['reference_consistency']:.6f}"
            )
        return 0
    except Exception as exc:
        print(f"ERROR: AU calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    finally:
        if detector is not None:
            detector.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
