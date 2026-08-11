#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


def log(message: str) -> None:
    print(f"[pyfeat_worker] {message}", file=sys.stderr, flush=True)


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def norm_path(value: Any) -> str:
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crash-isolated Py-Feat AU worker")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Detectorv2 device, normally 'cpu' or 'cuda'.",
    )
    return parser.parse_args()


def load_detector(device: str):
    """
    Import and initialize Detectorv2 while redirecting third-party stdout to
    stderr so stdout remains a clean JSON protocol channel.
    """
    log(f"Importing Py-Feat Detectorv2 on device={device!r}...")
    with contextlib.redirect_stdout(sys.stderr):
        from feat import Detectorv2

    log("Py-Feat import succeeded; initializing Detectorv2...")
    with contextlib.redirect_stdout(sys.stderr):
        detector = Detectorv2(device=device, identity_model=None)

    log("Detectorv2 initialized successfully.")
    return detector


def extract_au_vectors(
    detector: Any,
    image_paths: list[str],
    output_size: int = 256,
) -> tuple[list[str], list[Optional[list[float]]]]:
    paths = [norm_path(path) for path in image_paths if str(path).strip()]
    if not paths:
        return [], []

    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"AU input image(s) not found: {missing}")

    log(f"Extracting AUs from {len(paths)} saved crop(s)...")
    with contextlib.redirect_stdout(sys.stderr):
        fex = detector.detect(
            paths if len(paths) > 1 else paths[0],
            data_type="image",
            batch_size=max(1, len(paths)),
            output_size=int(output_size),
        )

    if fex is None or len(fex) == 0:
        log("Detectorv2 returned no rows.")
        return [], [None for _ in paths]

    au_df = fex.aus
    au_columns = [str(column) for column in au_df.columns]
    if not au_columns:
        log("Detectorv2 result contained no AU columns.")
        return [], [None for _ in paths]

    # Py-Feat normally supplies an `input` column.  Use it to associate each
    # face row with the correct image.  This remains deterministic even when
    # an image unexpectedly contains more than one detected face.
    if "input" in fex.columns:
        inputs = [norm_path(value) for value in fex["input"].tolist()]
    else:
        inputs = []

    vectors: list[Optional[list[float]]] = []
    for path_index, path in enumerate(paths):
        if inputs:
            row_indices = [
                row_index
                for row_index, source in enumerate(inputs)
                if source == path
            ]
        elif len(fex) == len(paths):
            # Conservative fallback for a Py-Feat version that omits `input`
            # but returns exactly one row per supplied crop in input order.
            row_indices = [path_index]
        else:
            row_indices = []

        if not row_indices:
            vectors.append(None)
            continue

        if len(row_indices) > 1 and "FaceScore" in fex.columns:
            best_idx = max(
                row_indices,
                key=lambda i: float(fex.iloc[i]["FaceScore"]),
            )
        else:
            best_idx = row_indices[0]

        vector = [float(fex.iloc[best_idx][column]) for column in au_columns]
        vectors.append(vector)

    log(
        f"AU extraction complete: columns={len(au_columns)}, "
        f"usable_vectors={sum(v is not None for v in vectors)}/{len(vectors)}"
    )
    return au_columns, vectors


def main() -> int:
    args = parse_args()

    try:
        detector = load_detector(args.device)
    except Exception as exc:
        # Python-level failures are reported cleanly.  A native SIGSEGV will
        # terminate this process before this handler, which is exactly why the
        # worker is isolated from experiment_warm_up.py.
        error = f"{type(exc).__name__}: {exc}"
        log(f"Initialization failed: {error}")
        emit({"type": "ready", "ok": False, "error": error})
        return 2

    emit(
        {
            "type": "ready",
            "ok": True,
            "device": args.device,
            # Detectorv2 AU columns are learned after the first detection.
            "au_columns": [],
        }
    )

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            emit(
                {
                    "request_id": None,
                    "ok": False,
                    "error": f"Invalid JSON request: {exc}",
                }
            )
            continue

        command = str(request.get("cmd", "")).strip().lower()
        request_id = request.get("request_id")

        if command == "shutdown":
            log("Shutdown requested.")
            return 0

        if command != "extract":
            emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"Unknown command: {command!r}",
                }
            )
            continue

        try:
            raw_paths = request.get("image_paths") or []
            if not isinstance(raw_paths, list):
                raise TypeError("image_paths must be a JSON list")

            output_size = int(request.get("output_size", 256))
            au_columns, vectors = extract_au_vectors(
                detector,
                [str(path) for path in raw_paths],
                output_size=output_size,
            )
            emit(
                {
                    "request_id": request_id,
                    "ok": True,
                    "au_columns": au_columns,
                    "vectors": vectors,
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log(f"Request {request_id!r} failed: {error}")
            traceback.print_exc(file=sys.stderr)
            emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": error,
                }
            )

    log("stdin closed; worker exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
