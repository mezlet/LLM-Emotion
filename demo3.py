#!/usr/bin/env python3
"""
test_robot_output.py

Standalone test harness for the Tritium TTS ("say") endpoint and the
Tritium sequence_player facial-expression endpoint, WITHOUT needing to run
the full Silero VAD + faster-whisper + Ollama + Self-RAG pipeline from
ameca_demo.py.

Use this to answer two questions quickly:
  1. Does RobotSpeaker.say() actually make the robot talk?
  2. Which sequence names in EMOTION_SEQUENCE_MAP actually trigger a
     facial expression on the robot (vs. a 404/error from Tritium)?

It's a copy of the RobotSpeaker/RobotExpression classes from
ameca_demo.py (kept in sync manually -- if you change those classes in
ameca_demo.py, mirror the change here), wrapped in a small interactive
CLI so you can type text to speak, pick an emotion to test, or type a
raw sequence name to probe Tritium directly.

USAGE
    python test_robot_output.py
    python test_robot_output.py --tts_url http://emah/tritium/text_to_speech/say?voice=Lucy
    python test_robot_output.py --tts_token YOUR_TOKEN --expression_host http://emah
    python test_robot_output.py --dry_run          # print what WOULD be sent, no network calls

Once you're inside the CLI, type "help" to see the command list.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests


# =========================
# Shared config (mirrors ameca_demo.py)
# =========================

PLUTCHIK_EMOTIONS = {
    "joy": "😊",
    "trust": "🙂",
    "fear": "😨",
    "surprise": "😮",
    "sadness": "😢",
    "disgust": "🤢",
    "anger": "😠",
    "anticipation": "🤔",
    "neutral": "🙂",
}

# FILL THESE IN with the actual sequence names registered in your Tritium
# sequence library (Behaviour > Sequences in the Tritium GUI). Placeholders
# below -- use this script's "cycle" and "seq" commands to find the real
# ones without needing to run the full pipeline.
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


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_ts(message: str) -> None:
    print(f"[{now_ts()}] {message}")


def clean_text_for_tts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[*_`~]", "", text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return text.strip()


# =========================
# RobotSpeaker (TTS) -- same PUT pattern as ameca_demo.py
# =========================

class RobotSpeaker:
    def __init__(
        self,
        tts_url: str,
        tts_token: str = "",
        speaking_cooldown_s: float = 0.3,
        dry_run: bool = False,
    ) -> None:
        self.tts_url = tts_url
        self.tts_token = tts_token
        self.speaking_cooldown_s = speaking_cooldown_s
        self.dry_run = dry_run
        self._speaking_until = 0.0

        parsed = urlparse(tts_url)
        self._host = f"{parsed.scheme}://{parsed.netloc}"

    def say(self, text: str) -> bool:
        spoken = clean_text_for_tts(text)
        if not spoken:
            print_ts("[TTS] Nothing to say after cleaning; skipping.")
            return False

        headers = {"Content-Type": "text/plain; charset=utf-8"}
        if self.tts_token:
            headers["X-Tritium-Auth-Token"] = self.tts_token

        if self.dry_run:
            print_ts(
                f"[DRY RUN][TTS] Would PUT {self.tts_url} "
                f"(token_set={bool(self.tts_token)}) text={spoken!r}"
            )
            return True

        print_ts(f"[TTS] PUT {self.tts_url} (token_set={bool(self.tts_token)}) text={spoken[:80]!r}")

        try:
            response = requests.put(self.tts_url, data=spoken.encode("utf-8"), headers=headers, timeout=5)
            ok = 200 <= response.status_code < 300
            print_ts(
                f"[TTS] status={response.status_code} {'OK' if ok else 'FAILED'}: "
                f"{response.text[:200]!r}"
            )
            return ok
        except Exception as exc:
            print_ts(f"[TTS] requests.put failed: {exc}")
            return False


# =========================
# RobotExpression (facial expression via sequence_player) -- same
# PUT pattern as ameca_demo.py
# =========================

class RobotExpression:
    def __init__(
        self,
        host: str = "http://emah",
        tts_token: str = "",
        timeout: float = 3.0,
        dry_run: bool = False,
    ) -> None:
        self.host = host.rstrip("/")
        self.token = tts_token
        self.timeout = timeout
        self.dry_run = dry_run
        self.last_emotion: Optional[str] = None

    def play_sequence_raw(self, sequence_name: str) -> bool:
        """Play an arbitrary sequence name, bypassing the emotion map entirely.
        Useful for probing Tritium to discover valid sequence names."""
        uri = f"{self.host}/tritium/sequence_player/play/{sequence_name}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["X-Tritium-Auth-Token"] = self.token

        if self.dry_run:
            print_ts(f"[DRY RUN][EXPRESSION] Would PUT {uri}")
            return True

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

    def set_emotion(self, emotion: str, confidence: float = 1.0, force: bool = True) -> bool:
        resolved_emotion = emotion if emotion in EMOTION_SEQUENCE_MAP else "trust"

        if not force and resolved_emotion == self.last_emotion:
            print_ts(f"[EXPRESSION] Emotion unchanged ({resolved_emotion}); skipping.")
            return False

        sequence_name = EMOTION_SEQUENCE_MAP.get(resolved_emotion, EMOTION_SEQUENCE_MAP["trust"])
        print_ts(f"[EXPRESSION] emotion={resolved_emotion} -> sequence='{sequence_name}'")
        success = self.play_sequence_raw(sequence_name)

        if success:
            self.last_emotion = resolved_emotion
        return success


# =========================
# Interactive CLI
# =========================

HELP_TEXT = """
Commands:
  say <text>            Speak <text> via Tritium TTS (RobotSpeaker.say).
  emotion <name>         Trigger the facial-expression sequence mapped to
                         <name>. Valid names: joy, trust, fear, surprise,
                         sadness, disgust, anger, anticipation.
  seq <sequence_name>    Play an ARBITRARY Tritium sequence name directly,
                         bypassing the emotion map. Use this to probe for
                         valid sequence names on your Tritium build.
  both <emotion> <text>  Trigger the expression for <emotion> AND speak
                         <text> in the same turn (mirrors what
                         ameca_demo.py does every conversational turn).
  cycle [delay_seconds]  Cycle through all 8 emotions in order, playing
                         each mapped sequence with a pause between them
                         (default delay: 2.5s). Watch the robot's face to
                         see which ones visibly do something.
  map                    Print the current EMOTION_SEQUENCE_MAP.
  help                   Show this help text.
  exit / quit            Quit.

Examples:
  say Hello, I am Ameca.
  emotion joy
  seq Emotion_Joy
  both sadness I am sorry to hear that.
  cycle 3
"""


def run_cli(speaker: RobotSpeaker, expression: RobotExpression) -> None:
    print_ts("Robot output test harness ready. Type 'help' for commands, 'exit' to quit.")
    print(HELP_TEXT)

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_ts("Goodbye.")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if command in {"exit", "quit"}:
            print_ts("Goodbye.")
            break

        if command == "help":
            print(HELP_TEXT)
            continue

        if command == "map":
            print_ts("Current EMOTION_SEQUENCE_MAP:")
            for emotion, sequence_name in EMOTION_SEQUENCE_MAP.items():
                print(f"    {emotion:14s} -> {sequence_name}")
            continue

        if command == "say":
            if not rest:
                print_ts("Usage: say <text>")
                continue
            speaker.say(rest)
            continue

        if command == "emotion":
            emotion_name = rest.strip().lower()
            if emotion_name not in EMOTION_SEQUENCE_MAP:
                print_ts(
                    f"Unknown emotion '{emotion_name}'. Valid options: "
                    f"{', '.join(EMOTION_SEQUENCE_MAP.keys())}"
                )
                continue
            expression.set_emotion(emotion_name, confidence=1.0, force=True)
            continue

        if command == "seq":
            sequence_name = rest.strip()
            if not sequence_name:
                print_ts("Usage: seq <sequence_name>")
                continue
            expression.play_sequence_raw(sequence_name)
            continue

        if command == "both":
            sub_parts = rest.split(maxsplit=1)
            if len(sub_parts) < 2:
                print_ts("Usage: both <emotion> <text>")
                continue
            emotion_name, text = sub_parts[0].lower(), sub_parts[1]
            if emotion_name not in EMOTION_SEQUENCE_MAP:
                print_ts(
                    f"Unknown emotion '{emotion_name}'. Valid options: "
                    f"{', '.join(EMOTION_SEQUENCE_MAP.keys())}"
                )
                continue
            # Mirrors ameca_demo.py: expression is set independent of/before
            # the TTS call, not gated on speech timing.
            expression.set_emotion(emotion_name, confidence=1.0, force=True)
            speaker.say(text)
            continue

        if command == "cycle":
            try:
                delay = float(rest.strip()) if rest.strip() else 2.5
            except ValueError:
                delay = 2.5
            print_ts(f"Cycling through all emotions with {delay}s delay between each...")
            for emotion_name in EMOTION_SEQUENCE_MAP:
                expression.set_emotion(emotion_name, confidence=1.0, force=True)
                time.sleep(delay)
            print_ts("Cycle complete. Check console output above for which PUTs returned 2xx vs failed.")
            continue

        print_ts(f"Unknown command: '{command}'. Type 'help' for the command list.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone test harness for Tritium TTS + facial-expression (sequence_player) endpoints."
    )
    parser.add_argument(
        "--tts_url",
        default=os.environ.get("TTS_URL", "http://emah/tritium/text_to_speech/say?voice=Lucy"),
        help="Tritium TTS 'say' endpoint URL.",
    )
    parser.add_argument(
        "--tts_token",
        default="",
        help="X-Tritium-Auth-Token used for both TTS and sequence_player calls.",
    )
    parser.add_argument(
        "--expression_host",
        default=os.environ.get("EXPRESSION_HOST", "http://emah"),
        help="Base host for the Tritium sequence_player endpoint (default: http://emah).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what WOULD be sent instead of making real network calls. Use this to sanity-check "
        "URLs/tokens/mappings before touching the physical robot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_ts("=== Robot output test harness ===")
    print_ts(f"TTS URL: {args.tts_url}")
    print_ts(f"Expression host: {args.expression_host}")
    print_ts(f"Token set: {bool(args.tts_token)}")
    print_ts(f"Dry run: {args.dry_run}")
    print()

    speaker = RobotSpeaker(
        tts_url=args.tts_url,
        tts_token=args.tts_token,
        dry_run=args.dry_run,
    )
    expression = RobotExpression(
        host=args.expression_host,
        tts_token=args.tts_token,
        dry_run=args.dry_run,
    )

    run_cli(speaker, expression)


if __name__ == "__main__":
    main()
