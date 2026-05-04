from __future__ import annotations

import re
from datetime import datetime


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_ts(message: str) -> None:
    print(f"[{now_ts()}] {message}")


def get_system_datetime() -> datetime:
    return datetime.now()


def looks_like_time_question(user_text: str) -> bool:
    text = user_text.strip().lower()
    patterns = [
        r"\bwhat(?:'s| is)? the time\b", r"\bcurrent time\b", r"\btime now\b",
        r"\bwhat time is it\b", r"\bcan you tell me the time\b", r"\btell me the time\b",
        r"\bwhat(?:'s| is)? the date\b", r"\bcurrent date\b", r"\bdate today\b",
        r"\bwhat(?:'s| is)? today'?s date\b", r"\btell me the date\b",
        r"\bwhat day is it\b", r"\bwhat day is today\b", r"\bcurrent day\b",
        r"\btoday is what day\b", r"\bwhat(?:'s| is)? the current time and date\b",
        r"\btoday'?s date and time\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def build_system_time_reply(user_text: str) -> str:
    now = get_system_datetime()
    text = user_text.strip().lower()
    current_time = now.strftime("%I:%M %p").lstrip("0")
    current_date = now.strftime("%A, %B %d, %Y")
    current_day = now.strftime("%A")

    asks_time = "time" in text
    asks_date = "date" in text
    asks_day = "what day" in text or "day is it" in text or "day is today" in text

    if (asks_time and asks_date) or "time and date" in text:
        return f"The current date and time is {current_date} at {current_time}."
    if asks_time:
        return f"The current time is {current_time}."
    if asks_date:
        return f"Today's date is {current_date}."
    if asks_day:
        return f"Today is {current_day}."
    return f"The current date and time is {current_date} at {current_time}."
