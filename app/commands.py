from __future__ import annotations

import re
from typing import Optional


def normalize_command(text: str) -> Optional[str]:
    stripped = text.strip().lower()
    if not stripped.startswith(("/", "\\")):
        return None
    return re.sub(r"^[\\/]+", "", stripped)
