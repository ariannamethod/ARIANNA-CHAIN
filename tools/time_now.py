from __future__ import annotations

from datetime import datetime
from typing import Dict, Any


def time_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


TOOL_SPEC: Dict[str, Any] = {
    "name": "time.now",
    "desc": "UTC timestamp now.",
    "args": {},
    "func": time_now,
}
