from __future__ import annotations

from typing import Dict, Any


def memory_note(text: str) -> str:
    from arianna_chain import SelfMonitor, _redact

    sm = SelfMonitor()
    sm.note(_redact(text)[:1000])
    return "ok"


TOOL_SPEC: Dict[str, Any] = {
    "name": "memory.note",
    "desc": "store a short note to memory.",
    "args": {"text": "string"},
    "func": memory_note,
}
