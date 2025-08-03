from __future__ import annotations

from typing import Dict, Any


def memory_search(query: str, limit: int = 3) -> str:
    from arianna_chain import SelfMonitor, _redact

    sm = SelfMonitor()
    hits = sm.search_prompts_and_notes(query, limit=limit)
    if not hits:
        return "(no hits)"
    out = [f"- {_redact(h)}" for h in hits]
    return "\n".join(out)


TOOL_SPEC: Dict[str, Any] = {
    "name": "memory.search",
    "desc": "search previous prompts/answers/notes (redacted).",
    "args": {"query": "string", "limit": "int"},
    "func": memory_search,
}
