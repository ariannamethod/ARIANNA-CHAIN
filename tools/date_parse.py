from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any


def date_parse(text: str) -> str:
    text = text.strip()
    fmts = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for f in fmts:
        try:
            dt = datetime.strptime(text, f)
            return dt.date().isoformat()
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except Exception:
        return json.dumps({"ok": False, "error": "unrecognized date"})


TOOL_SPEC: Dict[str, Any] = {
    "name": "date.parse",
    "desc": "parse common date formats to ISO date.",
    "args": {"text": "string"},
    "func": date_parse,
}
