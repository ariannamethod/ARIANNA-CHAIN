from __future__ import annotations

import json
import re
from typing import Dict, Any


def text_regex_extract(pattern: str, text: str, limit: int = 10, flags: str = "") -> str:
    fl = 0
    if "i" in flags:
        fl |= re.IGNORECASE
    if "m" in flags:
        fl |= re.MULTILINE
    if "s" in flags:
        fl |= re.DOTALL
    try:
        rgx = re.compile(pattern, fl)
        matches = rgx.findall(text)
        if isinstance(matches, list):
            matches = matches[: max(1, min(limit, 50))]
            flat = []
            for m in matches:
                if isinstance(m, tuple):
                    flat.append("".join(map(str, m)))
                else:
                    flat.append(str(m))
            uniq, seen = [], set()
            for x in flat:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return json.dumps(uniq, ensure_ascii=False)
        return "[]"
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


TOOL_SPEC: Dict[str, Any] = {
    "name": "text.regex_extract",
    "desc": "regex matches as JSON list.",
    "args": {"pattern": "string", "text": "string", "limit": "int", "flags": "string"},
    "func": text_regex_extract,
}
