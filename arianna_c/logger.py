from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import List


@dataclass
class ThoughtLogEntry:
    timestamp: str
    message: str
    complexity: int
    entropy: float


class ThoughtComplexityLogger:
    """Track complexity and entropy of generated thoughts."""

    def __init__(self, log_file: str | Path = "logs/thought_log.jsonl") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logs: List[ThoughtLogEntry] = []

    def log_turn(self, message: str, complexity_scale: int, entropy: float) -> ThoughtLogEntry:
        entry = ThoughtLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            message=message,
            complexity=max(1, min(5, complexity_scale)),
            entropy=float(min(1.0, entropy)),
        )
        self.logs.append(entry)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
        return entry

    def recent(self, n: int = 7) -> List[ThoughtLogEntry]:
        return self.logs[-n:]


def estimate_complexity_and_entropy(message: str) -> tuple[int, float]:
    complexity = 1
    lowered = message.lower()
    if any(keyword in lowered for keyword in ["why", "paradox", "recursive"]):
        complexity += 2
    if len(message) > 300:
        complexity += 1
    complexity = max(1, min(5, complexity))
    unique_words = len(set(message.split()))
    entropy = min(1.0, unique_words / 40)
    return complexity, entropy


thought_logger = ThoughtComplexityLogger()

__all__ = [
    "ThoughtComplexityLogger",
    "estimate_complexity_and_entropy",
    "thought_logger",
    "ThoughtLogEntry",
]
