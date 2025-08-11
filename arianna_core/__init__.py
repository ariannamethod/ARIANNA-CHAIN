from __future__ import annotations

from pathlib import Path

from .tokenizer import ByteTokenizer, TokenTensor, tokenizer
from .model import (
    AriannaC,
    AriannaCConfig,
    LinearW2A8,
    quantize_2bit,
    _pack2,
    _unpack2,
)
from .reasoning import (
    SelfMonitor,
    ThoughtComplexityLogger,
    estimate_complexity_and_entropy,
    format_reward,
    reasoning_steps_reward,
    thought_logger,
    validate_reasoning_tags,
)
from .http import RobustHTTPClient, call_liquid, call_liquid_stream

PERSONA_PATH = Path(__file__).resolve().parent.parent / "prompts" / "core.txt"
PERSONA = PERSONA_PATH.read_text(encoding="utf-8").strip()
CORE_PROMPT = PERSONA

__all__ = [
    "AriannaC",
    "AriannaCConfig",
    "LinearW2A8",
    "quantize_2bit",
    "_pack2",
    "_unpack2",
    "SelfMonitor",
    "ThoughtComplexityLogger",
    "estimate_complexity_and_entropy",
    "format_reward",
    "reasoning_steps_reward",
    "thought_logger",
    "validate_reasoning_tags",
    "tokenizer",
    "ByteTokenizer",
    "TokenTensor",
    "RobustHTTPClient",
    "call_liquid",
    "call_liquid_stream",
    "CORE_PROMPT",
    "PERSONA",
]
