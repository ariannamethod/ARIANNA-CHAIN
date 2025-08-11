"""Application configuration loaded from environment variables.

This module centralizes access to all environment settings used across the
project. Import :data:`settings` and use its attributes instead of calling
``os.getenv`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get_env(
    name: str,
    default: Any | None = None,
    cast: Callable[[str], Any] = str,
    *,
    required: bool = False,
) -> Any:
    """Read and cast an environment variable."""
    value = os.getenv(name, None)
    if value is None or value == "":
        if required and default is None:
            raise ConfigError(f"Environment variable {name} is required")
        value = default
    if value is None:
        return None
    try:
        return cast(value)
    except Exception as exc:  # pragma: no cover - cast errors are rare
        raise ConfigError(f"Invalid value for {name!r}: {exc}") from exc


def _bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes"}


def _tuple(v: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in v.split(";") if x.strip())


def _railway_host() -> str | None:
    host = os.getenv("RAILWAY_STATIC_URL") or os.getenv("RAILWAY_URL")
    if host:
        return host if host.startswith("http") else f"https://{host}"
    return None


def _default_server_url() -> str:
    host = _railway_host()
    return f"{host.rstrip('/')}/generate" if host else "http://127.0.0.1:8000/generate"


def _default_server_sse_url() -> str:
    host = _railway_host()
    return f"{host.rstrip('/')}/generate_sse" if host else "http://127.0.0.1:8000/generate_sse"


@dataclass(frozen=True)
class Settings:
    """Container for environment configuration."""

    openai_api_key: str = field(
        default_factory=lambda: _get_env("OPENAI_API_KEY", required=True)
    )
    openai_base_url: str | None = field(
        default_factory=lambda: _get_env("OPENAI_BASE_URL", None)
    )
    arianna_server_token: str = field(
        default_factory=lambda: _get_env("ARIANNA_SERVER_TOKEN", "")
    )
    arianna_server_url: str = field(
        default_factory=lambda: _get_env("ARIANNA_SERVER_URL", _default_server_url())
    )
    arianna_server_sse_url: str = field(
        default_factory=lambda: _get_env(
            "ARIANNA_SERVER_SSE_URL", _default_server_sse_url()
        )
    )
    telegram_token: str | None = field(
        default_factory=lambda: _get_env("TELEGRAM_TOKEN", None)
    )
    arianna_model: str = field(
        default_factory=lambda: _get_env("ARIANNA_MODEL", "gpt-4.1")
    )
    arianna_model_light: str = field(
        default_factory=lambda: _get_env(
            "ARIANNA_MODEL_LIGHT",
            _get_env("ARIANNA_MODEL", "gpt-4.1"),
        )
    )
    arianna_model_heavy: str = field(
        default_factory=lambda: _get_env(
            "ARIANNA_MODEL_HEAVY",
            _get_env("ARIANNA_MODEL", "gpt-4.1"),
        )
    )
    heavy_trigger_tokens: int = field(
        default_factory=lambda: _get_env("HEAVY_TRIGGER_TOKENS", 3500, int)
    )
    heavy_hints: Tuple[str, ...] = field(
        default_factory=lambda: _get_env(
            "HEAVY_HINTS",
            "deep analysis;докажи;пошагово;reason;рефлексия",
            _tuple,
        )
    )
    prompt_limit_chars: int = field(
        default_factory=lambda: _get_env("PROMPT_LIMIT_CHARS", 16000, int)
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: _get_env("CACHE_TTL_SECONDS", 120, int)
    )
    cache_max_items: int = field(
        default_factory=lambda: _get_env("CACHE_MAX_ITEMS", 256, int)
    )
    schema_version: str = field(
        default_factory=lambda: _get_env("SCHEMA_VERSION", "1.3")
    )
    rate_capacity: int = field(
        default_factory=lambda: _get_env("RATE_CAPACITY", 20, int)
    )
    rate_refill_per_sec: float = field(
        default_factory=lambda: _get_env("RATE_REFILL_PER_SEC", 0.5, float)
    )
    rate_state_ttl: int = field(
        default_factory=lambda: _get_env("RATE_STATE_TTL", 3600, int)
    )
    rate_state_cleanup: int = field(
        default_factory=lambda: _get_env("RATE_STATE_CLEANUP", 1000, int)
    )
    sse_heartbeat_every: int = field(
        default_factory=lambda: _get_env("SSE_HEARTBEAT_EVERY", 10, int)
    )
    sse_time_heartbeat_sec: float = field(
        default_factory=lambda: _get_env("SSE_TIME_HEARTBEAT_SEC", 12.0, float)
    )
    code_block_limit: int = field(
        default_factory=lambda: _get_env("CODE_BLOCK_LIMIT", 65536, int)
    )
    simhash_hamming_thr: int = field(
        default_factory=lambda: _get_env("SIMHASH_HAMMING_THR", 3, int)
    )
    port: int = field(default_factory=lambda: _get_env("PORT", 8000, int))
    arianna_max_tokens: int = field(
        default_factory=lambda: _get_env("ARIANNA_MAX_TOKENS", 3000, int)
    )
    arianna_last_usage_summary_tokens: int = field(
        default_factory=lambda: _get_env(
            "ARIANNA_LAST_USAGE_SUMMARY_TOKENS", 8000, int
        )
    )
    arianna_disable_embed: bool = field(
        default_factory=lambda: _get_env("ARIANNA_DISABLE_EMBED", "", _bool)
    )
    arianna_snapshot_codebase: bool = field(
        default_factory=lambda: _get_env("ARIANNA_SNAPSHOT_CODEBASE", "0", _bool)
    )
    arianna_embed_model: str = field(
        default_factory=lambda: _get_env(
            "ARIANNA_EMBED_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    )


settings = Settings()
