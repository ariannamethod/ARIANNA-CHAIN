from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get_env(name: str, default: Any | None = None, cast: Callable[[str], Any] = str) -> Any:
    value = os.getenv(name, None)
    if value is None or value == "":
        return default
    try:
        return cast(value)
    except Exception as exc:  # pragma: no cover - cast errors are rare
        raise ConfigError(f"Invalid value for {name!r}: {exc}") from exc


def _bool(v: str) -> bool:
    return v.lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    """Server-specific configuration."""

    server_token_hash: str = field(
        default_factory=lambda: _hash_token(_get_env("ARIANNA_SERVER_TOKEN", ""))
    )
    request_max_bytes: int = field(
        default_factory=lambda: _get_env("REQUEST_MAX_BYTES", 1_000_000, int)
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: _get_env("REQUEST_TIMEOUT_SECONDS", 30, int)
    )
    log_dir: str = field(
        default_factory=lambda: _get_env("SERVER_LOG_DIR", os.path.join("logs", "server"))
    )
    log_file: str = field(
        default_factory=lambda: _get_env("SERVER_LOG_FILE", "server.log")
    )
    snapshot_codebase: bool = field(
        default_factory=lambda: _get_env("ARIANNA_SNAPSHOT_CODEBASE", "0", _bool)
    )


def _hash_token(token: str) -> str:
    if not token:
        return ""
    return hashlib.sha256(token.encode()).hexdigest()


settings = Settings()
