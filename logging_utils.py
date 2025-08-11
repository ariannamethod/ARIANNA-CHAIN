import json
import logging
import re
from datetime import datetime
from typing import Any

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token)[\"'\s:]*[A-Za-z0-9\-_]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\\.[A-Za-z0-9_\-]{10,}\\.[A-Za-z0-9_\-]{10,}"),
]


def _redact(text: str) -> str:
    red = text
    for pat in _SECRET_PATTERNS:
        red = pat.sub("[REDACTED]", red)
    return red


class SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        if isinstance(record.msg, str):
            record.msg = _redact(record.msg)
        if record.args:
            record.args = tuple(_redact(str(a)) for a in record.args)
        return True


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        message = record.getMessage()
        log: dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(),
            "lvl": record.levelname,
            "msg": message,
        }
        trace_id = getattr(record, "trace_id", None)
        if trace_id is not None:
            log["trace_id"] = trace_id
        return json.dumps(log, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.addFilter(SecretFilter())
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


__all__ = ["get_logger", "_SECRET_PATTERNS", "SecretFilter", "JSONFormatter"]
