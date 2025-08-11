from __future__ import annotations

import hashlib
import threading
import time
import uuid
from typing import Dict, Tuple

from flask import Blueprint, g, jsonify, request

from arianna_core.config import settings as core_settings
from .config import settings as server_settings

security_bp = Blueprint("security", __name__)


def _extract_auth_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return request.headers.get("X-Auth-Token", "").strip()


def _client_key() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    tok = _extract_auth_token()
    if tok:
        return tok
    return request.remote_addr or "unknown"


class RateLimiter:
    def __init__(self, capacity: int, refill_per_sec: float, state_ttl: int, cleanup_threshold: int):
        self.capacity, self.refill = capacity, refill_per_sec
        self.state: Dict[str, Tuple[float, float]] = {}
        self.lock = threading.Lock()
        self.ttl = state_ttl
        self._cleanup_threshold = cleanup_threshold
        self._cleanup_running = False

    def _cleanup(self):
        now = time.time()
        with self.lock:
            stale = [k for k, (_, last) in self.state.items() if now - last > self.ttl]
            for k in stale:
                self.state.pop(k, None)
            self._cleanup_running = False

    def allow(self, key: str) -> bool:
        now = time.time()
        start_cleanup = False
        with self.lock:
            tokens, last = self.state.get(key, (self.capacity, now))
            if now - last > self.ttl:
                self.state.pop(key, None)
                tokens, last = self.capacity, now
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens < 1.0:
                self.state[key] = (tokens, now)
                return False
            self.state[key] = (tokens - 1.0, now)
            if len(self.state) > self._cleanup_threshold and not self._cleanup_running:
                self._cleanup_running = True
                start_cleanup = True
        if start_cleanup:
            threading.Thread(target=self._cleanup, daemon=True).start()
        return True

    def remaining(self, key: str) -> int:
        now = time.time()
        with self.lock:
            tokens, last = self.state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            return int(tokens)


limiter = RateLimiter(
    core_settings.rate_capacity,
    core_settings.rate_refill_per_sec,
    core_settings.rate_state_ttl,
    core_settings.rate_state_cleanup,
)


@security_bp.before_app_request
def _auth_and_limits():
    g.req_id = uuid.uuid4().hex
    if request.content_length and request.content_length > server_settings.request_max_bytes:
        return jsonify({"error": "request_too_large", "req_id": g.req_id}), 413
    if server_settings.server_token_hash:
        token = _extract_auth_token()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash != server_settings.server_token_hash:
            return jsonify({"error": "unauthorized", "req_id": g.req_id}), 401
    g.key_id = _client_key()


@security_bp.after_app_request
def _inject_req_id(resp):
    resp.headers.setdefault("X-Req-Id", getattr(g, "req_id", ""))
    return resp


__all__ = [
    "security_bp",
    "limiter",
    "_client_key",
]
