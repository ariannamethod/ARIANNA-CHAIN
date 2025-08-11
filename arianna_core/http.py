import json
import logging
import os
import threading
import time
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ---- simple HTTP client ------------------------------------------------------

REQUIRED_FIELDS: Dict[str, Tuple[type, ...]] = {
    "mode": (str,),
    "think": (str,),
    "answer": (str,),
    "stop": (bool,),
    "step": (int,),
    "confidence": (int, float),
    "halt_reason": (str,),
}


class HTTPClient:
    """Minimal HTTP client with basic validation."""

    def __init__(self, timeout: float = 60.0):
        self._local = threading.local()
        self.timeout = timeout

    def _sess(self) -> requests.Session:
        s = getattr(self._local, "sess", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            self._local.sess = s
        return s

    def post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(url, str):
            raise TypeError("url must be str, got %s" % type(url).__name__)
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        r = self._sess().post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError as e:  # pragma: no cover
            raise ValueError("Response is not valid JSON") from e
        if not isinstance(data, dict):
            raise ValueError("Response JSON must be object")
        missing = [k for k in REQUIRED_FIELDS if k not in data]
        if missing:
            raise ValueError(f"Missing fields in response: {', '.join(missing)}")
        wrong = [k for k, t in REQUIRED_FIELDS.items() if not isinstance(data.get(k), t)]
        if wrong:
            raise ValueError(f"Invalid field types in response: {', '.join(wrong)}")
        return data

    def stream_sse(
        self, url: str, payload: Dict[str, Any]
    ) -> Generator[Tuple[str, Dict[str, Any] | None], None, None]:
        if not isinstance(url, str):
            raise TypeError("url must be str")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        r = self._sess().post(url, json=payload, timeout=self.timeout, stream=True)
        r.raise_for_status()
        current_event = "message"
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event_name = line.split("event:", 1)[1].strip()
                if event_name:
                    current_event = event_name
                else:
                    logger.warning("Malformed event line: %s", line)
                continue
            if line.startswith("data:"):
                data_raw = line.split("data:", 1)[1].strip()
                try:
                    data = json.loads(data_raw)
                except Exception:
                    logger.warning("Malformed JSON line: %s", line)
                    yield (current_event, None)
                    continue
                yield (current_event, data)
            else:
                logger.warning("Unrecognized line: %s", line)


# ---- Liquid server wrappers ---------------------------------------------------

def _srv() -> str:
    return os.getenv("ARIANNA_SERVER_URL", "http://127.0.0.1:8000/generate")


def _srv_sse() -> str:
    return os.getenv("ARIANNA_SERVER_SSE_URL", "http://127.0.0.1:8000/generate_sse")


def _token() -> Optional[str]:
    return os.getenv("ARIANNA_SERVER_TOKEN")


class _HTTP:
    def __init__(self, timeout: float = 60.0):
        self._local = threading.local()
        self.timeout = timeout

    def _sess(self) -> requests.Session:
        s = getattr(self._local, "sess", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            if _token():
                s.headers["Authorization"] = f"Bearer {_token()}"
            self._local.sess = s
        return s

    def post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        base = 0.35
        last_exc = None
        for i in range(3):
            try:
                r = self._sess().post(url, json=payload, timeout=self.timeout)
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        sleep_s = float(retry_after) if retry_after else (base * (2 ** i))
                    except Exception:
                        sleep_s = base * (2 ** i)
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:  # pragma: no cover
                last_exc = e
                time.sleep(base * (2 ** i) + 0.2)
        raise last_exc  # type: ignore[misc]

    def stream_sse(self, url: str, payload: Dict[str, Any]):
        s = self._sess()
        if _token():
            s.headers["Authorization"] = f"Bearer {_token()}"
        r = s.post(url, json=payload, timeout=self.timeout, stream=True)
        r.raise_for_status()
        return r.iter_lines(decode_unicode=True)


_http = _HTTP(timeout=90.0)


def call_liquid(
    prompt: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: float = 60.0,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": prompt}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if trace_id:
        payload["trace_id"] = trace_id
    url = _srv()
    _http.timeout = timeout
    data = _http.post_json(url, payload)
    resp = data.get("response", data)
    if not isinstance(resp, dict):
        return {
            "mode": "final",
            "think": "",
            "answer": str(resp),
            "stop": True,
            "step": 1,
            "confidence": 0.5,
            "halt_reason": "error",
        }
    return resp


def call_liquid_stream(
    prompt: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    timeout: float = 60.0,
) -> Iterable[Tuple[str, Dict[str, Any] | None]]:
    payload: Dict[str, Any] = {"prompt": prompt}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    url = _srv_sse()
    _http.timeout = timeout
    current_event = "message"
    for line in _http.stream_sse(url, payload):
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line.split("event:", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_raw = line.split("data:", 1)[1].strip()
            try:
                data = json.loads(data_raw)
            except Exception:
                data = {"raw": data_raw}
            yield (current_event, data)


__all__ = ["HTTPClient", "call_liquid", "call_liquid_stream"]
