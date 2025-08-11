import json
import logging
import threading
import time
from typing import Any, Dict, Generator, Iterable, Optional, Tuple

import requests
from .config import settings

logger = logging.getLogger(__name__)


# ---- robust HTTP client ------------------------------------------------------


class RobustHTTPClient:
    """HTTP client with headers, retries and SSE support."""

    def __init__(
        self,
        *,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        backoff: float = 0.35,
    ) -> None:
        self._local = threading.local()
        self.timeout = timeout
        self.base_headers = headers or {}
        self.max_retries = max_retries
        self.backoff = backoff

    def _sess(self) -> requests.Session:
        s = getattr(self._local, "sess", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            s.headers.update(self.base_headers)
            self._local.sess = s
        return s

    def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(url, str):
            raise TypeError("url must be str")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        sess = self._sess()
        last_exc: Exception | None = None
        for i in range(self.max_retries):
            try:
                r = sess.post(url, json=payload, timeout=self.timeout, headers=headers)
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        sleep_s = float(retry_after) if retry_after else self.backoff * (2**i)
                    except Exception:
                        sleep_s = self.backoff * (2**i)
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                try:
                    data = r.json()
                except ValueError as e:  # pragma: no cover
                    raise ValueError("Response is not valid JSON") from e
                if not isinstance(data, dict):
                    raise ValueError("Response JSON must be object")
                return data
            except requests.RequestException as e:  # pragma: no cover
                last_exc = e
                time.sleep(self.backoff * (2**i) + 0.2)
        raise last_exc  # type: ignore[misc]

    def stream_sse(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Tuple[str, Dict[str, Any] | None], None, None]:
        if not isinstance(url, str):
            raise TypeError("url must be str")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        sess = self._sess()
        last_exc: Exception | None = None
        for i in range(self.max_retries):
            try:
                r = sess.post(
                    url, json=payload, timeout=self.timeout, stream=True, headers=headers
                )
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        sleep_s = float(retry_after) if retry_after else self.backoff * (2**i)
                    except Exception:
                        sleep_s = self.backoff * (2**i)
                    time.sleep(sleep_s)
                    continue
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
                return
            except requests.RequestException as e:  # pragma: no cover
                last_exc = e
                time.sleep(self.backoff * (2**i) + 0.2)
        raise last_exc  # type: ignore[misc]


# ---- Liquid server wrappers ---------------------------------------------------

def _srv() -> str:
    return settings.arianna_server_url


def _srv_sse() -> str:
    return settings.arianna_server_sse_url


def _token() -> Optional[str]:
    return settings.arianna_server_token or None


def _headers() -> Dict[str, str]:
    if _token():
        return {"Authorization": f"Bearer {_token()}"}
    return {}


_http = RobustHTTPClient(timeout=90.0)


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
    data = _http.post_json(url, payload, headers=_headers())
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
    for event, data in _http.stream_sse(url, payload, headers=_headers()):
        if data is None:
            yield (event, None)
        else:
            yield (event, data)


__all__ = ["RobustHTTPClient", "call_liquid", "call_liquid_stream"]
