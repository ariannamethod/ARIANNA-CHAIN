import json
import logging
import threading
from typing import Any, Dict, Tuple, Generator

import requests

logger = logging.getLogger(__name__)

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

    # Internal -----------------------------------------------------------------
    def _sess(self) -> requests.Session:
        s = getattr(self._local, "sess", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Content-Type": "application/json"})
            self._local.sess = s
        return s

    # Requests -----------------------------------------------------------------
    def post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(url, str):
            raise TypeError(f"url must be str, got {type(url).__name__}")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        r = self._sess().post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError as e:  # pragma: no cover - network errors
            raise ValueError("Response is not valid JSON") from e
        if not isinstance(data, dict):
            raise ValueError(f"Response JSON must be object, got {type(data).__name__}")

        missing = [k for k in REQUIRED_FIELDS if k not in data]
        if missing:
            raise ValueError(f"Missing fields in response: {', '.join(missing)}")
        wrong = [k for k, t in REQUIRED_FIELDS.items() if not isinstance(data.get(k), t)]
        if wrong:
            raise ValueError(f"Invalid field types in response: {', '.join(wrong)}")
        return data

    # Streaming ----------------------------------------------------------------
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
