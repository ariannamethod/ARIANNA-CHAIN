import logging
import time
from typing import Any, Dict

from arianna_core.http import RobustHTTPClient


class DummyResponse:
    def __init__(
        self,
        data: Any,
        *,
        lines: list[str] | None = None,
        status_code: int = 200,
        headers: Dict[str, str] | None = None,
    ):
        self._data = data
        self.status_code = status_code
        self._lines = lines or []
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception("error")

    def json(self) -> Any:
        return self._data

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line


def test_post_json_retries_and_headers(monkeypatch):
    class DummySession:
        def __init__(self):
            self.calls = 0
            self.last_headers: Dict[str, str] | None = None

        def post(self, url: str, json: Dict[str, Any], timeout: float, headers=None, stream: bool = False):
            self.calls += 1
            self.last_headers = headers
            if self.calls == 1:
                return DummyResponse({}, status_code=429)
            return DummyResponse({"ok": True})

    c = RobustHTTPClient()
    sess = DummySession()
    monkeypatch.setattr(c, "_sess", lambda: sess)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    data = c.post_json("http://example.com", {}, headers={"X-Test": "1"})
    assert data == {"ok": True}
    assert sess.calls == 2
    assert sess.last_headers["X-Test"] == "1"


def test_stream_sse_invalid_event_and_json(monkeypatch, caplog):
    lines = [
        "event:",  # invalid event
        "data: {\"a\": 1",  # truncated JSON
    ]
    resp = DummyResponse({}, lines=lines)

    class DummySession:
        def post(self, url: str, json: Dict[str, Any], timeout: float, headers=None, stream: bool = False):
            return resp

    c = RobustHTTPClient()
    monkeypatch.setattr(c, "_sess", lambda: DummySession())
    with caplog.at_level(logging.WARNING):
        items = list(c.stream_sse("http://example.com", {}))
    assert items == [("message", None)]
    assert any("Malformed event" in m for m in caplog.messages)
    assert any("Malformed JSON" in m for m in caplog.messages)
