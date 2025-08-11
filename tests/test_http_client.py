import logging
from typing import Any, Dict

import pytest

from arianna_core.http import HTTPClient


class DummyResponse:
    def __init__(self, data: Any, *, lines: list[str] | None = None):
        self._data = data
        self.status_code = 200
        self._lines = lines or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._data

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line


class DummySession:
    def __init__(self, resp: DummyResponse):
        self.resp = resp

    def post(self, url: str, json: Dict[str, Any], timeout: float, stream: bool = False):
        return self.resp


@pytest.fixture
def client(monkeypatch):
    c = HTTPClient()
    # prevent real HTTP session creation
    monkeypatch.setattr(c, "_sess", lambda: DummySession(DummyResponse({})))
    return c


def test_post_json_missing_fields(client, monkeypatch):
    data = {"mode": "final"}  # missing required fields
    resp = DummyResponse(data)
    monkeypatch.setattr(client, "_sess", lambda: DummySession(resp))
    with pytest.raises(ValueError) as exc:
        client.post_json("http://example.com", {})
    assert "Missing fields" in str(exc.value)


def test_stream_sse_invalid_event_and_json(monkeypatch, caplog):
    lines = [
        "event:",  # invalid event
        "data: {\"a\": 1",  # truncated JSON
    ]
    resp = DummyResponse({}, lines=lines)
    c = HTTPClient()
    monkeypatch.setattr(c, "_sess", lambda: DummySession(resp))
    with caplog.at_level(logging.WARNING):
        items = list(c.stream_sse("http://example.com", {}))
    assert items == [("message", None)]
    assert any("Malformed event" in m for m in caplog.messages)
    assert any("Malformed JSON" in m for m in caplog.messages)
