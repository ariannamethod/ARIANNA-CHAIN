import os
import json
from typing import Generator

# Ensure required environment variables are set before importing the server
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ARIANNA_SERVER_TOKEN", "secret-token")

import server  # noqa: E402  # import after setting env vars


VALID_HEADERS = {"Authorization": "Bearer secret-token"}
INVALID_HEADERS = {"Authorization": "Bearer wrong"}


def test_generate_valid_token(monkeypatch):
    def fake_create(prompt: str, *, model=None, temperature=0.3, top_p=0.95, check_flags=True):
        return (
            {
                "mode": "final",
                "think": "",
                "answer": "Hello",
                "stop": True,
                "step": 1,
                "halt_reason": "stop",
            },
            {"total": 1},
            "id-123",
        )

    monkeypatch.setattr(server, "_responses_create", fake_create)
    client = server.app.test_client()
    resp = client.post("/generate", json={"prompt": "hi"}, headers=VALID_HEADERS)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["response"]["answer"] == "Hello"


def test_generate_invalid_token():
    client = server.app.test_client()
    resp = client.post("/generate", json={"prompt": "hi"}, headers=INVALID_HEADERS)
    assert resp.status_code == 401


def test_generate_sse(monkeypatch):
    def fake_stream(prompt: str, *, model=None, temperature=0.3, top_p=0.95) -> Generator[str, None, None]:
        yield "retry: 10000\n\n"
        yield (
            "event: response.output_text.delta\n"
            f"data: {json.dumps({'delta': 'chunk1'})}\n\n"
        )
        completed = json.dumps(
            {
                'mode': 'final',
                'think': '',
                'answer': 'done',
                'stop': True,
                'step': 1,
                'halt_reason': 'stop',
            }
        )
        yield (
            "event: response.completed\n"
            f"data: {completed}\n\n"
        )

    monkeypatch.setattr(server, "_responses_stream", fake_stream)
    client = server.app.test_client()
    resp = client.post("/generate_sse", json={"prompt": "hi"}, headers=VALID_HEADERS)

    chunks = []
    for chunk in resp.response:
        chunks.append(chunk)
        if len(chunks) >= 4:
            break

    assert chunks[0] == b": ready\n\n"
    joined = b"".join(chunks[1:])
    assert b"event: response.output_text.delta" in joined
    assert b"event: response.completed" in joined
