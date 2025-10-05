import json

import server


def test_responses_create_fallback(monkeypatch):
    server._RESPONSE_FORMAT_SUPPORTED = None

    calls = []

    class DummyResponse:
        def __init__(self) -> None:
            self.output_text = json.dumps(
                {
                    "mode": "final",
                    "think": "t",
                    "answer": "a",
                    "stop": True,
                    "confidence": 0.8,
                    "step": 1,
                    "halt_reason": "final",
                }
            )

        def to_dict_recursive(self):  # pragma: no cover - simple structure
            return {"usage": {"input_tokens": 5, "output_tokens": 3}, "id": "dummy"}

    class DummyClient:
        def __init__(self) -> None:
            self.responses = self

        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("Responses.create() got an unexpected keyword argument 'response_format'")
            return DummyResponse()

    monkeypatch.setattr(server, "_openai_client", lambda: DummyClient())

    obj, usage, openai_id = server._responses_create(
        "prompt", model=None, temperature=0.1, top_p=0.9
    )

    assert server._RESPONSE_FORMAT_SUPPORTED is False
    assert len(calls) == 2
    assert server.JSON_FALLBACK_MARKER in calls[-1]["input"]
    assert "response_format" not in calls[-1]
    assert obj["mode"] == "final"
    assert obj["answer"] == "a"
    assert usage == {"input": 5, "output": 3, "total": 8}
    assert openai_id == "dummy"
