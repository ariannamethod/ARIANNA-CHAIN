from types import SimpleNamespace
from unittest.mock import patch

from arianna_chain import generate_text, tokenizer


class DummyMonitor:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def search_prompts(self, *_args, **_kwargs):
        return []

    def log(self, *_args, **_kwargs):
        pass


def _patch_env():
    return (
        patch("arianna_chain.SelfMonitor", DummyMonitor),
        patch("arianna_chain.quantize_2bit"),
        patch(
            "arianna_chain.thought_logger.log_turn",
            return_value=SimpleNamespace(tokens=1, entropy=0.1, perplexity=None, timestamp="t", confidence=0.5),
        ),
    )


def test_reflection_revises_answer_when_critique_negative() -> None:
    draft = tokenizer.encode("draft")
    revised = tokenizer.encode("revised")
    p1, p2, p3 = _patch_env()
    with (
        p1,
        p2,
        p3,
        patch("arianna_chain.reflect", return_value="Needs work"),
        patch("arianna_chain.AriannaC") as MockModel,
    ):
        mock = MockModel.return_value
        mock.generate.side_effect = [draft, revised]
        mock.eval.return_value = None
        result = generate_text("prompt", self_reflect=True)
    assert result == tokenizer.decode(revised)
    assert mock.generate.call_count == 2


def test_reflection_keeps_answer_when_critique_positive() -> None:
    draft = tokenizer.encode("draft")
    p1, p2, p3 = _patch_env()
    with (
        p1,
        p2,
        p3,
        patch("arianna_chain.reflect", return_value="Looks good"),
        patch("arianna_chain.AriannaC") as MockModel,
    ):
        mock = MockModel.return_value
        mock.generate.return_value = draft
        mock.eval.return_value = None
        result = generate_text("prompt", self_reflect=True)
    assert result == tokenizer.decode(draft)
    assert mock.generate.call_count == 1
