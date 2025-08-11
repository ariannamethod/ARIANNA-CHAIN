from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from arianna_chain import VectorStore
from arianna_core import format_reward, reasoning_steps_reward


def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self._dim = 2

        def encode(self, texts: list[str]) -> np.ndarray:  # type: ignore[no-untyped-def]
            return np.zeros((len(texts), 2), dtype=np.float32)

        def get_sentence_embedding_dimension(self) -> int:  # type: ignore[no-untyped-def]
            return self._dim

    module = types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_format_reward_valid_and_invalid() -> None:
    good = "<think>step</think><answer>done</answer>"
    bad = "<think>missing answer"
    assert format_reward(good) == 1.0
    assert format_reward(bad) == 0.0


def test_reasoning_steps_reward() -> None:
    text_ok = "<think>1. a\n- b\n* c</think><answer>x</answer>"
    text_bad = "<think>1. a\n- b</think><answer>x</answer>"
    assert reasoning_steps_reward(text_ok) == 1.0
    assert reasoning_steps_reward(text_bad) == 0.0


def test_vector_store_search(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sentence_transformer(monkeypatch)
    store = VectorStore()
    store.add(["hello world", "foo bar", "baz"])
    results = store.search("foo")
    assert "foo bar" in results
