from __future__ import annotations

from arianna_chain import VectorStore
from arianna_core import format_reward, reasoning_steps_reward


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


def test_vector_store_search() -> None:
    store = VectorStore()
    store.add(["hello world", "foo bar", "baz"])
    results = store.search("foo")
    assert "foo bar" in results
