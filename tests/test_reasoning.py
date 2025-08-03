from __future__ import annotations

from unittest.mock import patch

import json
from pathlib import Path

import torch

from arianna_chain import (
    generate_consistent_text,
    generate_with_think,
    reason_loop,
    tree_reason_loop,
    multi_reason,
    tokenizer,
)


def test_generate_with_think_returns_thought_and_final() -> None:
    """Ensure ``generate_with_think`` yields both text and metadata."""

    with patch("arianna_chain.generate_text", return_value=("thought", {"c": 1})) as mock_gen:
        result = generate_with_think("prompt")

    # The wrapper should request reasoning metadata and return the tuple as-is
    assert result == ("thought", {"c": 1})
    mock_gen.assert_called_once_with("prompt", max_new_tokens=50, config=None, log_reasoning=True)


def test_consistency_improves_with_multiple_attempts() -> None:
    """Using ``n>1`` should select the most frequent answer."""

    side_effect = ["B", "A", "A"]

    # Single attempt may yield an inconsistent answer
    with patch("arianna_chain.generate_with_think", side_effect=side_effect):
        single = generate_consistent_text("prompt", n=1)

    # Multiple attempts should recover the majority answer "A"
    with patch("arianna_chain.generate_with_think", side_effect=side_effect):
        multi = generate_consistent_text("prompt", n=3)

    assert single != "A"
    assert multi == "A"


def test_reason_loop_alternates_and_logs() -> None:
    """The reasoning loop should log intermediate thoughts and answers."""

    class DummyModel:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def eval(self) -> None:  # pragma: no cover - simple stub
            pass

        def generate(self, idx, max_new_tokens):  # pragma: no cover - simple stub
            self.calls += 1
            if self.calls % 2:
                addition = tokenizer.encode(" thought")
            else:
                addition = tokenizer.encode(" answer")
            return torch.cat([idx, addition], dim=1)

    with (
        patch("arianna_chain.AriannaC", DummyModel),
        patch("arianna_chain.quantize_2bit", lambda _: None),
        patch("arianna_chain.SelfMonitor.__init__", return_value=None),
        patch("arianna_chain.SelfMonitor.log") as mock_log,
    ):
        result = reason_loop("Q", max_steps=1)

    assert isinstance(result, str)
    assert mock_log.call_args_list[0][0][0] == "<think>"
    assert mock_log.call_args_list[1][0][0] == "<answer>"


def test_reason_loop_beam_selects_highest_scoring() -> None:
    """Multi-path mode should return highest scoring answer."""

    responses = [
        {"mode": "final", "think": "", "answer": "plain", "stop": True, "confidence": 0.7},
        {"mode": "final", "think": "", "answer": "1. numbered", "stop": True, "confidence": 0.6},
    ]

    with (
        patch("arianna_chain.call_liquid", side_effect=responses),
        patch("arianna_chain.SelfMonitor.__init__", return_value=None),
        patch("arianna_chain.SelfMonitor.log"),
    ):
        result = reason_loop("Q", max_steps=1, beams=2)

    assert result == "1. numbered"


def test_tree_reason_loop_selects_best_branch() -> None:
    """Tree search should evaluate multiple branches and pick the best."""

    with (
        patch("arianna_chain.reason_loop", side_effect=["bad", "good"]) as mock_loop,
        patch(
            "arianna_chain.estimate_complexity_and_entropy",
            side_effect=lambda ans: (1, {"bad": 0.1, "good": 0.9}[ans]),
        ),
    ):
        result = tree_reason_loop("Q", beam_size=2, depth=1)

    assert result == "good"
    assert mock_loop.call_count == 2


def test_multi_reason_majority_selection() -> None:
    """``multi_reason`` should choose the majority answer across paths."""

    outputs = ["A", "B", "A", "A"]
    temps: list[float] = []

    def fake_reason(prompt, base_temperature=0.3, **kwargs):
        temps.append(base_temperature)
        return outputs.pop(0)

    with (
        patch("arianna_chain.reason_loop", side_effect=fake_reason) as mock_loop,
        patch("arianna_chain.SelfMonitor.__init__", return_value=None),
        patch("arianna_chain.SelfMonitor.log") as mock_log,
    ):
        result = multi_reason("Q", paths=4)

    assert result == "A"
    assert mock_loop.call_count == 4
    assert len({round(t, 2) for t in temps}) > 1
    path_logs = [c for c in mock_log.call_args_list if c[0][0] == "<path>"]
    assert len(path_logs) == 4


def test_gsm8k_subset_accuracy() -> None:
    """Evaluate simple math questions and compute accuracy."""

    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "gsm8k_subset.jsonl"
    samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    answers = {sample["question"]: sample["answer"] for sample in samples}

    def fake_generate(prompt: str, **kwargs) -> str:
        return answers[prompt]

    with patch("tests.test_reasoning.generate_consistent_text", side_effect=fake_generate):
        correct = sum(
            generate_consistent_text(sample["question"]) == sample["answer"]
            for sample in samples
        )

    accuracy = correct / len(samples)
    assert accuracy == 1.0
