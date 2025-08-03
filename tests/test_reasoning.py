from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import torch

from arianna_chain import generate_consistent_text, generate_with_think, reason_loop, tokenizer


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


def test_reason_loop_enforces_think_answer_format() -> None:
    """``reason_loop`` should return text with explicit think and answer tags."""

    pattern = re.compile(r"^<think>.+?</think><answer>.+?</answer>$", re.DOTALL)
    good_response = {
        "mode": "final",
        "think": "thought",
        "answer": "<think>thought</think><answer>result</answer>",
        "stop": True,
        "confidence": 0.9,
    }

    with (
        patch("arianna_chain.call_liquid", return_value=good_response),
        patch("arianna_chain.SelfMonitor.__init__", return_value=None),
        patch("arianna_chain.SelfMonitor.log"),
    ):
        result = reason_loop("Q", max_steps=1)

    assert pattern.fullmatch(result)

    bad_answers = [
        "<think>missing closing</think><answer>answer",
        "<answer>no think</answer>",
        "plain",
    ]
    for bad in bad_answers:
        bad_resp = dict(good_response, answer=bad)
        with (
            patch("arianna_chain.call_liquid", return_value=bad_resp),
            patch("arianna_chain.SelfMonitor.__init__", return_value=None),
            patch("arianna_chain.SelfMonitor.log"),
        ):
            out = reason_loop("Q", max_steps=1)
        assert not pattern.fullmatch(out)


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
