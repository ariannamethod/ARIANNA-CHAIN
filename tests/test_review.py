from unittest.mock import patch

from arianna_chain import generate_with_review


def test_generate_with_review_corrects_arithmetic_error() -> None:
    calls: list[str] = []

    def fake_generate_text(prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return "2+2=5"
        assert "2+2=5" in prompt
        return "4"

    with patch("arianna_chain.generate_text", side_effect=fake_generate_text):
        result = generate_with_review("What is 2+2?", use_liquid=False)

    assert result == "4"
    assert len(calls) == 2
