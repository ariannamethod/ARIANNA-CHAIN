from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from arianna_chain import SelfMonitor, generate_text, tokenizer


def test_generate_text_uses_provided_monitor() -> None:
    monitor = MagicMock(spec=SelfMonitor)
    monitor.search_embedding.return_value = []
    monitor.search.return_value = []
    monitor.log.return_value = None
    tokens = tokenizer.encode("result")
    with (
        patch("arianna_chain.AriannaC") as MockModel,
        patch("arianna_chain.quantize_2bit", lambda _: None),
        patch(
            "arianna_chain.estimate_complexity_and_entropy", return_value=(1, 0.1, None)
        ),
        patch(
            "arianna_chain.thought_logger.log_turn",
            return_value=SimpleNamespace(tokens=1, entropy=0.1, perplexity=None, timestamp="t"),
        ),
        patch("arianna_chain.SelfMonitor") as MockSM,
    ):
        mock_model = MockModel.return_value
        mock_model.generate.return_value = tokens
        mock_model.eval.return_value = None
        result = generate_text("prompt", use_liquid=False, monitor=monitor)
    MockSM.assert_not_called()
    monitor.log.assert_called_once_with("prompt", tokenizer.decode(tokens))
    assert result == tokenizer.decode(tokens)
