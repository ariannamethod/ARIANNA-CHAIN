from arianna_chain import ThoughtComplexityLogger, estimate_complexity_and_entropy, tokenizer


def test_estimate_complexity_and_entropy_keywords():
    msg = "This is a paradox that asks why it is recursive"
    tokens, entropy, _ = estimate_complexity_and_entropy(msg)
    assert tokens == len(tokenizer.encode(msg))
    assert 0 <= entropy <= 1


def test_logger_records_and_recent():
    logger = ThoughtComplexityLogger(log_file="logs/test_log.jsonl")
    entry = logger.log_turn("test message", 2, 0.5, confidence=0.8)
    assert entry.tokens == 2
    assert entry.perplexity is None
    assert logger.recent(1)[0].message == "test message"
