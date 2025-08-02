from arianna_c.logger import estimate_complexity_and_entropy, ThoughtComplexityLogger


def test_estimate_complexity_and_entropy_keywords():
    msg = "This is a paradox that asks why it is recursive"
    complexity, entropy = estimate_complexity_and_entropy(msg)
    assert complexity >= 3
    assert 0 <= entropy <= 1


def test_logger_records_and_recent():
    logger = ThoughtComplexityLogger(log_file="logs/test_log.jsonl")
    entry = logger.log_turn("test message", 2, 0.5)
    assert entry.complexity == 2
    assert logger.recent(1)[0].message == "test message"
