import json
from logging_utils import get_logger


def test_secret_redaction(capfd):
    logger = get_logger("test_redaction")
    secret = "sk-1234567890ABCDEFGHIJKLMNOP"
    logger.info("token=%s", secret)
    err = capfd.readouterr().err.strip()
    data = json.loads(err)
    assert secret not in data["msg"]
    assert "[REDACTED]" in data["msg"]
