import os
import base64

os.environ.setdefault("OPENAI_API_KEY", "test")
from server import _sanitize_prompt  # noqa: E402


def test_long_non_base64_string_is_untouched():
    data = "abc" * 333  # length 999, invalid Base64
    sanitized = _sanitize_prompt(data, limit=len(data) + 10)
    assert sanitized == data


def test_valid_base64_payload_is_redacted():
    payload = base64.b64encode(b"A" * 500).decode()
    sanitized = _sanitize_prompt(payload, limit=len(payload) + 10)
    assert sanitized == "[BASE64_REDACTED]"


def test_large_valid_base64_payload_is_redacted():
    payload = base64.b64encode(b"A" * 20000).decode()
    sanitized = _sanitize_prompt(payload, limit=len(payload) + 10)
    assert sanitized == "[BASE64_REDACTED]"
