import base64

from server import _sanitize_prompt


def test_sanitize_prompt_leaves_invalid_base64():
    invalid = 'A' * 201  # length not multiple of 4
    assert _sanitize_prompt(invalid) == invalid


def test_sanitize_prompt_redacts_valid_base64():
    payload = base64.b64encode(b'x' * 300).decode()
    sanitized = _sanitize_prompt(payload)
    assert sanitized == '[BASE64_REDACTED]'
