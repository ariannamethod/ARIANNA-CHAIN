import asyncio
import types

from main import handle_message


class DummyContext:
    pass


def test_handle_message_without_message():
    update = types.SimpleNamespace(message=None, effective_message=None)
    ctx = DummyContext()
    # Should not raise
    asyncio.run(handle_message(update, ctx))
