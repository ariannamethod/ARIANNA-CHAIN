import asyncio
import types

from main import _build_application, _idle_until_cancelled, handle_message


class DummyContext:
    pass


def test_handle_message_without_message():
    update = types.SimpleNamespace(message=None, effective_message=None)
    ctx = DummyContext()
    # Should not raise
    asyncio.run(handle_message(update, ctx))


def test_build_application_without_token():
    assert _build_application(None) is None


def test_idle_until_cancelled_handles_cancellation():
    async def runner():
        task = asyncio.create_task(_idle_until_cancelled(sleep_interval=0.01))
        await asyncio.sleep(0.02)
        task.cancel()
        await task

    asyncio.run(runner())
