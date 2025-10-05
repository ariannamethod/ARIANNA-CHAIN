"""Telegram bot interface for Arianna Chain via internal server."""
from __future__ import annotations

import asyncio
import logging
import re

from arianna_core import SelfMonitor
from arianna_core.config import settings
from arianna_core.http import call_liquid
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _preview(text: str, limit: int = 160) -> str:
    """Return a trimmed preview of *text* for logging."""
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _log_sync(prompt: str, output: str) -> None:
    """Log the interaction in a worker thread."""
    with SelfMonitor(check_same_thread=False) as sm:
        sm.log(prompt, output)


async def _log_interaction(prompt: str, output: str) -> None:
    """Persist the interaction without blocking the event loop."""
    try:
        await asyncio.to_thread(_log_sync, prompt, output)
    except Exception:  # pragma: no cover - logging failures are non-critical
        logger.exception("Failed to persist interaction in SelfMonitor")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greet the user."""
    await update.message.reply_text("Привет! Я Арианна Чейн.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the user's message to Arianna-C server and return the answer."""
    message = update.effective_message
    if not message:
        logging.warning("Update without message: %s", update)
        return
    prompt = message.text or ""
    user = getattr(message, "from_user", None)
    user_id = getattr(user, "id", "unknown")
    logger.info("Received Telegram message from %s: %s", user_id, _preview(prompt))
    try:
        result = await asyncio.to_thread(call_liquid, prompt)
        plan = result.get("plan", "")
        think = result.get("think", "")
        answer = result.get("answer", "")
        text = f"<plan>{plan}</plan>\n<think>{think}</think>\n<answer>{answer}</answer>"
        asyncio.create_task(_log_interaction(prompt, text))
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        reply = (match.group(1).strip() if match else answer.strip()) or "(пустой ответ)"
        logger.info("Replying to %s: %s", user_id, _preview(reply))
        await message.reply_text(reply)
    except Exception as exc:  # pragma: no cover
        logging.exception("Arianna chain request failed")
        error_reply = f"Ошибка: {exc}"
        logger.info("Replying to %s with error: %s", user_id, _preview(error_reply))
        await message.reply_text(error_reply)


async def _idle_until_cancelled(sleep_interval: float = 3600.0) -> None:
    """Keep the process alive until the event loop is cancelled."""
    try:
        while True:
            await asyncio.sleep(sleep_interval)
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        logging.info("Shutting down idle Telegram worker")


def _build_application(token: str | None) -> Application | None:
    """Create the Telegram application if a token is provided."""
    if not token:
        return None
    return ApplicationBuilder().token(token).build()


async def main() -> None:
    app = _build_application(settings.telegram_token)
    if app is None:
        logging.warning(
            "TELEGRAM_TOKEN is not set; Telegram worker is disabled."
        )
        await _idle_until_cancelled()
        return
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
