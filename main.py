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
    try:
        with SelfMonitor() as sm:
            result = await asyncio.to_thread(call_liquid, prompt)
            plan = result.get("plan", "")
            think = result.get("think", "")
            answer = result.get("answer", "")
            text = f"<plan>{plan}</plan>\n<think>{think}</think>\n<answer>{answer}</answer>"
            sm.log(prompt, text)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        reply = (match.group(1).strip() if match else answer.strip()) or "(пустой ответ)"
        await message.reply_text(reply)
    except Exception as exc:  # pragma: no cover
        logging.exception("Arianna chain request failed")
        await message.reply_text(f"Ошибка: {exc}")


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
