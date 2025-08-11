"""Telegram bot interface for Arianna Chain via internal server."""

import asyncio
import logging
import os
import re

from arianna_chain import generate_text
from arianna_core import SelfMonitor
from telegram import Update
from telegram.ext import (
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
    """Send the user's message to Arianna-C and return the answer."""
    prompt = update.message.text or ""
    try:
        with SelfMonitor() as sm:
            result = await asyncio.to_thread(generate_text, prompt, monitor=sm)
        text = result[0] if isinstance(result, tuple) else result
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        reply = (match.group(1).strip() if match else text.strip()) or "(пустой ответ)"
        await update.message.reply_text(reply)
    except Exception as exc:  # pragma: no cover
        logging.exception("Arianna chain request failed")
        await update.message.reply_text(f"Ошибка: {exc}")


async def main() -> None:
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is required")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
