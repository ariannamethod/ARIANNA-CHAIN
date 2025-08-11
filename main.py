"""Telegram bot interface for Arianna Chain using OpenAI Responses API."""
import asyncio
import logging
import os

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(level=logging.INFO)

client = OpenAI()
MODEL = os.getenv("ARIANNA_MODEL", "gpt-4.1")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greet the user."""
    await update.message.reply_text("Привет! Я Арианна Чейн.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the user's message to OpenAI and return the reply."""
    prompt = update.message.text or ""
    try:
        resp = client.responses.create(model=MODEL, input=prompt)
        text = getattr(resp, "output_text", "").strip() or "(пустой ответ)"
        await update.message.reply_text(text)
    except Exception as exc:  # pragma: no cover
        logging.exception("OpenAI request failed")
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
