"""Telegram bot interface for Arianna Chain via internal server."""
import asyncio
import logging
import re

from arianna_core import SelfMonitor
from arianna_core.config import settings
from arianna_core.http import call_liquid
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


async def main() -> None:
    token = settings.telegram_token
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is required")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
