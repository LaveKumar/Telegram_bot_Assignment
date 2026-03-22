"""
bot.py — Telegram Vision Captioning Bot.

Commands:
  /start | /help  — usage info
  /summarize      — show last 3 cached captions
  <photo>         — describe image + return tags
"""

import asyncio
import io
import logging
import os
from collections import defaultdict

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from cache  import ImageCache
from vision import describe_image

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Shared state ───────────────────────────────────────────────────────────────
CACHE = ImageCache()

# Per-user interaction history (last 3 turns for context-awareness)
# { user_id: [{"role": "user"|"assistant", "content": str}, ...] }
user_history: dict[int, list] = defaultdict(list)
MAX_HISTORY = 3


def _push_history(uid: int, role: str, content: str) -> None:
    hist = user_history[uid]
    hist.append({"role": role, "content": content})
    # Keep only last MAX_HISTORY assistant replies
    user_history[uid] = hist[-(MAX_HISTORY * 2):]


def _fmt(result: dict, cached: bool = False) -> str:
    tags = "  •  ".join(t for t in result["tags"] if t)
    suffix = "\n\n_⚡ cached result_" if cached else ""
    return (
        f"🖼 *Caption*\n{result['caption']}\n\n"
        f"🏷 *Tags*\n`{tags}`"
        f"{suffix}"
    )


# ── Handlers ───────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👁 *Vision Captioning Bot*\n\n"
        "Send me any photo and I'll describe it and suggest tags.\n\n"
        "*Commands*\n"
        "• /help — show this message\n"
        "• /summarize — recap your last session\n\n"
        "_Powered by Ollama · LLaVA · sentence-transformers_",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, ctx)


async def cmd_summarize(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    uid   = update.effective_user.id
    hist  = user_history.get(uid, [])

    # Build summary from per-user history first …
    lines = [
        f"• {e['content'][:120]}"
        for e in hist
        if e["role"] == "assistant"
    ]

    # … fall back to global recent cache if user has no history yet
    if not lines:
        recent = CACHE.recent_captions(3)
        lines  = [f"• {r['caption'][:120]}" for r in recent]

    if not lines:
        await update.message.reply_text("No history yet — send me an image first!")
        return

    await update.message.reply_text(
        "📋 *Recent captions:*\n\n" + "\n".join(lines),
        parse_mode="Markdown",
    )


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    uid   = update.effective_user.id
    photo = update.message.photo[-1]          # largest size

    # ── Download ───────────────────────────────────────────────────────────────
    tg_file = await ctx.bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    img_bytes = buf.getvalue()
    sha = ImageCache.sha256(img_bytes)

    # ── Layer-1 cache: exact hash ──────────────────────────────────────────────
    result = CACHE.get_by_hash(sha)
    if result:
        await update.message.reply_text(_fmt(result, cached=True), parse_mode="Markdown")
        _push_history(uid, "user",      "[image]")
        _push_history(uid, "assistant", result["caption"])
        return

    # ── Ollama call (blocking → run in thread) ─────────────────────────────────
    await update.message.reply_text("🔍 Analysing with LLaVA …")
    try:
        result = await asyncio.to_thread(describe_image, img_bytes)
    except Exception as exc:
        logger.exception("Ollama error")
        await update.message.reply_text(f"⚠️ Ollama error:\n`{exc}`", parse_mode="Markdown")
        return

    # ── Layer-2 cache: semantic (check before storing) ─────────────────────────
    sem = CACHE.get_by_embedding(result["caption"])
    if sem:
        await update.message.reply_text(_fmt(sem, cached=True), parse_mode="Markdown")
        _push_history(uid, "user",      "[image]")
        _push_history(uid, "assistant", sem["caption"])
        return

    # ── Store & reply ──────────────────────────────────────────────────────────
    CACHE.store(sha, result)
    await update.message.reply_text(_fmt(result), parse_mode="Markdown")
    _push_history(uid, "user",      "[image]")
    _push_history(uid, "assistant", result["caption"])


async def handle_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Please send a *photo* for me to describe, or use /help.",
        parse_mode="Markdown",
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app   = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("help",      cmd_help))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot polling …")
    app.run_polling()


if __name__ == "__main__":
    main()
