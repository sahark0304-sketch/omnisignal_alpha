"""
ingestion/discord_listener.py — discord.py + Embed parsing.

Listens to configured Discord channels for trading signal messages.
Handles both plain text and embed-formatted signals.
"""

import asyncio
import discord
from discord.ext import commands
import config
from ingestion.signal_queue import push, RawSignal
from utils.logger import get_logger

logger = get_logger(__name__)

intents = discord.Intents.default()
intents.message_content = True  # Requires "Message Content Intent" in bot settings
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    logger.info(f"[Discord] Bot connected as {bot.user} ✅")
    logger.info(f"[Discord] Monitoring {len(config.DISCORD_CHANNEL_IDS)} channel(s)")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if message.channel.id not in config.DISCORD_CHANNEL_IDS:
        return

    text = message.content
    source = f"discord:{message.guild.name}:{message.channel.name}" if message.guild else "discord:dm"

    # Extract text from embeds if present
    if message.embeds:
        embed_texts = []
        for embed in message.embeds:
            parts = []
            if embed.title:
                parts.append(embed.title)
            if embed.description:
                parts.append(embed.description)
            for field in embed.fields:
                parts.append(f"{field.name}: {field.value}")
            embed_texts.append("\n".join(parts))
        if embed_texts:
            text = (text + "\n" if text else "") + "\n---\n".join(embed_texts)

    if not text or len(text) < 10:
        return

    # Download first image attachment if present
    image_bytes = None
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            try:
                image_bytes = await attachment.read()
            except Exception as e:
                logger.warning(f"[Discord] Could not download attachment: {e}")
            break

    raw = RawSignal(content=text, source=source, image_bytes=image_bytes)
    await push(raw)
    logger.info(f"[Discord] Signal received from {source}: {text[:60]}...")


async def run_discord_listener():
    """Start the Discord bot — runs as a long-lived async task."""
    if not config.DISCORD_BOT_TOKEN:
        logger.warning("[Discord] No bot token configured — listener disabled.")
        return

    logger.info("[Discord] Starting bot...")
    try:
        await bot.start(config.DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.error("[Discord] Invalid bot token — check DISCORD_BOT_TOKEN in .env")
    except Exception as e:
        logger.error(f"[Discord] Bot error: {e}")
