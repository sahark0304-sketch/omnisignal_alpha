"""
utils/notifier.py — Telegram back-channel alerts.

Sends real-time notifications to your personal Telegram chat
for trade opens, closes, halts, and errors.
"""

import asyncio
import ssl
import aiohttp
import certifi
import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Bot API token only — MTProto uses TELEGRAM_API_ID / TELEGRAM_API_HASH in telegram_listener.
BOT_TOKEN = config.TELEGRAM_BOT_TOKEN
NOTIFY_CHAT_ID = config.NOTIFY_CHAT_ID
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

_ssl_ctx = ssl.create_default_context(cafile=certifi.where())


def notify(message: str):
    """
    Send a Telegram notification (fire-and-forget).
    Safe to call from sync context -- schedules async send.
    """
    if not NOTIFY_CHAT_ID:
        logger.debug("[Notifier] No NOTIFY_CHAT_ID configured -- skipping notification.")
        return
    if not (BOT_TOKEN and str(BOT_TOKEN).strip()):
        logger.debug("[Notifier] No TELEGRAM_BOT_TOKEN configured -- skipping notification.")
        return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_async(message))
    except RuntimeError:
        asyncio.run(_send_async(message))


async def _send_async(message: str):
    """Send message via Telegram Bot API."""
    url = TELEGRAM_API_URL.format(token=BOT_TOKEN)
    payload = {
        "chat_id": NOTIFY_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    timeout = aiohttp.ClientTimeout(total=10)

    # Try with certifi CA bundle first
    try:
        conn = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    logger.debug(f"[Notifier] Sent notification: {message[:60]}...")
                    return
                body = await resp.text()
                logger.warning(f"[Notifier] Telegram API returned {resp.status}: {body}")
                return
    except (aiohttp.ClientConnectorSSLError, ssl.SSLError) as e:
        logger.warning(f"[Notifier] SSL with certifi failed, retrying without verification: {e}")
    except Exception as e:
        logger.warning(f"[Notifier] Failed to send notification: {e}")
        return

    # Fallback: disable SSL verification entirely
    try:
        conn = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    logger.debug(f"[Notifier] Sent notification (ssl=False): {message[:60]}...")
                else:
                    body = await resp.text()
                    logger.warning(f"[Notifier] Telegram API returned {resp.status}: {body}")
    except Exception as e:
        logger.warning(f"[Notifier] Failed to send notification (ssl=False fallback): {e}")


def send_autopsy(report: str):
    """Send a trade autopsy report to Telegram (same channel as notifications)."""
    if not NOTIFY_CHAT_ID:
        return
    if not (BOT_TOKEN and str(BOT_TOKEN).strip()):
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_async(report))
    except RuntimeError:
        asyncio.run(_send_async(report))
