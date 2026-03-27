"""
ingestion/telegram_listener.py -- Telethon async monitor.

v3.5: Uses StringSession (in-memory) instead of SQLite session file.
This permanently eliminates the 'database is locked' crash that occurs
on Windows when Telethon's internal SQLite session gets locked under
heavy async load.

The session string is persisted to a plain text file (.session_str)
for auth persistence across restarts.  No SQLite involved.
"""

import asyncio
import os
from telethon import TelegramClient, events
from telethon.sessions import StringSession
import config
from ingestion.signal_queue import push, RawSignal
from utils.logger import get_logger

logger = get_logger(__name__)

_telethon_client: TelegramClient | None = None
_SESSION_STR_FILE = "omnisignal_session.session_str"
_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_DELAY_SECS = 10


def _load_session_string() -> str:
    """Load saved session string from disk, or return empty string."""
    try:
        if os.path.exists(_SESSION_STR_FILE):
            with open(_SESSION_STR_FILE, "r") as f:
                s = f.read().strip()
                if s:
                    return s
    except Exception:
        pass
    return ""


def _save_session_string(client: TelegramClient):
    """Persist the current session string to disk."""
    try:
        session_str = client.session.save()
        if session_str:
            with open(_SESSION_STR_FILE, "w") as f:
                f.write(session_str)
    except Exception as e:
        logger.debug(f"[Telegram] Could not save session string: {e}")


async def disconnect():
    """Cleanly disconnect the Telethon client."""
    global _telethon_client
    if _telethon_client is not None:
        try:
            _save_session_string(_telethon_client)
            await _telethon_client.disconnect()
            logger.info("[Telegram] Client disconnected cleanly.")
        except Exception as e:
            logger.warning(f"[Telegram] Disconnect error (non-fatal): {e}")
        finally:
            _telethon_client = None


async def run_telegram_listener():
    """Start the Telegram listener with StringSession (no SQLite)."""
    global _telethon_client

    if not config.TELEGRAM_API_ID or not config.TELEGRAM_API_HASH:
        logger.warning("[Telegram] No API credentials configured -- listener disabled.")
        return

    channels = [c.strip() for c in config.TELEGRAM_CHANNELS if c.strip()]
    if not channels:
        logger.warning("[Telegram] No channels configured -- listener disabled.")
        return

    chat_ids = [int(c) if c.lstrip("-").isdigit() else c for c in channels]
    attempt = 0

    while attempt < _MAX_RECONNECT_ATTEMPTS:
        attempt += 1

        session_str = _load_session_string()
        client = TelegramClient(
            StringSession(session_str),
            config.TELEGRAM_API_ID,
            config.TELEGRAM_API_HASH,
            connection_retries=5,
            retry_delay=5,
        )

        @client.on(events.NewMessage(chats=chat_ids))
        async def handler(event):
            text = event.raw_text
            if not text or len(text) < 10:
                return

            source = f"telegram:{event.chat_id}"
            image_bytes = None

            if event.photo:
                try:
                    image_bytes = await event.download_media(bytes)
                except Exception as e:
                    logger.warning(f"[Telegram] Could not download image: {e}")

            raw = RawSignal(content=text, source=source, image_bytes=image_bytes)
            await push(raw)
            logger.info(f"[Telegram] Signal received from chat {event.chat_id}: {text[:60]}...")

        logger.info(f"[Telegram] Connecting -- monitoring {len(channels)} channel(s)... (attempt {attempt})")

        try:
            # v6.3.1 FIX: Always use MTProto user session, never bot_token.
            # Bot tokens cannot read public channels the bot is not admin of.
            await client.start()

            me = await client.get_me()
            if me.bot:
                logger.error(
                    '[Telegram] CONNECTED AS BOT — cannot read channels! '
                    'Delete omnisignal_session.session_str and restart '
                    'to trigger user phone login.'
                )
                await client.disconnect()
                return
            else:
                logger.info(
                    '[Telegram] Connected as user: %s (ID: %d) — can read all channels',
                    me.first_name, me.id,
                )

            _telethon_client = client
            _save_session_string(client)
            logger.info("[Telegram] Listener active \u2705")
            attempt = 0
            await client.run_until_disconnected()

        except Exception as e:
            logger.error(
                f"[Telegram] Connection error (attempt {attempt}): {e}. "
                f"Reconnecting in {_RECONNECT_DELAY_SECS}s..."
            )
            try:
                await client.disconnect()
            except Exception:
                pass
            _telethon_client = None
            await asyncio.sleep(_RECONNECT_DELAY_SECS)

    logger.critical(f"[Telegram] Failed after {_MAX_RECONNECT_ATTEMPTS} attempts. Listener stopped.")
