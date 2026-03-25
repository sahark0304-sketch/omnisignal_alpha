"""
ingestion/signal_queue.py — Thread-safe async queue for raw signals.

All ingestors (Telegram, Discord, manual) push RawSignal objects here.
The main orchestrator pulls and processes them sequentially.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
#  DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class RawSignal:
    content: str                              # Raw text of the signal message
    source: str                               # e.g. "telegram:channel_name", "discord:server"
    received_at: datetime = field(default_factory=datetime.now)
    image_bytes: Optional[bytes] = None       # Attached chart screenshot, if any


# ─────────────────────────────────────────────
#  ASYNC QUEUE
# ─────────────────────────────────────────────

_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
_current_task_done: bool = True


async def push(signal: RawSignal):
    """Add a raw signal to the processing queue."""
    await _queue.put(signal)
    logger.info(f"[Queue] Signal queued from {signal.source} | Queue size: {_queue.qsize()}")


async def pull() -> RawSignal:
    """Block until a signal is available, then return it."""
    return await _queue.get()


def done():
    """Mark the current queue task as done."""
    try:
        _queue.task_done()
    except ValueError:
        pass


def queue_size() -> int:
    return _queue.qsize()
