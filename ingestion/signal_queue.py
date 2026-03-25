"""
ingestion/signal_queue.py - Priority-aware async queue for raw signals.

All ingestors (Telegram, Discord, manual) push RawSignal objects here.
The main orchestrator pulls and processes them sequentially.

Priority tiers:
  1 (highest) - AUTO_CONVERGENCE, S-Tier VIP Telegram signals
  2 (normal)  - Standard AUTO_ scanners, unrated Telegram signals
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RawSignal:
    content: str
    source: str
    received_at: datetime = field(default_factory=datetime.now)
    image_bytes: Optional[bytes] = None
    retry_count: int = 0


_PRIORITY_1_SOURCES = {"AUTO_CONVERGENCE"}

_counter = 0


def _assign_priority(source: str) -> int:
    """Assign queue priority: 1 = highest, 2 = normal."""
    if source in _PRIORITY_1_SOURCES:
        return 1
    if not source.startswith("AUTO_"):
        return 1
    return 2


_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=500)


async def push(signal: RawSignal):
    """Add a raw signal to the priority processing queue."""
    global _counter
    _counter += 1
    priority = _assign_priority(signal.source)
    await _queue.put((priority, _counter, signal))
    logger.info(
        "[Queue] Signal queued from %s | P%d | Queue size: %d",
        signal.source, priority, _queue.qsize(),
    )


async def pull() -> RawSignal:
    """Block until a signal is available, then return highest priority."""
    _, _, signal = await _queue.get()
    return signal


def done():
    """Mark the current queue task as done."""
    try:
        _queue.task_done()
    except ValueError:
        pass


def queue_size() -> int:
    return _queue.qsize()
