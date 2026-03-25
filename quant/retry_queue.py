"""
quant/retry_queue.py -- Signal Retry Queue.

Recovers valid signals that were rejected for transient reasons (spread too
wide, news block, max concurrent trades reached).  Instead of permanently
losing these signals, they're held for up to 90 seconds and re-submitted
when the blocking condition may have cleared.
"""

import asyncio
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Set

from ingestion.signal_queue import RawSignal, push
from utils.logger import get_logger

logger = get_logger(__name__)

TRANSIENT_STAGES: Set[str] = {
    "SPREAD",
    "NEWS",
    "MAX_TRADES",
    "FREQUENCY",
    "EXPOSURE",
    "LATENCY",
    "DEVIATION",
}

MAX_QUEUE_SIZE = 15
MAX_RETRIES = 4
RETRY_INTERVAL_SECS = 15
MAX_AGE_SECS = 120


class _RetryEntry:
    __slots__ = ("signal", "first_rejected_at", "rejection_stage", "attempts")

    def __init__(self, signal: RawSignal, stage: str):
        self.signal = signal
        self.first_rejected_at = time.time()
        self.rejection_stage = stage
        self.attempts = 0


class SignalRetryQueue:

    def __init__(self):
        self._queue: Deque[_RetryEntry] = deque(maxlen=MAX_QUEUE_SIZE)
        self._total_retried = 0
        self._total_expired = 0
        self._total_received = 0

    def maybe_retry(self, raw_signal: RawSignal, rejection_reason: str, rejection_stage: str):
        """Called when a signal is rejected. Enqueues it if the rejection is transient."""
        if raw_signal.retry_count >= MAX_RETRIES:
            return

        if rejection_stage not in TRANSIENT_STAGES:
            return

        if len(self._queue) >= MAX_QUEUE_SIZE:
            oldest = self._queue[0]
            logger.debug(
                f"[RetryQ] Queue full, dropping oldest: {oldest.signal.source}"
            )
            self._queue.popleft()

        entry = _RetryEntry(raw_signal, rejection_stage)
        self._queue.append(entry)
        self._total_received += 1
        logger.info(
            f"[RetryQ] Queued for retry: {raw_signal.source} "
            f"(reason={rejection_stage}, attempt={raw_signal.retry_count + 1})"
        )

    async def run(self):
        """Background loop: re-submit queued signals periodically."""
        logger.info(
            f"[RetryQ] Started (interval={RETRY_INTERVAL_SECS}s, "
            f"max_retries={MAX_RETRIES}, max_age={MAX_AGE_SECS}s)"
        )
        await asyncio.sleep(30)
        while True:
            try:
                await self._process_cycle()
            except Exception as e:
                logger.error(f"[RetryQ] Cycle error: {e}")
            await asyncio.sleep(RETRY_INTERVAL_SECS)

    async def _process_cycle(self):
        if not self._queue:
            return

        now = time.time()
        to_retry = []
        remaining = deque(maxlen=MAX_QUEUE_SIZE)

        while self._queue:
            entry = self._queue.popleft()
            age = now - entry.first_rejected_at

            if age > MAX_AGE_SECS:
                self._total_expired += 1
                logger.debug(
                    f"[RetryQ] Expired ({age:.0f}s): {entry.signal.source}"
                )
                continue

            entry.attempts += 1
            if entry.attempts > MAX_RETRIES:
                self._total_expired += 1
                continue

            to_retry.append(entry)

        for entry in to_retry:
            entry.signal.retry_count += 1
            entry.signal.received_at = datetime.now()

            await push(entry.signal)
            self._total_retried += 1
            logger.info(
                f"[RetryQ] Re-submitted: {entry.signal.source} "
                f"(attempt={entry.signal.retry_count}, "
                f"original_stage={entry.rejection_stage})"
            )

    def get_stats(self) -> Dict:
        return {
            "queued": len(self._queue),
            "total_received": self._total_received,
            "total_retried": self._total_retried,
            "total_expired": self._total_expired,
        }


retry_queue = SignalRetryQueue()
