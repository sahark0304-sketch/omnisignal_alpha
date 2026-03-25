"""
ai_engine/consensus_engine.py — Multi-source conviction tracker.

When multiple independent signal sources agree on the same trade
within a time window, the signal is flagged as "High Conviction"
which can influence position sizing or execution priority.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict

import config
from ai_engine.ai_engine import ParsedSignal
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _SignalRecord:
    symbol: str
    action: str
    source: str
    timestamp: datetime


class ConsensusEngine:
    """Tracks recent signals and detects multi-source agreement."""

    def __init__(self):
        self._recent: List[_SignalRecord] = []

    def add_and_check(self, signal: ParsedSignal) -> bool:
        """
        Register a new signal and check if it forms a consensus.

        Returns True if >= CONSENSUS_MIN_SOURCES agree on the same
        symbol + direction within the time window.
        """
        now = datetime.now()
        cutoff = now - timedelta(minutes=config.CONSENSUS_WINDOW_MINUTES)

        # Prune stale records
        self._recent = [r for r in self._recent if r.timestamp >= cutoff]

        # Add new record
        self._recent.append(_SignalRecord(
            symbol=signal.symbol,
            action=signal.action,
            source=signal.raw_source,
            timestamp=now,
        ))

        # Count unique sources agreeing on this symbol + action
        matching_sources = set()
        for record in self._recent:
            if record.symbol == signal.symbol and record.action == signal.action:
                matching_sources.add(record.source)

        is_consensus = len(matching_sources) >= config.CONSENSUS_MIN_SOURCES

        if is_consensus:
            logger.info(
                f"[Consensus] HIGH CONVICTION: {signal.symbol} {signal.action} — "
                f"{len(matching_sources)} sources agree: {matching_sources}"
            )
        else:
            logger.debug(
                f"[Consensus] {signal.symbol} {signal.action} — "
                f"{len(matching_sources)}/{config.CONSENSUS_MIN_SOURCES} sources so far"
            )

        return is_consensus

    def get_active_consensuses(self) -> Dict[str, int]:
        """Return a dict of 'SYMBOL_ACTION' -> source count for active windows."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=config.CONSENSUS_WINDOW_MINUTES)
        active = [r for r in self._recent if r.timestamp >= cutoff]

        counts: Dict[str, set] = defaultdict(set)
        for r in active:
            key = f"{r.symbol}_{r.action}"
            counts[key].add(r.source)

        return {k: len(v) for k, v in counts.items()}


# Module-level singleton
consensus_engine = ConsensusEngine()
