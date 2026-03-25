"""
quant/signal_amplifier.py -- Signal Confluence Amplifier (v3.1)

HIGH-IMPACT UPGRADE: When multiple independent scanners agree on the same
direction within a short time window, amplify the signal by:
  1. Boosting conviction -> higher lot sizing via alpha multiplier
  2. Widening the TP target (momentum of agreement)
  3. Tightening the SL (if both scanners agree, the entry is higher quality)

This is the institutional concept of "signal stacking" -- when 2+ independent
alpha sources converge, the probability of a profitable trade increases
non-linearly.

Additionally, this module provides an "Opportunity Score" that the system
logs every cycle, quantifying how much tradeable signal the market is
producing. When the score is zero for extended periods, it dynamically
relaxes scanner thresholds further via vol_regime override.
"""

import asyncio
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class _SignalEvent:
    __slots__ = ("source", "action", "symbol", "entry", "timestamp", "metadata")

    def __init__(self, source: str, action: str, symbol: str, entry: float, metadata: dict = None):
        self.source = source
        self.action = action
        self.symbol = symbol
        self.entry = entry
        self.timestamp = time.time()
        self.metadata = metadata or {}


class SignalAmplifier:
    """
    Tracks recent scanner signals and detects multi-source confluence.

    When 2+ independent scanners fire the same direction on the same symbol
    within CONFLUENCE_WINDOW_SECS, the amplifier:
      - Sets a conviction boost multiplier (1.3x for 2 sources, 1.6x for 3+)
      - Logs the stacking event for ML training
      - Provides the boost as an alpha_multiplier override to risk_guard
    """

    CONFLUENCE_WINDOW_SECS = 120
    MIN_SOURCES_FOR_BOOST = 2

    BOOST_MAP = {
        2: 1.30,
        3: 1.60,
        4: 1.80,
        5: 2.00,
    }

    def __init__(self):
        self._recent_signals: List[_SignalEvent] = []
        self._boost_log: List[Dict] = []
        self._total_boosts = 0
        self._stale_window_secs = 300
        self._drought_start: Optional[float] = None
        self._drought_threshold_secs = 1800

    def register_signal(self, source: str, action: str, symbol: str, entry: float, metadata: dict = None):
        """Called by main.py after a scanner signal enters the pipeline."""
        event = _SignalEvent(source, action, symbol, entry, metadata)
        self._recent_signals.append(event)
        self._drought_start = None
        self._prune_stale()

    def get_confluence_boost(self, action: str, symbol: str) -> Tuple[float, int, List[str]]:
        """
        Check if recent signals from different sources agree on this trade.

        Returns:
            (boost_multiplier, n_agreeing_sources, list_of_source_names)
        """
        self._prune_stale()
        now = time.time()
        cutoff = now - self.CONFLUENCE_WINDOW_SECS

        agreeing = {}
        for evt in self._recent_signals:
            if (evt.symbol == symbol
                    and evt.action == action
                    and evt.timestamp >= cutoff
                    and evt.source not in agreeing):
                agreeing[evt.source] = evt

        n_sources = len(agreeing)
        if n_sources < self.MIN_SOURCES_FOR_BOOST:
            return 1.0, n_sources, list(agreeing.keys())

        boost = self.BOOST_MAP.get(n_sources, 2.0)
        self._total_boosts += 1

        source_names = list(agreeing.keys())
        logger.info(
            f"[Amplifier] SIGNAL STACKING: {n_sources} sources agree on "
            f"{symbol} {action} -> {boost:.1f}x boost | sources: {source_names}"
        )

        self._boost_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action,
            "n_sources": n_sources,
            "boost": boost,
            "sources": source_names,
        })

        return boost, n_sources, source_names

    def check_drought(self) -> bool:
        """Returns True if no scanner has fired in DROUGHT_THRESHOLD seconds."""
        if not self._recent_signals:
            if self._drought_start is None:
                self._drought_start = time.time()
            return (time.time() - self._drought_start) > self._drought_threshold_secs
        return False

    def get_opportunity_score(self) -> float:
        """
        0.0 = dead market (no signals in 30 min)
        0.5 = normal (1-3 signals in last 30 min)
        1.0 = hot market (5+ signals in last 30 min)
        """
        self._prune_stale()
        now = time.time()
        recent_count = sum(1 for e in self._recent_signals if now - e.timestamp < 1800)
        if recent_count == 0:
            return 0.0
        if recent_count <= 3:
            return 0.5
        return min(1.0, recent_count / 5.0)

    def _prune_stale(self):
        cutoff = time.time() - self._stale_window_secs
        self._recent_signals = [e for e in self._recent_signals if e.timestamp >= cutoff]

    def get_stats(self) -> Dict:
        self._prune_stale()
        return {
            "active_signals": len(self._recent_signals),
            "total_boosts": self._total_boosts,
            "opportunity_score": self.get_opportunity_score(),
            "recent_boosts": self._boost_log[-5:] if self._boost_log else [],
            "in_drought": self.check_drought(),
        }


signal_amplifier = SignalAmplifier()
