"""
quant/shadow_ledger.py — OmniSignal Alpha v6.0
The Shadow Ledger: Virtual Execution Tracker for Rejected Signals

WHAT THIS IS:
  When Risk Guard blocks a valid scanner signal (spread, news, DD mode, etc.)
  the actual dollar cost of that decision is currently unknown. The Shadow Ledger
  fixes this. It tracks every rejected scanner signal as a "virtual trade" and
  simulates what would have happened by watching subsequent price action.

  At the end of each week you get:
    "Filter SPREAD blocked 12 signals. Virtual outcome: +$840 missed profit."
    "Filter NEWS blocked 4 signals. Virtual outcome: -$120 (correct blocks)."
    "Filter DD_BLOCK blocked 8 signals. Virtual outcome: +$1,400 missed profit."

DESIGN PRINCIPLES:
  - Zero modifications to core risk_guard logic
  - Risk guard calls shadow_ledger.track_rejection() via a registered callback
    after its own work completes — one line added to _reject()
  - All state is in-memory + SQLite (shadow_ledger table in omnisignal.db)
  - Weekly report generated every Monday 00:00 UTC
  - Never blocks or modifies any real order

INTEGRATION:
  1. risk_guard.py adds ONE line to _reject():
       from quant.shadow_ledger import shadow_ledger as _sl
       _sl.track_rejection(signal, stage, reason, current_bid, current_ask)
  2. main.py startup() adds one task:
       asyncio.create_task(_supervise("shadow_ledger", shadow_ledger.run_monitor))
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Only track AUTO_* scanner signals — telegram signals have other filters that are not "costs"
TRACKED_SOURCES = {"AUTO_SMC", "AUTO_TFI", "AUTO_CATCD", "AUTO_PULLBACK",
                   "AUTO_MR", "AUTO_SCANNER", "AUTO_CONVERGENCE"}

# Only track transient rejections — structural blocks (TOXIC, HALT) are correct by definition
TRACKED_STAGES = {"SPREAD", "NEWS", "MAX_TRADES", "DD_BLOCK", "FREQUENCY",
                  "LATENCY", "EXEC_DEDUP", "SESSION_BLACKOUT", "HTF_TREND_GATE",
                  "BREAKOUT_BLOCK", "DIRECTION_BLOCK", "SESSION_BUDGET"}

MAX_VIRTUAL_TRADES  = 200   # cap in-memory
MONITOR_INTERVAL    = 15    # seconds between price checks
MAX_TRADE_AGE_SECS  = 3600  # close virtual trades after 1 hour
PIP_VALUE_APPROX    = 1.0   # $1 per pip per 0.01 lot (approximate for report)


@dataclass
class VirtualTrade:
    id: int
    source: str
    stage: str
    action: str               # BUY or SELL
    symbol: str
    entry: float
    sl: float
    tp1: Optional[float]
    opened_at: float          # unix timestamp
    opened_at_iso: str
    virtual_lots: float = 0.01
    virtual_pnl: float = 0.0
    exit_reason: str = "OPEN"  # OPEN, TP_HIT, SL_HIT, EXPIRED, REVERSED
    closed_at: Optional[float] = None


class ShadowLedger:
    """Tracks rejected scanner signals as virtual trades to quantify filter costs."""

    def __init__(self):
        self._open_trades: Dict[int, VirtualTrade] = {}
        self._closed_trades: List[VirtualTrade] = []
        self._next_id: int = 1
        self._total_tracked: int = 0
        self._last_weekly_report: Optional[float] = None
        self._pip_size: float = 0.1  # XAUUSD default

    # ──────────────────────────────────────────────────────────────────────────
    #  Public API — called from risk_guard
    # ──────────────────────────────────────────────────────────────────────────

    def track_rejection(
        self,
        signal,
        stage: str,
        reason: str,
        current_bid: float,
        current_ask: float,
    ):
        """
        Called by risk_guard._reject() for every rejected signal.
        Creates a virtual trade if the signal meets tracking criteria.
        """
        try:
            if not hasattr(signal, "raw_source"):
                return
            if signal.raw_source not in TRACKED_SOURCES:
                return
            if stage not in TRACKED_STAGES:
                return
            if not signal.stop_loss:
                return
            if len(self._open_trades) >= MAX_VIRTUAL_TRADES:
                return

            entry = (current_ask if signal.action == "BUY" else current_bid)
            if entry <= 0:
                return

            vt = VirtualTrade(
                id          = self._next_id,
                source      = signal.raw_source,
                stage       = stage,
                action      = signal.action,
                symbol      = signal.symbol or "XAUUSD",
                entry       = entry,
                sl          = signal.stop_loss,
                tp1         = signal.tp1,
                opened_at   = time.time(),
                opened_at_iso = datetime.now(timezone.utc).isoformat(),
            )
            self._open_trades[self._next_id] = vt
            self._next_id += 1
            self._total_tracked += 1

            logger.debug(
                "[ShadowLedger] Virtual trade #%d: %s %s @ %.2f SL=%.2f | blocked by %s",
                vt.id, vt.action, vt.symbol, vt.entry, vt.sl, stage,
            )
        except Exception as e:
            logger.debug("[ShadowLedger] track_rejection error: %s", e)

    # ──────────────────────────────────────────────────────────────────────────
    #  Background monitor loop
    # ──────────────────────────────────────────────────────────────────────────

    async def run_monitor(self):
        logger.info("[ShadowLedger] Monitor started (interval=%ds)", MONITOR_INTERVAL)
        while True:
            try:
                await self._check_open_trades()
                await self._maybe_emit_weekly_report()
            except Exception as e:
                logger.debug("[ShadowLedger] Monitor error: %s", e)
            await asyncio.sleep(MONITOR_INTERVAL)

    async def _check_open_trades(self):
        if not self._open_trades:
            return

        loop = asyncio.get_event_loop()
        prices = {}
        to_close = []

        for trade_id, vt in list(self._open_trades.items()):
            now = time.time()
            age = now - vt.opened_at

            if age > MAX_TRADE_AGE_SECS:
                vt.exit_reason = "EXPIRED"
                vt.closed_at = now
                to_close.append(trade_id)
                continue

            # Fetch current price (cached per symbol per cycle)
            sym = vt.symbol
            if sym not in prices:
                try:
                    import MetaTrader5 as mt5
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        prices[sym] = (float(tick.bid), float(tick.ask))
                    else:
                        continue
                except Exception:
                    continue

            bid, ask = prices[sym]
            current = bid if vt.action == "BUY" else ask

            # Check SL and TP
            if vt.action == "BUY":
                if current <= vt.sl:
                    vt.virtual_pnl = (vt.sl - vt.entry) / self._pip_size * PIP_VALUE_APPROX
                    vt.exit_reason = "SL_HIT"
                    vt.closed_at = now
                    to_close.append(trade_id)
                elif vt.tp1 and current >= vt.tp1:
                    vt.virtual_pnl = (vt.tp1 - vt.entry) / self._pip_size * PIP_VALUE_APPROX
                    vt.exit_reason = "TP_HIT"
                    vt.closed_at = now
                    to_close.append(trade_id)
                else:
                    vt.virtual_pnl = (current - vt.entry) / self._pip_size * PIP_VALUE_APPROX
            else:
                if current >= vt.sl:
                    vt.virtual_pnl = (vt.entry - vt.sl) / self._pip_size * PIP_VALUE_APPROX
                    vt.exit_reason = "SL_HIT"
                    vt.closed_at = now
                    to_close.append(trade_id)
                elif vt.tp1 and current <= vt.tp1:
                    vt.virtual_pnl = (vt.entry - vt.tp1) / self._pip_size * PIP_VALUE_APPROX
                    vt.exit_reason = "TP_HIT"
                    vt.closed_at = now
                    to_close.append(trade_id)
                else:
                    vt.virtual_pnl = (vt.entry - current) / self._pip_size * PIP_VALUE_APPROX

        for trade_id in to_close:
            vt = self._open_trades.pop(trade_id)
            self._closed_trades.append(vt)
            if vt.exit_reason in ("TP_HIT", "SL_HIT"):
                logger.debug(
                    "[ShadowLedger] Virtual #%d closed: %s %s | %s | pnl=$%.2f | was blocked by %s",
                    vt.id, vt.action, vt.symbol, vt.exit_reason, vt.virtual_pnl, vt.stage,
                )

        # Trim closed trades to last 500
        if len(self._closed_trades) > 500:
            self._closed_trades = self._closed_trades[-500:]

    # ──────────────────────────────────────────────────────────────────────────
    #  Weekly report
    # ──────────────────────────────────────────────────────────────────────────

    async def _maybe_emit_weekly_report(self):
        now_utc = datetime.now(timezone.utc)
        if now_utc.weekday() != 0 or now_utc.hour != 0:
            return
        if self._last_weekly_report and (time.time() - self._last_weekly_report) < 82800:
            return
        self._last_weekly_report = time.time()
        report = self.generate_report(days=7)
        logger.info("[ShadowLedger] === WEEKLY FILTER COST REPORT ===\n%s", report)
        try:
            from utils.notifier import notify
            notify(f"📊 *Shadow Ledger — Weekly Filter Cost Report*\n\n```\n{report}\n```")
        except Exception:
            pass
        try:
            from database import db_manager
            db_manager.log_audit("SHADOW_LEDGER_WEEKLY", {"report": report})
        except Exception:
            pass

    def generate_report(self, days: int = 7) -> str:
        """Generate a human-readable filter cost report for the last N days."""
        cutoff = time.time() - days * 86400
        relevant = [vt for vt in self._closed_trades
                    if vt.opened_at >= cutoff and vt.exit_reason in ("TP_HIT", "SL_HIT", "EXPIRED")]

        if not relevant:
            return f"No completed virtual trades in the last {days} days."

        # Aggregate by stage
        stage_stats: Dict[str, Dict] = {}
        for vt in relevant:
            s = vt.stage
            if s not in stage_stats:
                stage_stats[s] = {"count": 0, "wins": 0, "losses": 0,
                                  "virtual_pnl": 0.0, "by_source": {}}
            stage_stats[s]["count"] += 1
            stage_stats[s]["virtual_pnl"] += vt.virtual_pnl
            if vt.virtual_pnl > 0:
                stage_stats[s]["wins"] += 1
            else:
                stage_stats[s]["losses"] += 1
            src = vt.source
            stage_stats[s]["by_source"][src] = (
                stage_stats[s]["by_source"].get(src, 0) + vt.virtual_pnl
            )

        lines = [f"Shadow Ledger Report — last {days} days", "-" * 54]
        total_missed = 0.0

        for stage, st in sorted(stage_stats.items(), key=lambda kv: -kv[1]["virtual_pnl"]):
            wr = st["wins"] / max(st["count"], 1) * 100
            pnl = st["virtual_pnl"]
            total_missed += pnl
            verdict = "CORRECT" if pnl <= 0 else "MISSED PROFIT"
            lines.append(
                f"  {stage:<24} {st['count']:>4} signals  "
                f"WR={wr:.0f}%  virtual=${pnl:+.0f}  [{verdict}]"
            )
            for src, src_pnl in sorted(st["by_source"].items(), key=lambda kv: -kv[1]):
                lines.append(f"    ↳ {src}: ${src_pnl:+.0f}")

        lines.append("-" * 54)
        lines.append(f"  Total virtual missed profit: ${total_missed:+.0f}")
        lines.append(f"  Total tracked: {self._total_tracked} | Open: {len(self._open_trades)}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            "total_tracked": self._total_tracked,
            "open_virtual_trades": len(self._open_trades),
            "closed_virtual_trades": len(self._closed_trades),
            "tracked_sources": list(TRACKED_SOURCES),
            "tracked_stages": list(TRACKED_STAGES),
        }


# Module-level singleton
shadow_ledger = ShadowLedger()
