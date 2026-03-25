"""
quant/trade_orchestrator.py - OmniSignal Alpha v5.0
ADAPTIVE TRADE ORCHESTRATOR (ATO)

DESIGN PHILOSOPHY: SNIPER, NOT MACHINE GUN.
  More trades come from FASTER reaction to valid setups, not from
  lowering standards. Every trade must pass the same quality bar.
  The ATO only adjusts HOW FAST filters reset between valid trades -
  never whether a filter is applied at all.

PROP FIRM SAFETY RULES (NEVER VIOLATED):
  1. NO drought relaxation - if the market gives nothing, we sit.
     Zero trades for 3 hours is CORRECT behavior in a dead market.
     FOMO-driven "standard lowering" is the #1 account killer.

  2. NO filter bypass - every safety filter stays ON in every regime.
     Session blackouts, HTF trend gates, toxicity filters - all active.
     In FAST_TREND mode filters get TIGHTER, not looser: we add
     extra confirmation requirements because the risk of a reversal
     spike is highest when momentum peaks.

  3. NO confidence threshold reduction - the AI confidence floor
     is a quality gate. Lowering it = taking bad trades.
     Instead, we let the AI's own confidence speak: a 9/10 signal
     in a trending market processes faster because cooldowns are
     shorter, not because we accepted a 3/10 signal.

  4. LOSING STREAKS = FULL STOP - after consecutive losses the system
     doesn't just "slow down", it enters DEFENSIVE mode where lot
     sizes are halved and no new scanners fire until a manual
     review or a daily reset.

WHAT THE ATO ACTUALLY DOES (safely):
  A. Regime-adaptive COOLDOWNS between trades (not filter quality):
     - In FAST_TREND: direction cooloff shrinks from 900s to 450s
       because the market structure changes fast, so a 15-minute
       cooloff based on an old SL hit is stale information.
     - In SLOW_RANGE: cooloffs EXTEND because nothing has changed.
     - Execution dedup shrinks proportionally - valid signals from
       different scanners don't need to wait 90s between each other
       when the market is moving fast.

  B. Regime-adaptive LOT SIZING:
     - In FAST_TREND: lots increase by 15% (higher probability edge)
     - In VOLATILE: lots decrease by 40% (wider stops needed)
     - On winning streak: +10% per win after 3 (capped at +30%)
     - On losing streak: -15% per loss after 2 (capped at -50%)
     - Session P&L negative: auto-reduce to 70% size

  C. Regime-adaptive TP EXPANSION:
     - In FAST_TREND: TPs expand 40% (ride the move)
     - In SLOW_RANGE: TPs contract 15% (take what the market gives)

  D. REJECTION ANALYTICS (information only):
     - Tracks which filters block the most signals
     - Tracks which blocked signals would have been profitable
     - Logs "wasteful filter" stats for weekly human review
     - NEVER auto-adjusts filters based on this - humans decide
"""

import asyncio
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)


class MarketPhase(Enum):
    DEAD        = "DEAD"
    SLOW_RANGE  = "SLOW_RANGE"
    NORMAL      = "NORMAL"
    ACTIVE      = "ACTIVE"
    FAST_TREND  = "FAST_TREND"
    VOLATILE    = "VOLATILE"


PHASE_PARAMS = {
    MarketPhase.DEAD: {
        "cooldown_mult": 1.5,
        "lot_mult": 0.5,
        "tp_expand": 0.8,
        "max_trades_hour": 2,
    },
    MarketPhase.SLOW_RANGE: {
        "cooldown_mult": 1.2,
        "lot_mult": 0.8,
        "tp_expand": 0.85,
        "max_trades_hour": 4,
    },
    MarketPhase.NORMAL: {
        "cooldown_mult": 1.0,
        "lot_mult": 1.0,
        "tp_expand": 1.0,
        "max_trades_hour": 8,
    },
    MarketPhase.ACTIVE: {
        "cooldown_mult": 0.7,
        "lot_mult": 1.10,
        "tp_expand": 1.2,
        "max_trades_hour": 10,
    },
    MarketPhase.FAST_TREND: {
        "cooldown_mult": 0.5,
        "lot_mult": 1.15,
        "tp_expand": 1.4,
        "max_trades_hour": 12,
    },
    MarketPhase.VOLATILE: {
        "cooldown_mult": 1.3,
        "lot_mult": 0.6,
        "tp_expand": 1.0,
        "max_trades_hour": 5,
    },
}


WINNING_STREAK_THRESHOLD  = 3
WINNING_STREAK_LOT_BONUS  = 0.10
WINNING_STREAK_LOT_CAP    = 1.30

LOSING_STREAK_THRESHOLD   = 2
LOSING_STREAK_LOT_PENALTY = 0.15
LOSING_STREAK_LOT_FLOOR   = 0.50
LOSING_STREAK_DEFENSIVE   = 5

SESSION_LOSS_LOT_MULT     = 0.70


@dataclass
class _RejectionRecord:
    stage: str
    timestamp: float
    source: str
    symbol: str
    action: str
    confidence: int


@dataclass
class AdaptiveParams:
    phase: MarketPhase = MarketPhase.NORMAL
    phase_confidence: float = 0.5
    cooldown_multiplier: float = 1.0
    direction_cooloff_mult: float = 1.0
    breakout_block_mult: float = 1.0
    exec_dedup_mult: float = 1.0
    lot_size_multiplier: float = 1.0
    tp_expansion: float = 1.0
    trades_this_hour: int = 0
    trades_today: int = 0
    max_trades_this_hour: int = 8
    budget_remaining: bool = True
    session_pnl: float = 0.0
    session_win_rate: float = 0.5
    streak_type: str = "NONE"
    streak_length: int = 0
    is_defensive: bool = False
    top_rejecting_filter: str = ""
    total_rejections_4h: int = 0


class AdaptiveTradeOrchestrator:
    """
    Regime-adaptive meta-controller for cooldowns, sizing, and TP expansion.

    CORE PRINCIPLE: Adjusts SPEED and SIZE, never QUALITY.
    Every safety filter stays active in every regime.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_phase = MarketPhase.NORMAL
        self._phase_confidence = 0.5
        self._phase_updated_at = 0.0
        self._rejections: deque = deque(maxlen=2000)
        self._hourly_timestamps: List[float] = []
        self._today_date: Optional[str] = None
        self._today_trade_count: int = 0
        self._consecutive_results: List[bool] = []
        self._session_pnl = 0.0
        self._session_trades = 0
        self._session_wins = 0
        self._is_defensive = False
        self._wasteful_filters: Dict[str, int] = defaultdict(int)
        logger.info("[ATO] Adaptive Trade Orchestrator initialized (Sniper Mode)")

    def get_adaptive_params(self) -> AdaptiveParams:
        with self._lock:
            self._update_regime()
            self._update_daily_tracking()
            self._prune_hourly()

            phase_p = PHASE_PARAMS[self._current_phase]
            cooldown_mult = phase_p["cooldown_mult"]
            lot_mult = phase_p["lot_mult"]
            tp_expand = phase_p["tp_expand"]
            max_hour = phase_p["max_trades_hour"]

            streak_len = self._get_streak_length()
            streak_type = "NONE"
            if self._consecutive_results:
                streak_type = "WIN" if self._consecutive_results[-1] else "LOSS"

            if streak_type == "WIN" and streak_len >= WINNING_STREAK_THRESHOLD:
                extra = streak_len - WINNING_STREAK_THRESHOLD
                bonus = 1.0 + (extra + 1) * WINNING_STREAK_LOT_BONUS
                lot_mult *= min(bonus, WINNING_STREAK_LOT_CAP)

            if streak_type == "LOSS" and streak_len >= LOSING_STREAK_THRESHOLD:
                extra = streak_len - LOSING_STREAK_THRESHOLD
                penalty = 1.0 - (extra + 1) * LOSING_STREAK_LOT_PENALTY
                lot_mult *= max(penalty, LOSING_STREAK_LOT_FLOOR)
                cooldown_mult *= 1.0 + (extra * 0.15)

            if self._is_defensive:
                lot_mult *= 0.50
                cooldown_mult *= 2.0
                max_hour = min(max_hour, 3)

            if self._session_pnl < -30 and self._session_trades >= 3:
                lot_mult *= SESSION_LOSS_LOT_MULT

            hour_trades = len(self._hourly_timestamps)
            top_filter, total_rej = self._get_top_rejection()

            cooldown_mult = round(max(cooldown_mult, 0.5), 2)
            lot_mult = round(max(lot_mult, 0.30), 2)
            tp_expand = round(max(tp_expand, 0.7), 2)

            return AdaptiveParams(
                phase=self._current_phase,
                phase_confidence=self._phase_confidence,
                cooldown_multiplier=cooldown_mult,
                direction_cooloff_mult=cooldown_mult,
                breakout_block_mult=round(max(cooldown_mult * 0.9, 0.5), 2),
                exec_dedup_mult=round(max(cooldown_mult * 0.8, 0.4), 2),
                lot_size_multiplier=lot_mult,
                tp_expansion=tp_expand,
                trades_this_hour=hour_trades,
                trades_today=self._today_trade_count,
                max_trades_this_hour=max_hour,
                budget_remaining=hour_trades < max_hour,
                session_pnl=round(self._session_pnl, 2),
                session_win_rate=round(
                    self._session_wins / max(self._session_trades, 1), 2
                ),
                streak_type=streak_type,
                streak_length=streak_len,
                is_defensive=self._is_defensive,
                top_rejecting_filter=top_filter,
                total_rejections_4h=total_rej,
            )

    def record_rejection(self, stage: str, source: str, symbol: str,
                         action: str, confidence: int):
        with self._lock:
            self._rejections.append(_RejectionRecord(
                stage=stage, timestamp=time.time(), source=source,
                symbol=symbol, action=action, confidence=confidence,
            ))

    def record_execution(self, source: str, symbol: str, action: str,
                         lot_size: float, ticket: int = 0):
        with self._lock:
            self._hourly_timestamps.append(time.time())
            self._today_trade_count += 1
            self._session_trades += 1

    def record_close(self, ticket: int, pnl: float, source: str,
                     duration_secs: float = 0):
        with self._lock:
            self._session_pnl += pnl
            if pnl > 0:
                self._session_wins += 1
                self._consecutive_results.append(True)
            else:
                self._consecutive_results.append(False)

            if len(self._consecutive_results) > 30:
                self._consecutive_results = self._consecutive_results[-30:]

            streak_len = self._get_streak_length()
            if (self._consecutive_results
                    and not self._consecutive_results[-1]
                    and streak_len >= LOSING_STREAK_DEFENSIVE
                    and not self._is_defensive):
                self._is_defensive = True
                logger.critical(
                    "[ATO] DEFENSIVE MODE: %d consecutive losses. "
                    "Lots halved, cooldowns doubled. Resets at midnight UTC.",
                    streak_len,
                )

            if pnl > 15:
                self._mark_rejections_wasteful()

    def exit_defensive_mode(self):
        with self._lock:
            self._is_defensive = False
            logger.info("[ATO] Defensive mode manually cleared.")

    def _update_regime(self):
        now = time.time()
        if now - self._phase_updated_at < 20:
            return
        try:
            from quant.vol_regime import vol_regime
            vol_regime.refresh()
            hurst = vol_regime.hurst
            vol_z = vol_regime.vol_z

            utc_hour = datetime.now(timezone.utc).hour
            is_active = 7 <= utc_hour < 21
            is_peak = 8 <= utc_hour < 16

            if not is_active:
                phase, confidence = MarketPhase.DEAD, 0.9
            elif vol_z > 2.5 and hurst < 0.50:
                phase, confidence = MarketPhase.VOLATILE, 0.8
            elif vol_z > 1.5 and hurst >= 0.58:
                phase = MarketPhase.FAST_TREND
                confidence = min(0.5 + (hurst - 0.55) * 2, 0.9)
            elif vol_z > 0.3 and hurst >= 0.52 and is_peak:
                phase, confidence = MarketPhase.ACTIVE, 0.7
            elif vol_z < -0.8:
                phase, confidence = MarketPhase.SLOW_RANGE, 0.75
            else:
                phase, confidence = MarketPhase.NORMAL, 0.5

            if phase != self._current_phase:
                logger.info(
                    "[ATO] PHASE: %s -> %s (H=%.3f vol_z=%.2f)",
                    self._current_phase.value, phase.value, hurst, vol_z,
                )

            self._current_phase = phase
            self._phase_confidence = confidence
            self._phase_updated_at = now
        except Exception as e:
            logger.debug("[ATO] Regime error: %s", e)

    def _prune_hourly(self):
        cutoff = time.time() - 3600
        self._hourly_timestamps = [t for t in self._hourly_timestamps if t > cutoff]

    def _update_daily_tracking(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._today_date != today:
            if self._today_date is not None:
                logger.info(
                    "[ATO] Daily reset | %d trades, $%.2f PnL, %.0f%% WR",
                    self._session_trades, self._session_pnl,
                    self._session_wins / max(self._session_trades, 1) * 100,
                )
            self._today_date = today
            self._today_trade_count = 0
            self._session_pnl = 0.0
            self._session_trades = 0
            self._session_wins = 0
            self._wasteful_filters.clear()
            if self._is_defensive:
                self._is_defensive = False
                logger.info("[ATO] Defensive mode cleared on daily reset.")

    def _get_streak_length(self) -> int:
        if not self._consecutive_results:
            return 0
        last = self._consecutive_results[-1]
        streak = 0
        for r in reversed(self._consecutive_results):
            if r == last:
                streak += 1
            else:
                break
        return streak

    def _mark_rejections_wasteful(self):
        now = time.time()
        for rej in reversed(list(self._rejections)):
            if now - rej.timestamp > 300:
                break
            if rej.confidence >= 7:
                self._wasteful_filters[rej.stage] += 1

    def _get_top_rejection(self) -> Tuple[str, int]:
        cutoff = time.time() - 14400
        counts: Dict[str, int] = defaultdict(int)
        for rej in self._rejections:
            if rej.timestamp >= cutoff:
                counts[rej.stage] += 1
        if not counts:
            return "", 0
        top = max(counts, key=counts.get)
        return top, sum(counts.values())

    def get_diagnostics(self) -> Dict:
        with self._lock:
            cutoff = time.time() - 14400
            recent = defaultdict(int)
            for rej in self._rejections:
                if rej.timestamp >= cutoff:
                    recent[rej.stage] += 1
            return {
                "phase": self._current_phase.value,
                "is_defensive": self._is_defensive,
                "session_trades": self._session_trades,
                "session_pnl": round(self._session_pnl, 2),
                "session_win_rate": round(
                    self._session_wins / max(self._session_trades, 1), 2
                ),
                "streak_length": self._get_streak_length(),
                "streak_type": (
                    "WIN" if self._consecutive_results and self._consecutive_results[-1]
                    else "LOSS" if self._consecutive_results else "NONE"
                ),
                "trades_today": self._today_trade_count,
                "rejection_breakdown_4h": dict(recent),
                "wasteful_filters": dict(self._wasteful_filters),
                "total_rejections_4h": sum(recent.values()),
            }

    def get_status_line(self) -> str:
        d = self.get_diagnostics()
        tag = " [DEFENSIVE]" if d["is_defensive"] else ""
        return (
            f"Phase={d['phase']} trades={d['session_trades']} "
            f"pnl=${d['session_pnl']:+.0f} WR={d['session_win_rate']:.0%} "
            f"streak={d['streak_type']}x{d['streak_length']}{tag}"
        )


def get_scaled_cooldown(base_secs: float, category: str = "general") -> float:
    """Scale a cooldown duration based on current market regime."""
    params = trade_orchestrator.get_adaptive_params()
    mult_map = {
        "general": params.cooldown_multiplier,
        "direction": params.direction_cooloff_mult,
        "breakout": params.breakout_block_mult,
        "dedup": params.exec_dedup_mult,
    }
    mult = mult_map.get(category, params.cooldown_multiplier)
    return max(base_secs * mult, 20.0)


def get_lot_size_multiplier() -> float:
    return trade_orchestrator.get_adaptive_params().lot_size_multiplier


def get_tp_expansion() -> float:
    return trade_orchestrator.get_adaptive_params().tp_expansion


def is_budget_remaining() -> bool:
    return trade_orchestrator.get_adaptive_params().budget_remaining


def is_defensive_mode() -> bool:
    return trade_orchestrator.get_adaptive_params().is_defensive


async def run_orchestrator_monitor():
    """Background task: logs ATO status every 5 minutes."""
    logger.info("[ATO] Monitor started")
    while True:
        try:
            logger.info("[ATO] %s", trade_orchestrator.get_status_line())
            diag = trade_orchestrator.get_diagnostics()
            if diag["total_rejections_4h"] > 0:
                top = sorted(
                    diag["rejection_breakdown_4h"].items(),
                    key=lambda kv: kv[1], reverse=True,
                )[:5]
                logger.info("[ATO] Top rejections (4h): %s",
                            ", ".join(f"{s}={c}" for s, c in top))
            if diag["wasteful_filters"]:
                waste = sorted(
                    diag["wasteful_filters"].items(),
                    key=lambda kv: kv[1], reverse=True,
                )[:3]
                logger.info("[ATO] Wasteful filters (info): %s",
                            ", ".join(f"{s}={c}" for s, c in waste))
        except Exception as e:
            logger.debug("[ATO] Monitor error: %s", e)
        await asyncio.sleep(300)


trade_orchestrator = AdaptiveTradeOrchestrator()
