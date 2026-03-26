"""
quant/prop_firm_finisher.py — OmniSignal Alpha v6.0
Prop-Firm Finisher Protocol: "Coast Mode"

PROBLEM IT SOLVES:
  You're at 7.5% profit. Target is 8%. You need $500 more.
  Normal ATO is still sizing for full risk. One bad $400 loss could push you
  to 7.1%, then you need $900 more, and the pressure creates revenge trading.
  The drawdown buffer also shrinks as the target nears.

WHAT COAST MODE DOES:
  When within COAST_TRIGGER_PCT of the profit target:
    1. Slashes lot sizes by 80% (COAST_LOT_MULTIPLIER = 0.20)
    2. Raises minimum AI confidence threshold to 10 (triple-convergence only)
    3. Requires convergence engine score >= COAST_MIN_CONSENSUS before entry
    4. Tightens daily DD guard to COAST_DD_TIGHTEN_FACTOR of normal limit
    5. Blocks all AUTO_* except AUTO_CONVERGENCE and AUTO_SMC

WHAT IT DOES NOT DO:
  - Never modifies risk_guard core logic
  - Never overrides the drawdown halt
  - Never bypasses news/spread filters
  - Does not completely freeze trading (that causes impatience errors)

INTEGRATION:
  - risk_guard.validate() calls finisher.check_override() at step 0.5
  - The finisher returns (override_active: bool, lot_multiplier, min_confidence)
  - risk_guard applies these on top of its normal sizing
  - Completely transparent: all ATO/alpha ranker logic still runs first

ACTIVATION:
  Set in config.py:
    PROP_FIRM_PHASE = "CHALLENGE"
    CHALLENGE_PROFIT_TARGET_PCT = 8.0
    CHALLENGE_PROFIT_CURRENT_PCT = 7.5   # update daily via .env or CLI

  When (TARGET - CURRENT) <= COAST_TRIGGER_PCT, Coast Mode activates.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import config
from utils.logger import get_logger
from database import db_manager

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

COAST_TRIGGER_PCT       = 0.5     # activate when within 0.5% of target
COAST_LOT_MULTIPLIER    = 0.20    # reduce lot sizes to 20% of normal
COAST_MIN_CONFIDENCE    = 10      # only 10/10 signals allowed
COAST_MIN_CONSENSUS     = 65      # convergence engine score threshold
COAST_DD_TIGHTEN_FACTOR = 0.60    # tighten daily DD limit to 60% of normal
COAST_ALLOWED_SOURCES   = {       # only these AUTO sources trade in coast mode
    "AUTO_CONVERGENCE",
    "AUTO_SMC",
}


@dataclass
class CoastModeStatus:
    active: bool
    reason: str
    lot_multiplier: float = 1.0
    min_confidence: int   = 4
    min_consensus: int    = 0
    gap_remaining_pct: float = 0.0
    gap_remaining_dollars: float = 0.0


class PropFirmFinisher:
    """
    Meta-layer above the ATO. Activates Coast Mode when near a prop firm
    profit target to protect gains during the final approach.
    """

    def __init__(self):
        self._coast_active: bool = False
        self._coast_entered_at: Optional[float] = None
        self._last_log_time: float = 0.0
        self._trades_in_coast: int = 0
        self._profit_locked_in_coast: float = 0.0

    def check_override(
        self,
        signal_source: str,
        signal_confidence: int,
        signal_action: str,
        convergence_score: int = 0,
        account_equity: float = 0.0,
    ) -> CoastModeStatus:
        """
        Called by risk_guard before sizing.
        Returns CoastModeStatus with override parameters.
        """
        # Only runs in CHALLENGE phase
        phase = getattr(config, "PROP_FIRM_PHASE", "PERSONAL")
        if phase != config.PropPhase.CHALLENGE:
            return CoastModeStatus(active=False, reason="Not in CHALLENGE phase")

        target_pct   = getattr(config, "CHALLENGE_PROFIT_TARGET_PCT", 8.0)
        current_pct  = getattr(config, "CHALLENGE_PROFIT_CURRENT_PCT", 0.0)
        gap_pct      = target_pct - current_pct

        # Calculate remaining dollar gap
        initial_bal  = getattr(config, "INITIAL_ACCOUNT_BALANCE", 10000.0)
        gap_dollars  = initial_bal * (gap_pct / 100.0)

        if gap_pct > COAST_TRIGGER_PCT:
            # Not near target yet
            if self._coast_active:
                self._exit_coast_mode(gap_pct)
            return CoastModeStatus(
                active=False, reason=f"Gap={gap_pct:.2f}% > trigger {COAST_TRIGGER_PCT}%",
                gap_remaining_pct=gap_pct, gap_remaining_dollars=gap_dollars,
            )

        # ── COAST MODE IS ACTIVE ─────────────────────────────────────────────
        if not self._coast_active:
            self._enter_coast_mode(gap_pct, gap_dollars)

        now = time.time()
        if now - self._last_log_time > 60:
            logger.info(
                "[Finisher] COAST MODE: need $%.0f (%.2f%%) | "
                "lots=%.0f%% | min_conf=%d | min_consensus=%d",
                gap_dollars, gap_pct,
                COAST_LOT_MULTIPLIER * 100,
                COAST_MIN_CONFIDENCE, COAST_MIN_CONSENSUS,
            )
            self._last_log_time = now

        # ── Block non-approved AUTO sources ───────────────────────────────────
        if signal_source.startswith("AUTO_") and signal_source not in COAST_ALLOWED_SOURCES:
            return CoastModeStatus(
                active=True,
                reason=f"COAST MODE: source {signal_source} not in approved list",
                lot_multiplier=0.0,  # 0.0 = blocked
                min_confidence=COAST_MIN_CONFIDENCE,
                gap_remaining_pct=gap_pct,
                gap_remaining_dollars=gap_dollars,
            )

        # ── Confidence gate ────────────────────────────────────────────────────
        if signal_confidence < COAST_MIN_CONFIDENCE:
            return CoastModeStatus(
                active=True,
                reason=f"COAST MODE: confidence {signal_confidence}/10 < {COAST_MIN_CONFIDENCE} required",
                lot_multiplier=0.0,
                min_confidence=COAST_MIN_CONFIDENCE,
                gap_remaining_pct=gap_pct,
                gap_remaining_dollars=gap_dollars,
            )

        # ── Consensus gate ─────────────────────────────────────────────────────
        if convergence_score < COAST_MIN_CONSENSUS:
            return CoastModeStatus(
                active=True,
                reason=f"COAST MODE: consensus {convergence_score} < {COAST_MIN_CONSENSUS} required",
                lot_multiplier=0.0,
                min_confidence=COAST_MIN_CONFIDENCE,
                gap_remaining_pct=gap_pct,
                gap_remaining_dollars=gap_dollars,
            )

        # ── APPROVED: return coast-mode parameters ────────────────────────────
        self._trades_in_coast += 1
        return CoastModeStatus(
            active=True,
            reason=f"COAST MODE APPROVED: conf={signal_confidence}/10 consensus={convergence_score}",
            lot_multiplier=COAST_LOT_MULTIPLIER,
            min_confidence=COAST_MIN_CONFIDENCE,
            min_consensus=COAST_MIN_CONSENSUS,
            gap_remaining_pct=gap_pct,
            gap_remaining_dollars=gap_dollars,
        )

    def record_coast_trade_result(self, pnl: float):
        """Call when a trade opened during coast mode closes."""
        if self._coast_active:
            self._profit_locked_in_coast += pnl
            logger.info(
                "[Finisher] Coast trade closed: $%.2f | session total: $%.2f",
                pnl, self._profit_locked_in_coast,
            )

    def get_status_line(self) -> str:
        phase = getattr(config, "PROP_FIRM_PHASE", "PERSONAL")
        if phase != "CHALLENGE":
            return "Finisher: PERSONAL mode (inactive)"
        target  = getattr(config, "CHALLENGE_PROFIT_TARGET_PCT", 8.0)
        current = getattr(config, "CHALLENGE_PROFIT_CURRENT_PCT", 0.0)
        gap     = target - current
        status  = "COAST" if self._coast_active else "NORMAL"
        return (
            f"Finisher: {status} | target={target}% current={current}% "
            f"gap={gap:.2f}% trigger={COAST_TRIGGER_PCT}%"
        )

    def get_tightened_dd_limit(self) -> Optional[float]:
        """
        Returns a tightened DD limit when in coast mode, or None if not active.
        risk_guard can use this to override config.DAILY_DRAWDOWN_LIMIT_PCT.
        """
        if not self._coast_active:
            return None
        return config.DAILY_DRAWDOWN_LIMIT_PCT * COAST_DD_TIGHTEN_FACTOR

    def _enter_coast_mode(self, gap_pct: float, gap_dollars: float):
        self._coast_active    = True
        self._coast_entered_at = time.time()
        logger.warning(
            "[Finisher] *** COAST MODE ACTIVATED *** "
            "Need %.2f%% ($%.0f) more | Slashing lots to %.0f%% | "
            "Min confidence raised to %d/10",
            gap_pct, gap_dollars, COAST_LOT_MULTIPLIER * 100, COAST_MIN_CONFIDENCE,
        )
        try:
            db_manager.log_audit("COAST_MODE_ENTERED", {
                "gap_pct": round(gap_pct, 3),
                "gap_dollars": round(gap_dollars, 2),
                "lot_multiplier": COAST_LOT_MULTIPLIER,
                "min_confidence": COAST_MIN_CONFIDENCE,
            })
            from utils.notifier import notify
            notify(
                f"🏁 *COAST MODE ACTIVATED*\n\n"
                f"Challenge target: almost there!\n"
                f"Gap remaining: `{gap_pct:.2f}%` (${gap_dollars:.0f})\n"
                f"Lot sizes slashed to `{COAST_LOT_MULTIPLIER*100:.0f}%`\n"
                f"Min confidence: `{COAST_MIN_CONFIDENCE}/10` only\n"
                f"_Protect the gains. Cross the line._"
            )
        except Exception:
            pass

    def _exit_coast_mode(self, gap_pct: float):
        if not self._coast_active:
            return
        self._coast_active = False
        duration = time.time() - (self._coast_entered_at or time.time())
        logger.info(
            "[Finisher] Coast mode exited: gap now %.2f%% | "
            "was active %.0fmin | trades=%d profit=$%.2f",
            gap_pct, duration / 60, self._trades_in_coast, self._profit_locked_in_coast,
        )
        try:
            db_manager.log_audit("COAST_MODE_EXITED", {
                "gap_pct": round(gap_pct, 3),
                "duration_mins": round(duration / 60, 1),
                "trades_in_coast": self._trades_in_coast,
                "profit_locked": round(self._profit_locked_in_coast, 2),
            })
        except Exception:
            pass
        self._trades_in_coast = 0
        self._profit_locked_in_coast = 0.0
        self._coast_entered_at = None


# Module-level singleton
prop_firm_finisher = PropFirmFinisher()
