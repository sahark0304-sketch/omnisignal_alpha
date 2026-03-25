"""
quant/convexity_engine.py -- OmniSignal Alpha v4.2
Breakthrough #4: Asymmetric Kelly Convexity Engine

Implements "House Money" profit buffer scaling for asymmetric convexity.
When today's unrealized P&L is positive, a portion of the daily profit is
treated as a risk-free buffer that allows more aggressive sizing on
high-conviction signals.

Mathematical basis:
  daily_profit = current_equity - opening_equity
  profit_fraction = daily_profit / opening_equity
  house_money_boost = 1.0 + (profit_fraction * AGGRESSION_FACTOR)
  effective_boost = min(house_money_boost, MAX_BOOST)

Activation conditions:
  - Daily P&L must be positive (we are playing with house money)
  - Signal confidence must be >= HIGH_CONVICTION_THRESHOLD
  - Boost is proportional to how much profit we have accumulated today
  - MAX_BOOST caps the maximum amplification (1.8x)
  - NEVER boosts when in drawdown (natural asymmetry)
"""

from typing import Tuple
from database import db_manager
from utils.logger import get_logger

logger = get_logger(__name__)

AGGRESSION_FACTOR           = 15.0
MAX_BOOST                   = 1.80
POST_WIN_DAMPENER           = 0.60  # multiply raw_boost when daily profit_pct > 5%
HARD_LOT_CEILING            = 0.20  # enforced in risk_guard
HIGH_CONVICTION_THRESHOLD   = 8
MIN_PROFIT_FOR_BOOST_PCT    = 0.15
CONVERGENCE_BONUS           = 0.40


def compute_convexity_boost(
    signal_confidence: int,
    current_equity: float,
    source: str = "",
    has_convergence: bool = False,
) -> Tuple[float, str]:
    """
    Compute the convexity boost for a trade based on daily profit buffer.
    Returns (boost_multiplier, reason_string). boost >= 1.0.
    """
    if signal_confidence < HIGH_CONVICTION_THRESHOLD:
        return 1.0, ""

    try:
        opening_equity = db_manager.get_opening_equity()
        if not opening_equity or opening_equity <= 0:
            return 1.0, ""
    except Exception:
        return 1.0, ""

    daily_profit = current_equity - opening_equity
    profit_pct = (daily_profit / opening_equity) * 100.0

    if profit_pct < MIN_PROFIT_FOR_BOOST_PCT:
        return 1.0, ""

    profit_fraction = daily_profit / opening_equity
    raw_boost = 1.0 + (profit_fraction * AGGRESSION_FACTOR)

    if has_convergence:
        raw_boost += CONVERGENCE_BONUS

    # v4.2: AMD Distribution phase = maximum conviction
    try:
        from quant.amd_engine import amd_engine, AMDPhase
        amd_state = amd_engine.get_state()
        if amd_state["phase"] == AMDPhase.DISTRIBUTION:
            amd_conf = amd_state.get("confidence", 0)
            amd_bonus = amd_conf * 0.50
            raw_boost += amd_bonus
            logger.info(
                "[Convexity] AMD DISTRIBUTION bonus +%.2f (conf=%.2f)",
                amd_bonus, amd_conf,
            )
    except Exception:
        pass

    # v4.3.2: After a big daily win, reduce aggression (house-money blow-up guard)
    if profit_pct > 5.0:
        raw_boost *= POST_WIN_DAMPENER

    boost = min(raw_boost, MAX_BOOST)
    boost = max(boost, 1.0)

    reason = (
        f"Convexity boost {boost:.2f}x | daily P&L: ${daily_profit:+.2f} "
        f"({profit_pct:+.1f}%) | AI:{signal_confidence}/10"
    )
    if has_convergence:
        reason += " | +convergence"

    logger.info(f"[Convexity] {reason}")
    db_manager.log_audit("CONVEXITY_BOOST", {
        "boost": round(boost, 3),
        "daily_profit": round(daily_profit, 2),
        "profit_pct": round(profit_pct, 2),
        "confidence": signal_confidence,
        "convergence": has_convergence,
        "source": source,
    })

    return boost, reason


def compute_institutional_scaling(
    signal_confidence: int,
    current_equity: float,
    source: str,
    has_convergence: bool,
    consensus_score: int = 0,
) -> Tuple[float, str]:
    """
    v4.2: Institutional lot scaling for maximum confluence setups.
    When consensus + AMD + AI confidence all align, apply aggressive sizing.
    """
    base_boost, base_reason = compute_convexity_boost(
        signal_confidence, current_equity, source, has_convergence,
    )

    if consensus_score < 60:
        return base_boost, base_reason

    # Full confluence: consensus 80+ AND high AI confidence
    if consensus_score >= 80 and signal_confidence >= 9:
        institutional_mult = 1.40
        reason = (
            f"INSTITUTIONAL SCALING: consensus={consensus_score} "
            f"AI={signal_confidence}/10 | base_boost={base_boost:.2f}x "
            f"* institutional={institutional_mult:.2f}x"
        )
        final = min(base_boost * institutional_mult, 2.50)
        logger.info("[Convexity] %s -> %.2fx", reason, final)
        db_manager.log_audit("INSTITUTIONAL_SCALING", {
            "consensus": consensus_score,
            "confidence": signal_confidence,
            "base_boost": round(base_boost, 3),
            "final": round(final, 3),
            "source": source,
        })
        return final, reason

    # Partial: consensus 60-79 with decent confidence
    if consensus_score >= 60 and signal_confidence >= 8:
        partial_mult = 1.15
        reason = (
            f"Partial institutional: consensus={consensus_score} "
            f"AI={signal_confidence}/10 | boost={base_boost:.2f}x * {partial_mult:.2f}x"
        )
        final = min(base_boost * partial_mult, 2.20)
        return final, reason

    # v4.3.2: Hard lot ceiling — no single trade should be so large that one loss wipes a session
    # The actual lot cap is enforced in risk_guard using this constant
    return base_boost, base_reason
