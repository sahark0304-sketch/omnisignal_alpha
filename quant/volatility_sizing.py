"""
quant/volatility_sizing.py — OmniSignal Alpha v2.0
Pillar 5: ATR-based position sizing + Kelly Criterion

Sizing priority:
  1. If Kelly data available for source → use fractional Kelly risk %
  2. Else → use base RISK_PER_TRADE_PCT
  3. ATR validates/replaces SL when it's dangerously tight
  4. Alpha Ranker multiplier applied last (Pillar 6)
  5. Hard cap: never exceed KELLY_MAX_RISK_PCT regardless
"""

from dataclasses import dataclass
from typing import Optional, Dict
import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SizingResult:
    lot_size: float
    risk_amount: float
    risk_pct: float
    sl_pips: float
    method: str                      # "BASE" | "KELLY" | "ATR" | "ATR+KELLY"
    kelly_f: Optional[float] = None
    atr_value: Optional[float] = None
    alpha_multiplier: float = 1.0
    notes: str = ""


def calculate_lot_size(
    equity: float,
    entry: float,
    stop_loss: float,
    pip_size: float,
    pip_value_per_lot: float,
    atr_value: float = 0.0,
    source_stats: Optional[Dict] = None,
    alpha_multiplier: float = 1.0,
) -> SizingResult:
    """
    Unified position sizing. All inputs in price space; pip_size converts to pips.
    """
    base_risk_pct = config.RISK_PER_TRADE_PCT
    kelly_f = None
    method = "BASE"

    # ── Kelly Criterion ──────────────────────────────────────────────────────
    if (config.KELLY_ENABLED and source_stats and
            source_stats.get("total", 0) >= config.KELLY_MIN_TRADES):
        kelly_f = _kelly(
            win_rate = source_stats["win_rate"],
            avg_win  = source_stats["avg_win"],
            avg_loss = source_stats["avg_loss"],
        )
        if kelly_f > 0:
            kelly_pct = min(kelly_f * 100.0, config.KELLY_MAX_RISK_PCT)
            if kelly_pct > base_risk_pct:
                base_risk_pct = kelly_pct
                method = "KELLY"
                logger.debug(f"[Sizing] Kelly f*={kelly_f:.4f} → risk={base_risk_pct:.2f}%")

    # ── SL distance ──────────────────────────────────────────────────────────
    sl_dist = abs(entry - stop_loss)
    atr_notes = ""

    if config.VOLATILITY_SIZING_ENABLED and atr_value > 0 and sl_dist > 0:
        ideal_sl = atr_value * config.ATR_MULTIPLIER
        if sl_dist < atr_value * 0.5:
            # SL is unrealistically tight — will get whipsawed
            logger.warning(
                f"[Sizing] SL {sl_dist:.5f} < 0.5×ATR {atr_value*0.5:.5f}. "
                f"Replacing with ATR×{config.ATR_MULTIPLIER}={ideal_sl:.5f}"
            )
            sl_dist = ideal_sl
            atr_notes = f"SL widened to ATR×{config.ATR_MULTIPLIER}"
            method = "ATR" if method == "BASE" else "ATR+" + method
        elif sl_dist > atr_value * 3.0:
            atr_notes = f"⚠ SL wider than 3×ATR — poor R:R"

    if sl_dist <= 0:
        return SizingResult(0.01, 0.0, 0.0, 0.0, "ERROR", notes="Zero SL distance")

    if pip_size <= 0:
        logger.warning("[Sizing] pip_size=%.6f invalid, defaulting to 0.0001", pip_size)
        pip_size = 0.0001
    sl_pips = sl_dist / pip_size
    if sl_pips <= 0:
        return SizingResult(0.01, 0.0, 0.0, 0.0, "ERROR", notes="Zero pip distance")

    if pip_value_per_lot <= 0:
        logger.warning("[Sizing] pip_value_per_lot=%.4f invalid, defaulting to 10.0", pip_value_per_lot)
        pip_value_per_lot = 10.0

    # ── Raw lot ───────────────────────────────────────────────────────────────
    risk_amount = equity * (base_risk_pct / 100.0)
    raw_lot = risk_amount / (sl_pips * pip_value_per_lot)

    # ── Alpha Ranker multiplier ───────────────────────────────────────────────
    raw_lot *= alpha_multiplier

    # ── Hard cap ─────────────────────────────────────────────────────────────
    max_risk   = equity * (config.KELLY_MAX_RISK_PCT / 100.0)
    max_lot    = max_risk / (sl_pips * pip_value_per_lot)
    lot        = round(min(raw_lot, max_lot, 100.0), 2)
    if equity >= 5000:
        lot = max(lot, 0.05)
    elif equity >= 3000:
        lot = max(lot, 0.03)
    else:
        lot = max(lot, 0.01)

    actual_risk     = lot * sl_pips * pip_value_per_lot
    actual_risk_pct = (actual_risk / equity) * 100.0

    logger.info(
        f"[Sizing] method={method} lots={lot} risk=${actual_risk:.2f} "
        f"({actual_risk_pct:.2f}%) SL={sl_pips:.1f}pips alpha={alpha_multiplier}× {atr_notes}"
    )
    return SizingResult(
        lot_size=lot, risk_amount=actual_risk, risk_pct=actual_risk_pct,
        sl_pips=sl_pips, method=method, kelly_f=kelly_f,
        atr_value=atr_value, alpha_multiplier=alpha_multiplier, notes=atr_notes,
    )


def _kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion: f* = (W×R - (1-W)) / R, where R = avg_win/avg_loss.
    Returns fractional Kelly (×KELLY_FRACTION). Returns 0 if edge is negative.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    R = avg_win / avg_loss
    W = win_rate
    full_kelly = (W * R - (1 - W)) / R
    frac = full_kelly * config.KELLY_FRACTION
    logger.debug(f"[Kelly] W={W:.2%} R={R:.2f} f*={full_kelly:.4f} frac={frac:.4f}")
    return max(0.0, frac)


def get_source_stats(source: str) -> Optional[Dict]:
    """Fetch win-rate & avg P&L per trade for Kelly calculation."""
    try:
        from database import db_manager
        with db_manager.get_connection() as conn:
            rows = conn.execute("""
                SELECT t.pnl
                FROM trades t
                JOIN signals s ON t.signal_id = s.id
                WHERE s.source = ? AND t.status = 'CLOSED' AND t.pnl IS NOT NULL
                ORDER BY t.close_time DESC
                LIMIT ?
            """, (source, config.KELLY_MIN_TRADES * 2)).fetchall()

        if not rows or len(rows) < 10:
            return None

        pnls   = [float(r[0]) for r in rows]
        wins   = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        return {
            "total":    len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_win":  sum(wins) / max(len(wins), 1),
            "avg_loss": sum(losses) / max(len(losses), 1),
        }
    except Exception as e:
        logger.debug(f"[Sizing] Could not load source stats: {e}")
        return None
