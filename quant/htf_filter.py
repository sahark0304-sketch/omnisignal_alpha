"""
quant/htf_filter.py -- OmniSignal Alpha v3.3
Institutional Proprietary Edge Module

Three predatory mechanisms that exploit microstructure data retail
bots cannot access or compute:

1. ORDER FLOW TOXICITY FILTER
   Replaces lagging M15 EMA with real-time tick-level adverse selection
   detection.  Computes two independent toxicity metrics:

   A) LIQUIDITY VOID:  price_velocity / sqrt(volume)
      Fast price movement on thin volume = no counterparties = price
      falling through air.  Classic flash-crash signature.

   B) TOXIC DUMP:  CVD_acceleration * abs(price_displacement)
      Massive negative CVD + fast displacement = institutional
      liquidation event.  Retail is the exit liquidity.

   Toxicity score = max(void_score, dump_score) normalized to [0, 1].
   When score >= TOXICITY_THRESHOLD, the detected direction is flagged.

2. SIGNAL INVERSION ENGINE
   When the Toxicity Filter detects SEVERE conditions AND a scanner
   fires a counter-trend signal, the system does NOT just block it.
   It INVERTS the signal: BUY becomes SELL, SELL becomes BUY.
   The SL/TP are recalculated from the inverted entry.

   Rationale: if AUTO_PULLBACK fires a BUY into a crash, retail
   traders are doing the same.  Their stops cluster below recent
   lows.  When those stops get hit, they become market SELL orders
   that accelerate the move.  We ride that cascade.

   Inversion only triggers when toxicity >= INVERSION_THRESHOLD (0.8).

3. GEOMETRIC R:R PRESERVATION
   When the SL is widened to meet the ATR floor, the TP is scaled by
   the exact same coefficient to preserve the original R:R ratio.

   coefficient = new_sl_dist / original_sl_dist
   new_tp1 = entry + (original_tp1_dist * coefficient)
   new_tp2 = entry + (original_tp2_dist * coefficient)

4. EXECUTION DEDUP (unchanged from v3.2)
"""

import asyncio
import time
from typing import Tuple, Optional, Dict
import numpy as np
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from utils.logger import get_logger

logger = get_logger(__name__)

TOXICITY_WINDOW_SECS   = 60
TOXICITY_THRESHOLD     = 0.60
INVERSION_THRESHOLD    = 0.80
MIN_TICKS_FOR_TOXICITY = 50
MIN_GAP_SECS           = 90
MIN_ATR_MULT           = 0.5
MAX_RR_MULTIPLIER      = 2.5

_last_executions: dict = {}
_toxicity_state: Dict[str, dict] = {}
_inversion_count: int = 0

# Toxicity-adaptive sizing coefficients
INVERSION_SIZE_BOOST   = 1.35   # 35% size increase on inverted signals (edge from cascade)
UNCERTAIN_DAMPENER     = 0.30   # reduce size by score*30% when toxicity is nonzero but below threshold
CASCADE_ACCEL_BOOST    = 1.50   # extra boost when CVD acceleration confirms cascade momentum
TOXICITY_POLL_SECS     = 3.0    # background monitor polling interval


def assess_toxicity(
    symbol: str,
    pip_size: float = 0.01,
) -> dict:
    """
    Real-time Order Flow Toxicity assessment.
    Returns {score, direction, void_score, dump_score, displacement, cvd_accel}.
    """
    result = {
        "score": 0.0, "direction": None,
        "void_score": 0.0, "dump_score": 0.0,
        "displacement": 0.0, "cvd_accel": 0.0,
    }

    if mt5 is None:
        return result

    try:
        tick_from = datetime.now() - timedelta(seconds=TOXICITY_WINDOW_SECS)
        ticks = mt5.copy_ticks_from(symbol, tick_from, 8000, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) < MIN_TICKS_FOR_TOXICITY:
            return result

        prices = (ticks['bid'] + ticks['ask']) / 2.0
        try:
            volumes = ticks['volume_real'].astype(float)
        except (ValueError, KeyError):
            volumes = ticks['volume'].astype(float)
        flags = ticks['flags'].astype(int)

        displacement = (prices[-1] - prices[0]) / pip_size
        total_vol = np.sum(volumes)

        if total_vol < 1:
            return result

        elapsed = max(1, len(ticks))
        price_velocity = abs(displacement) / (TOXICITY_WINDOW_SECS / 60.0)
        vol_per_tick = total_vol / elapsed

        void_raw = price_velocity / max(np.sqrt(vol_per_tick), 0.1)
        void_score = min(1.0, void_raw / 50.0)

        buy_mask = (flags & 0x20) != 0
        sell_mask = (flags & 0x40) != 0
        neutral = ~(buy_mask | sell_mask)
        buy_vol = np.where(buy_mask, volumes, np.where(neutral, volumes * 0.5, 0.0))
        sell_vol = np.where(sell_mask, volumes, np.where(neutral, volumes * 0.5, 0.0))
        cvd = np.cumsum(buy_vol - sell_vol)

        half = len(cvd) // 2
        if half > 5:
            cvd_accel = (np.mean(cvd[half:]) - np.mean(cvd[:half])) / max(total_vol * 0.01, 0.1)
        else:
            cvd_accel = 0.0

        dump_raw = abs(cvd_accel) * abs(displacement) / 1000.0
        dump_score = min(1.0, dump_raw / 3.0)

        score = max(void_score, dump_score)

        direction = None
        if score >= TOXICITY_THRESHOLD:
            if displacement < 0 and (cvd_accel < 0 or void_score > dump_score):
                direction = "BEARISH"
            elif displacement > 0 and (cvd_accel > 0 or void_score > dump_score):
                direction = "BULLISH"

        result = {
            "score": round(score, 3),
            "direction": direction,
            "void_score": round(void_score, 3),
            "dump_score": round(dump_score, 3),
            "displacement": round(displacement, 1),
            "cvd_accel": round(cvd_accel, 2),
        }

        _toxicity_state[symbol] = result

        if score >= TOXICITY_THRESHOLD and direction:
            logger.warning(
                f"[Toxicity] {direction} toxicity detected | "
                f"score={score:.2f} (void={void_score:.2f} dump={dump_score:.2f}) | "
                f"disp={displacement:+.0f}p cvd_accel={cvd_accel:+.2f}"
            )

    except Exception as e:
        logger.debug(f"[Toxicity] Assessment error: {e}")

    return result


def check_toxicity_gate(
    symbol: str,
    action: str,
    pip_size: float = 0.01,
) -> Tuple[bool, str, Optional[str]]:
    """
    Check if a signal should be blocked or inverted by the toxicity filter.

    Returns:
        (allowed, reason, inversion_action)
        - allowed=True, inversion_action=None: proceed normally
        - allowed=False, inversion_action=None: hard block
        - allowed=False, inversion_action="BUY"/"SELL": invert to this direction
    """
    cached = _toxicity_state.get(symbol)
    if cached and cached.get("score", 0) > 0:
        tox = cached
    else:
        tox = assess_toxicity(symbol, pip_size)

    if tox["score"] < TOXICITY_THRESHOLD or tox["direction"] is None:
        return True, "", None

    is_counter_trend = (
        (tox["direction"] == "BEARISH" and action == "BUY") or
        (tox["direction"] == "BULLISH" and action == "SELL")
    )

    if not is_counter_trend:
        return True, "", None

    if tox["score"] >= INVERSION_THRESHOLD:
        inverted = "SELL" if action == "BUY" else "BUY"
        reason = (
            f"SIGNAL INVERSION: {action}->{inverted} | "
            f"Toxicity={tox['score']:.2f} ({tox['direction']}) | "
            f"void={tox['void_score']:.2f} dump={tox['dump_score']:.2f} | "
            f"Riding trapped {action}ers' stop cascade"
        )
        global _inversion_count
        _inversion_count += 1
        return False, reason, inverted

    reason = (
        f"Toxicity block: {action} blocked in {tox['direction']} regime | "
        f"score={tox['score']:.2f} (void={tox['void_score']:.2f} "
        f"dump={tox['dump_score']:.2f}) | disp={tox['displacement']:+.0f}p"
    )
    return False, reason, None


def compute_sl_floor_with_rr(
    symbol: str,
    entry_price: float,
    signal_sl: float,
    signal_tp1: Optional[float],
    signal_tp2: Optional[float],
    signal_tp3: Optional[float],
    action: str,
    pip_size: float = 0.01,
) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    """
    ATR SL floor with Geometric R:R Preservation.

    If SL is widened, TP1/TP2/TP3 are scaled by the same coefficient
    so the original R:R ratio is strictly preserved.

    Returns (adjusted_sl, adjusted_tp1, adjusted_tp2, adjusted_tp3).
    """
    if mt5 is None:
        return signal_sl, signal_tp1, signal_tp2, signal_tp3

    try:
        m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
        if m5 is None or len(m5) < 14:
            return signal_sl, signal_tp1, signal_tp2, signal_tp3

        highs = m5[-14:]['high'].astype(float)
        lows = m5[-14:]['low'].astype(float)
        closes = m5[-15:-1]['close'].astype(float)

        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - closes), np.abs(lows - closes))
        )
        atr = np.mean(tr)

        min_sl_dist = atr * MIN_ATR_MULT
        original_sl_dist = abs(entry_price - signal_sl)

        if original_sl_dist >= min_sl_dist or original_sl_dist < pip_size:
            return signal_sl, signal_tp1, signal_tp2, signal_tp3

        coefficient = min_sl_dist / original_sl_dist

        # v3.8 FIX: Hard cap - reject if ATR floor needs >1.5x SL widening
        try:
            import config as _cfg
            _max_rr = getattr(_cfg, "MAX_RR_MULTIPLIER", MAX_RR_MULTIPLIER)
        except Exception:
            _max_rr = MAX_RR_MULTIPLIER
        if coefficient > _max_rr:
            logger.warning(
                "[Toxicity] REJECT_ATR_TOO_WIDE: %s %s coeff=%.2fx > cap %.1fx "
                "SL:%.0fp needs %.0fp ATR=%.0fp",
                symbol, action, coefficient, _max_rr,
                original_sl_dist / pip_size, min_sl_dist / pip_size, atr / pip_size,
            )
            from database import db_manager
            db_manager.log_audit("REJECT_ATR_TOO_WIDE", {
                "symbol": symbol, "action": action,
                "coefficient": round(coefficient, 3),
                "sl_pips": round(original_sl_dist / pip_size, 1),
                "atr_pips": round(atr / pip_size, 1),
            })
            return None, None, None, None

        if action == "BUY":
            new_sl = round(entry_price - min_sl_dist, 2)
            new_tp1 = round(entry_price + abs(signal_tp1 - entry_price) * coefficient, 2) if signal_tp1 else None
            new_tp2 = round(entry_price + abs(signal_tp2 - entry_price) * coefficient, 2) if signal_tp2 else None
            new_tp3 = round(entry_price + abs(signal_tp3 - entry_price) * coefficient, 2) if signal_tp3 else None
        else:
            new_sl = round(entry_price + min_sl_dist, 2)
            new_tp1 = round(entry_price - abs(entry_price - signal_tp1) * coefficient, 2) if signal_tp1 else None
            new_tp2 = round(entry_price - abs(entry_price - signal_tp2) * coefficient, 2) if signal_tp2 else None
            new_tp3 = round(entry_price - abs(entry_price - signal_tp3) * coefficient, 2) if signal_tp3 else None

        logger.info(
            f"[Toxicity] R:R preserved SL/TP scaling: {symbol} {action} | "
            f"coeff={coefficient:.2f}x | SL:{original_sl_dist/pip_size:.0f}p->"
            f"{min_sl_dist/pip_size:.0f}p | ATR={atr/pip_size:.0f}p"
        )

        from database import db_manager
        db_manager.log_audit("RR_PRESERVATION", {
            "symbol": symbol, "action": action,
            "coefficient": round(coefficient, 3),
            "old_sl_pips": round(original_sl_dist / pip_size, 1),
            "new_sl_pips": round(min_sl_dist / pip_size, 1),
            "atr_pips": round(atr / pip_size, 1),
        })

        return new_sl, new_tp1, new_tp2, new_tp3

    except Exception as e:
        logger.debug(f"[Toxicity] SL floor error: {e}")

    return signal_sl, signal_tp1, signal_tp2, signal_tp3


def invert_signal_levels(
    entry_price: float,
    original_sl: float,
    original_tp1: Optional[float],
    action: str,
    inverted_action: str,
    pip_size: float = 0.01,
) -> Tuple[float, float, Optional[float]]:
    """
    Recalculate SL/TP for an inverted signal.

    Uses the original SL distance as the new TP distance (we ride
    the stop cascade), and sets the new SL at a tighter level
    (the old TP1 area acts as invalidation for the inversion thesis).
    """
    original_sl_dist = abs(entry_price - original_sl)
    original_tp_dist = abs(original_tp1 - entry_price) if original_tp1 else original_sl_dist

    new_tp_dist = max(original_sl_dist, original_tp_dist) * 1.2
    new_sl_dist = original_sl_dist * 0.7

    if inverted_action == "SELL":
        new_sl = round(entry_price + new_sl_dist, 2)
        new_tp = round(entry_price - new_tp_dist, 2)
    else:
        new_sl = round(entry_price - new_sl_dist, 2)
        new_tp = round(entry_price + new_tp_dist, 2)

    return new_sl, new_tp, None


def check_execution_dedup(
    symbol: str,
    action: str,
) -> Tuple[bool, str]:
    """Check if a trade was already opened in the same direction recently."""
    key = f"{symbol}_{action}"
    now = time.time()

    if key in _last_executions:
        elapsed = now - _last_executions[key]
        try:
            from quant.trade_orchestrator import get_scaled_cooldown
            _eff_gap = get_scaled_cooldown(MIN_GAP_SECS, "dedup")
        except ImportError:
            _eff_gap = MIN_GAP_SECS
        if elapsed < _eff_gap:
            remaining = int(_eff_gap - elapsed)
            return False, (
                f"Execution dedup: {symbol} {action} already opened "
                f"{elapsed:.0f}s ago ({remaining}s cooloff remaining)"
            )
    return True, ""


def register_execution(symbol: str, action: str):
    """Call after a trade is successfully placed."""
    _last_executions[f"{symbol}_{action}"] = time.time()


def get_current_toxicity(symbol: str = "XAUUSD") -> dict:
    """Get the most recent toxicity assessment for a symbol."""
    return _toxicity_state.get(symbol, {"score": 0.0, "direction": None})



def get_sizing_coefficient(
    symbol: str,
    action: str,
    was_inverted: bool = False,
) -> Tuple[float, str]:
    """
    Returns (multiplier, reason) for toxicity-adaptive position sizing.

    Three regimes:
    A) Inverted signal (was_inverted=True):
       Boost by INVERSION_SIZE_BOOST.  If CVD acceleration confirms
       the cascade, apply CASCADE_ACCEL_BOOST instead.
       Rationale: we have asymmetric information edge --- retail is
       trapped, their stops are our fuel.

    B) Normal signal, nonzero toxicity but below threshold:
       Dampen by (1 - score * UNCERTAIN_DAMPENER).
       Rationale: elevated toxicity = uncertain regime.  Reduce
       exposure proportionally to the uncertainty.

    C) Clean regime (toxicity near zero):
       Return 1.0 --- standard sizing.
    """
    tox = _toxicity_state.get(symbol, {"score": 0.0, "direction": None, "cvd_accel": 0.0})
    score = tox.get("score", 0.0)
    cvd_accel = tox.get("cvd_accel", 0.0)
    direction = tox.get("direction")

    if was_inverted:
        cascade_confirmed = (
            (action == "SELL" and cvd_accel < -1.0) or
            (action == "BUY" and cvd_accel > 1.0)
        )
        if cascade_confirmed:
            return CASCADE_ACCEL_BOOST, (
                f"CASCADE BOOST: {CASCADE_ACCEL_BOOST:.0%} | "
                f"CVD accel={cvd_accel:+.2f} confirms cascade in {action} direction"
            )
        return INVERSION_SIZE_BOOST, (
            f"INVERSION BOOST: {INVERSION_SIZE_BOOST:.0%} | "
            f"Riding trapped traders' stop cascade"
        )

    if 0.15 < score < TOXICITY_THRESHOLD:
        dampener = 1.0 - (score * UNCERTAIN_DAMPENER)
        return round(dampener, 3), (
            f"TOXICITY DAMPENER: {dampener:.0%} sizing | "
            f"score={score:.2f} (elevated but sub-threshold)"
        )

    return 1.0, ""


def get_inversion_stats() -> dict:
    """Return inversion performance metrics for monitoring."""
    return {
        "total_inversions": _inversion_count,
        "current_toxicity": {
            sym: {
                "score": s.get("score", 0),
                "direction": s.get("direction"),
                "void": s.get("void_score", 0),
                "dump": s.get("dump_score", 0),
            }
            for sym, s in _toxicity_state.items()
        },
    }


class ToxicityMonitor:
    """
    Background toxicity pre-computation engine.

    Runs every TOXICITY_POLL_SECS, continuously updating _toxicity_state
    for all monitored symbols.  This means check_toxicity_gate() reads
    pre-computed state instead of blocking on MT5 tick fetches ---
    reducing gate latency from ~50ms to <0.1ms.
    """

    def __init__(self, symbols: list = None, poll_secs: float = None):
        self.symbols = symbols or ["XAUUSD"]
        self.poll_secs = poll_secs or TOXICITY_POLL_SECS
        self._running = False

    async def run(self):
        self._running = True
        logger.info(
            f"[ToxicityMonitor] Started | symbols={self.symbols} "
            f"poll={self.poll_secs}s"
        )
        while self._running:
            try:
                for symbol in self.symbols:
                    assess_toxicity(symbol)
            except Exception as e:
                logger.debug(f"[ToxicityMonitor] Poll error: {e}")
            await asyncio.sleep(self.poll_secs)

    def stop(self):
        self._running = False


toxicity_monitor = ToxicityMonitor()


def check_htf_trend_gate(
    symbol: str,
    action: str,
    pip_size: float = 0.01,
) -> Tuple[bool, str]:
    """
    v4.4: Multi-Timeframe Trend Gate.
    Checks M15 and M5 EMA structure to block counter-trend M1 signals.

    Uses EMA(20) vs EMA(50) crossover + price position relative to EMA(20)
    to determine the higher-timeframe structural trend.

    Returns (allowed, reason).
    """
    if mt5 is None:
        return True, ""

    try:
        m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 60)
        m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60)

        if m15 is None or len(m15) < 55 or m5 is None or len(m5) < 55:
            return True, ""

        def _ema(data, period):
            k = 2.0 / (period + 1)
            ema = np.empty(len(data))
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = data[i] * k + ema[i - 1] * (1 - k)
            return ema

        def _assess_trend(rates):
            closes = rates['close'].astype(float)
            ema20 = _ema(closes, 20)
            ema50 = _ema(closes, 50)

            current_price = closes[-1]
            ema20_now = ema20[-1]
            ema50_now = ema50[-1]

            ema20_above_50 = ema20_now > ema50_now
            price_above_ema20 = current_price > ema20_now
            ema_gap_pips = abs(ema20_now - ema50_now) / pip_size

            if ema20_above_50 and price_above_ema20 and ema_gap_pips > 5:
                return "BULLISH", ema_gap_pips
            elif not ema20_above_50 and not price_above_ema20 and ema_gap_pips > 5:
                return "BEARISH", ema_gap_pips
            else:
                return "NEUTRAL", ema_gap_pips

        m15_trend, m15_gap = _assess_trend(m15)
        m5_trend, m5_gap = _assess_trend(m5)

        both_bearish = m15_trend == "BEARISH" and m5_trend == "BEARISH"
        both_bullish = m15_trend == "BULLISH" and m5_trend == "BULLISH"
        combined_gap = m15_gap + m5_gap

        # v7.1: Softened HTF gate - only block STRONG disagreement
        # Research: HTF gate rejected 50 signals, 37 would have won (74% false reject)
        # New: block only when combined gap > 30 (strong trend), allow weak with warning
        _strong_threshold = 30
        _weak_threshold = 15

        if action == "BUY" and both_bearish:
            if combined_gap > _strong_threshold:
                reason = (
                    f"HTF TREND GATE: BUY blocked | M15={m15_trend} ({m15_gap:.0f}p) "
                    f"M5={m5_trend} ({m5_gap:.0f}p) | STRONG bearish ({combined_gap:.0f}p)"
                )
                logger.info(f"[HTF] {reason}")
                return False, reason
            elif combined_gap > _weak_threshold:
                reason = (
                    f"HTF_WEAK_AGAINST: BUY allowed with caution | M15={m15_trend} ({m15_gap:.0f}p) "
                    f"M5={m5_trend} ({m5_gap:.0f}p) | Weak bearish ({combined_gap:.0f}p) - lot penalty advised"
                )
                logger.info(f"[HTF] {reason}")
                return True, reason

        if action == "SELL" and both_bullish:
            if combined_gap > _strong_threshold:
                reason = (
                    f"HTF TREND GATE: SELL blocked | M15={m15_trend} ({m15_gap:.0f}p) "
                    f"M5={m5_trend} ({m5_gap:.0f}p) | STRONG bullish ({combined_gap:.0f}p)"
                )
                logger.info(f"[HTF] {reason}")
                return False, reason
            elif combined_gap > _weak_threshold:
                reason = (
                    f"HTF_WEAK_AGAINST: SELL allowed with caution | M15={m15_trend} ({m15_gap:.0f}p) "
                    f"M5={m5_trend} ({m5_gap:.0f}p) | Weak bullish ({combined_gap:.0f}p) - lot penalty advised"
                )
                logger.info(f"[HTF] {reason}")
                return True, reason

        return True, ""

    except Exception as e:
        logger.debug(f"[HTF] Trend gate error: {e}")
        return True, ""
