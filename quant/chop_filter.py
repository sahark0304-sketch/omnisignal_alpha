"""
quant/chop_filter.py — OmniSignal Alpha v6.2
Chop Regime Filter: Institutional Tradeability Gate
"""

import time
import math
import numpy as np
from typing import Tuple, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    _mt5_available = True
except ImportError:
    mt5 = None
    _mt5_available = False


CI_PERIOD           = 14
CI_CHOP_THRESHOLD   = 61.8
CI_TREND_THRESHOLD  = 38.2

WDR_LOOKBACK        = 10
WDR_CHOP_THRESHOLD  = 0.55

DCS_LOOKBACK        = 20
DCS_CHOP_THRESHOLD  = 0.55

W_CI                = 0.45
W_WDR               = 0.30
W_DCS               = 0.25

CHOP_BLOCK_THRESHOLD = 0.35
CHOP_WARN_THRESHOLD  = 0.50

CACHE_TTL_SECS      = 15
TIMEFRAME_STR       = "M5"

_cache: Dict[str, Tuple[float, dict, float]] = {}


def _compute_choppiness_index(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = CI_PERIOD,
) -> float:
    n = len(highs)
    if n < period + 1:
        return 50.0

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    recent_tr = tr[-period:]
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]

    sum_tr = float(np.sum(recent_tr))
    hh = float(np.max(recent_highs))
    ll = float(np.min(recent_lows))

    range_hl = hh - ll
    if range_hl <= 0 or sum_tr <= 0:
        return 50.0

    log_period = math.log10(period)
    if log_period <= 0:
        return 50.0

    ci = 100.0 * math.log10(sum_tr / range_hl) / log_period
    return float(np.clip(ci, 0.0, 100.0))


def _compute_wick_dominance(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = WDR_LOOKBACK,
) -> float:
    n = len(opens)
    if n < lookback:
        lookback = n
    if lookback < 3:
        return 0.5

    ratios = []
    for i in range(-lookback, 0):
        o = float(opens[i])
        h = float(highs[i])
        l = float(lows[i])
        c = float(closes[i])
        candle_range = h - l
        if candle_range < 1e-8:
            continue
        body_top = max(o, c)
        body_bot = min(o, c)
        upper_wick = h - body_top
        lower_wick = body_bot - l
        wick_total = upper_wick + lower_wick
        ratios.append(wick_total / candle_range)

    if not ratios:
        return 0.5
    return float(np.mean(ratios))


def _compute_directional_consistency(
    closes: np.ndarray,
    lookback: int = DCS_LOOKBACK,
) -> float:
    n = len(closes)
    if n < lookback + 1:
        lookback = n - 1
    if lookback < 5:
        return 0.5

    recent = closes[-(lookback + 1):]
    diffs = np.diff(recent)
    up_count = int(np.sum(diffs > 0))
    down_count = int(np.sum(diffs < 0))
    total = up_count + down_count
    if total == 0:
        return 0.5

    return max(up_count, down_count) / total


def _compute_tradeability_score(ci: float, wdr: float, dcs: float) -> float:
    ci_norm = 1.0 - (ci / 100.0)
    wdr_inv = 1.0 - wdr
    score = W_CI * ci_norm + W_WDR * wdr_inv + W_DCS * dcs
    return float(np.clip(score, 0.0, 1.0))


def check(symbol: str = "XAUUSD") -> Tuple[bool, float, Dict]:
    now = time.time()

    cached = _cache.get(symbol)
    if cached and (now - cached[2]) < CACHE_TTL_SECS:
        _, details, _ = cached
        score = details["tradeability_score"]
        tradeable = score >= CHOP_BLOCK_THRESHOLD
        return tradeable, score, details

    details = _compute(symbol)
    score = details["tradeability_score"]
    _cache[symbol] = (score, details, now)

    tradeable = score >= CHOP_BLOCK_THRESHOLD

    if not tradeable:
        logger.warning(
            "[ChopFilter] BLOCKED: %s tradeability=%.2f (CI=%.1f WDR=%.2f DCS=%.2f) "
            "threshold=%.2f",
            symbol, score, details["choppiness_index"],
            details["wick_dominance"], details["directional_consistency"],
            CHOP_BLOCK_THRESHOLD,
        )
    elif score < CHOP_WARN_THRESHOLD:
        logger.info(
            "[ChopFilter] WARNING: %s tradeability=%.2f (marginal)",
            symbol, score,
        )

    return tradeable, score, details


def get_lot_dampener(symbol: str = "XAUUSD") -> float:
    _, score, _ = check(symbol)
    if score >= CHOP_WARN_THRESHOLD:
        return 1.0
    if score >= CHOP_BLOCK_THRESHOLD:
        ratio = (score - CHOP_BLOCK_THRESHOLD) / (CHOP_WARN_THRESHOLD - CHOP_BLOCK_THRESHOLD)
        return max(0.50, ratio)
    return 0.0


def get_metrics(symbol: str = "XAUUSD") -> Dict:
    _, _, details = check(symbol)
    return details


def _compute(symbol: str) -> Dict:
    result = {
        "choppiness_index": 50.0,
        "wick_dominance": 0.50,
        "directional_consistency": 0.50,
        "tradeability_score": 0.50,
        "regime": "UNKNOWN",
        "ci_regime": "NEUTRAL",
        "tradeable": True,
    }

    if not _mt5_available or mt5 is None:
        return result

    try:
        tf_map = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
        tf = tf_map.get(TIMEFRAME_STR, mt5.TIMEFRAME_M5)
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, max(CI_PERIOD, DCS_LOOKBACK) + 5)

        if rates is None or len(rates) < CI_PERIOD + 1:
            return result

        opens = rates["open"].astype(np.float64)
        highs = rates["high"].astype(np.float64)
        lows = rates["low"].astype(np.float64)
        closes = rates["close"].astype(np.float64)

        ci = _compute_choppiness_index(highs, lows, closes, CI_PERIOD)
        wdr = _compute_wick_dominance(opens, highs, lows, closes, WDR_LOOKBACK)
        dcs = _compute_directional_consistency(closes, DCS_LOOKBACK)
        score = _compute_tradeability_score(ci, wdr, dcs)

        if ci > CI_CHOP_THRESHOLD:
            ci_regime = "CHOP"
        elif ci < CI_TREND_THRESHOLD:
            ci_regime = "TRENDING"
        else:
            ci_regime = "NEUTRAL"

        if score < CHOP_BLOCK_THRESHOLD:
            regime = "UNTRADEABLE"
        elif score < CHOP_WARN_THRESHOLD:
            regime = "MARGINAL"
        elif score >= 0.70:
            regime = "CLEAN_TREND"
        else:
            regime = "NORMAL"

        result = {
            "choppiness_index": round(ci, 2),
            "wick_dominance": round(wdr, 4),
            "directional_consistency": round(dcs, 4),
            "tradeability_score": round(score, 4),
            "regime": regime,
            "ci_regime": ci_regime,
            "tradeable": score >= CHOP_BLOCK_THRESHOLD,
        }

    except Exception as e:
        logger.error("[ChopFilter] Compute error for %s: %s", symbol, e)

    return result


def invalidate_cache(symbol: str = None):
    if symbol:
        _cache.pop(symbol, None)
    else:
        _cache.clear()
