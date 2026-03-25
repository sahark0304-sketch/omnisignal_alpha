"""
quant/regime_detector.py -- Market Regime Detection Engine (OmniSignal Alpha v3.0)

Classifies XAUUSD into TRENDING, RANGING, or VOLATILE regimes using Hurst exponent,
ADX, ATR regime ratio, and price structure analysis, then adapts trading parameters.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum

import numpy as np

import config
from database import db_manager
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    _mt5_available = True
except ImportError:
    mt5 = None
    _mt5_available = False


# ---------------------------------------------------------------------------
#  Enums & Data Structures
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeState:
    regime: MarketRegime
    confidence: float
    trend_direction: str
    hurst: float
    adx: float
    atr_regime_ratio: float
    spread_z_score: float
    support_level: float
    resistance_level: float

    sizing_multiplier: float
    sl_atr_mult: float
    tp_atr_mult: float
    trail_atr_mult: float
    min_conviction: int

    raw_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_PIP_SIZE = 0.1
_CACHE_TTL_SECS = 30

_WEIGHT_HURST = 0.35
_WEIGHT_ADX = 0.30
_WEIGHT_ATR_REGIME = 0.20
_WEIGHT_STRUCTURE = 0.15

_HURST_TREND_THRESHOLD = 0.55
_HURST_RANGE_THRESHOLD = 0.48
_ADX_TREND_THRESHOLD = 25.0
_ADX_RANGE_THRESHOLD = 20.0
_ATR_VOLATILE_RATIO = 1.5
_ATR_EXPANDING_RATIO = 1.0
_ATR_CONTRACTING_RATIO = 0.7
_STRUCTURE_TREND_PCT = 0.60
_STRUCTURE_RANGE_WIDTH = 3.0

_REGIME_PARAMS = {
    MarketRegime.TRENDING: {
        "sizing_multiplier": 1.2,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "trail_atr_mult": 1.5,
        "min_conviction": 6,
        "allow_counter_trend": False,
        "min_ai_confidence": 6,
        "min_confluence_score": 4,
    },
    MarketRegime.RANGING: {
        "sizing_multiplier": 0.8,
        "sl_atr_mult": 1.0,
        "tp_atr_mult": 1.2,
        "trail_atr_mult": 0.8,
        "min_conviction": 7,
        "allow_counter_trend": True,
        "min_ai_confidence": 7,
        "min_confluence_score": 5,
    },
    MarketRegime.VOLATILE: {
        "sizing_multiplier": 0.5,
        "sl_atr_mult": 2.5,
        "tp_atr_mult": 3.5,
        "trail_atr_mult": 2.0,
        "min_conviction": 9,
        "allow_counter_trend": False,
        "min_ai_confidence": 9,
        "min_confluence_score": 6,
    },
    MarketRegime.UNKNOWN: {
        "sizing_multiplier": 0.6,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.0,
        "trail_atr_mult": 1.0,
        "min_conviction": 8,
        "allow_counter_trend": False,
        "min_ai_confidence": 8,
        "min_confluence_score": 5,
    },
}


# ---------------------------------------------------------------------------
#  MT5 Helpers
# ---------------------------------------------------------------------------

_TF_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _get_mt5_timeframe(tf_str: str):
    if mt5 is None:
        return None
    return getattr(mt5, _TF_MAP.get(tf_str, "TIMEFRAME_M15"), None)


def _fetch_rates(symbol: str, timeframe_str: str, count: int) -> Optional[np.ndarray]:
    """Fetch OHLCV bars from MT5 with full error handling."""
    if not _mt5_available or mt5 is None:
        return None
    try:
        tf = _get_mt5_timeframe(timeframe_str)
        if tf is None:
            return None
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) < 10:
            return None
        return rates
    except Exception as e:
        logger.warning(f"[RegimeDetector] MT5 fetch failed ({symbol} {timeframe_str}): {e}")
        return None


# ---------------------------------------------------------------------------
#  Statistical Computations (pure numpy)
# ---------------------------------------------------------------------------

def _compute_hurst(closes: np.ndarray, lookback: int = 100) -> float:
    """Rescaled Range (R/S) Hurst exponent on log returns."""
    if len(closes) < lookback:
        lookback = len(closes)
    if lookback < 20:
        return 0.50

    series = closes[-lookback:]
    mask = series[:-1] > 0
    if not np.all(mask):
        return 0.50
    log_returns = np.log(series[1:] / series[:-1])
    n = len(log_returns)
    if n < 16:
        return 0.50

    divisors = [k for k in [2, 4, 8, 16, 32] if n // k >= 8]
    if len(divisors) < 2:
        return 0.50

    log_sizes = []
    log_rs = []

    for k in divisors:
        chunk_size = n // k
        rs_accum = []
        for start in range(0, n - chunk_size + 1, chunk_size):
            chunk = log_returns[start:start + chunk_size]
            mean_c = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_c)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(chunk, ddof=0)
            if s > 1e-12:
                rs_accum.append(r / s)
        if rs_accum:
            log_sizes.append(np.log(chunk_size))
            log_rs.append(np.log(np.mean(rs_accum)))

    if len(log_sizes) < 2:
        return 0.50

    x = np.array(log_sizes)
    y = np.array(log_rs)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den < 1e-15:
        return 0.50

    h = float(num / den)
    return float(np.clip(h, 0.0, 1.0))


def _compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 14) -> Tuple[float, float, float]:
    """Wilder's ADX with +DI / -DI. Returns (adx, plus_di, minus_di)."""
    n = len(highs)
    if n < period + 1:
        return 0.0, 0.0, 0.0

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    smoothed_tr = float(np.sum(tr[1:period + 1]))
    smoothed_plus = float(np.sum(plus_dm[1:period + 1]))
    smoothed_minus = float(np.sum(minus_dm[1:period + 1]))

    dx_values = []
    last_plus_di = 0.0
    last_minus_di = 0.0

    for i in range(period, n):
        if i > period:
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + tr[i]
            smoothed_plus = smoothed_plus - (smoothed_plus / period) + plus_dm[i]
            smoothed_minus = smoothed_minus - (smoothed_minus / period) + minus_dm[i]

        if smoothed_tr > 1e-12:
            plus_di = 100.0 * smoothed_plus / smoothed_tr
            minus_di = 100.0 * smoothed_minus / smoothed_tr
        else:
            plus_di = 0.0
            minus_di = 0.0

        di_sum = plus_di + minus_di
        if di_sum > 1e-12:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0.0

        dx_values.append(dx)
        last_plus_di = plus_di
        last_minus_di = minus_di

    if not dx_values:
        return 0.0, 0.0, 0.0

    if len(dx_values) < period:
        adx_val = float(np.mean(dx_values))
    else:
        adx_val = float(np.mean(dx_values[:period]))
        for i in range(period, len(dx_values)):
            adx_val = (adx_val * (period - 1) + dx_values[i]) / period

    return adx_val, last_plus_di, last_minus_di


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """Wilder's ATR. Returns array of ATR values aligned with input length."""
    n = len(highs)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr_arr = np.zeros(n)
    if n < period:
        atr_arr[:] = np.mean(tr[:n]) if n > 0 else 0.0
        return atr_arr

    atr_arr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period

    for i in range(period - 1):
        atr_arr[i] = atr_arr[period - 1]

    return atr_arr


def _compute_atr_regime_ratio(highs: np.ndarray, lows: np.ndarray,
                              closes: np.ndarray, period: int = 14) -> float:
    """Current ATR(14) vs trailing 24-hour (96 M15 bars) average ATR."""
    atr_arr = _compute_atr(highs, lows, closes, period)
    current_atr = float(atr_arr[-1])

    lookback_24h = min(96, len(atr_arr))
    avg_atr_24h = float(np.mean(atr_arr[-lookback_24h:]))

    if avg_atr_24h < 1e-12:
        return 1.0
    return current_atr / avg_atr_24h


def _compute_spread_z_score(rates: np.ndarray, lookback: int = 50) -> float:
    """Z-score of current spread vs recent average spread."""
    if len(rates) < 10:
        return 0.0
    spreads = rates['spread'][-lookback:].astype(np.float64)
    if len(spreads) < 5:
        return 0.0
    current = float(spreads[-1])
    mean_s = float(np.mean(spreads))
    std_s = float(np.std(spreads, ddof=0))
    if std_s < 1e-12:
        return 0.0
    return (current - mean_s) / std_s


# ---------------------------------------------------------------------------
#  Support / Resistance Detection
# ---------------------------------------------------------------------------

def _detect_support_resistance(highs: np.ndarray, lows: np.ndarray,
                               lookback: int = 50,
                               window: int = 3) -> Tuple[float, float]:
    """
    Local minima in lows = support, local maxima in highs = resistance.
    A point is a local extremum if it is the min/max within `window` bars
    on each side.
    """
    n = len(highs)
    if n < lookback:
        lookback = n
    h = highs[-lookback:]
    l = lows[-lookback:]
    m = len(h)

    support = float(np.min(l))
    resistance = float(np.max(h))

    for i in range(m - 1, window - 1, -1):
        right = min(i + window + 1, m)
        left = max(0, i - window)
        if l[i] == np.min(l[left:right]):
            support = float(l[i])
            break

    for i in range(m - 1, window - 1, -1):
        right = min(i + window + 1, m)
        left = max(0, i - window)
        if h[i] == np.max(h[left:right]):
            resistance = float(h[i])
            break

    return support, resistance


# ---------------------------------------------------------------------------
#  Regime Scoring Functions
# ---------------------------------------------------------------------------

def _score_hurst(hurst: float) -> Dict[str, float]:
    trending = 0.0
    ranging = 0.0

    if hurst > _HURST_TREND_THRESHOLD:
        trending = min((hurst - _HURST_TREND_THRESHOLD) / 0.15, 1.0)
    elif hurst < _HURST_RANGE_THRESHOLD:
        ranging = min((_HURST_RANGE_THRESHOLD - hurst) / 0.15, 1.0)

    return {"trending": trending, "ranging": ranging, "volatile": 0.0}


def _score_adx(adx: float) -> Dict[str, float]:
    trending = 0.0
    ranging = 0.0

    if adx > _ADX_TREND_THRESHOLD:
        trending = min((adx - _ADX_TREND_THRESHOLD) / 15.0, 1.0)
    elif adx < _ADX_RANGE_THRESHOLD:
        ranging = min((_ADX_RANGE_THRESHOLD - adx) / 10.0, 1.0)

    return {"trending": trending, "ranging": ranging, "volatile": 0.0}


def _score_atr_regime(ratio: float) -> Dict[str, float]:
    trending = 0.0
    ranging = 0.0
    volatile = 0.0

    if ratio > _ATR_VOLATILE_RATIO:
        volatile = min((ratio - _ATR_VOLATILE_RATIO) / 0.5, 1.0)
    elif ratio > _ATR_EXPANDING_RATIO:
        trending = min((ratio - _ATR_EXPANDING_RATIO) / 0.5, 1.0)
    elif ratio < _ATR_CONTRACTING_RATIO:
        ranging = min((_ATR_CONTRACTING_RATIO - ratio) / 0.3, 1.0)

    return {"trending": trending, "ranging": ranging, "volatile": volatile}


def _score_price_structure(highs: np.ndarray, lows: np.ndarray,
                           atr_arr: np.ndarray) -> Dict[str, float]:
    """Higher-highs/higher-lows or lower-lows/lower-highs sequence analysis."""
    trending = 0.0
    ranging = 0.0

    n = min(20, len(highs))
    if n < 5:
        return {"trending": 0.0, "ranging": 0.0, "volatile": 0.0}

    h = highs[-n:]
    l = lows[-n:]

    hh_hl_count = 0
    ll_lh_count = 0
    for i in range(1, len(h)):
        if h[i] > h[i - 1] and l[i] > l[i - 1]:
            hh_hl_count += 1
        if h[i] < h[i - 1] and l[i] < l[i - 1]:
            ll_lh_count += 1

    max_seq = max(hh_hl_count, ll_lh_count)
    seq_pct = max_seq / (n - 1) if n > 1 else 0.0

    if seq_pct > _STRUCTURE_TREND_PCT:
        trending = min((seq_pct - _STRUCTURE_TREND_PCT) / 0.20, 1.0)

    lookback_sr = min(50, len(highs))
    h_sr = highs[-lookback_sr:]
    l_sr = lows[-lookback_sr:]
    range_width = float(np.max(h_sr) - np.min(l_sr))
    current_atr = float(atr_arr[-1]) if len(atr_arr) > 0 else 1.0
    if current_atr > 1e-12:
        normalized_width = range_width / current_atr
        if normalized_width < _STRUCTURE_RANGE_WIDTH:
            ranging = max(ranging, min((_STRUCTURE_RANGE_WIDTH - normalized_width) / 2.0, 1.0))

    return {"trending": trending, "ranging": ranging, "volatile": 0.0}


# ---------------------------------------------------------------------------
#  Trend Direction
# ---------------------------------------------------------------------------

def _determine_direction(plus_di: float, minus_di: float,
                         closes: np.ndarray) -> str:
    """Resolve trend direction from DI spread with EMA fallback."""
    di_diff = plus_di - minus_di
    if abs(di_diff) < 2.0:
        if len(closes) >= 20:
            ema_fast = float(np.mean(closes[-8:]))
            ema_slow = float(np.mean(closes[-20:]))
            if ema_fast > ema_slow:
                return "BULLISH"
            elif ema_fast < ema_slow:
                return "BEARISH"
        return "NEUTRAL"
    return "BULLISH" if di_diff > 0 else "BEARISH"


# ---------------------------------------------------------------------------
#  Regime Detector Class
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Stateful regime detector with TTL-based caching and audit logging."""

    def __init__(self):
        self._cache: Optional[RegimeState] = None
        self._cache_ts: float = 0.0
        self._last_regime: Optional[MarketRegime] = None

    def detect_regime(self, symbol: str = "XAUUSD",
                      timeframe_str: str = "M15") -> RegimeState:
        """
        Run the full regime detection pipeline.

        Uses the specified timeframe as primary (default M15).
        Caches results for _CACHE_TTL_SECS to avoid redundant computation.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < _CACHE_TTL_SECS:
            return self._cache

        if not _mt5_available:
            state = self._build_unknown_state()
            self._update_cache(state)
            logger.info(
                f"[RegimeDetector] {symbol}: UNKNOWN (MT5 not available)"
            )
            return state

        rates = _fetch_rates(symbol, timeframe_str, 200)
        if rates is None:
            state = self._build_unknown_state()
            self._update_cache(state)
            logger.info(
                f"[RegimeDetector] {symbol}: UNKNOWN (no data)"
            )
            return state

        try:
            state = self._run_pipeline(rates, symbol, timeframe_str)
        except Exception as e:
            logger.error(f"[RegimeDetector] Pipeline error: {e}")
            state = self._build_unknown_state()

        self._update_cache(state)
        self._log_regime(state, symbol)
        self._check_regime_change(state, symbol)
        return state

    def _run_pipeline(self, rates: np.ndarray, symbol: str,
                      timeframe_str: str) -> RegimeState:
        closes = rates['close'].astype(np.float64)
        highs = rates['high'].astype(np.float64)
        lows = rates['low'].astype(np.float64)

        hurst = _compute_hurst(closes, lookback=100)

        adx, plus_di, minus_di = _compute_adx(highs, lows, closes, period=14)

        atr_ratio = _compute_atr_regime_ratio(highs, lows, closes, period=14)
        atr_arr = _compute_atr(highs, lows, closes, period=14)

        spread_z = _compute_spread_z_score(rates, lookback=50)

        support, resistance = _detect_support_resistance(
            highs, lows, lookback=50, window=3
        )

        hurst_scores = _score_hurst(hurst)
        adx_scores = _score_adx(adx)
        atr_scores = _score_atr_regime(atr_ratio)
        structure_scores = _score_price_structure(highs, lows, atr_arr)

        trending_score = (
            _WEIGHT_HURST * hurst_scores["trending"]
            + _WEIGHT_ADX * adx_scores["trending"]
            + _WEIGHT_ATR_REGIME * atr_scores["trending"]
            + _WEIGHT_STRUCTURE * structure_scores["trending"]
        )
        ranging_score = (
            _WEIGHT_HURST * hurst_scores["ranging"]
            + _WEIGHT_ADX * adx_scores["ranging"]
            + _WEIGHT_ATR_REGIME * atr_scores["ranging"]
            + _WEIGHT_STRUCTURE * structure_scores["ranging"]
        )
        volatile_score = (
            _WEIGHT_ATR_REGIME * atr_scores["volatile"]
        )

        if spread_z > 2.0:
            volatile_score += 0.15 * min((spread_z - 2.0) / 2.0, 1.0)

        scores = {
            "trending": trending_score,
            "ranging": ranging_score,
            "volatile": volatile_score,
        }
        winner_key = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[winner_key] / total if total > 1e-12 else 0.0

        regime_map = {
            "trending": MarketRegime.TRENDING,
            "ranging": MarketRegime.RANGING,
            "volatile": MarketRegime.VOLATILE,
        }
        regime = regime_map.get(winner_key, MarketRegime.UNKNOWN)

        if total < 0.05:
            regime = MarketRegime.UNKNOWN
            confidence = 0.0

        direction = _determine_direction(plus_di, minus_di, closes)
        params = _REGIME_PARAMS[regime]

        raw_scores = {
            "hurst": round(hurst, 4),
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "atr_ratio": round(atr_ratio, 4),
            "spread_z": round(spread_z, 2),
            "trending_score": round(trending_score, 4),
            "ranging_score": round(ranging_score, 4),
            "volatile_score": round(volatile_score, 4),
            "hurst_trending": round(hurst_scores["trending"], 4),
            "hurst_ranging": round(hurst_scores["ranging"], 4),
            "adx_trending": round(adx_scores["trending"], 4),
            "adx_ranging": round(adx_scores["ranging"], 4),
            "atr_volatile": round(atr_scores["volatile"], 4),
            "structure_trending": round(structure_scores["trending"], 4),
            "structure_ranging": round(structure_scores["ranging"], 4),
            "support": round(support, 2),
            "resistance": round(resistance, 2),
        }

        return RegimeState(
            regime=regime,
            confidence=round(confidence, 4),
            trend_direction=direction,
            hurst=round(hurst, 4),
            adx=round(adx, 2),
            atr_regime_ratio=round(atr_ratio, 4),
            spread_z_score=round(spread_z, 2),
            support_level=round(support, 2),
            resistance_level=round(resistance, 2),
            sizing_multiplier=params["sizing_multiplier"],
            sl_atr_mult=params["sl_atr_mult"],
            tp_atr_mult=params["tp_atr_mult"],
            trail_atr_mult=params["trail_atr_mult"],
            min_conviction=params["min_conviction"],
            raw_scores=raw_scores,
        )

    def _build_unknown_state(self) -> RegimeState:
        params = _REGIME_PARAMS[MarketRegime.UNKNOWN]
        return RegimeState(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            trend_direction="NEUTRAL",
            hurst=0.50,
            adx=0.0,
            atr_regime_ratio=1.0,
            spread_z_score=0.0,
            support_level=0.0,
            resistance_level=0.0,
            sizing_multiplier=params["sizing_multiplier"],
            sl_atr_mult=params["sl_atr_mult"],
            tp_atr_mult=params["tp_atr_mult"],
            trail_atr_mult=params["trail_atr_mult"],
            min_conviction=params["min_conviction"],
            raw_scores={},
        )

    def _update_cache(self, state: RegimeState) -> None:
        self._cache = state
        self._cache_ts = time.time()

    def _log_regime(self, state: RegimeState, symbol: str) -> None:
        logger.info(
            f"[RegimeDetector] {symbol}: {state.regime.value} "
            f"(conf={state.confidence:.2f}, H={state.hurst:.2f}, "
            f"ADX={state.adx:.0f}, dir={state.trend_direction})"
        )

    def _check_regime_change(self, state: RegimeState, symbol: str) -> None:
        if self._last_regime is not None and self._last_regime != state.regime:
            try:
                db_manager.log_audit("REGIME_CHANGE", {
                    "symbol": symbol,
                    "from": self._last_regime.value,
                    "to": state.regime.value,
                    "confidence": state.confidence,
                    "hurst": state.hurst,
                    "adx": state.adx,
                    "atr_ratio": state.atr_regime_ratio,
                    "direction": state.trend_direction,
                })
            except Exception as e:
                logger.warning(f"[RegimeDetector] Audit log failed: {e}")
            logger.info(
                f"[RegimeDetector] REGIME CHANGE: "
                f"{self._last_regime.value} -> {state.regime.value} "
                f"({symbol}, conf={state.confidence:.2f})"
            )
        self._last_regime = state.regime

    def get_regime_summary(self) -> str:
        """One-line human-readable summary of the current cached regime."""
        if self._cache is None:
            return "Regime: not yet computed"
        s = self._cache
        return (
            f"Regime={s.regime.value} conf={s.confidence:.2f} "
            f"dir={s.trend_direction} H={s.hurst:.3f} ADX={s.adx:.0f} "
            f"ATR_ratio={s.atr_regime_ratio:.2f} S={s.support_level:.2f} "
            f"R={s.resistance_level:.2f}"
        )

    def invalidate_cache(self) -> None:
        """Force recomputation on the next detect_regime call."""
        self._cache = None
        self._cache_ts = 0.0


# ---------------------------------------------------------------------------
#  Public API Functions
# ---------------------------------------------------------------------------

def detect_regime(symbol: str = "XAUUSD",
                  timeframe_str: str = "M15") -> RegimeState:
    """
    Run full regime detection pipeline.

    Uses M15 as primary timeframe. Delegates to the module-level
    RegimeDetector singleton with built-in caching.
    """
    return regime_detector.detect_regime(symbol, timeframe_str)


def get_trading_params(regime_state: RegimeState) -> Dict:
    """
    Convert a RegimeState into concrete trading parameters.

    ATR-based pip values are computed from the regime multipliers and
    a fresh ATR fetch (or a sensible default if MT5 is unavailable).
    """
    current_atr_pips = _get_current_atr_pips()

    params = _REGIME_PARAMS.get(
        regime_state.regime, _REGIME_PARAMS[MarketRegime.UNKNOWN]
    )
    sl_pips = round(current_atr_pips * regime_state.sl_atr_mult, 1)
    tp1_pips = round(current_atr_pips * regime_state.tp_atr_mult * 0.6, 1)
    tp2_pips = round(current_atr_pips * regime_state.tp_atr_mult, 1)
    trail_activation_pips = round(current_atr_pips * regime_state.trail_atr_mult, 1)
    trail_step_pips = round(trail_activation_pips * 0.5, 1)

    return {
        "sizing_mult": regime_state.sizing_multiplier,
        "sl_pips": sl_pips,
        "tp1_pips": tp1_pips,
        "tp2_pips": tp2_pips,
        "trail_activation_pips": trail_activation_pips,
        "trail_step_pips": trail_step_pips,
        "min_ai_confidence": params["min_ai_confidence"],
        "min_confluence_score": params["min_confluence_score"],
        "allow_counter_trend": params["allow_counter_trend"],
        "regime": regime_state.regime.value,
        "direction": regime_state.trend_direction,
    }


def should_trade(regime_state: RegimeState, signal_action: str,
                 signal_confidence: int) -> Tuple[bool, str]:
    """
    Regime-aware trade filter.

    Returns (allowed, reason) based on current regime, signal direction,
    and signal confidence score.
    """
    regime = regime_state.regime
    direction = regime_state.trend_direction
    action_upper = signal_action.upper()

    if regime == MarketRegime.UNKNOWN:
        if signal_confidence >= 8:
            return True, "UNKNOWN regime but high confidence"
        return False, "UNKNOWN regime, insufficient confidence"

    if regime == MarketRegime.TRENDING:
        is_aligned = (
            (direction == "BULLISH" and action_upper == "BUY")
            or (direction == "BEARISH" and action_upper == "SELL")
        )
        if is_aligned:
            return True, f"Aligned with {direction} trend"
        if signal_confidence >= 9:
            return True, f"Counter-trend override (confidence={signal_confidence})"
        return (
            False,
            f"Counter-trend blocked: {action_upper} vs {direction} trend "
            f"(need confidence>=9, got {signal_confidence})",
        )

    if regime == MarketRegime.RANGING:
        current_price = _get_last_close(regime_state)
        support = regime_state.support_level
        resistance = regime_state.resistance_level
        range_width = resistance - support if resistance > support else 1.0
        boundary_zone = range_width * 0.20

        if action_upper == "BUY" and current_price <= support + boundary_zone:
            return True, f"BUY near support ({current_price:.2f} near {support:.2f})"
        if action_upper == "SELL" and current_price >= resistance - boundary_zone:
            return True, f"SELL near resistance ({current_price:.2f} near {resistance:.2f})"
        if signal_confidence >= 8:
            return True, f"Range mid-zone override (confidence={signal_confidence})"
        return (
            False,
            f"RANGING: {action_upper} not near boundary "
            f"(price={current_price:.2f}, S={support:.2f}, R={resistance:.2f})",
        )

    if regime == MarketRegime.VOLATILE:
        if signal_confidence >= regime_state.min_conviction:
            return True, f"VOLATILE but ultra-high conviction ({signal_confidence})"
        return (
            False,
            f"VOLATILE regime: blocked (need confidence>={regime_state.min_conviction}, "
            f"got {signal_confidence})",
        )

    return False, "Unhandled regime state"


# ---------------------------------------------------------------------------
#  Internal Helpers for Public API
# ---------------------------------------------------------------------------

def _get_current_atr_pips() -> float:
    """Fetch live ATR(14) in pips from M15 data, with conservative fallback."""
    if not _mt5_available:
        return 15.0
    try:
        rates = _fetch_rates("XAUUSD", "M15", 30)
        if rates is not None and len(rates) >= 15:
            highs = rates['high'].astype(np.float64)
            lows = rates['low'].astype(np.float64)
            closes = rates['close'].astype(np.float64)
            atr_arr = _compute_atr(highs, lows, closes, period=14)
            return float(atr_arr[-1]) / _PIP_SIZE
    except Exception:
        pass
    return 15.0


def _get_last_close(regime_state: RegimeState) -> float:
    """Get most recent close price for boundary proximity checks."""
    if not _mt5_available:
        midpoint = (regime_state.support_level + regime_state.resistance_level) / 2.0
        return midpoint if midpoint > 0 else 0.0
    try:
        rates = _fetch_rates("XAUUSD", "M1", 1)
        if rates is not None and len(rates) > 0:
            return float(rates['close'][-1])
    except Exception:
        pass
    midpoint = (regime_state.support_level + regime_state.resistance_level) / 2.0
    return midpoint if midpoint > 0 else 0.0


# ---------------------------------------------------------------------------
#  Module-Level Singleton
# ---------------------------------------------------------------------------

regime_detector = RegimeDetector()
