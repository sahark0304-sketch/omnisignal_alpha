"""
quant/feature_engineering.py -- Advanced feature engineering for WinModel.
Computes an 18-dimensional feature vector per signal from live MT5 market data.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_mt5_available = False
try:
    import MetaTrader5 as mt5
    _mt5_available = True
except ImportError:
    logger.warning("[FeatureEng] MetaTrader5 not installed -- features will return defaults")
    mt5 = None


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_XAUUSD_PIP_SIZE = 0.1

_SESSION_SCHEDULE = {
    "ASIA":    {"start_utc_hour": 22, "duration_hours": 9},
    "LONDON":  {"start_utc_hour": 7,  "duration_hours": 5},
    "OVERLAP": {"start_utc_hour": 12, "duration_hours": 2},
    "NY":      {"start_utc_hour": 14, "duration_hours": 8},
}

_FEATURE_NAMES: List[str] = [
    # Group 1: Volatility Regime
    "vol_regime_ratio",
    "vol_regime_label",
    "realized_vol_rank",
    # Group 2: Multi-Timeframe RSI Divergence
    "rsi_m5",
    "rsi_m15",
    "rsi_h1",
    "rsi_divergence_score",
    # Group 3: Micro-Structure / Institutional Footprint
    "tick_density_ratio",
    "bid_ask_imbalance",
    "volume_surge",
    "spread_volatility",
    # Group 4: Momentum & Mean-Reversion
    "momentum_m15",
    "mean_rev_z",
    "bar_body_ratio",
    "wick_rejection_pct",
    # Group 5: Session & Time Context
    "minutes_into_session",
    "session_atr_rank",
    "is_session_open_30min",
    # Group 6: Trade Recency & Account State
    "time_since_last_trade_mins",
    "current_dd_pct",
    # Group 7: Momentum Slope (v6.1 anti-decay features)
    "pullback_slope",
    "slope_delta",
    "sl_to_atr_ratio",
    "choppiness_index",
    "wick_dominance_ratio",
    "directional_consistency",
]


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def get_feature_names() -> List[str]:
    """Return ordered list of all feature names this module produces."""
    return list(_FEATURE_NAMES)


def compute_features(symbol: str, action: str, entry_price: float) -> Dict[str, float]:
    """
    Compute the full feature vector for a signal.
    Returns a dict of {feature_name: float_value}.
    All features are floats. Missing data -> 0.0 with a warning log.
    This function is blocking (calls MT5 synchronously).
    """
    features: Dict[str, float] = {name: 0.0 for name in _FEATURE_NAMES}

    if not _mt5_available:
        logger.warning("[FeatureEng] MT5 unavailable -- returning zero features")
        return features

    try:
        pip_size = _get_pip_size(symbol)

        g1 = _compute_volatility_regime(symbol)
        g2 = _compute_rsi_divergence(symbol)
        g3 = _compute_microstructure(symbol, pip_size)
        g4 = _compute_momentum(symbol)
        g5 = _compute_session_context(symbol)

        for group in (g1, g2, g3, g4, g5):
            for key, val in group.items():
                if key in features:
                    features[key] = float(val) if np.isfinite(val) else 0.0

        # Group 6: Trade recency & account state
        try:
            from database import db_manager
            last_close = db_manager.get_last_trade_close_time(symbol)
            if last_close is not None:
                from datetime import datetime as _dt
                if isinstance(last_close, str):
                    last_close = _dt.fromisoformat(last_close)
                _mins = (_dt.now() - last_close).total_seconds() / 60.0
                features["time_since_last_trade_mins"] = min(_mins, 1440.0)
            else:
                features["time_since_last_trade_mins"] = 999.0
        except Exception:
            features["time_since_last_trade_mins"] = 999.0

        try:
            from database import db_manager as _db
            _opening = _db.get_opening_equity()
            if _opening and _opening > 0 and _mt5_available:
                _acct = mt5.account_info()
                if _acct and _acct.equity > 0:
                    features["current_dd_pct"] = max(0.0, (_opening - _acct.equity) / _opening * 100.0)
                else:
                    features["current_dd_pct"] = 0.0
            else:
                features["current_dd_pct"] = 0.0
        except Exception:
            features["current_dd_pct"] = 0.0

        # Group 7: Momentum slope features (v6.1)
        features.update(_compute_slope_features(symbol))
        features.update(_compute_chop_features(symbol))

    except Exception as exc:
        logger.error(f"[FeatureEng] Top-level failure: {exc}", exc_info=True)

    return features



def _compute_slope_features(symbol: str) -> Dict[str, float]:
    """
    v6.1: Extract M1 EMA(20) momentum slope and its rate of change.
    These features let the ML model detect momentum decay -- the exact
    failure mode that caused the -.84 Trade 2 loss.
    """
    result = {
        "pullback_slope": 0.0,
        "slope_delta": 0.0,
        "sl_to_atr_ratio": 0.0,
    }
    try:
        rates = _safe_copy_rates(symbol, mt5.TIMEFRAME_M1, 0, 60)
        if rates is None or len(rates) < 25:
            return result

        closes = rates["close"].astype(np.float64)
        highs = rates["high"].astype(np.float64)
        lows = rates["low"].astype(np.float64)
        pip_size = _get_pip_size(symbol)

        ema = _compute_ema(closes, 20)
        if len(ema) < 10:
            return result

        current_slope = (ema[-1] - ema[-5]) / (5 * pip_size)
        result["pullback_slope"] = round(float(current_slope), 4)

        older_slope = (ema[-6] - ema[-10]) / (4 * pip_size)
        if abs(older_slope) > 1e-6:
            result["slope_delta"] = round(
                float((current_slope - older_slope) / abs(older_slope)), 4
            )
        else:
            result["slope_delta"] = round(float(current_slope), 4)

        atr_series = _compute_atr(highs, lows, closes, 14)
        if len(atr_series) > 0 and atr_series[-1] > 1e-10:
            result["sl_to_atr_ratio"] = round(float(atr_series[-1] / pip_size), 2)

    except Exception as e:
        logger.warning(f"[FeatureEng] Slope features failed: {e}")

    return result


def _compute_chop_features(symbol: str) -> Dict[str, float]:
    """v6.2: Choppiness Index, Wick Dominance, and Directional Consistency."""
    result = {
        "choppiness_index": 50.0,
        "wick_dominance_ratio": 0.50,
        "directional_consistency": 0.50,
    }
    try:
        rates = _safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 0, 30)
        if rates is None or len(rates) < 16:
            return result

        highs = rates["high"].astype(np.float64)
        lows = rates["low"].astype(np.float64)
        closes = rates["close"].astype(np.float64)
        opens = rates["open"].astype(np.float64)

        period = 14
        if len(highs) >= period + 1:
            tr = np.empty(len(highs))
            tr[0] = highs[0] - lows[0]
            for i in range(1, len(highs)):
                tr[i] = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            recent_tr = tr[-period:]
            hh = float(np.max(highs[-period:]))
            ll = float(np.min(lows[-period:]))
            range_hl = hh - ll
            if range_hl > 0:
                import math
                sum_tr = float(np.sum(recent_tr))
                log_period = math.log10(period)
                if log_period > 0:
                    ci = 100.0 * math.log10(sum_tr / range_hl) / log_period
                    result["choppiness_index"] = round(float(np.clip(ci, 0, 100)), 2)

        lookback = min(10, len(opens))
        if lookback >= 3:
            wick_ratios = []
            for i in range(-lookback, 0):
                cr = highs[i] - lows[i]
                if cr < 1e-8:
                    continue
                body_top = max(opens[i], closes[i])
                body_bot = min(opens[i], closes[i])
                wicks = (highs[i] - body_top) + (body_bot - lows[i])
                wick_ratios.append(wicks / cr)
            if wick_ratios:
                result["wick_dominance_ratio"] = round(float(np.mean(wick_ratios)), 4)

        lookback_dcs = min(20, len(closes) - 1)
        if lookback_dcs >= 5:
            diffs = np.diff(closes[-(lookback_dcs + 1):])
            up = int(np.sum(diffs > 0))
            down = int(np.sum(diffs < 0))
            total = up + down
            if total > 0:
                result["directional_consistency"] = round(max(up, down) / total, 4)

    except Exception as e:
        logger.warning("[FeatureEng] Chop features error: %s", e)

    return result


def compute_dynamic_sl_tp(
    symbol: str,
    action: str,
    entry_price: float,
    features: Dict[str, float],
) -> Dict:
    """
    Compute volatility-adjusted SL and TP based on current regime.
    Uses ATR and swing structure; returns regime-aware risk parameters.
    """
    pip_size = _get_pip_size(symbol)
    atr_pips = _current_atr_pips(symbol, pip_size)

    regime_label = features.get("vol_regime_label", 1.0)
    if regime_label <= 0.0:
        regime, sl_mult, tp1_mult, tp2_mult = "LOW", 1.2, 1.8, 3.0
    elif regime_label >= 2.0:
        regime, sl_mult, tp1_mult, tp2_mult = "HIGH", 2.0, 2.5, 4.0
    else:
        regime, sl_mult, tp1_mult, tp2_mult = "NORMAL", 1.5, 2.0, 3.5

    atr_sl = atr_pips * sl_mult
    atr_tp1 = atr_pips * tp1_mult
    atr_tp2 = atr_pips * tp2_mult

    swing_sl = _swing_structure_sl(symbol, action, entry_price, pip_size)
    sl_method = "ATR_REGIME"

    if swing_sl is not None and swing_sl < atr_sl:
        final_sl = swing_sl
        sl_method = "SWING_STRUCTURE"
    else:
        final_sl = atr_sl

    hard_cap = 50.0
    final_sl = min(final_sl, hard_cap)

    trail_activation = max(round(0.5 * atr_pips * sl_mult, 1), 3.0)
    trail_step = max(round(0.3 * atr_pips * sl_mult, 1), 1.0)

    return {
        "dynamic_sl_pips": round(final_sl, 1),
        "dynamic_tp1_pips": round(atr_tp1, 1),
        "dynamic_tp2_pips": round(atr_tp2, 1),
        "sl_method": sl_method,
        "regime": regime,
        "atr_current": round(atr_pips, 2),
        "trail_activation_pips": trail_activation,
        "trail_step_pips": trail_step,
    }


# ---------------------------------------------------------------------------
#  MT5 data helpers
# ---------------------------------------------------------------------------

def _get_pip_size(symbol: str) -> float:
    if not _mt5_available:
        return _XAUUSD_PIP_SIZE
    try:
        info = mt5.symbol_info(symbol)
        if info is not None and info.trade_tick_size > 0:
            return info.trade_tick_size * 10
    except Exception:
        pass
    return _XAUUSD_PIP_SIZE


def _safe_copy_rates(symbol: str, timeframe, start_pos: int, count: int):
    """Fetch OHLCV bars from MT5, returning None on failure."""
    if not _mt5_available:
        return None
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
        if rates is None or len(rates) == 0:
            return None
        return rates
    except Exception as exc:
        logger.debug(f"[FeatureEng] copy_rates failed: {exc}")
        return None


def _safe_copy_ticks(symbol: str, count: int, flags=None):
    """Fetch recent ticks from MT5, returning None on failure."""
    if not _mt5_available:
        return None
    try:
        if flags is None:
            flags = mt5.COPY_TICKS_ALL
        ticks = mt5.copy_ticks_from_pos(symbol, 0, count, flags)
        if ticks is None or len(ticks) == 0:
            return None
        return ticks
    except Exception as exc:
        logger.debug(f"[FeatureEng] copy_ticks failed: {exc}")
        return None


# ---------------------------------------------------------------------------
#  Technical indicator primitives
# ---------------------------------------------------------------------------

def _compute_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """Compute ATR series using Wilder smoothing. Returns array of len(closes)."""
    n = len(closes)
    if n < 2:
        return np.zeros(n)

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(n)
    if n < period:
        atr[:] = np.mean(tr[:n])
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute the latest RSI value using Wilder smoothing."""
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    ema = np.empty_like(data, dtype=np.float64)
    if len(data) == 0:
        return ema
    alpha = 2.0 / (period + 1)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1]
    return ema


# ---------------------------------------------------------------------------
#  Group 1: Volatility Regime
# ---------------------------------------------------------------------------

def _compute_volatility_regime(symbol: str) -> Dict[str, float]:
    result = {
        "vol_regime_ratio": 0.0,
        "vol_regime_label": 1.0,
        "realized_vol_rank": 0.5,
    }

    rates_m15 = _safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 0, 110)
    if rates_m15 is None or len(rates_m15) < 30:
        logger.warning("[FeatureEng] Insufficient M15 data for volatility regime")
        return result

    highs = rates_m15["high"].astype(np.float64)
    lows = rates_m15["low"].astype(np.float64)
    closes = rates_m15["close"].astype(np.float64)

    atr_series = _compute_atr(highs, lows, closes, 14)
    current_atr = atr_series[-1]

    lookback = min(96, len(atr_series))
    ma_atr_24h = np.mean(atr_series[-lookback:])

    if ma_atr_24h > 1e-10:
        ratio = current_atr / ma_atr_24h
    else:
        ratio = 1.0

    if ratio < 0.7:
        label = 0.0
    elif ratio > 1.3:
        label = 2.0
    else:
        label = 1.0

    result["vol_regime_ratio"] = round(ratio, 4)
    result["vol_regime_label"] = label

    # Realized vol rank: 1h vol percentile vs last 5 days on M5
    rates_m5 = _safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 0, 1500)
    if rates_m5 is not None and len(rates_m5) >= 24:
        m5_closes = rates_m5["close"].astype(np.float64)
        log_returns = np.diff(np.log(m5_closes))

        window_1h = 12  # 12 x M5 bars = 1 hour
        if len(log_returns) >= window_1h:
            current_vol = np.std(log_returns[-window_1h:])
            total_windows = (len(log_returns) - window_1h) // window_1h
            if total_windows > 0:
                historical_vols = np.array([
                    np.std(log_returns[i * window_1h:(i + 1) * window_1h])
                    for i in range(total_windows)
                ])
                rank = np.sum(historical_vols <= current_vol) / len(historical_vols)
                result["realized_vol_rank"] = round(float(rank), 4)

    return result


# ---------------------------------------------------------------------------
#  Group 2: Multi-Timeframe RSI Divergence
# ---------------------------------------------------------------------------

def _compute_rsi_divergence(symbol: str) -> Dict[str, float]:
    result = {
        "rsi_m5": 50.0,
        "rsi_m15": 50.0,
        "rsi_h1": 50.0,
        "rsi_divergence_score": 0.0,
    }

    rsi_vals = {}
    tf_map = {
        "rsi_m5":  (mt5.TIMEFRAME_M5, 50),
        "rsi_m15": (mt5.TIMEFRAME_M15, 50),
        "rsi_h1":  (mt5.TIMEFRAME_H1, 50),
    }

    for key, (tf, bars) in tf_map.items():
        rates = _safe_copy_rates(symbol, tf, 0, bars)
        if rates is not None and len(rates) >= 16:
            val = _compute_rsi(rates["close"].astype(np.float64), 14)
            result[key] = round(val, 2)
            rsi_vals[key] = val
        else:
            logger.warning(f"[FeatureEng] Insufficient data for {key}")

    if "rsi_m5" in rsi_vals and "rsi_h1" in rsi_vals:
        result["rsi_divergence_score"] = round(
            abs(rsi_vals["rsi_m5"] - rsi_vals["rsi_h1"]) / 100.0, 4
        )

    return result


# ---------------------------------------------------------------------------
#  Group 3: Micro-Structure / Institutional Footprint
# ---------------------------------------------------------------------------

def _compute_microstructure(symbol: str, pip_size: float) -> Dict[str, float]:
    result = {
        "tick_density_ratio": 1.0,
        "bid_ask_imbalance": 0.0,
        "volume_surge": 1.0,
        "spread_volatility": 0.0,
    }

    ticks = _safe_copy_ticks(symbol, 50000)
    if ticks is not None and len(ticks) >= 10:
        tick_times = ticks["time"].astype(np.float64)
        latest_time = tick_times[-1]

        mask_60s = tick_times >= (latest_time - 60)
        recent_count = float(np.sum(mask_60s))

        mask_30m = tick_times >= (latest_time - 1800)
        ticks_30m = float(np.sum(mask_30m))
        avg_60s_count = ticks_30m / 30.0 if ticks_30m > 0 else 1.0

        if avg_60s_count > 0:
            result["tick_density_ratio"] = round(recent_count / avg_60s_count, 4)

        mask_120s = tick_times >= (latest_time - 120)
        n_120 = int(np.sum(mask_120s))
        if n_120 > 1:
            bids_120 = ticks["bid"][mask_120s].astype(np.float64)
            diffs = np.diff(bids_120)
            upticks = float(np.sum(diffs > 0))
            downticks = float(np.sum(diffs < 0))
            total = upticks + downticks
            if total > 0:
                result["bid_ask_imbalance"] = round((upticks - downticks) / total, 4)

    rates_m5 = _safe_copy_rates(symbol, mt5.TIMEFRAME_M5, 0, 25)
    if rates_m5 is not None and len(rates_m5) >= 21:
        vols = rates_m5["tick_volume"].astype(np.float64)
        current_vol = vols[-1]
        avg_vol_20 = np.mean(vols[-21:-1])
        if avg_vol_20 > 0:
            result["volume_surge"] = round(float(current_vol / avg_vol_20), 4)

    rates_m1 = _safe_copy_rates(symbol, mt5.TIMEFRAME_M1, 0, 30)
    if rates_m1 is not None and len(rates_m1) >= 10:
        spreads_raw = rates_m1["spread"].astype(np.float64)
        try:
            point = mt5.symbol_info(symbol).point if _mt5_available else 0.01
        except Exception:
            point = 0.01
        spreads_pips = spreads_raw * point / pip_size if pip_size > 1e-10 else spreads_raw
        mean_sp = np.mean(spreads_pips)
        if mean_sp > 1e-10:
            result["spread_volatility"] = round(float(np.std(spreads_pips) / mean_sp), 4)

    return result


# ---------------------------------------------------------------------------
#  Group 4: Momentum & Mean-Reversion
# ---------------------------------------------------------------------------

def _compute_momentum(symbol: str) -> Dict[str, float]:
    result = {
        "momentum_m15": 0.0,
        "mean_rev_z": 0.0,
        "bar_body_ratio": 0.5,
        "wick_rejection_pct": 0.5,
    }

    rates = _safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 0, 60)
    if rates is None or len(rates) < 21:
        logger.warning("[FeatureEng] Insufficient M15 data for momentum features")
        return result

    closes = rates["close"].astype(np.float64)
    opens = rates["open"].astype(np.float64)
    highs = rates["high"].astype(np.float64)
    lows = rates["low"].astype(np.float64)

    atr_series = _compute_atr(highs, lows, closes, 14)
    current_atr = atr_series[-1]

    if len(closes) >= 21 and current_atr > 1e-10:
        result["momentum_m15"] = round(
            float((closes[-1] - closes[-21]) / current_atr), 4
        )

    if len(closes) >= 50:
        ema50 = _compute_ema(closes, 50)
        std_50 = np.std(closes[-50:])
        if std_50 > 1e-10:
            result["mean_rev_z"] = round(
                float((closes[-1] - ema50[-1]) / std_50), 4
            )

    last5_ranges = highs[-5:] - lows[-5:]
    last5_bodies = np.abs(closes[-5:] - opens[-5:])
    valid = last5_ranges > 1e-10
    if np.any(valid):
        ratios = np.where(valid, last5_bodies / last5_ranges, 0.0)
        result["bar_body_ratio"] = round(float(np.mean(ratios)), 4)

    upper_wicks = highs[-3:] - np.maximum(opens[-3:], closes[-3:])
    lower_wicks = np.minimum(opens[-3:], closes[-3:]) - lows[-3:]
    total_wicks = upper_wicks + lower_wicks
    ranges_3 = highs[-3:] - lows[-3:]
    valid_3 = ranges_3 > 1e-10
    if np.any(valid_3):
        wick_pcts = np.where(valid_3, total_wicks / ranges_3, 0.0)
        result["wick_rejection_pct"] = round(float(np.mean(wick_pcts)), 4)

    return result


# ---------------------------------------------------------------------------
#  Group 5: Session & Time Context
# ---------------------------------------------------------------------------

def _get_current_session(utc_hour: int) -> Optional[str]:
    """Determine the active trading session from a UTC hour."""
    for name, sched in _SESSION_SCHEDULE.items():
        start = sched["start_utc_hour"]
        end = (start + sched["duration_hours"]) % 24
        if start < end:
            if start <= utc_hour < end:
                return name
        else:
            if utc_hour >= start or utc_hour < end:
                return name
    return None


def _compute_session_context(symbol: str) -> Dict[str, float]:
    result = {
        "minutes_into_session": 0.5,
        "session_atr_rank": 1.0,
        "is_session_open_30min": 0.0,
    }

    now = datetime.now(timezone.utc)
    utc_hour = now.hour
    utc_minute = now.minute

    session = _get_current_session(utc_hour)
    if session is None:
        return result

    sched = _SESSION_SCHEDULE[session]
    start_hour = sched["start_utc_hour"]
    duration_minutes = sched["duration_hours"] * 60

    if utc_hour >= start_hour:
        minutes_in = (utc_hour - start_hour) * 60 + utc_minute
    else:
        minutes_in = (utc_hour + 24 - start_hour) * 60 + utc_minute

    if duration_minutes > 0:
        result["minutes_into_session"] = round(
            min(float(minutes_in) / duration_minutes, 1.0), 4
        )

    is_london_open = session == "LONDON" and minutes_in <= 30
    is_ny_open = session == "NY" and minutes_in <= 30
    result["is_session_open_30min"] = 1.0 if (is_london_open or is_ny_open) else 0.0

    # Session ATR rank: current 2h ATR vs avg 2h ATR for last 5 occurrences
    bars_2h = 8  # 8 x M15 = 2 hours
    rates = _safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 0, bars_2h)
    if rates is not None and len(rates) >= bars_2h:
        h = rates["high"].astype(np.float64)
        l = rates["low"].astype(np.float64)
        c = rates["close"].astype(np.float64)
        atr_now = _compute_atr(h, l, c, min(bars_2h, 14))
        current_2h_atr = float(atr_now[-1])

        lookback_bars = bars_2h * 5
        hist_rates = _safe_copy_rates(
            symbol, mt5.TIMEFRAME_M15, 0, lookback_bars + bars_2h
        )
        if hist_rates is not None and len(hist_rates) >= lookback_bars:
            hh = hist_rates["high"].astype(np.float64)
            hl = hist_rates["low"].astype(np.float64)
            hc = hist_rates["close"].astype(np.float64)
            window_atrs = []
            for i in range(0, len(hh) - bars_2h, bars_2h):
                seg_h = hh[i : i + bars_2h]
                seg_l = hl[i : i + bars_2h]
                seg_c = hc[i : i + bars_2h]
                a = _compute_atr(seg_h, seg_l, seg_c, min(bars_2h, 14))
                window_atrs.append(float(a[-1]))

            if len(window_atrs) > 0:
                avg_hist = np.mean(window_atrs)
                if avg_hist > 1e-10:
                    result["session_atr_rank"] = round(
                        float(current_2h_atr / avg_hist), 4
                    )

    return result


# ---------------------------------------------------------------------------
#  Dynamic SL/TP helpers
# ---------------------------------------------------------------------------

def _current_atr_pips(symbol: str, pip_size: float) -> float:
    """Get current ATR(14) on M15 expressed in pips."""
    if not _mt5_available:
        return 15.0
    rates = _safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 0, 30)
    if rates is None or len(rates) < 15:
        return 15.0
    h = rates["high"].astype(np.float64)
    l = rates["low"].astype(np.float64)
    c = rates["close"].astype(np.float64)
    atr = _compute_atr(h, l, c, 14)
    if pip_size > 1e-10:
        return float(atr[-1] / pip_size)
    return 15.0


def _swing_structure_sl(
    symbol: str,
    action: str,
    entry_price: float,
    pip_size: float,
) -> Optional[float]:
    """
    Find recent swing high/low for SL placement.
    Returns SL distance in pips, or None if structure not found.
    """
    if not _mt5_available:
        return None
    rates = _safe_copy_rates(symbol, mt5.TIMEFRAME_M15, 0, 30)
    if rates is None or len(rates) < 10:
        return None

    highs = rates["high"].astype(np.float64)
    lows = rates["low"].astype(np.float64)
    action_upper = action.upper()
    buffer_pips = 2.0

    if action_upper == "BUY":
        swing_low = float(np.min(lows[-10:]))
        distance = (entry_price - swing_low) / pip_size + buffer_pips
        return round(distance, 1) if distance > 0 else None

    if action_upper == "SELL":
        swing_high = float(np.max(highs[-10:]))
        distance = (swing_high - entry_price) / pip_size + buffer_pips
        return round(distance, 1) if distance > 0 else None

    return None
