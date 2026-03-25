"""
quant/confluence_engine.py â OmniSignal Alpha v3.0
Institutional Confluence Gate: ATR sizing, Hurst regime, session context,
macro alignment advisory, and ML feature computation.

This engine does NOT use retail oscillators (RSI, MACD, Bollinger) as
pass/fail gates.  If the AI Engine approved the signal, confluence passes.
The engine computes institutional metrics (Hurst, ATR, session, vol regime)
for the ML pipeline and risk sizing.
"""

import asyncio
import math
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_TF_MAP: Dict[str, int] = {}


def _get_tf_map():
    global _TF_MAP
    if not _TF_MAP:
        try:
            import MetaTrader5 as mt5
            _TF_MAP = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
            }
        except Exception:
            pass
    return _TF_MAP


@dataclass
class ConfluenceResult:
    score: int
    max_score: int
    passed: bool
    checks: Dict[str, bool]   = field(default_factory=dict)
    details: Dict[str, str]   = field(default_factory=dict)
    atr: float                = 0.0
    hurst_50: float           = 0.0
    hurst_100: float          = 0.0
    hurst_regime: str         = "UNKNOWN"
    vwap_distance_atr: float  = 0.0
    har_rv_forecast: float    = 0.0
    session: str              = "UNKNOWN"
    realized_var: float       = 0.0
    vol_z_score: float        = 0.0
    spread_percentile: float  = 0.5
    htf_h4_bias: int          = 0
    htf_d1_bias: int          = 0
    ob_zone: Optional[Tuple]  = None
    feature_row_id: int       = -1
    whale_detected: bool      = False
    sweep_detected: bool      = False

    @property
    def pct(self) -> float:
        return self.score / max(self.max_score, 1)


# â DATA FETCH ââââ

def _fetch_rates(symbol: str, tf_str: str, n: int = 250):
    import MetaTrader5 as mt5
    tf = _get_tf_map().get(tf_str, mt5.TIMEFRAME_H1)
    mt5.symbol_select(symbol, True)
    return mt5.copy_rates_from_pos(symbol, tf, 0, n)


async def _get_rates(symbol: str, tf: str = "H1", n: int = 250):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _fetch_rates(symbol, tf, n))


# â ATR ââââ

def _atr(rates, period: int = 14) -> np.ndarray:
    h  = rates["high"].astype(float)
    lo = rates["low"].astype(float)
    c  = rates["close"].astype(float)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    atr = np.zeros(len(tr))
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# â HURST (DFA) ââââ

def _hurst_dfa(prices: np.ndarray, min_window: int = 10) -> float:
    if len(prices) < 30:
        return 0.5
    try:
        log_returns = np.diff(np.log(prices.astype(float) + 1e-10))
        y = np.cumsum(log_returns - np.mean(log_returns))
        N = len(y)
        max_window = N // 4
        if max_window < min_window:
            return 0.5
        windows = np.unique(
            np.logspace(np.log10(min_window), np.log10(max_window), 20).astype(int)
        )
        windows = windows[windows >= min_window]
        F_vals = []
        for n in windows:
            n_segments = N // n
            if n_segments < 2:
                continue
            rms_list = []
            for seg in range(n_segments):
                seg_y = y[seg * n: (seg + 1) * n]
                x_seg = np.arange(n, dtype=float)
                coeffs = np.polyfit(x_seg, seg_y, 1)
                trend = np.polyval(coeffs, x_seg)
                residuals = seg_y - trend
                rms_list.append(np.sqrt(np.mean(residuals ** 2)))
            if rms_list:
                F_vals.append((n, np.mean(rms_list)))
        if len(F_vals) < 4:
            return 0.5
        log_n = np.log(np.array([f[0] for f in F_vals], dtype=float))
        log_F = np.log(np.array([f[1] for f in F_vals], dtype=float) + 1e-15)
        slope, _ = np.polyfit(log_n, log_F, 1)
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def _classify_hurst(h: float) -> str:
    if h >= config.HURST_MIN_THRESHOLD:
        return "TRENDING"
    elif h <= config.HURST_MEAN_REVERSION_MAX:
        return "MEAN_REVERTING"
    return "RANDOM"


# â VOLATILITY REGIME ââââ

def _vol_z_score(atr_arr: np.ndarray, lookback: int = 50) -> float:
    if len(atr_arr) < lookback:
        return 0.0
    window = atr_arr[-lookback:]
    mean_atr = float(np.mean(window))
    std_atr = float(np.std(window))
    if std_atr < 1e-10:
        return 0.0
    return (float(atr_arr[-1]) - mean_atr) / std_atr


# â WHALE + SWEEP (informational, never blocks) ââââ

def _whale_volume(rates, lookback: int = 20) -> Tuple[bool, float]:
    vols = rates["tick_volume"].astype(float)
    if len(vols) < lookback + 1:
        return False, 1.0
    avg = float(np.mean(vols[-(lookback+1):-1]))
    cur = float(vols[-1])
    ratio = cur / avg if avg > 0 else 1.0
    return ratio >= config.WHALE_VOLUME_SPIKE_MULT, ratio


def _liquidity_sweep(rates, action: str) -> Tuple[bool, str]:
    r = rates[-(config.WHALE_SWEEP_LOOKBACK + 2):-1]
    cur = rates[-1]
    min_wick = config.WHALE_SWEEP_REJECTION_PCT
    if action == "BUY":
        low5 = float(np.min(r["low"][-5:]))
        wick = (float(cur["close"]) - float(cur["low"])) / max(float(cur["close"]), 1e-9)
        swept = float(cur["low"]) < low5 and wick >= min_wick
        return swept, (f"Bullish sweep: low < {low5:.2f}" if swept else "")
    else:
        high5 = float(np.max(r["high"][-5:]))
        wick = (float(cur["high"]) - float(cur["close"])) / max(float(cur["high"]), 1e-9)
        swept = float(cur["high"]) > high5 and wick >= min_wick
        return swept, (f"Bearish sweep: high > {high5:.2f}" if swept else "")


# â MASTER CONFLUENCE: Institutional Pass-Through ââââ
#
# This does NOT block AI-approved trades.  It computes institutional metrics
# (ATR, Hurst, session, vol regime, whale/sweep) that feed the ML pipeline
# and position sizing engine.  Macro/COT advisory is logged, never hard-blocks.

async def check_confluence(
    symbol: str,
    action: str,
    entry_price: Optional[float] = None,
    timeframe: str = None,
    current_spread_pips: float = 0.0,
    signal_id: int = None,
) -> ConfluenceResult:
    """Compute institutional metrics and always pass the trade through.
    The AI Engine is the decision gate, not retail oscillators."""
    session = get_current_session()
    tf = timeframe or config.CONFLUENCE_TIMEFRAME

    if not config.CONFLUENCE_ENABLED:
        return ConfluenceResult(score=1, max_score=1, passed=True, session=session)

    rates = await _get_rates(symbol, tf, n=250)

    if rates is None or len(rates) < 30:
        logger.warning(f"[Confluence] No data for {symbol} â pass-through")
        return ConfluenceResult(score=1, max_score=1, passed=True, session=session)

    closes = rates["close"].astype(float)

    # ATR for position sizing
    atr_arr = _atr(rates, config.ATR_PERIOD)
    atr_val = float(atr_arr[-1])

    # Hurst exponent (DFA)
    h50 = _hurst_dfa(closes[-50:]) if len(closes) >= 50 else 0.5
    h100 = _hurst_dfa(closes[-100:]) if len(closes) >= 100 else h50
    hurst_regime = _classify_hurst(h50)

    # Volatility regime
    vol_z = _vol_z_score(atr_arr)

    # Whale volume + Liquidity sweep (informational only)
    whale, whale_ratio = _whale_volume(rates) if config.WHALE_ENABLED else (False, 1.0)
    sweep, sweep_msg = _liquidity_sweep(rates, action) if config.WHALE_ENABLED else (False, "")

    # Macro/COT advisory (logged, never blocks)
    macro_note = ""
    try:
        from quant.macro_filter import macro_filter
        state = macro_filter.state
        if state.is_fresh:
            bias = state.cot.bias
            if (bias == "SHORT_ONLY" and action == "BUY") or \
               (bias == "LONG_ONLY" and action == "SELL"):
                macro_note = f"COT contra-structural ({bias} vs {action})"
                logger.info(f"[Confluence] ADVISORY: {macro_note}")
    except Exception:
        pass

    checks = {
        "ai_approved": True,
        "atr_valid": atr_val > 0,
        "hurst_computed": True,
    }
    details = {
        "atr": f"{atr_val:.4f}",
        "hurst_50": f"H={h50:.3f} ({hurst_regime})",
        "vol_z": f"z={vol_z:.2f}",
        "whale": f"ratio={whale_ratio:.2f}x detected={whale}",
        "sweep": sweep_msg or "none",
        "macro": macro_note or "aligned",
        "session": session,
    }

    score = sum(1 for v in checks.values() if v)
    max_score = len(checks)

    result = ConfluenceResult(
        score=score, max_score=max_score, passed=True,
        checks=checks, details=details, atr=atr_val,
        hurst_50=h50, hurst_100=h100, hurst_regime=hurst_regime,
        session=session, vol_z_score=vol_z,
        whale_detected=whale, sweep_detected=sweep,
    )

    logger.info(
        f"[Confluence] {symbol} {action} PASSED | "
        f"ATR={atr_val:.4f} H={h50:.3f}({hurst_regime}) "
        f"Vol_z={vol_z:.2f} Session={session} "
        f"Whale={'YES' if whale else 'no'} Sweep={'YES' if sweep else 'no'}"
    )
    return result


# â PUBLIC HELPERS ââââ

async def get_atr(symbol: str, timeframe: str = None, period: int = None) -> float:
    tf = timeframe or config.ATR_TIMEFRAME
    per = period or config.ATR_PERIOD
    rates = await _get_rates(symbol, tf, n=per * 3)
    if rates is None or len(rates) < per + 1:
        return 0.0
    return float(_atr(rates, per)[-1])


def get_current_session() -> str:
    hour = datetime.now(timezone.utc).hour
    for name, (start, end) in config.SESSION_HOURS_UTC.items():
        if start < end:
            if start <= hour < end:
                return name
        else:
            if hour >= start or hour < end:
                return name
    return "ASIA"
