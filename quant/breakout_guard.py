"""
quant/breakout_guard.py -- OmniSignal Alpha v3.1
Tick-Level Breakout Kill Switch + Directional Circuit Breaker

Two institutional-grade protections that prevent the bot from fighting
the dominant trend:

1. BREAKOUT KILL SWITCH
   Polls raw ticks every 3 seconds.  When price moves >= BREAKOUT_PIPS
   within BREAKOUT_WINDOW_SECS with accelerating volume, it declares
   a breakout in that direction and blocks ALL counter-trend scanners
   for BREAKOUT_BLOCK_SECS.

2. DIRECTIONAL CIRCUIT BREAKER
   When a trade hits its stop loss, the direction is locked out for
   DIRECTION_COOLOFF_SECS.  During cooloff, new signals in the SAME
   direction are blocked UNLESS price has broken the previous local
   structure (below prior swing low for sells, above prior swing high
   for buys).
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from utils.logger import get_logger

logger = get_logger(__name__)

# Breakout Kill Switch Config
BREAKOUT_PIPS        = 25.0
BREAKOUT_WINDOW_SECS = 45
BREAKOUT_BLOCK_SECS  = 120
VOL_ACCEL_MIN        = 1.3
POLL_INTERVAL        = 3.0

COUNTER_TREND_SOURCES = {"AUTO_MR", "AUTO_SCANNER", "AUTO_CATCD"}

# Directional Circuit Breaker Config
DIRECTION_COOLOFF_SECS = 900
STRUCTURE_LOOKBACK     = 120

# v4.3.2: Consecutive loss tracking with escalating pause
_consecutive_losses: Dict[str, int] = {}  # symbol -> count
_loss_pause_until: Dict[str, float] = {}  # symbol -> timestamp

CONSECUTIVE_LOSS_PAUSE_BASE_SECS = 300  # 5 min base pause after 3 losses
CONSECUTIVE_LOSS_MAX = 3  # trigger after this many


def register_consecutive_loss(symbol: str):
    """Called when a trade closes at a loss. Escalates pause duration."""
    _consecutive_losses[symbol] = _consecutive_losses.get(symbol, 0) + 1
    count = _consecutive_losses[symbol]
    if count >= CONSECUTIVE_LOSS_MAX:
        pause_secs = CONSECUTIVE_LOSS_PAUSE_BASE_SECS * (count - CONSECUTIVE_LOSS_MAX + 1)
        pause_secs = min(pause_secs, 1800)  # cap at 30 minutes
        _loss_pause_until[symbol] = time.time() + pause_secs
        logger.warning(
            "[BreakoutGuard] CONSECUTIVE LOSS PAUSE: %s has %d consecutive losses | "
            "Pausing ALL trades for %ds",
            symbol, count, pause_secs,
        )


def register_consecutive_win(symbol: str):
    """Called when a trade closes in profit. Resets the counter."""
    _consecutive_losses[symbol] = 0
    _loss_pause_until.pop(symbol, None)


def is_loss_paused(symbol: str) -> Tuple[bool, str]:
    """Check if trading is paused due to consecutive losses."""
    until = _loss_pause_until.get(symbol, 0)
    if time.time() < until:
        remaining = int(until - time.time())
        count = _consecutive_losses.get(symbol, 0)
        return True, (
            f"Consecutive loss pause: {count} losses on {symbol} | "
            f"{remaining}s remaining | Stop trading, let the market settle"
        )
    return False, ""


# v6.2: Session-Level Single-Loss Dampener
_session_loss_until: Dict[str, float] = {}
_session_loss_amount: Dict[str, float] = {}


def register_session_loss(symbol: str, pnl: float):
    """Called when a trade closes at a loss. If loss exceeds threshold, activate dampener."""
    import config as _cfg
    threshold = getattr(_cfg, "SESSION_SINGLE_LOSS_THRESHOLD", 50.0)
    duration = getattr(_cfg, "SESSION_LOSS_DAMPENER_DURATION_SECS", 3600)
    if abs(pnl) >= threshold:
        _session_loss_until[symbol] = time.time() + duration
        _session_loss_amount[symbol] = pnl
        logger.warning(
            "[BreakoutGuard] SESSION LOSS DAMPENER: %s lost $%.2f >= $%.0f threshold | "
            "Lots reduced 50%% for %ds",
            symbol, pnl, threshold, duration,
        )


def get_session_loss_dampener(symbol: str) -> float:
    """Returns lot multiplier based on session loss state. 1.0 = normal, 0.5 = dampened."""
    import config as _cfg
    until = _session_loss_until.get(symbol, 0)
    if time.time() < until:
        return getattr(_cfg, "SESSION_LOSS_LOT_REDUCTION", 0.50)
    _session_loss_until.pop(symbol, None)
    _session_loss_amount.pop(symbol, None)
    return 1.0


# v4.4: Post-Win Trend Bias Lock
TREND_WIN_BIAS_SECS = 300  # 5 minute counter-trend block after trend-aligned win
_trend_win_bias: Dict[str, Dict] = {}  # symbol -> {direction, until_ts}


def register_trend_win(symbol: str, action: str, regime: str):
    """Called when a profitable trade closes in FAST_TREND regime.
    Blocks counter-trend signals for TREND_WIN_BIAS_SECS."""
    if regime != "FAST_TREND":
        return
    until = time.time() + TREND_WIN_BIAS_SECS
    _trend_win_bias[symbol] = {"direction": action, "until_ts": until}
    logger.warning(
        "[BreakoutGuard] TREND BIAS LOCK: %s %s won in %s | "
        "counter-trend blocked for %ds",
        symbol, action, regime, TREND_WIN_BIAS_SECS,
    )


def is_trend_bias_blocked(signal_action: str, signal_symbol: str = "XAUUSD") -> Tuple[bool, str]:
    """Block signals that oppose a recent trend-aligned win."""
    bias = _trend_win_bias.get(signal_symbol)
    if not bias:
        return False, ""
    if time.time() > bias["until_ts"]:
        _trend_win_bias.pop(signal_symbol, None)
        return False, ""
    if signal_action == bias["direction"]:
        return False, ""
    remaining = int(bias["until_ts"] - time.time())
    return True, (
        f"Trend bias lock: {bias['direction']} won in FAST_TREND | "
        f"counter-trend {signal_action} blocked ({remaining}s remaining)"
    )


# Module State
_breakout_direction: Optional[str] = None
_breakout_until: float = 0.0

_sl_events: Dict[str, float] = {}
_sl_entry_prices: Dict[str, float] = {}


class BreakoutGuard:
    def __init__(self, symbol: str = "XAUUSD"):
        self._symbol = symbol
        self._pip_size = 0.01

    async def run(self):
        logger.info(
            f"[BreakoutGuard] Started for {self._symbol} "
            f"(poll {POLL_INTERVAL}s | trigger {BREAKOUT_PIPS}p in {BREAKOUT_WINDOW_SECS}s | "
            f"block {BREAKOUT_BLOCK_SECS}s)"
        )
        while True:
            try:
                self._check_breakout()
            except Exception as e:
                logger.debug(f"[BreakoutGuard] Cycle error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    def _check_breakout(self):
        global _breakout_direction, _breakout_until

        if mt5 is None:
            return

        tick_from = datetime.now() - timedelta(seconds=BREAKOUT_WINDOW_SECS)

        try:
            ticks = mt5.copy_ticks_from(
                self._symbol, tick_from, 5000, mt5.COPY_TICKS_ALL
            )
        except Exception:
            return

        if ticks is None or len(ticks) < 30:
            return

        prices = (ticks['bid'] + ticks['ask']) / 2.0
        try:
            volumes = ticks['volume_real'].astype(float)
        except (ValueError, KeyError):
            volumes = ticks['volume'].astype(float)

        displacement_pips = (prices[-1] - prices[0]) / self._pip_size

        half = len(volumes) // 2
        if half < 10:
            return
        vol_older = np.mean(volumes[:half])
        vol_recent = np.mean(volumes[half:])
        vol_accel = vol_recent / max(vol_older, 0.001)

        now = time.time()
        if abs(displacement_pips) >= BREAKOUT_PIPS and vol_accel >= VOL_ACCEL_MIN:
            direction = "BUY" if displacement_pips > 0 else "SELL"

            if _breakout_direction != direction or now > _breakout_until:
                _breakout_direction = direction
                try:
                    from quant.trade_orchestrator import get_scaled_cooldown
                    _scaled_block = get_scaled_cooldown(BREAKOUT_BLOCK_SECS, "breakout")
                except ImportError:
                    _scaled_block = BREAKOUT_BLOCK_SECS
                _breakout_until = now + _scaled_block
                logger.warning(
                    f"[BreakoutGuard] BREAKOUT DETECTED: {direction} "
                    f"| {displacement_pips:+.1f}p in {BREAKOUT_WINDOW_SECS}s "
                    f"| vol_accel={vol_accel:.1f}x "
                    f"| Counter-trend blocked for {BREAKOUT_BLOCK_SECS}s"
                )
                from database import db_manager
                db_manager.log_audit("BREAKOUT_DETECTED", {
                    "direction": direction,
                    "displacement_pips": round(displacement_pips, 1),
                    "vol_accel": round(vol_accel, 2),
                    "block_until": _breakout_until,
                })


def is_counter_trend_blocked(signal_action: str, signal_source: str) -> Tuple[bool, str]:
    """Check if a signal should be blocked by the breakout kill switch."""
    global _breakout_direction, _breakout_until

    now = time.time()
    if now > _breakout_until or _breakout_direction is None:
        return False, ""

    is_counter = False
    if _breakout_direction == "BUY" and signal_action == "SELL":
        is_counter = True
    elif _breakout_direction == "SELL" and signal_action == "BUY":
        is_counter = True

    if not is_counter:
        return False, ""

    if signal_source in COUNTER_TREND_SOURCES or signal_source.startswith("AUTO_"):
        remaining = int(_breakout_until - now)
        reason = (
            f"Breakout kill switch: {_breakout_direction} breakout active | "
            f"counter-trend {signal_source} {signal_action} blocked "
            f"({remaining}s remaining)"
        )
        return True, reason

    return False, ""


def register_sl_hit(action: str, entry_price: float, symbol: str = "XAUUSD"):
    """Called by trade_manager when a trade is stopped out."""
    global _sl_events, _sl_entry_prices

    key = f"{symbol}_{action}"
    _sl_events[key] = time.time()
    _sl_entry_prices[key] = entry_price

    logger.info(
        f"[BreakoutGuard] SL HIT registered: {symbol} {action} @ {entry_price:.2f} "
        f"| {action} direction locked for {DIRECTION_COOLOFF_SECS}s"
    )


def is_direction_blocked(
    signal_action: str,
    signal_symbol: str = "XAUUSD",
) -> Tuple[bool, str]:
    """Check if a direction is blocked by the circuit breaker."""
    loss_paused, loss_reason = is_loss_paused(signal_symbol)
    if loss_paused:
        return True, loss_reason

    key = f"{signal_symbol}_{signal_action}"
    if key not in _sl_events:
        return False, ""

    now = time.time()
    elapsed = now - _sl_events[key]

    try:
        from quant.trade_orchestrator import get_scaled_cooldown
        _eff_cooloff = get_scaled_cooldown(DIRECTION_COOLOFF_SECS, "direction")
    except ImportError:
        _eff_cooloff = DIRECTION_COOLOFF_SECS
    if elapsed > _eff_cooloff:
        _sl_events.pop(key, None)
        _sl_entry_prices.pop(key, None)
        return False, ""

    remaining = int(_eff_cooloff - elapsed)

    if _has_structure_break(signal_action, signal_symbol):
        logger.info(
            f"[BreakoutGuard] Structure break detected -- "
            f"{signal_action} cooloff lifted for {signal_symbol}"
        )
        _sl_events.pop(key, None)
        _sl_entry_prices.pop(key, None)
        return False, ""

    reason = (
        f"Directional circuit breaker: {signal_action} locked after SL hit "
        f"({remaining}s remaining) | Stop fighting the trend"
    )
    return True, reason


def _has_structure_break(action: str, symbol: str) -> bool:
    """
    Check if the market structure has broken in the signal's favor,
    which would justify lifting the directional cooloff.

    For a BUY block to be lifted: price must break above recent swing high.
    For a SELL block to be lifted: price must break below recent swing low.
    """
    if mt5 is None:
        return False

    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, STRUCTURE_LOOKBACK)
        if rates is None or len(rates) < 20:
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False

        current_price = (tick.bid + tick.ask) / 2.0
        highs = rates['high'].astype(float)
        lows = rates['low'].astype(float)

        lookback_bars = min(15, len(rates) - 5)

        if action == "SELL":
            recent_low = np.min(lows[-lookback_bars:-1])
            return current_price < recent_low

        elif action == "BUY":
            recent_high = np.max(highs[-lookback_bars:-1])
            return current_price > recent_high

    except Exception:
        pass

    return False


breakout_guard = BreakoutGuard()
