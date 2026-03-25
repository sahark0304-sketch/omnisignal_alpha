"""
quant/flow_exit.py -- OmniSignal Alpha v3.1
Breakthrough #3: Flow-Based Microstructure Exits

Detects CVD (Cumulative Volume Delta) exhaustion to exit profitable positions
dynamically, BEFORE static TP levels.  When the underlying order flow reverses
against our position while we're in profit, we exit immediately to lock in
gains before the flow reversal fully materializes in price.

Mathematical basis:
  CVD = cumsum(buy_volume - sell_volume) over last N seconds
  CVD_slope = (mean(CVD[recent_half]) - mean(CVD[older_half])) / half_period
  Exhaustion = CVD_slope reverses against our direction while position is profitable

Only triggers when:
  - Position is profitable (unrealized P&L > 0)
  - Profit exceeds MIN_PROFIT_PIPS (covers spread + commission)
  - CVD slope reversal exceeds EXHAUSTION_THRESHOLD
  - At least MIN_TICKS ticks in the analysis window
"""

import time
from typing import Optional, Tuple
import numpy as np
import MetaTrader5 as mt5

from utils.logger import get_logger

logger = get_logger(__name__)

CVD_WINDOW_SECS      = 120
EXHAUSTION_THRESHOLD = 0.55
MIN_PROFIT_PIPS      = 20.0
MIN_TICKS            = 80
COOLDOWN_PER_TICKET  = 45

_last_check: dict = {}


def check_flow_exit(
    ticket: int,
    symbol: str,
    action: str,
    entry_price: float,
    current_price: float,
    pip_size: float = 0.01,
) -> Tuple[bool, str]:
    """
    Check if a position should be exited based on CVD exhaustion.

    Returns:
        (should_exit, reason)
    """
    now = time.time()
    if ticket in _last_check and (now - _last_check[ticket]) < COOLDOWN_PER_TICKET:
        return False, ""

    _last_check[ticket] = now

    if action == "BUY":
        profit_pips = (current_price - entry_price) / pip_size
    else:
        profit_pips = (entry_price - current_price) / pip_size

    if profit_pips < MIN_PROFIT_PIPS:
        return False, ""

    try:
        from datetime import datetime, timedelta
        tick_from = datetime.now() - timedelta(seconds=CVD_WINDOW_SECS)
        ticks = mt5.copy_ticks_from(
            symbol,
            tick_from,
            5000,
            mt5.COPY_TICKS_ALL,
        )
    except Exception:
        return False, ""

    if ticks is None or len(ticks) < MIN_TICKS:
        return False, ""

    try:
        try:
            volumes = ticks['volume_real'].astype(float)
        except (ValueError, KeyError):
            volumes = ticks['volume'].astype(float)
        flags = ticks['flags'].astype(int)

        buy_mask = (flags & 0x20) != 0
        sell_mask = (flags & 0x40) != 0
        neutral_mask = ~(buy_mask | sell_mask)

        buy_vol = np.where(buy_mask, volumes, np.where(neutral_mask, volumes * 0.5, 0.0))
        sell_vol = np.where(sell_mask, volumes, np.where(neutral_mask, volumes * 0.5, 0.0))

        cvd = np.cumsum(buy_vol - sell_vol)

        n = len(cvd)
        half = n // 2
        if half < 20:
            return False, ""

        cvd_recent = np.mean(cvd[half:])
        cvd_older = np.mean(cvd[:half])
        total_vol = np.sum(volumes)
        if total_vol <= 0:
            return False, ""

        cvd_slope_norm = (cvd_recent - cvd_older) / (total_vol * 0.01)

    except Exception as e:
        logger.debug(f"[FlowExit] CVD computation error for {ticket}: {e}")
        return False, ""

    should_exit = False
    reason = ""

    if action == "BUY" and cvd_slope_norm < -EXHAUSTION_THRESHOLD:
        should_exit = True
        reason = (
            f"CVD exhaustion: slope={cvd_slope_norm:+.2f} (sell pressure building) "
            f"| profit={profit_pips:.1f}p | ticks={n}"
        )
    elif action == "SELL" and cvd_slope_norm > EXHAUSTION_THRESHOLD:
        should_exit = True
        reason = (
            f"CVD exhaustion: slope={cvd_slope_norm:+.2f} (buy pressure building) "
            f"| profit={profit_pips:.1f}p | ticks={n}"
        )

    if should_exit:
        logger.info(f"[FlowExit] EXIT SIGNAL for ticket {ticket}: {reason}")

    return should_exit, reason


def cleanup_ticket(ticket: int):
    """Remove tracking for a closed ticket."""
    _last_check.pop(ticket, None)
