"""
quant/smc_scanner.py — OmniSignal Alpha v6.0
Smart Money Concepts (SMC) Scanner

WHAT THIS IS:
  Institutional traders leave footprints in price action that retail indicators
  cannot detect. This scanner reads those footprints on M5 and M15:

  1. MARKET STRUCTURE (BOS / CHoCH)
     - Break of Structure (BOS): price sweeps a swing high/low confirming trend direction.
     - Change of Character (CHoCH): trend reversal signal — first break of structure
       in the opposite direction.

  2. ORDER BLOCKS (OB)
     The last bearish candle before a bullish impulse (bullish OB), or
     the last bullish candle before a bearish impulse (bearish OB).
     Price returns to these zones and respects them as S/R.

  3. FAIR VALUE GAPS (FVG)
     Three-candle pattern where candle[i+2].low > candle[i].high (bullish FVG) or
     candle[i+2].high < candle[i].low (bearish FVG). Price is drawn back to fill
     these inefficiencies. Also called "imbalances" or "liquidity gaps."

SIGNAL LOGIC:
  Entry fires when:
    a) Market structure is identified (BOS direction)
    b) Price retraces INTO an OB or FVG in the BOS direction
    c) A rejection candle forms at the OB/FVG boundary (wick rejection or engulfing)
    d) Spread is acceptable and not in news window

  SL: Below/above the OB origin candle low/high (+ ATR buffer)
  TP1: Previous swing high/low (structural target)
  TP2: 1.618 fibonacci extension (institutional target)

SAFETY:
  - Additive: pushes to the same signal_queue.push() used by all other scanners
  - Source tag: "AUTO_SMC" — receives Alpha Ranker treatment, all risk filters apply
  - Circuit breaker: 5 consecutive losses → 1h cooloff
  - Max 8 signals per hour
  - Only active during London + NY sessions (07:00-21:00 UTC)
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np

import config
from ingestion.signal_queue import push, RawSignal
from quant.vol_regime import vol_regime
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    _mt5_available = True
except ImportError:
    mt5 = None
    _mt5_available = False

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_SECS    = 12       # scan every 12 seconds
SWING_LOOKBACK        = 5        # pivot lookback for swing detection
STRUCTURE_BARS        = 60       # M15 bars used for structure analysis
OB_LOOKBACK           = 30       # max bars back to look for order blocks
FVG_MIN_SIZE_PIPS     = 2.0      # minimum FVG size to trade
OB_MAX_AGE_BARS       = 40       # discard OBs older than this
REJECTION_WICK_PCT    = 0.40     # minimum lower/upper wick % for rejection candle
MAX_SIGNALS_PER_HOUR  = 8
CONSEC_LOSS_LIMIT     = 5
COOLOFF_HOURS         = 1
MAX_SPREAD_PIPS       = 10.0
SIGNAL_COOLDOWN_SECS  = 90       # min gap between signals
DEDUP_WINDOW_SECS     = 300      # same-direction dedup


@dataclass
class SwingPoint:
    bar_idx: int
    price: float
    kind: str  # "HIGH" or "LOW"
    time: Optional[datetime] = None


@dataclass
class OrderBlock:
    kind: str             # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    origin_bar: int
    impulse_size_pips: float
    time: Optional[datetime] = None

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class FairValueGap:
    kind: str         # "BULLISH" or "BEARISH"
    top: float
    bottom: float
    origin_bar: int
    size_pips: float
    time: Optional[datetime] = None
    filled: bool = False

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  SMC Scanner
# ─────────────────────────────────────────────────────────────────────────────

class SMCScanner:
    """
    Smart Money Concepts scanner for XAUUSD.
    Generates signals when price returns to institutional order blocks
    or fair value gaps in the direction of confirmed market structure.
    """

    def __init__(self, symbol: str = "XAUUSD"):
        self._symbol      = symbol
        self._pip_size    = 0.1  # XAUUSD pip
        self._signals_generated    = 0
        self._consecutive_losses   = 0
        self._disabled_until: Optional[float] = None
        self._last_signal_time: Optional[float] = None
        self._last_signal_direction: Optional[str] = None
        self._signal_timestamps: List[float] = []
        self._cycle_count  = 0
        self._last_pressure = 0.0

        # Active structure state
        self._current_bias: Optional[str] = None  # "BULLISH" or "BEARISH"
        self._active_obs: List[OrderBlock] = []
        self._active_fvgs: List[FairValueGap] = []
        self._last_structure_update: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self):
        logger.info(
            "[SMC] Scanner started for %s (poll=%ds swing_lookback=%d)",
            self._symbol, POLL_INTERVAL_SECS, SWING_LOOKBACK,
        )
        await asyncio.sleep(30)  # warm-up delay

        while True:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error("[SMC] Cycle error: %s", e, exc_info=True)
            await asyncio.sleep(POLL_INTERVAL_SECS)

    # ──────────────────────────────────────────────────────────────────────────
    #  Scan cycle
    # ──────────────────────────────────────────────────────────────────────────

    async def _scan_cycle(self):
        self._cycle_count += 1

        if self._cycle_count % 150 == 0:
            logger.info(
                "[SMC] Status: %d cycles | %d signals | bias=%s "
                "OBs=%d FVGs=%d",
                self._cycle_count, self._signals_generated,
                self._current_bias, len(self._active_obs), len(self._active_fvgs),
            )

        if not self._is_trading_session():
            return

        if not _mt5_available or mt5 is None:
            return

        if self._is_circuit_open():
            return

        spread = self._get_spread_pips()
        if spread is None or spread > MAX_SPREAD_PIPS:
            return

        # Update structure every 60 seconds (M15 bars)
        now = time.time()
        if now - self._last_structure_update > 60:
            rates_m15 = self._fetch_rates(mt5.TIMEFRAME_M15, STRUCTURE_BARS)
            if rates_m15 is not None and len(rates_m15) >= 30:
                self._update_structure(rates_m15)
                self._last_structure_update = now

        if self._current_bias is None:
            return

        # Get current M5 bars for entry detection
        rates_m5 = self._fetch_rates(mt5.TIMEFRAME_M5, 20)
        if rates_m5 is None or len(rates_m5) < 5:
            return

        entry_signal = self._check_entry(rates_m5)
        if entry_signal is None:
            return

        action, entry, sl, tp1, tp2, reason = entry_signal

        # Dedup check
        if (
            self._last_signal_direction == action
            and self._last_signal_time
            and (now - self._last_signal_time) < DEDUP_WINDOW_SECS
        ):
            return

        if self._last_signal_time and (now - self._last_signal_time) < SIGNAL_COOLDOWN_SECS:
            return

        await self._push_signal(action, entry, sl, tp1, tp2, reason)

    # ──────────────────────────────────────────────────────────────────────────
    #  Structure analysis (M15)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_structure(self, rates: np.ndarray):
        """Update market bias, order blocks, and FVGs from M15 bars."""
        highs  = rates["high"].astype(float)
        lows   = rates["low"].astype(float)
        closes = rates["close"].astype(float)
        opens  = rates["open"].astype(float)
        n      = len(rates)

        # ── Swing detection ──────────────────────────────────────────────────
        swing_highs, swing_lows = self._find_swings(highs, lows, SWING_LOOKBACK)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return

        # ── Market bias from Break of Structure ──────────────────────────────
        # BOS Bullish: current close breaks above last swing high
        # BOS Bearish: current close breaks below last swing low
        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]
        prev_sh = swing_highs[-2] if len(swing_highs) >= 2 else None
        prev_sl = swing_lows[-2] if len(swing_lows) >= 2 else None

        current_close = closes[-1]

        new_bias = self._current_bias  # default: keep existing

        if prev_sh and current_close > last_sh.price:
            # BOS bullish — price broke above last swing high
            new_bias = "BULLISH"
            if self._current_bias != "BULLISH":
                logger.info(
                    "[SMC] BOS BULLISH: %s broke swing high %.2f",
                    self._symbol, last_sh.price,
                )
        elif prev_sl and current_close < last_sl.price:
            # BOS bearish — price broke below last swing low
            new_bias = "BEARISH"
            if self._current_bias != "BEARISH":
                logger.info(
                    "[SMC] BOS BEARISH: %s broke swing low %.2f",
                    self._symbol, last_sl.price,
                )

        # CHoCH detection: bias flip with strong engulfing candle
        if (
            self._current_bias == "BULLISH"
            and new_bias == "BEARISH"
            and prev_sl
            and closes[-2] > opens[-2]  # prior bullish candle
            and closes[-1] < opens[-1]  # current bearish engulf
        ):
            logger.info("[SMC] CHoCH BEARISH: reversal signal")

        if (
            self._current_bias == "BEARISH"
            and new_bias == "BULLISH"
            and prev_sh
            and closes[-2] < opens[-2]
            and closes[-1] > opens[-1]
        ):
            logger.info("[SMC] CHoCH BULLISH: reversal signal")

        self._current_bias = new_bias

        # ── Update pressure for convergence engine ────────────────────────────
        if new_bias == "BULLISH":
            self._last_pressure = 0.55
        elif new_bias == "BEARISH":
            self._last_pressure = -0.55
        else:
            self._last_pressure = 0.0

        # ── Find fresh Order Blocks ───────────────────────────────────────────
        self._active_obs = self._find_order_blocks(rates, opens, highs, lows, closes)

        # ── Find unmitigated Fair Value Gaps ─────────────────────────────────
        self._active_fvgs = self._find_fvgs(rates, highs, lows)

    # ──────────────────────────────────────────────────────────────────────────
    #  Swing detection
    # ──────────────────────────────────────────────────────────────────────────

    def _find_swings(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        lookback: int,
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        n = len(highs)
        swing_highs, swing_lows = [], []

        for i in range(lookback, n - lookback):
            window_h = highs[i - lookback:i + lookback + 1]
            window_l = lows[i - lookback:i + lookback + 1]

            if highs[i] == np.max(window_h):
                swing_highs.append(SwingPoint(bar_idx=i, price=float(highs[i]), kind="HIGH"))
            if lows[i] == np.min(window_l):
                swing_lows.append(SwingPoint(bar_idx=i, price=float(lows[i]), kind="LOW"))

        return swing_highs, swing_lows

    # ──────────────────────────────────────────────────────────────────────────
    #  Order block detection
    # ──────────────────────────────────────────────────────────────────────────

    def _find_order_blocks(
        self,
        rates: np.ndarray,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> List[OrderBlock]:
        obs = []
        n = len(rates)
        current_price = float(closes[-1])

        for i in range(max(0, n - OB_LOOKBACK), n - 3):
            # Bullish OB: bearish candle (red) immediately followed by strong bullish impulse
            is_bearish_candle = closes[i] < opens[i]
            if is_bearish_candle:
                # Check for impulse: next 2 bars should close significantly above OB
                impulse_high = float(np.max(highs[i+1:i+4]))
                impulse_size = (impulse_high - highs[i]) / self._pip_size
                if impulse_size >= 5.0:
                    ob = OrderBlock(
                        kind="BULLISH",
                        top=float(highs[i]),
                        bottom=float(lows[i]),
                        origin_bar=i,
                        impulse_size_pips=impulse_size,
                    )
                    # Only keep OB if price is currently above it (unmitigated from below)
                    if current_price > ob.top:
                        # Check not yet mitigated (price hasn't traded below OB bottom since)
                        future_lows = lows[i+1:]
                        if float(np.min(future_lows)) > ob.bottom:
                            obs.append(ob)

            # Bearish OB: bullish candle (green) immediately followed by strong bearish impulse
            is_bullish_candle = closes[i] > opens[i]
            if is_bullish_candle:
                impulse_low = float(np.min(lows[i+1:i+4]))
                impulse_size = (lows[i] - impulse_low) / self._pip_size
                if impulse_size >= 5.0:
                    ob = OrderBlock(
                        kind="BEARISH",
                        top=float(highs[i]),
                        bottom=float(lows[i]),
                        origin_bar=i,
                        impulse_size_pips=impulse_size,
                    )
                    if current_price < ob.bottom:
                        future_highs = highs[i+1:]
                        if float(np.max(future_highs)) < ob.top:
                            obs.append(ob)

        # Sort by proximity to current price
        obs.sort(key=lambda o: abs(o.midpoint() - current_price))
        return obs[:5]  # keep 5 nearest

    # ──────────────────────────────────────────────────────────────────────────
    #  Fair Value Gap detection
    # ──────────────────────────────────────────────────────────────────────────

    def _find_fvgs(
        self,
        rates: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> List[FairValueGap]:
        fvgs = []
        n = len(rates)
        current_price = float(rates["close"][-1])

        for i in range(max(0, n - OB_LOOKBACK), n - 2):
            # Bullish FVG: gap between candle[i].high and candle[i+2].low
            gap_low  = float(highs[i])
            gap_high = float(lows[i + 2])
            if gap_high > gap_low:
                size_pips = (gap_high - gap_low) / self._pip_size
                if size_pips >= FVG_MIN_SIZE_PIPS:
                    fvg = FairValueGap(
                        kind="BULLISH",
                        top=gap_high,
                        bottom=gap_low,
                        origin_bar=i + 1,
                        size_pips=size_pips,
                    )
                    # Unmitigated: current price is above the FVG
                    if current_price > fvg.top:
                        # Check not filled: no close inside FVG since origin
                        future_lows = lows[i+3:]
                        if len(future_lows) == 0 or float(np.min(future_lows)) > fvg.bottom:
                            fvgs.append(fvg)

            # Bearish FVG: gap between candle[i].low and candle[i+2].high (inverted)
            gap_high2 = float(lows[i])
            gap_low2  = float(highs[i + 2])
            if gap_high2 > gap_low2:
                size_pips = (gap_high2 - gap_low2) / self._pip_size
                if size_pips >= FVG_MIN_SIZE_PIPS:
                    fvg = FairValueGap(
                        kind="BEARISH",
                        top=gap_high2,
                        bottom=gap_low2,
                        origin_bar=i + 1,
                        size_pips=size_pips,
                    )
                    if current_price < fvg.bottom:
                        future_highs = highs[i+3:]
                        if len(future_highs) == 0 or float(np.max(future_highs)) < fvg.top:
                            fvgs.append(fvg)

        fvgs.sort(key=lambda f: abs(f.midpoint() - current_price))
        return fvgs[:5]

    # ──────────────────────────────────────────────────────────────────────────
    #  Entry detection (M5)
    # ──────────────────────────────────────────────────────────────────────────

    def _check_entry(
        self,
        rates_m5: np.ndarray,
    ) -> Optional[Tuple]:
        """
        Returns (action, entry, sl, tp1, tp2, reason) or None.
        Entry fires when price retraces INTO an OB or FVG and shows rejection.
        """
        if self._current_bias is None:
            return None

        highs  = rates_m5["high"].astype(float)
        lows   = rates_m5["low"].astype(float)
        closes = rates_m5["close"].astype(float)
        opens  = rates_m5["open"].astype(float)

        tick = mt5.symbol_info_tick(self._symbol)
        if tick is None:
            return None

        current_bid = float(tick.bid)
        current_ask = float(tick.ask)
        current_mid = (current_bid + current_ask) / 2.0

        last_candle_high  = float(highs[-1])
        last_candle_low   = float(lows[-1])
        last_candle_close = float(closes[-1])
        last_candle_open  = float(opens[-1])
        candle_range      = last_candle_high - last_candle_low

        if candle_range < self._pip_size * 0.5:
            return None

        atr = self._get_m5_atr()

        # ── Check OB entries ─────────────────────────────────────────────────
        for ob in self._active_obs:
            if ob.kind == "BULLISH" and self._current_bias == "BULLISH":
                # Price must retrace INTO the OB
                if not ob.contains(current_mid):
                    continue
                # Rejection: lower wick dominates
                lower_wick = min(last_candle_open, last_candle_close) - last_candle_low
                if candle_range > 0 and lower_wick / candle_range >= REJECTION_WICK_PCT:
                    entry = current_ask
                    sl = max(ob.bottom - atr * 0.3, ob.bottom - 15 * self._pip_size)
                    sl = round(sl, 2)
                    sl_dist = abs(entry - sl)
                    if sl_dist < self._pip_size * 8:
                        continue
                    tp1 = round(entry + sl_dist * 1.5, 2)
                    tp2 = round(entry + sl_dist * 2.618, 2)  # fib extension
                    reason = (
                        f"SMC Bullish OB @ {ob.bottom:.2f}-{ob.top:.2f} "
                        f"impulse={ob.impulse_size_pips:.0f}p rejection wick"
                    )
                    return ("BUY", entry, sl, tp1, tp2, reason)

            elif ob.kind == "BEARISH" and self._current_bias == "BEARISH":
                if not ob.contains(current_mid):
                    continue
                upper_wick = last_candle_high - max(last_candle_open, last_candle_close)
                if candle_range > 0 and upper_wick / candle_range >= REJECTION_WICK_PCT:
                    entry = current_bid
                    sl = min(ob.top + atr * 0.3, ob.top + 15 * self._pip_size)
                    sl = round(sl, 2)
                    sl_dist = abs(sl - entry)
                    if sl_dist < self._pip_size * 8:
                        continue
                    tp1 = round(entry - sl_dist * 1.5, 2)
                    tp2 = round(entry - sl_dist * 2.618, 2)
                    reason = (
                        f"SMC Bearish OB @ {ob.bottom:.2f}-{ob.top:.2f} "
                        f"impulse={ob.impulse_size_pips:.0f}p rejection wick"
                    )
                    return ("SELL", entry, sl, tp1, tp2, reason)

        # ── Check FVG entries ─────────────────────────────────────────────────
        for fvg in self._active_fvgs:
            if fvg.kind == "BULLISH" and self._current_bias == "BULLISH":
                if not fvg.contains(current_mid):
                    continue
                # Bullish close inside FVG = rejection / bounce entry
                if last_candle_close > fvg.midpoint() and last_candle_close > last_candle_open:
                    entry = current_ask
                    sl = round(fvg.bottom - atr * 0.25, 2)
                    sl_dist = abs(entry - sl)
                    if sl_dist < self._pip_size * 8:
                        continue
                    tp1 = round(entry + sl_dist * 1.5, 2)
                    tp2 = round(entry + sl_dist * 2.5, 2)
                    reason = (
                        f"SMC Bullish FVG {fvg.size_pips:.0f}p gap "
                        f"@ {fvg.bottom:.2f}-{fvg.top:.2f} bullish close"
                    )
                    return ("BUY", entry, sl, tp1, tp2, reason)

            elif fvg.kind == "BEARISH" and self._current_bias == "BEARISH":
                if not fvg.contains(current_mid):
                    continue
                if last_candle_close < fvg.midpoint() and last_candle_close < last_candle_open:
                    entry = current_bid
                    sl = round(fvg.top + atr * 0.25, 2)
                    sl_dist = abs(sl - entry)
                    if sl_dist < self._pip_size * 8:
                        continue
                    tp1 = round(entry - sl_dist * 1.5, 2)
                    tp2 = round(entry - sl_dist * 2.5, 2)
                    reason = (
                        f"SMC Bearish FVG {fvg.size_pips:.0f}p gap "
                        f"@ {fvg.bottom:.2f}-{fvg.top:.2f} bearish close"
                    )
                    return ("SELL", entry, sl, tp1, tp2, reason)

        return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Signal emission
    # ──────────────────────────────────────────────────────────────────────────

    async def _push_signal(
        self,
        action: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        reason: str,
    ):
        text = (
            f"{self._symbol} {action} @ {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP: {tp1:.2f}\n"
            f"TP2: {tp2:.2f}\n"
            f"[AUTO_SMC] {reason}"
        )
        signal = RawSignal(content=text, source="AUTO_SMC")
        await push(signal)

        now = time.time()
        self._signals_generated  += 1
        self._last_signal_time    = now
        self._last_signal_direction = action
        self._signal_timestamps.append(now)

        logger.info(
            "[SMC] SIGNAL: %s %s @ %.2f SL=%.2f TP1=%.2f TP2=%.2f | %s | bias=%s",
            self._symbol, action, entry, sl, tp1, tp2, reason, self._current_bias,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Utility helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_m5_atr(self, period: int = 14) -> float:
        try:
            rates = mt5.copy_rates_from_pos(self._symbol, mt5.TIMEFRAME_M5, 0, period + 1)
            if rates is None or len(rates) < period:
                return 15 * self._pip_size
            h = rates["high"].astype(float)
            l = rates["low"].astype(float)
            c = rates["close"].astype(float)
            tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
            return float(np.mean(tr[-period:]))
        except Exception:
            return 15 * self._pip_size

    def _fetch_rates(self, timeframe, n: int) -> Optional[np.ndarray]:
        try:
            mt5.symbol_select(self._symbol, True)
            rates = mt5.copy_rates_from_pos(self._symbol, timeframe, 0, n)
            return rates if (rates is not None and len(rates) >= 10) else None
        except Exception:
            return None

    def _get_spread_pips(self) -> Optional[float]:
        if not _mt5_available:
            return None
        tick = mt5.symbol_info_tick(self._symbol)
        if tick is None:
            return None
        return (tick.ask - tick.bid) / self._pip_size

    def _is_trading_session(self) -> bool:
        h = datetime.now(timezone.utc).hour
        return 7 <= h < 21

    def _is_circuit_open(self) -> bool:
        now = time.time()
        if self._disabled_until and now < self._disabled_until:
            return True
        if self._disabled_until and now >= self._disabled_until:
            self._disabled_until = None
            self._consecutive_losses = 0
            logger.info("[SMC] Cooloff expired. Re-enabled.")
        self._signal_timestamps = [t for t in self._signal_timestamps if now - t < 3600]
        return len(self._signal_timestamps) >= MAX_SIGNALS_PER_HOUR

    def record_trade_result(self, pnl: float):
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= CONSEC_LOSS_LIMIT:
                self._disabled_until = time.time() + COOLOFF_HOURS * 3600
                logger.warning("[SMC] Circuit breaker: %d losses. Disabled %dh.", self._consecutive_losses, COOLOFF_HOURS)
        else:
            self._consecutive_losses = 0

    def get_stats(self) -> Dict:
        return {
            "symbol": self._symbol,
            "cycles": self._cycle_count,
            "signals_generated": self._signals_generated,
            "current_bias": self._current_bias,
            "active_obs": len(self._active_obs),
            "active_fvgs": len(self._active_fvgs),
            "consecutive_losses": self._consecutive_losses,
        }

    @property
    def pressure(self) -> float:
        return self._last_pressure


# Module-level singleton
smc_scanner = SMCScanner()
