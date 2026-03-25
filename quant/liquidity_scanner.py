"""
quant/liquidity_scanner.py -- M1 Liquidity Sweep Scanner (OmniSignal Alpha v3.0)

Autonomous background scanner that polls XAUUSD M1 data for institutional
liquidity grab patterns and pushes auto-generated signals into the pipeline.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Optional

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


# -----------------------------------------------------------------------
#  Liquidity Sweep Scanner
# -----------------------------------------------------------------------

class LiquidityScanner:
    """Detects M1 liquidity sweeps on XAUUSD and generates trading signals."""

    def __init__(self, symbol: str = "XAUUSD", poll_interval: float = 10.0):
        self._symbol = symbol
        self._poll_interval = poll_interval
        self._pip_size = 0.1
        self._base_min_sweep_pips = 7.0
        self._min_sweep_pips = 7.0
        self._base_volume_mult = 1.5
        self._volume_mult = 1.5
        self._base_wick_pct = 0.60
        self._wick_pct_threshold = 0.60
        self._cooldown_secs = 90
        self._range_lookback = 50
        self._volume_avg_lookback = 20
        self._dedup_window_secs = 180
        self._max_spread_pips = 12.0

        self._last_signal_time: Optional[float] = None
        self._last_sweep_level: Optional[float] = None
        self._last_sweep_time: Optional[float] = None
        self._signals_generated: int = 0
        self._last_pressure: float = 0.0
        self._sweeps_detected: int = 0
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    async def run(self):
        """Main async loop. Call as asyncio.create_task(scanner.run())."""
        logger.info(
            f"[LiquidityScanner] Started for {self._symbol} "
            f"(poll every {self._poll_interval}s)"
        )
        await asyncio.sleep(15)
        while True:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[LiquidityScanner] Scan error: {e}")
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    #  Single scan cycle
    # ------------------------------------------------------------------

    async def _scan_cycle(self):
        """One scan iteration: fetch data, detect sweep, optionally signal."""
        sf = vol_regime.scale_factor()
        self._min_sweep_pips = max(3.0, self._base_min_sweep_pips * sf)
        self._volume_mult = max(1.15, self._base_volume_mult * sf)
        self._wick_pct_threshold = max(0.40, self._base_wick_pct * sf)
        self._cycle_count += 1

        if self._cycle_count % 100 == 0:
            logger.info(
                f"[LiquidityScanner] Status: {self._cycle_count} cycles | "
                f"{self._sweeps_detected} sweeps detected | "
                f"{self._signals_generated} signals generated"
            )

        if not _mt5_available or mt5 is None:
            logger.warning("[LiquidityScanner] MT5 not available, skipping cycle")
            return

        rates = self._fetch_m1_rates()
        if rates is None:
            return

        needed = self._range_lookback + 1
        if len(rates) < needed:
            logger.warning(
                f"[LiquidityScanner] Insufficient data: got {len(rates)}, "
                f"need {needed}"
            )
            return

        history = rates[:-1]
        range_high = float(np.max(history['high'][-self._range_lookback:]))
        range_low = float(np.min(history['low'][-self._range_lookback:]))

        vol_window = min(self._volume_avg_lookback, len(history))
        vol_avg = float(np.mean(history['tick_volume'][-vol_window:]))

        sweep = self._check_sweep(rates, range_high, range_low, vol_avg)
        if sweep is None:
            return

        self._sweeps_detected += 1

        now = time.time()
        if self._last_signal_time and (now - self._last_signal_time) < self._cooldown_secs:
            logger.debug(
                "[LiquidityScanner] Sweep found but in cooldown, skipping signal"
            )
            return

        if self._last_sweep_level is not None and self._last_sweep_time is not None:
            level_match = abs(sweep["sweep_level"] - self._last_sweep_level) < (5 * self._pip_size)
            time_match = (now - self._last_sweep_time) < self._dedup_window_secs
            if level_match and time_match:
                logger.debug(
                    "[LiquidityScanner] Duplicate sweep at same level, skipping"
                )
                return

        spread_pips = self._get_spread_pips()
        if spread_pips is None:
            logger.warning(
                "[LiquidityScanner] Could not read bid/ask spread, skipping signal"
            )
            return
        if spread_pips > self._max_spread_pips:
            logger.debug(
                f"[LiquidityScanner] Spread too wide "
                f"({spread_pips:.1f} pips > {self._max_spread_pips}), skipping signal"
            )
            return

        self._last_sweep_level = sweep["sweep_level"]
        self._last_sweep_time = now

        await self._generate_signal(sweep)

    # ------------------------------------------------------------------
    #  MT5 data fetch
    # ------------------------------------------------------------------

    def _fetch_m1_rates(self) -> Optional[np.ndarray]:
        """Fetch M1 bars from MT5."""
        try:
            mt5.symbol_select(self._symbol, True)
            rates = mt5.copy_rates_from_pos(
                self._symbol, mt5.TIMEFRAME_M1, 0, self._range_lookback + 5
            )
            if rates is None or len(rates) == 0:
                logger.warning(
                    f"[LiquidityScanner] No M1 data returned for {self._symbol}"
                )
                return None
            return rates
        except Exception as e:
            logger.warning(f"[LiquidityScanner] MT5 fetch failed: {e}")
            return None

    def _get_spread_pips(self) -> Optional[float]:
        """Current spread in pips from live bid/ask (XAUUSD pip = 0.1)."""
        if not _mt5_available or mt5 is None:
            return None
        tick = mt5.symbol_info_tick(self._symbol)
        if tick is None:
            return None
        ask = float(tick.ask)
        bid = float(tick.bid)
        if ask <= 0 or bid <= 0 or ask < bid:
            return None
        return (ask - bid) / self._pip_size

    # ------------------------------------------------------------------
    #  Sweep detection
    # ------------------------------------------------------------------

    def _check_sweep(
        self,
        rates: np.ndarray,
        range_high: float,
        range_low: float,
        vol_avg: float,
    ) -> Optional[Dict]:
        """
        Check last 2 candles for a liquidity sweep.

        Returns dict with keys: action, sweep_level, entry_price, sl, tp,
        volume_ratio, wick_pct, candle_low, candle_high, reason
        -- or None if no sweep detected.
        """
        min_pierce = self._min_sweep_pips * self._pip_size
        check_indices = [-1, -2]

        for idx in check_indices:
            if abs(idx) > len(rates):
                continue

            candle = rates[idx]
            c_open = float(candle['open'])
            c_high = float(candle['high'])
            c_low = float(candle['low'])
            c_close = float(candle['close'])
            c_volume = float(candle['tick_volume'])

            candle_range = c_high - c_low
            if candle_range < 1e-8:
                continue

            volume_ratio = c_volume / vol_avg if vol_avg > 0 else 0.0
            if volume_ratio < self._volume_mult:
                continue

            entry_price = float(rates[-1]['close'])

            # --- Bullish sweep (wick below range_low, close back above) ---
            if c_low < range_low - min_pierce and c_close > range_low:
                lower_wick = min(c_open, c_close) - c_low
                wick_pct = lower_wick / candle_range

                if wick_pct >= self._wick_pct_threshold:
                    sl = c_low - (5 * self._pip_size)
                    sl_distance = abs(entry_price - sl)
                    max_sl_distance = max(35 * self._pip_size, 1.5 * (c_high - c_low))
                    if sl_distance > max_sl_distance:
                        sl = entry_price - max_sl_distance

                    tp = range_high

                    logger.info(
                        f"[LiquidityScanner] SWEEP DETECTED: {self._symbol} "
                        f"BULLISH at {c_low:.2f} | vol={volume_ratio:.1f}x | "
                        f"wick={wick_pct * 100:.0f}% | generating BUY signal"
                    )

                    return {
                        "action": "BUY",
                        "sweep_level": c_low,
                        "entry_price": entry_price,
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "volume_ratio": volume_ratio,
                        "wick_pct": wick_pct,
                        "candle_low": c_low,
                        "candle_high": c_high,
                        "reason": "bullish rejection at range low",
                    }

            # --- Bearish sweep (wick above range_high, close back below) ---
            if c_high > range_high + min_pierce and c_close < range_high:
                upper_wick = c_high - max(c_open, c_close)
                wick_pct = upper_wick / candle_range

                if wick_pct >= self._wick_pct_threshold:
                    sl = c_high + (5 * self._pip_size)
                    sl_distance = abs(sl - entry_price)
                    max_sl_distance = max(35 * self._pip_size, 1.5 * (c_high - c_low))
                    if sl_distance > max_sl_distance:
                        sl = entry_price + max_sl_distance

                    tp = range_low

                    logger.info(
                        f"[LiquidityScanner] SWEEP DETECTED: {self._symbol} "
                        f"BEARISH at {c_high:.2f} | vol={volume_ratio:.1f}x | "
                        f"wick={wick_pct * 100:.0f}% | generating SELL signal"
                    )

                    return {
                        "action": "SELL",
                        "sweep_level": c_high,
                        "entry_price": entry_price,
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "volume_ratio": volume_ratio,
                        "wick_pct": wick_pct,
                        "candle_low": c_low,
                        "candle_high": c_high,
                        "reason": "bearish rejection at range high",
                    }

        return None

    # ------------------------------------------------------------------
    #  Signal generation
    # ------------------------------------------------------------------

    def _is_counter_trend_in_fast_regime(self, action: str) -> bool:
        """v4.4: Block counter-trend sweeps in FAST_TREND regime."""
        try:
            ticks = mt5.copy_ticks_from(
                self._symbol,
                datetime.now() - __import__('datetime').timedelta(seconds=45),
                5000, mt5.COPY_TICKS_ALL,
            )
            if ticks is None or len(ticks) < 30:
                return False

            prices = (ticks['bid'] + ticks['ask']) / 2.0
            displacement = (prices[-1] - prices[0]) / self._pip_size
            dt = (ticks['time'][-1] - ticks['time'][0])
            if dt <= 0:
                return False

            velocity = abs(displacement) / max(dt, 1) * 10

            if velocity < 0.4:
                return False

            if displacement > 15 and action == "SELL":
                logger.info(
                    "[LiquidityScanner] REGIME GATE: suppressing SELL sweep "
                    "in FAST_TREND bullish (disp=+%.0fp vel=%.2f)",
                    displacement, velocity,
                )
                return True
            if displacement < -15 and action == "BUY":
                logger.info(
                    "[LiquidityScanner] REGIME GATE: suppressing BUY sweep "
                    "in FAST_TREND bearish (disp=%.0fp vel=%.2f)",
                    displacement, velocity,
                )
                return True
        except Exception as e:
            logger.debug(f"[LiquidityScanner] Regime check error: {e}")
        return False

    async def _generate_signal(self, sweep: Dict):
        """Build a RawSignal and push it into the processing queue."""
        text = (
            f"{self._symbol} {sweep['action']} @ {sweep['entry_price']:.2f}\n"
            f"SL: {sweep['sl']:.2f}\n"
            f"TP: {sweep['tp']:.2f}\n"
            f"[Auto-Scanner] Liquidity sweep detected at {sweep['sweep_level']:.2f} "
            f"with {sweep['volume_ratio']:.1f}x volume surge"
        )

        signal = RawSignal(content=text, source="AUTO_SCANNER")
        await push(signal)

        self._signals_generated += 1
        self._last_signal_time = time.time()

        logger.info(
            f"[LiquidityScanner] Signal pushed: {sweep['action']} "
            f"@ {sweep['entry_price']:.2f} | SL {sweep['sl']:.2f} | "
            f"TP {sweep['tp']:.2f} | {sweep['reason']}"
        )

    # ------------------------------------------------------------------
    #  Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return scanner statistics for logging."""
        return {
            "symbol": self._symbol,
            "cycles": self._cycle_count,
            "sweeps_detected": self._sweeps_detected,
            "signals_generated": self._signals_generated,
            "last_signal_time": (
                datetime.fromtimestamp(self._last_signal_time).isoformat()
                if self._last_signal_time else None
            ),
            "last_sweep_level": self._last_sweep_level,
            "poll_interval": self._poll_interval,
        }


# -----------------------------------------------------------------------
#  Module-Level Singleton
# -----------------------------------------------------------------------


    @property
    def pressure(self) -> float:
        return self._last_pressure

liquidity_scanner = LiquidityScanner()
