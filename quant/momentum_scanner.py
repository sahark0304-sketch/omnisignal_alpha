"""
quant/momentum_scanner.py -- M1 Momentum Pullback Scanner (OmniSignal Alpha v3.0)

Autonomous background scanner that polls XAUUSD M1 data for EMA(20) momentum
pullback patterns and pushes auto-generated signals into the pipeline.
"""

import asyncio
import time
from datetime import datetime, timezone
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
#  Momentum Pullback Scanner
# -----------------------------------------------------------------------

class MomentumScanner:
    """Detects M1 EMA(20) momentum pullbacks on XAUUSD and generates trading signals."""

    def __init__(self, symbol: str = "XAUUSD", poll_interval: float = 8.0):
        self._symbol = symbol
        self._poll_interval = poll_interval
        self._pip_size = 0.1
        self._ema_period = 20
        self._lookback = 60
        self._base_slope_threshold = 0.8
        self._slope_threshold = 0.8
        self._cooldown_secs = 120
        self._dedup_secs = 300
        self._max_spread_pips = 12.0

        self._last_signal_time: Optional[float] = None
        self._last_signal_direction: Optional[str] = None
        self._last_direction_time: Optional[float] = None
        self._signals_generated: int = 0
        self._max_signals_per_hour: int = 10
        self._hourly_timestamps: list = []
        self._last_pressure: float = 0.0
        self._last_slope: float = 0.0
        self._slope_history: list = []
        self._cycle_count: int = 0

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    async def run(self):
        """Main async loop. Call as asyncio.create_task(scanner.run())."""
        logger.info(
            f"[MomentumScanner] Started for {self._symbol} "
            f"(poll every {self._poll_interval}s)"
        )
        await asyncio.sleep(12)
        while True:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[MomentumScanner] Scan error: {e}")
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    #  Single scan cycle
    # ------------------------------------------------------------------

    async def _scan_cycle(self):
        """One scan iteration: fetch data, compute EMA, detect pullback."""
        sf = vol_regime.scale_factor()
        self._slope_threshold = self._base_slope_threshold * sf

        # Convergence pressure: normalized slope / threshold
        self._last_pressure = 0.0
        self._cycle_count += 1

        if self._cycle_count % 100 == 0:
            logger.info(
                f"[MomentumScanner] Status: {self._cycle_count} cycles | "
                f"{self._signals_generated} signals generated"
            )

        if not self._is_active_session():
            return

        if not _mt5_available or mt5 is None:
            logger.warning("[MomentumScanner] MT5 not available, skipping cycle")
            return

        rates = self._fetch_m1_rates()
        if rates is None:
            return

        if len(rates) < self._lookback:
            logger.warning(
                f"[MomentumScanner] Insufficient data: got {len(rates)}, "
                f"need {self._lookback}"
            )
            return

        spread_raw = float(rates[-1]["spread"])
        if spread_raw > self._max_spread_pips:
            logger.debug(
                f"[MomentumScanner] Spread too wide: "
                f"{spread_raw:.1f} pips, skipping"
            )
            return

        closes = rates["close"].astype(float)
        ema = self._compute_ema(closes, self._ema_period)

        if len(ema) < 6:
            return

        slope = (ema[-1] - ema[-5]) / (5 * self._pip_size)
        self._last_slope = slope
        self._slope_history.append((time.time(), slope))
        self._slope_history = [(t, s) for t, s in self._slope_history if time.time() - t < 600]

        if abs(slope) < self._slope_threshold:
            return

        # Update convergence pressure
        self._last_pressure = max(-1.0, min(1.0, slope / max(self._slope_threshold, 0.01)))

        setup = self._check_pullback(rates, ema, slope)
        if setup is None:
            return

        now = time.time()
        if self._last_signal_time and (now - self._last_signal_time) < self._cooldown_secs:
            logger.debug("[MomentumScanner] In cooldown, skipping signal")
            return

        direction = setup["action"]
        if (self._last_signal_direction == direction
                and self._last_direction_time is not None
                and (now - self._last_direction_time) < self._dedup_secs):
            logger.debug(
                f"[MomentumScanner] Duplicate {direction} within dedup window, skipping"
            )
            return

        label = "BULLISH" if direction == "BUY" else "BEARISH"
        logger.info(
            f"[MomentumScanner] PULLBACK: {self._symbol} {label} | "
            f"slope={slope:.1f}p/bar | vol={setup['vol_ratio']:.1f}x avg | "
            f"generating {direction}"
        )

        # Hourly signal cap
        self._hourly_timestamps = [t for t in self._hourly_timestamps if (now - t) < 3600]
        if len(self._hourly_timestamps) >= self._max_signals_per_hour:
            logger.debug('[MomentumScanner] Max signals/hour reached, throttling')
            return

        await self._generate_signal(setup)

    # ------------------------------------------------------------------
    #  Session filter
    # ------------------------------------------------------------------

    def _is_active_session(self) -> bool:
        """Only scan during London (07-13 UTC) and New York (13-21 UTC)."""
        hour = datetime.now(timezone.utc).hour
        return 7 <= hour < 21

    # ------------------------------------------------------------------
    #  MT5 data fetch
    # ------------------------------------------------------------------

    def _fetch_m1_rates(self) -> Optional[np.ndarray]:
        """Fetch M1 bars from MT5."""
        try:
            mt5.symbol_select(self._symbol, True)
            rates = mt5.copy_rates_from_pos(
                self._symbol, mt5.TIMEFRAME_M1, 0, self._lookback
            )
            if rates is None or len(rates) == 0:
                logger.warning(
                    f"[MomentumScanner] No M1 data returned for {self._symbol}"
                )
                return None
            return rates
        except Exception as e:
            logger.warning(f"[MomentumScanner] MT5 fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    #  EMA computation
    # ------------------------------------------------------------------

    def _compute_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA using numpy with exponential weights."""
        if len(closes) < period:
            return np.array([])
        alpha = 2.0 / (period + 1)
        ema = np.empty(len(closes), dtype=float)
        ema[0] = closes[0]
        for i in range(1, len(closes)):
            ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
        return ema

    # ------------------------------------------------------------------
    #  Pullback detection
    # ------------------------------------------------------------------

    def _check_pullback(
        self,
        rates: np.ndarray,
        ema: np.ndarray,
        slope: float,
    ) -> Optional[Dict]:
        """
        Check last 3 candles for a momentum pullback to EMA(20).

        Returns dict with signal parameters or None if no pullback found.
        """
        bullish = slope > self._slope_threshold
        vol_avg_window = min(10, len(rates) - 3)
        if vol_avg_window < 1:
            return None
        vol_avg = float(np.mean(rates["tick_volume"][-3 - vol_avg_window:-3]))

        for offset in range(1, 4):
            idx = -offset
            if abs(idx) > len(rates) or abs(idx) > len(ema):
                continue

            candle = rates[idx]
            c_high = float(candle["high"])
            c_low = float(candle["low"])
            c_close = float(candle["close"])
            c_volume = float(candle["tick_volume"])
            ema_val = ema[idx]

            if vol_avg > 0 and c_volume >= vol_avg:
                continue

            vol_ratio = c_volume / vol_avg if vol_avg > 0 else 0.0

            if bullish:
                touched_ema = c_low <= ema_val
                close_above = c_close > ema_val
                if touched_ema and close_above:
                    entry = float(rates[-1]["close"])
                    m5_atr = self._get_m5_atr()
                    sl_dist = max(1.0 * m5_atr, 40 * self._pip_size)
                    sl_dist = min(sl_dist, 2.0 * m5_atr)
                    sl = round(entry - sl_dist, 2)
                    risk = sl_dist
                    tp = round(entry + 1.5 * risk, 2)

                    return {
                        "action": "BUY",
                        "entry": round(entry, 2),
                        "sl": sl,
                        "tp": tp,
                        "slope": slope,
                        "vol_ratio": vol_ratio,
                    }
            else:
                touched_ema = c_high >= ema_val
                close_below = c_close < ema_val
                if touched_ema and close_below:
                    entry = float(rates[-1]["close"])
                    m5_atr = self._get_m5_atr()
                    sl_dist = max(1.0 * m5_atr, 40 * self._pip_size)
                    sl_dist = min(sl_dist, 2.0 * m5_atr)
                    sl = round(entry + sl_dist, 2)
                    risk = sl_dist
                    tp = round(entry - 1.5 * risk, 2)

                    return {
                        "action": "SELL",
                        "entry": round(entry, 2),
                        "sl": sl,
                        "tp": tp,
                        "slope": slope,
                        "vol_ratio": vol_ratio,
                    }

        return None

    # ------------------------------------------------------------------
    #  ATR computation
    # ------------------------------------------------------------------

    def _get_atr(self, rates: np.ndarray, period: int = 14) -> float:
        """Compute ATR from rate data."""
        if len(rates) < period + 1:
            return 10 * self._pip_size

        highs = rates["high"].astype(float)
        lows = rates["low"].astype(float)
        closes = rates["close"].astype(float)

        tr_values = np.empty(len(rates) - 1, dtype=float)
        for i in range(1, len(rates)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr_values[i - 1] = max(hl, hc, lc)

        atr = float(np.mean(tr_values[-period:]))
        return atr

    def _get_m5_atr(self, period: int = 14) -> float:
        """Fetch fresh M5 ATR(14) from MT5 for wider stop placement."""
        try:
            rates = mt5.copy_rates_from_pos(
                self._symbol, mt5.TIMEFRAME_M5, 0, period + 1
            )
            if rates is None or len(rates) < period:
                return 15 * self._pip_size
            highs = rates["high"].astype(float)
            lows = rates["low"].astype(float)
            closes = rates["close"].astype(float)
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]),
                ),
            )
            return float(np.mean(tr[-period:]))
        except Exception:
            return 15 * self._pip_size

    # ------------------------------------------------------------------
    #  Signal generation
    # ------------------------------------------------------------------

    async def _generate_signal(self, setup: Dict):
        """Build a RawSignal and push it into the processing queue."""
        text = (
            f"{self._symbol} {setup['action']} @ {setup['entry']:.2f}\n"
            f"SL: {setup['sl']:.2f}\n"
            f"TP: {setup['tp']:.2f}\n"
            f"[Auto-Pullback] EMA(20) momentum pullback | "
            f"slope={setup['slope']:.1f} pips/bar | "
            f"vol_ratio={setup['vol_ratio']:.1f}x"
        )

        signal = RawSignal(content=text, source="AUTO_PULLBACK")
        await push(signal)

        now = time.time()
        self._signals_generated += 1
        self._last_signal_time = now
        self._hourly_timestamps.append(now)
        self._last_signal_direction = setup["action"]
        self._last_direction_time = now

        logger.info(
            f"[MomentumScanner] Signal pushed: {setup['action']} "
            f"@ {setup['entry']:.2f} | SL {setup['sl']:.2f} | "
            f"TP {setup['tp']:.2f}"
        )

    # ------------------------------------------------------------------
    #  Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return scanner statistics for logging."""
        return {
            "symbol": self._symbol,
            "cycles": self._cycle_count,
            "signals_generated": self._signals_generated,
            "last_signal_time": (
                datetime.fromtimestamp(self._last_signal_time).isoformat()
                if self._last_signal_time else None
            ),
            "last_signal_direction": self._last_signal_direction,
            "poll_interval": self._poll_interval,
        }

    @property
    def pressure(self) -> float:
        return self._last_pressure

    @property
    def last_slope(self) -> float:
        return self._last_slope

    @property
    def slope_delta(self) -> float:
        """Rate of change of slope over last 2 readings (10-bar window)."""
        if len(self._slope_history) < 2:
            return 0.0
        _, older = self._slope_history[0]
        _, newer = self._slope_history[-1]
        if abs(older) < 1e-6:
            return 0.0
        return (newer - older) / abs(older)


# -----------------------------------------------------------------------
#  Module-Level Singleton
# -----------------------------------------------------------------------

momentum_scanner = MomentumScanner()
