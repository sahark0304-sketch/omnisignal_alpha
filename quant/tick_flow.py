"""
quant/tick_flow.py -- Tick Flow Imbalance (TFI) Engine.

Microstructure-level signal generator that detects institutional accumulation
and distribution from real-time tick data on XAUUSD.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List

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

_FLAG_BUY = 32
_FLAG_SELL = 64


class TickFlowEngine:
    """Detects institutional order flow imbalance from tick-level data."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        poll_interval: float = 4.0,
        tfi_threshold: float = 0.28,
        window_secs: int = 180,
        max_signals_per_hour: int = 12,
        consecutive_loss_limit: int = 5,
        cooloff_hours: int = 1,
    ):
        self._symbol = symbol
        self._poll_interval = poll_interval
        self._pip_size = 0.1
        self._base_tfi_threshold = tfi_threshold
        self._tfi_threshold = tfi_threshold
        self._window_secs = window_secs

        self._max_signals_hour = max_signals_per_hour
        self._loss_limit = consecutive_loss_limit
        self._cooloff_secs = cooloff_hours * 3600

        self._signal_timestamps: List[float] = []
        self._consecutive_losses = 0
        self._disabled_until: Optional[float] = None
        self._last_signal_time: Optional[float] = None
        self._last_signal_direction: Optional[str] = None
        self._signals_generated = 0
        self._cycle_count = 0
        self._last_pressure = 0.0

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------

    async def run(self):
        """Main async loop. Polls tick data every ~4 seconds."""
        logger.info(
            f"[TFI] Engine started for {self._symbol} "
            f"(poll {self._poll_interval}s | threshold {self._tfi_threshold} "
            f"| window {self._window_secs}s)"
        )
        await asyncio.sleep(20)
        while True:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[TFI] Scan error: {e}")
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    #  Circuit breaker
    # ------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        now = time.time()

        if self._disabled_until and now < self._disabled_until:
            return True
        if self._disabled_until and now >= self._disabled_until:
            self._disabled_until = None
            self._consecutive_losses = 0
            logger.info("[TFI] Cooloff period ended. Engine re-enabled.")

        self._signal_timestamps = [
            t for t in self._signal_timestamps if now - t < 3600
        ]
        if len(self._signal_timestamps) >= self._max_signals_hour:
            return True

        return False

    def record_trade_result(self, pnl: float):
        """Called externally when a TFI-sourced trade closes."""
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._loss_limit:
                self._disabled_until = time.time() + self._cooloff_secs
                logger.warning(
                    f"[TFI] CIRCUIT BREAKER: {self._consecutive_losses} "
                    f"consecutive losses. Disabled for "
                    f"{self._cooloff_secs // 3600}h."
                )
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------
    #  Scan cycle
    # ------------------------------------------------------------------

    async def _scan_cycle(self):
        self._cycle_count += 1

        if self._cycle_count % 200 == 0:
            logger.info(
                f"[TFI] Status: {self._cycle_count} cycles | "
                f"{self._signals_generated} signals | "
                f"losses_streak={self._consecutive_losses}"
            )

        sf = vol_regime.scale_factor()
        self._tfi_threshold = max(0.15, self._base_tfi_threshold * sf)
        self._last_pressure = 0.0

        if not _mt5_available or mt5 is None:
            return

        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        if hour < 7 or hour >= 21:
            return

        if self._is_circuit_open():
            return

        ticks = self._fetch_ticks()
        if ticks is None or len(ticks) < 200:
            return

        analysis = self._analyze_flow(ticks)
        if analysis is None:
            return

        tfi = analysis["tfi_ratio"]
        self._last_pressure = max(-1.0, min(1.0, tfi / max(self._tfi_threshold, 0.01)))
        compression = analysis["vol_compression"]
        cvd_accel = analysis["cvd_acceleration"]

        action = None
        if tfi > self._tfi_threshold and cvd_accel > 0 and compression:
            action = "BUY"
        elif tfi < -self._tfi_threshold and cvd_accel < 0 and compression:
            action = "SELL"

        if action is None:
            return

        spread = (analysis["current_ask"] - analysis["current_bid"]) / self._pip_size
        if spread > 12.0:
            return

        now = time.time()
        if self._last_signal_time and (now - self._last_signal_time) < 30:
            return
        if (
            self._last_signal_direction == action
            and self._last_signal_time
            and (now - self._last_signal_time) < 120
        ):
            return

        await self._generate_signal(action, analysis)

    # ------------------------------------------------------------------
    #  Tick fetching
    # ------------------------------------------------------------------

    def _fetch_ticks(self) -> Optional[np.ndarray]:
        try:
            mt5.symbol_select(self._symbol, True)
            utc_now = datetime.now(timezone.utc)
            from_time = utc_now - timedelta(seconds=self._window_secs + 60)
            ticks = mt5.copy_ticks_from(
                self._symbol, from_time, 5000, mt5.COPY_TICKS_ALL
            )
            if ticks is None or len(ticks) == 0:
                return None
            return ticks
        except Exception as e:
            logger.debug(f"[TFI] Tick fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    #  Flow analysis (core microstructure logic)
    # ------------------------------------------------------------------

    def _analyze_flow(self, ticks: np.ndarray) -> Optional[Dict]:
        now_epoch = time.time()
        tick_times = ticks["time"].astype(float)
        window_start = now_epoch - self._window_secs
        mask = tick_times >= window_start
        if mask.sum() < 100:
            return None

        w_ticks = ticks[mask]
        bids = w_ticks["bid"].astype(float)
        asks = w_ticks["ask"].astype(float)
        flags = w_ticks["flags"].astype(int)
        volumes = w_ticks["volume"].astype(float)
        lasts = w_ticks["last"].astype(float)

        has_flags = bool(np.any((flags & _FLAG_BUY) | (flags & _FLAG_SELL)))
        has_last = bool(np.any(lasts > 0))

        buy_vol = np.zeros(len(w_ticks))
        sell_vol = np.zeros(len(w_ticks))

        if has_flags:
            is_buy = (flags & _FLAG_BUY).astype(bool)
            is_sell = (flags & _FLAG_SELL).astype(bool)
            vol = np.maximum(volumes, 1.0)
            buy_vol = np.where(is_buy, vol, 0.0)
            sell_vol = np.where(is_sell, vol, 0.0)
            neutral = ~is_buy & ~is_sell
            if has_last:
                mids = (bids + asks) / 2.0
                buy_vol = np.where(neutral & (lasts >= mids), vol, buy_vol)
                sell_vol = np.where(neutral & (lasts < mids), vol, sell_vol)
        elif has_last:
            mids = (bids + asks) / 2.0
            vol = np.maximum(volumes, 1.0)
            buy_vol = np.where(lasts >= mids, vol, 0.0)
            sell_vol = np.where(lasts < mids, vol, 0.0)
        else:
            bid_diff = np.diff(bids, prepend=bids[0])
            buy_vol = np.where(bid_diff > 0, 1.0, 0.0)
            sell_vol = np.where(bid_diff < 0, 1.0, 0.0)

        total_buy = float(np.sum(buy_vol))
        total_sell = float(np.sum(sell_vol))
        total_vol = total_buy + total_sell

        if total_vol < 10:
            return None

        tfi_ratio = (total_buy - total_sell) / total_vol

        cvd = np.cumsum(buy_vol - sell_vol)
        half = len(cvd) // 2
        if half < 10:
            return None
        cvd_first_half = float(np.mean(cvd[:half]))
        cvd_second_half = float(np.mean(cvd[half:]))
        cvd_acceleration = cvd_second_half - cvd_first_half

        w_times = w_ticks["time"].astype(float)
        duration = float(w_times[-1] - w_times[0])
        tick_count = len(w_ticks)

        price_range = float(np.max(asks) - np.min(bids))
        atr_m1 = self._get_m1_atr()
        if atr_m1 <= 0:
            atr_m1 = price_range * 0.5

        range_ratio = price_range / atr_m1 if atr_m1 > 0 else 1.0

        n_thirds = max(len(w_ticks) // 3, 1)
        t_first = float(w_times[min(n_thirds - 1, len(w_times) - 1)] - w_times[0])
        t_last = float(w_times[-1] - w_times[max(-n_thirds, -len(w_times))])
        rate_first = n_thirds / max(t_first, 1.0)
        rate_last = n_thirds / max(t_last, 1.0)
        rate_accelerating = rate_last > rate_first * 1.2

        vol_compression = range_ratio < (0.92 / max(vol_regime.scale_factor(), 0.5)) and (rate_accelerating or vol_regime.is_low_vol)

        current_bid = float(bids[-1])
        current_ask = float(asks[-1])

        recent = min(50, len(bids))
        accum_low = float(np.min(bids[-recent:]))
        accum_high = float(np.max(asks[-recent:]))

        return {
            "tfi_ratio": round(tfi_ratio, 4),
            "total_buy": total_buy,
            "total_sell": total_sell,
            "cvd_final": float(cvd[-1]),
            "cvd_acceleration": cvd_acceleration,
            "tick_rate": round(tick_count / max(duration, 1.0), 1),
            "range_ratio": round(range_ratio, 3),
            "rate_accelerating": rate_accelerating,
            "vol_compression": vol_compression,
            "tick_count": tick_count,
            "current_bid": current_bid,
            "current_ask": current_ask,
            "accum_low": accum_low,
            "accum_high": accum_high,
            "atr_m1": atr_m1,
        }

    # ------------------------------------------------------------------
    #  ATR helper
    # ------------------------------------------------------------------

    def _get_m1_atr(self, period: int = 14) -> float:
        try:
            rates = mt5.copy_rates_from_pos(
                self._symbol, mt5.TIMEFRAME_M1, 0, period + 1
            )
            if rates is None or len(rates) < period:
                return 0.0
            highs = np.array([r[2] for r in rates])
            lows = np.array([r[3] for r in rates])
            closes = np.array([r[4] for r in rates])
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]),
                ),
            )
            return float(np.mean(tr[-period:]))
        except Exception:
            return 0.0

    def _get_m5_atr(self, period: int = 14) -> float:
        """Fetch M5 ATR(14) for wider stop placement."""
        try:
            rates = mt5.copy_rates_from_pos(
                self._symbol, mt5.TIMEFRAME_M5, 0, period + 1
            )
            if rates is None or len(rates) < period:
                return 15 * self._pip_size
            highs = np.array([r[2] for r in rates])
            lows = np.array([r[3] for r in rates])
            closes = np.array([r[4] for r in rates])
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

    async def _generate_signal(self, action: str, analysis: Dict):
        atr = analysis["atr_m1"]
        if atr <= 0:
            atr = 15 * self._pip_size

        m5_atr = self._get_m5_atr()

        if action == "BUY":
            entry = analysis["current_ask"]
            sl_micro = analysis["accum_low"] - 3 * self._pip_size
            sl_atr = entry - max(1.0 * m5_atr, 30 * self._pip_size)
            sl = min(sl_micro, sl_atr)
            sl_dist = abs(entry - sl)
            _max_sl = max(40 * self._pip_size, 2.0 * m5_atr)
            if sl_dist > _max_sl:
                sl = entry - _max_sl
                sl_dist = _max_sl
            if sl_dist < 8 * self._pip_size:
                sl = entry - 10 * self._pip_size
                sl_dist = 10 * self._pip_size
            tp = entry + max(sl_dist * 1.5, 10 * self._pip_size)
        else:
            entry = analysis["current_bid"]
            sl_micro = analysis["accum_high"] + 3 * self._pip_size
            sl_atr = entry + max(1.0 * m5_atr, 30 * self._pip_size)
            sl = max(sl_micro, sl_atr)
            sl_dist = abs(sl - entry)
            _max_sl = max(40 * self._pip_size, 2.0 * m5_atr)
            if sl_dist > _max_sl:
                sl = entry + _max_sl
                sl_dist = _max_sl
            if sl_dist < 8 * self._pip_size:
                sl = entry + 10 * self._pip_size
                sl_dist = 10 * self._pip_size
            tp = entry - max(sl_dist * 1.5, 10 * self._pip_size)

        text = (
            f"{self._symbol} {action} @ {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP: {tp:.2f}\n"
            f"[Auto-TFI] Tick flow imbalance {analysis['tfi_ratio']:+.2f} "
            f"| CVD accel={analysis['cvd_acceleration']:.0f} "
            f"| ticks={analysis['tick_count']} "
            f"| range_ratio={analysis['range_ratio']:.2f}"
        )

        signal = RawSignal(content=text, source="AUTO_TFI")
        await push(signal)

        now = time.time()
        self._signals_generated += 1
        self._signal_timestamps.append(now)
        self._last_signal_time = now
        self._last_signal_direction = action

        logger.info(
            f"[TFI] SIGNAL: {self._symbol} {action} @ {entry:.2f} "
            f"| TFI={analysis['tfi_ratio']:+.3f} "
            f"| CVD_accel={analysis['cvd_acceleration']:.0f} "
            f"| compression={analysis['vol_compression']} "
            f"| SL={sl:.2f} TP={tp:.2f}"
        )

    # ------------------------------------------------------------------
    #  Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        return {
            "symbol": self._symbol,
            "cycles": self._cycle_count,
            "signals_generated": self._signals_generated,
            "consecutive_losses": self._consecutive_losses,
            "disabled_until": (
                datetime.fromtimestamp(self._disabled_until).isoformat()
                if self._disabled_until
                else None
            ),
            "signals_this_hour": len(
                [t for t in self._signal_timestamps if time.time() - t < 3600]
            ),
            "last_signal_time": (
                datetime.fromtimestamp(self._last_signal_time).isoformat()
                if self._last_signal_time
                else None
            ),
        }

    @property
    def pressure(self) -> float:
        return self._last_pressure


# -----------------------------------------------------------------------
#  Module-Level Singleton
# -----------------------------------------------------------------------

tick_flow_engine = TickFlowEngine()
