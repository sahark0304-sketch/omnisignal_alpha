"""
quant/mean_reversion_engine.py -- Scanner #5: Tick-Level Mean-Reversion Engine
OmniSignal Alpha v3.0

Institutional microstructure mean-reversion scanner that ONLY activates when
the market is in a range-bound regime (Hurst < 0.52, Vol_z < 0).

Strategy: Fade tick-level VWAP deviations with CVD divergence confirmation.
When price extends > N sigma from rolling tick-VWAP while CVD is already
reversing, the engine fades the extreme for a reversion to VWAP.

This scanner is idle during trending regimes (where scanners 1-4 operate).
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


class MeanReversionEngine:
    """
    Tick-level mean-reversion scanner for range-bound / random-walk regimes.

    Activation condition: Hurst < 0.52 (ranging market)
    Core logic:
        1. Compute rolling tick-VWAP over the window
        2. Compute rolling standard deviation of price around VWAP
        3. When price extends beyond entry_sigma bands AND CVD diverges,
           fade the move targeting VWAP
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        poll_interval: float = 5.0,
        window_secs: int = 300,
        entry_sigma: float = 1.8,
        min_cvd_divergence: float = 0.6,
        max_signals_per_hour: int = 12,
        consecutive_loss_limit: int = 5,
        cooloff_hours: int = 1,
        max_hurst_activation: float = 0.52,
    ):
        self._symbol = symbol
        self._poll_interval = poll_interval
        self._pip_size = 0.1
        self._window_secs = window_secs
        self._entry_sigma = entry_sigma
        self._min_cvd_div = min_cvd_divergence
        self._max_hurst = max_hurst_activation

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
        logger.info(
            f"[MR] Mean-Reversion Engine started for {self._symbol} "
            f"(poll {self._poll_interval}s | sigma {self._entry_sigma} "
            f"| max_hurst {self._max_hurst})"
        )
        await asyncio.sleep(25)
        while True:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[MR] Scan error: {e}")
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
            logger.info("[MR] Cooloff expired. Re-enabled.")

        self._signal_timestamps = [
            t for t in self._signal_timestamps if now - t < 3600
        ]
        if len(self._signal_timestamps) >= self._max_signals_hour:
            return True
        return False

    def record_trade_result(self, pnl: float):
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._loss_limit:
                self._disabled_until = time.time() + self._cooloff_secs
                logger.warning(
                    f"[MR] Circuit breaker: {self._consecutive_losses} losses. "
                    f"Disabled for {self._cooloff_secs // 3600}h."
                )
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------
    #  Session filter
    # ------------------------------------------------------------------

    def _is_active_session(self) -> bool:
        hour = datetime.now(timezone.utc).hour
        return 7 <= hour < 21

    # ------------------------------------------------------------------
    #  Scan cycle
    # ------------------------------------------------------------------

    async def _scan_cycle(self):
        self._cycle_count += 1

        if self._cycle_count % 200 == 0:
            logger.info(
                f"[MR] Status: {self._cycle_count} cycles | "
                f"{self._signals_generated} signals | "
                f"H={vol_regime.hurst:.2f} vol_z={vol_regime.vol_z:.2f}"
            )

        if not self._is_active_session():
            return

        # Only activate in range-bound regimes
        if vol_regime.hurst >= self._max_hurst:
            return

        if not _mt5_available or mt5 is None:
            return

        if self._is_circuit_open():
            return

        ticks = self._fetch_ticks()
        if ticks is None or len(ticks) < 150:
            return

        analysis = self._analyze_reversion(ticks)
        if analysis is None:
            return

        action = analysis.get("action")
        if action is None:
            return

        spread = (analysis["current_ask"] - analysis["current_bid"]) / self._pip_size
        if spread > 12.0:
            return

        now = time.time()
        if self._last_signal_time and (now - self._last_signal_time) < 20:
            return
        if (
            self._last_signal_direction == action
            and self._last_signal_time
            and (now - self._last_signal_time) < 90
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
                self._symbol, from_time, 8000, mt5.COPY_TICKS_ALL
            )
            if ticks is None or len(ticks) == 0:
                return None
            return ticks
        except Exception as e:
            logger.debug(f"[MR] Tick fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    #  Core mean-reversion analysis
    # ------------------------------------------------------------------

    def _analyze_reversion(self, ticks: np.ndarray) -> Optional[Dict]:
        now_epoch = time.time()
        tick_times = ticks["time"].astype(float)
        window_start = now_epoch - self._window_secs
        mask = tick_times >= window_start
        if mask.sum() < 100:
            return None

        w_ticks = ticks[mask]
        bids = w_ticks["bid"].astype(float)
        asks = w_ticks["ask"].astype(float)
        mids = (bids + asks) / 2.0
        flags = w_ticks["flags"].astype(int)
        volumes = w_ticks["volume"].astype(float)
        lasts = w_ticks["last"].astype(float)

        # ── Tick-VWAP computation ──
        vol = np.maximum(volumes, 1.0)
        cum_vol = np.cumsum(vol)
        cum_pv = np.cumsum(mids * vol)
        vwap = cum_pv / np.maximum(cum_vol, 1.0)

        current_mid = float(mids[-1])
        current_vwap = float(vwap[-1])

        # ── Rolling deviation from VWAP ──
        deviations = mids - vwap
        lookback = min(200, len(deviations))
        recent_devs = deviations[-lookback:]
        sigma = float(np.std(recent_devs))
        if sigma < 1e-6:
            return None

        current_dev = current_mid - current_vwap
        z_from_vwap = current_dev / sigma

        # Update convergence pressure (negative z = bullish for MR)
        self._last_pressure = max(-1.0, min(1.0, -z_from_vwap / max(self._entry_sigma, 0.1)))

        # ── CVD computation ──
        buy_vol = np.zeros(len(w_ticks))
        sell_vol = np.zeros(len(w_ticks))

        has_flags = bool(np.any((flags & _FLAG_BUY) | (flags & _FLAG_SELL)))
        has_last = bool(np.any(lasts > 0))

        if has_flags:
            is_buy = (flags & _FLAG_BUY).astype(bool)
            is_sell = (flags & _FLAG_SELL).astype(bool)
            buy_vol = np.where(is_buy, vol, 0.0)
            sell_vol = np.where(is_sell, vol, 0.0)
            neutral = ~is_buy & ~is_sell
            if has_last:
                mid_arr = (bids + asks) / 2.0
                buy_vol = np.where(neutral & (lasts >= mid_arr), vol, buy_vol)
                sell_vol = np.where(neutral & (lasts < mid_arr), vol, sell_vol)
        elif has_last:
            mid_arr = (bids + asks) / 2.0
            buy_vol = np.where(lasts >= mid_arr, vol, 0.0)
            sell_vol = np.where(lasts < mid_arr, vol, 0.0)
        else:
            bid_diff = np.diff(bids, prepend=bids[0])
            buy_vol = np.where(bid_diff > 0, 1.0, 0.0)
            sell_vol = np.where(bid_diff < 0, 1.0, 0.0)

        cvd = np.cumsum(buy_vol - sell_vol)

        # ── CVD divergence detection ──
        # Price at upper extreme but CVD declining = bearish divergence (SELL)
        # Price at lower extreme but CVD rising = bullish divergence (BUY)
        third = max(len(cvd) // 3, 1)
        cvd_recent = float(np.mean(cvd[-third:]))
        cvd_middle = float(np.mean(cvd[third:2*third]))

        cvd_delta = cvd_recent - cvd_middle
        total_vol_sum = float(np.sum(vol))
        cvd_norm = cvd_delta / max(total_vol_sum * 0.01, 1.0)

        action = None

        if z_from_vwap >= self._entry_sigma:
            # Price extended ABOVE VWAP -- look for bearish CVD divergence
            if cvd_norm < -self._min_cvd_div:
                action = "SELL"
        elif z_from_vwap <= -self._entry_sigma:
            # Price extended BELOW VWAP -- look for bullish CVD divergence
            if cvd_norm > self._min_cvd_div:
                action = "BUY"

        current_bid = float(bids[-1])
        current_ask = float(asks[-1])

        return {
            "action": action,
            "current_bid": current_bid,
            "current_ask": current_ask,
            "vwap": current_vwap,
            "z_from_vwap": z_from_vwap,
            "sigma": sigma,
            "cvd_norm": cvd_norm,
            "tick_count": len(w_ticks),
            "hurst": vol_regime.hurst,
            "vol_z": vol_regime.vol_z,
        }

    # ------------------------------------------------------------------
    #  ATR helper
    # ------------------------------------------------------------------

    def _get_m5_atr(self, period: int = 14) -> float:
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

    async def _generate_signal(self, action: str, analysis: Dict):
        atr = self._get_m5_atr()

        # Mean-reversion trades use tighter stops + TP at VWAP
        if action == "BUY":
            entry = analysis["current_ask"]
            # SL: below recent low or 2x ATR, whichever is tighter
            sl_dist = max(1.0 * atr, 15 * self._pip_size)
            sl_dist = min(sl_dist, 2.0 * atr)
            sl = round(entry - sl_dist, 2)
            tp_dist = max(abs(entry - analysis["vwap"]) * 0.8, 8 * self._pip_size)
            tp_dist = min(tp_dist, max(25 * self._pip_size, 1.5 * atr))
            tp = round(entry + tp_dist, 2)
        else:
            entry = analysis["current_bid"]
            sl_dist = max(1.0 * atr, 15 * self._pip_size)
            sl_dist = min(sl_dist, 2.0 * atr)
            sl = round(entry + sl_dist, 2)
            tp_dist = max(abs(entry - analysis["vwap"]) * 0.8, 8 * self._pip_size)
            tp_dist = min(tp_dist, max(25 * self._pip_size, 1.5 * atr))
            tp = round(entry - tp_dist, 2)

        text = (
            f"{self._symbol} {action} @ {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP: {tp:.2f}\n"
            f"[Auto-MR] Mean-reversion | z={analysis['z_from_vwap']:+.2f}sigma "
            f"| CVD_div={analysis['cvd_norm']:+.2f} "
            f"| VWAP={analysis['vwap']:.2f} "
            f"| H={analysis['hurst']:.2f}"
        )

        signal = RawSignal(content=text, source="AUTO_MR")
        await push(signal)

        now = time.time()
        self._signals_generated += 1
        self._signal_timestamps.append(now)
        self._last_signal_time = now
        self._last_signal_direction = action

        logger.info(
            f"[MR] SIGNAL: {self._symbol} {action} @ {entry:.2f} "
            f"| z={analysis['z_from_vwap']:+.2f} "
            f"| CVD_div={analysis['cvd_norm']:+.2f} "
            f"| VWAP={analysis['vwap']:.2f} "
            f"| H={analysis['hurst']:.2f} vol_z={analysis['vol_z']:.2f} "
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
            "regime_active": vol_regime.hurst < self._max_hurst,
            "current_hurst": vol_regime.hurst,
            "current_vol_z": vol_regime.vol_z,
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

mr_engine = MeanReversionEngine()
