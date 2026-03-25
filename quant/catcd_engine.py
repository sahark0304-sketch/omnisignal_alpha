"""
quant/catcd_engine.py -- Cross-Asset Tick Correlation Decay (CATCD) Engine.

Detects temporary breakdowns in the XAUUSD-DXY inverse correlation at the
tick level.  When the rolling 90-second correlation deviates > 2.5 sigma
from its 2-hour reference distribution, the engine trades the mean-reversion
snap-back.
"""

import asyncio
import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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


_DXY_CANDIDATES = [
    "USDX", "DXY", "DX", "USDX.f", "DXY.f", "DOLLAR", "USDollar", "USDIndex",
    "DX_Z4", "DX_H5", "DX_M5", "DX_U5", "DX_Z5", "DX_H6", "DX_M6", "DX_U6",
    "DXFUTURES", "DXY_CASH", "USDX_CASH",
]
_EURUSD_FALLBACKS = ["EURUSD", "EURUSD.f", "EURUSDm", "EURUSD.raw"]


class CATCDEngine:
    """Cross-Asset Tick Correlation Decay scanner."""

    def __init__(
        self,
        gold_symbol: str = "XAUUSD",
        poll_interval: float = 4.0,
        corr_window_secs: int = 90,
        ref_window_secs: int = 7200,
        z_threshold: float = 2.5,
        max_signals_per_hour: int = 10,
        consecutive_loss_limit: int = 4,
        cooloff_hours: int = 1,
    ):
        self._gold = gold_symbol
        self._dxy: Optional[str] = None
        self._poll_interval = poll_interval
        self._pip_size = 0.1

        self._corr_window = corr_window_secs
        self._ref_window = ref_window_secs
        self._base_z_threshold = z_threshold
        self._z_threshold = z_threshold

        self._max_signals_hour = max_signals_per_hour
        self._loss_limit = consecutive_loss_limit
        self._cooloff_secs = cooloff_hours * 3600

        self._ref_correlations: deque = deque(maxlen=2000)
        self._signal_timestamps: List[float] = []
        self._consecutive_losses = 0
        self._disabled_until: Optional[float] = None
        self._last_signal_time: Optional[float] = None
        self._signals_generated = 0
        self._last_pressure = 0.0
        self._invert_proxy = False

    def _resolve_dxy_symbol(self) -> Optional[str]:
        """Find a valid dollar-index symbol, falling back to inverse EURUSD."""
        if not _mt5_available:
            return None
        for candidate in _DXY_CANDIDATES:
            try:
                mt5.symbol_select(candidate, True)
            except Exception:
                pass
            info = mt5.symbol_info(candidate)
            if info is not None and info.visible:
                logger.info(f"[CATCD] DXY symbol resolved: {candidate}")
                self._invert_proxy = False
                return candidate

        logger.warning("[CATCD] No direct DXY symbol. Trying EURUSD synthetic proxy...")
        for candidate in _EURUSD_FALLBACKS:
            try:
                mt5.symbol_select(candidate, True)
            except Exception:
                pass
            info = mt5.symbol_info(candidate)
            if info is not None and info.visible:
                self._invert_proxy = True
                logger.info(f"[CATCD] Using Inverse {candidate} as Synthetic USD Proxy")
                return candidate

        logger.warning("[CATCD] No DXY or EURUSD symbol found. Engine disabled.")
        return None

    def _is_trading_session(self) -> bool:
        """Only run during London + NY overlap (07:00-20:00 UTC)."""
        h = datetime.now(timezone.utc).hour
        return 7 <= h < 20

    def _circuit_breaker_ok(self) -> bool:
        now = time.time()
        if self._disabled_until and now < self._disabled_until:
            return False
        if self._disabled_until and now >= self._disabled_until:
            self._disabled_until = None
            self._consecutive_losses = 0
            logger.info("[CATCD] Cooloff expired. Re-enabled.")

        cutoff = now - 3600
        self._signal_timestamps = [t for t in self._signal_timestamps if t > cutoff]
        if len(self._signal_timestamps) >= self._max_signals_hour:
            return False
        return True

    def record_trade_result(self, pnl: float) -> None:
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._loss_limit:
                self._disabled_until = time.time() + self._cooloff_secs
                logger.warning(
                    f"[CATCD] Circuit breaker: {self._consecutive_losses} consecutive "
                    f"losses. Disabled for {self._cooloff_secs/3600:.0f}h."
                )
        else:
            self._consecutive_losses = 0

    def _fetch_ticks(self, symbol: str, secs: int) -> Optional[np.ndarray]:
        """Fetch recent ticks for a symbol."""
        if not _mt5_available:
            return None
        try:
            ticks = mt5.copy_ticks_from(
                symbol,
                datetime.now(timezone.utc),
                secs * 200,
                mt5.COPY_TICKS_ALL,
            )
            if ticks is None or len(ticks) < 20:
                return None
            now_ts = time.time()
            mask = ticks["time"] >= (now_ts - secs)
            filtered = ticks[mask]
            return filtered if len(filtered) >= 20 else None
        except Exception as e:
            logger.debug(f"[CATCD] Tick fetch error for {symbol}: {e}")
            return None

    def _compute_tick_returns(self, ticks: np.ndarray) -> np.ndarray:
        """Compute log-returns from tick midpoints."""
        bids = ticks["bid"].astype(np.float64)
        asks = ticks["ask"].astype(np.float64)
        mids = (bids + asks) / 2.0
        mids = mids[mids > 0]
        if len(mids) < 10:
            return np.array([])
        returns = np.diff(np.log(mids))
        return returns

    def _align_returns(
        self, gold_ticks: np.ndarray, dxy_ticks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align tick returns by resampling both series to 1-second bars,
        then computing returns on the aligned series.
        """
        def _to_second_bars(ticks: np.ndarray) -> Dict[int, float]:
            bars = {}
            for i in range(len(ticks)):
                ts = int(ticks["time"][i])
                bid = float(ticks["bid"][i])
                ask = float(ticks["ask"][i])
                mid = (bid + ask) / 2.0
                if mid > 0:
                    bars[ts] = mid
            return bars

        gold_bars = _to_second_bars(gold_ticks)
        dxy_bars = _to_second_bars(dxy_ticks)

        common_secs = sorted(set(gold_bars.keys()) & set(dxy_bars.keys()))
        if len(common_secs) < 15:
            return np.array([]), np.array([])

        gold_prices = np.array([gold_bars[s] for s in common_secs], dtype=np.float64)
        dxy_prices = np.array([dxy_bars[s] for s in common_secs], dtype=np.float64)

        if self._invert_proxy:
            dxy_prices = 1.0 / dxy_prices

        gold_ret = np.diff(np.log(gold_prices))
        dxy_ret = np.diff(np.log(dxy_prices))

        return gold_ret, dxy_ret

    def _pearson_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        if len(x) < 10 or len(x) != len(y):
            return 0.0
        x_dm = x - np.mean(x)
        y_dm = y - np.mean(y)
        denom = np.sqrt(np.sum(x_dm**2) * np.sum(y_dm**2))
        if denom < 1e-15:
            return 0.0
        return float(np.sum(x_dm * y_dm) / denom)

    def _compute_z_score(self, current_corr: float) -> float:
        """Z-score of current correlation vs reference distribution."""
        if len(self._ref_correlations) < 100:
            return 0.0
        ref = np.array(self._ref_correlations, dtype=np.float64)
        mu = np.mean(ref)
        sigma = np.std(ref)
        if sigma < 0.01:
            return 0.0
        return (current_corr - mu) / sigma

    def _determine_direction(
        self, gold_ticks: np.ndarray, dxy_ticks: np.ndarray, z_score: float
    ) -> Optional[str]:
        """
        Determine which asset decoupled and the mean-reversion trade direction.
        Gold and DXY are inversely correlated: DXY up => gold down, and vice versa.
        """
        def _recent_move(ticks: np.ndarray, lookback: int = 30) -> float:
            if len(ticks) < lookback:
                return 0.0
            recent = ticks[-lookback:]
            mids = (recent["bid"].astype(np.float64) + recent["ask"].astype(np.float64)) / 2.0
            mids = mids[mids > 0]
            if len(mids) < 5:
                return 0.0
            return float(mids[-1] - mids[0])

        gold_move = _recent_move(gold_ticks)
        dxy_move = _recent_move(dxy_ticks)

        gold_pip_move = abs(gold_move) / self._pip_size
        dxy_pct_move = abs(dxy_move) / max(float(dxy_ticks["bid"][-1]), 1.0) * 10000

        if gold_pip_move < 2 and dxy_pct_move < 2:
            return None

        if gold_pip_move > dxy_pct_move * 1.5:
            return "SELL" if gold_move > 0 else "BUY"
        elif dxy_pct_move > gold_pip_move * 1.5:
            return "BUY" if dxy_move > 0 else "SELL"
        else:
            if z_score > 0:
                return "SELL" if gold_move > 0 else "BUY"
            else:
                return "BUY" if gold_move < 0 else "SELL"

    def _get_m5_atr(self) -> float:
        if not _mt5_available:
            return 20 * self._pip_size
        try:
            rates = mt5.copy_rates_from_pos(self._gold, mt5.TIMEFRAME_M5, 0, 20)
            if rates is None or len(rates) < 14:
                return 20 * self._pip_size
            highs = rates["high"].astype(np.float64)
            lows = rates["low"].astype(np.float64)
            closes = rates["close"].astype(np.float64)
            tr = np.maximum(highs[1:] - lows[1:],
                            np.maximum(np.abs(highs[1:] - closes[:-1]),
                                       np.abs(lows[1:] - closes[:-1])))
            return float(np.mean(tr[-14:]))
        except Exception:
            return 20 * self._pip_size

    def _get_spread_pips(self) -> float:
        if not _mt5_available:
            return 99.0
        try:
            tick = mt5.symbol_info_tick(self._gold)
            if tick is None:
                return 99.0
            return (tick.ask - tick.bid) / self._pip_size
        except Exception:
            return 99.0

    async def _generate_signal(self, action: str, z_score: float, corr: float):
        """Build and push a RawSignal for the detected correlation decay."""
        if not _mt5_available:
            return
        tick = mt5.symbol_info_tick(self._gold)
        if tick is None:
            return

        entry = tick.ask if action == "BUY" else tick.bid
        atr = self._get_m5_atr()

        sl_dist = max(0.8 * atr, 20 * self._pip_size)
        sl_dist = min(sl_dist, 1.5 * atr)
        if sl_dist < 8 * self._pip_size:
            sl_dist = 10 * self._pip_size

        tp_dist = max(sl_dist * 2.0, 20 * self._pip_size)

        if action == "BUY":
            sl = round(entry - sl_dist, 2)
            tp = round(entry + tp_dist, 2)
        else:
            sl = round(entry + sl_dist, 2)
            tp = round(entry - tp_dist, 2)

        text = (
            f"AUTO_CATCD | {action} {self._gold} @ {entry:.2f} | "
            f"SL:{sl:.2f} TP:{tp:.2f} | "
            f"z={z_score:.2f} corr={corr:.3f}"
        )

        signal = RawSignal(content=text, source="AUTO_CATCD")
        await push(signal)

        now = time.time()
        self._signals_generated += 1
        self._signal_timestamps.append(now)
        self._last_signal_time = now

        logger.info(
            f"[CATCD] Signal #{self._signals_generated}: {action} {self._gold} "
            f"z={z_score:.2f} corr={corr:.3f} atr={atr/self._pip_size:.1f}p"
        )

    async def run(self):
        """Main async loop. Polls tick data every ~4 seconds."""
        if not _mt5_available:
            logger.warning("[CATCD] MetaTrader5 not available. Engine idle.")
            while True:
                await asyncio.sleep(60)

        self._dxy = self._resolve_dxy_symbol()
        if self._dxy is None:
            logger.error("[CATCD] No DXY or EURUSD symbol available. Engine idle.")
            while True:
                await asyncio.sleep(300)

        proxy_tag = " (Inverse EURUSD Proxy)" if self._invert_proxy else ""
        logger.info(
            f"[CATCD] Engine started: {self._gold} vs {self._dxy}{proxy_tag} | "
            f"poll {self._poll_interval}s | z-threshold {self._z_threshold} | "
            f"corr-window {self._corr_window}s | ref-window {self._ref_window}s"
        )

        while True:
            try:
                await asyncio.sleep(self._poll_interval)

                if not self._is_trading_session():
                    continue

                if not self._circuit_breaker_ok():
                    continue

                if self._get_spread_pips() > 12.0:
                    continue

                gold_ticks = self._fetch_ticks(self._gold, self._corr_window)
                dxy_ticks = self._fetch_ticks(self._dxy, self._corr_window)

                if gold_ticks is None or dxy_ticks is None:
                    continue

                gold_ret, dxy_ret = self._align_returns(gold_ticks, dxy_ticks)
                if len(gold_ret) < 10:
                    continue

                corr = self._pearson_corr(gold_ret, dxy_ret)
                self._ref_correlations.append(corr)

                z_score = self._compute_z_score(corr)

                sf = vol_regime.scale_factor()
                self._z_threshold = max(1.8, self._base_z_threshold * sf)
                pass

                self._last_pressure = max(-1.0, min(1.0, z_score / max(self._z_threshold, 0.1)))

                if abs(z_score) < self._z_threshold:
                    continue

                if self._last_signal_time and (time.time() - self._last_signal_time) < 60:
                    continue

                direction = self._determine_direction(gold_ticks, dxy_ticks, z_score)
                if direction is None:
                    continue

                # v4.4: CVD confirmation -- verify flow supports the signal direction
                gold_ret_dir = self._compute_tick_returns(gold_ticks)
                if len(gold_ret_dir) >= 20:
                    cvd = np.cumsum(gold_ret_dir)
                    half = len(cvd) // 2
                    cvd_slope = float(np.mean(cvd[half:]) - np.mean(cvd[:half]))
                    if direction == "BUY" and cvd_slope < -0.0001:
                        logger.debug("[CATCD] CVD opposes BUY direction, skipping")
                        continue
                    if direction == "SELL" and cvd_slope > 0.0001:
                        logger.debug("[CATCD] CVD opposes SELL direction, skipping")
                        continue

                await self._generate_signal(direction, z_score, corr)

            except asyncio.CancelledError:
                logger.info("[CATCD] Engine cancelled.")
                break
            except Exception as e:
                logger.error(f"[CATCD] Loop error: {e}", exc_info=True)
                await asyncio.sleep(10)



    @property
    def pressure(self) -> float:
        return self._last_pressure

catcd_engine = CATCDEngine()
