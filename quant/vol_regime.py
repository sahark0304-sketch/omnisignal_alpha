"""
quant/vol_regime.py -- Shared Volatility Regime State for all Alpha Scanners.

Provides a single source of truth for the current market regime (ATR, Vol_z,
Hurst) that scanners use to dynamically scale their thresholds.
"""

import time
import numpy as np
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    _mt5 = True
except ImportError:
    mt5 = None
    _mt5 = False


class VolRegime:
    """Cached volatility regime state, refreshed every ~30 seconds."""

    def __init__(self, symbol: str = "XAUUSD"):
        self._symbol = symbol
        self._atr: float = 0.0
        self._vol_z: float = 0.0
        self._hurst: float = 0.5
        self._last_update: float = 0.0
        self._ttl_secs: float = 30.0

    def refresh(self):
        now = time.time()
        if now - self._last_update < self._ttl_secs:
            return
        if not _mt5 or mt5 is None:
            return
        try:
            rates = mt5.copy_rates_from_pos(
                self._symbol, mt5.TIMEFRAME_M15, 0, 120
            )
            if rates is None or len(rates) < 60:
                return

            h = rates["high"].astype(float)
            lo = rates["low"].astype(float)
            c = rates["close"].astype(float)
            pc = np.roll(c, 1); pc[0] = c[0]
            tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
            atr_arr = np.zeros(len(tr))
            atr_arr[13] = float(np.mean(tr[:14]))
            for i in range(14, len(tr)):
                atr_arr[i] = (atr_arr[i-1] * 13 + tr[i]) / 14

            self._atr = float(atr_arr[-1])
            window = atr_arr[-50:]
            mu = float(np.mean(window))
            sigma = float(np.std(window))
            self._vol_z = (self._atr - mu) / sigma if sigma > 1e-10 else 0.0

            # DFA Hurst on last 50 closes
            prices = c[-50:]
            log_ret = np.diff(np.log(prices + 1e-10))
            y = np.cumsum(log_ret - np.mean(log_ret))
            N = len(y)
            max_w = N // 4
            if max_w >= 4:
                windows = np.unique(np.logspace(np.log10(4), np.log10(max_w), 12).astype(int))
                F_vals = []
                for n in windows:
                    segs = N // n
                    if segs < 2:
                        continue
                    rms = []
                    for s in range(segs):
                        seg_y = y[s*n:(s+1)*n]
                        x_seg = np.arange(n, dtype=float)
                        coeffs = np.polyfit(x_seg, seg_y, 1)
                        res = seg_y - np.polyval(coeffs, x_seg)
                        rms.append(np.sqrt(np.mean(res**2)))
                    F_vals.append((n, np.mean(rms)))
                if len(F_vals) >= 3:
                    log_n = np.log([f[0] for f in F_vals])
                    log_F = np.log([f[1] + 1e-15 for f in F_vals])
                    slope, _ = np.polyfit(log_n, log_F, 1)
                    self._hurst = float(np.clip(slope, 0.0, 1.0))

            self._last_update = now
        except Exception as e:
            logger.debug(f"[VolRegime] Refresh error: {e}")

    @property
    def atr(self) -> float:
        self.refresh()
        return self._atr

    @property
    def vol_z(self) -> float:
        self.refresh()
        return self._vol_z

    @property
    def hurst(self) -> float:
        self.refresh()
        return self._hurst

    @property
    def is_low_vol(self) -> bool:
        self.refresh()
        return self._vol_z < -0.5

    @property
    def is_trending(self) -> bool:
        self.refresh()
        return self._hurst >= 0.52

    @property
    def is_ranging(self) -> bool:
        self.refresh()
        return self._hurst < 0.52

    def scale_factor(self) -> float:
        """
        Dynamic scaling multiplier for scanner thresholds.

        In low vol (vol_z < 0): thresholds shrink so scanners fire more.
        In high vol (vol_z > 0): thresholds grow to avoid noise.

        Formula: scale = clamp(1 + vol_z * 0.25, 0.50, 1.80)

        vol_z = -1.25 => scale = 0.69  (thresholds drop ~31%)
        vol_z =  0.00 => scale = 1.00  (nominal)
        vol_z = +2.00 => scale = 1.50  (thresholds rise 50%)
        """
        self.refresh()
        raw = 1.0 + self._vol_z * 0.25
        return max(0.50, min(raw, 1.80))


vol_regime = VolRegime()
