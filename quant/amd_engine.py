"""
quant/amd_engine.py -- OmniSignal Alpha v4.2
AMD Cycle Engine: Accumulation-Manipulation-Distribution Detection

Institutional Logic:
  Smart money operates in 3 phases:
  1. ACCUMULATION: Price consolidates in a tight range. ATR compresses.
     Institutions build positions quietly.
  2. MANIPULATION: A false breakout beyond the range traps retail traders.
     Volume spikes on the inducement candle, then reverses sharply.
  3. DISTRIBUTION: The real move begins. Strong directional follow-through
     in the opposite direction of the manipulation fake-out.

Power of 3 Session Mapping (XAUUSD):
  - Asian (00:00-07:00 UTC): Accumulation range forms
  - London Open (07:00-09:00 UTC): Manipulation spike (liquidity grab)
  - NY Session (13:00-21:00 UTC): Distribution trend

The engine continuously tracks which phase the market is in and exposes
a directional bias + confidence score for the consensus layer.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple
import numpy as np

try:
    import MetaTrader5 as mt5
    _mt5_available = True
except ImportError:
    _mt5_available = False
    mt5 = None

from utils.logger import get_logger

logger = get_logger(__name__)

POLL_INTERVAL = 5.0
ATR_COMPRESSION_RATIO = 0.55
MANIPULATION_WICK_RATIO = 0.65
MANIPULATION_VOL_MULT = 1.4
MIN_RANGE_CANDLES = 8
LOOKBACK_CANDLES = 60
DISTRIBUTION_MOMENTUM_BARS = 3


class AMDPhase:
    UNKNOWN = "UNKNOWN"
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"


class AMDEngine:
    def __init__(self, symbol: str = "XAUUSD"):
        self._symbol = symbol
        self._pip_size = 0.1
        self._phase = AMDPhase.UNKNOWN
        self._bias: Optional[str] = None
        self._confidence = 0.0
        self._range_high = 0.0
        self._range_low = 0.0
        self._manipulation_direction: Optional[str] = None
        self._phase_start_time = 0.0
        self._last_update = 0.0
        self._accumulation_bars = 0

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def bias(self) -> Optional[str]:
        return self._bias

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def range_high(self) -> float:
        return self._range_high

    @property
    def range_low(self) -> float:
        return self._range_low

    async def run(self):
        logger.info(
            f"[AMD] Engine started for {self._symbol} | "
            f"poll={POLL_INTERVAL}s | compression_ratio={ATR_COMPRESSION_RATIO}"
        )
        while True:
            try:
                self._update_cycle()
            except Exception as e:
                logger.error(f"[AMD] Cycle error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    def _update_cycle(self):
        if not _mt5_available or mt5 is None:
            return

        rates = mt5.copy_rates_from_pos(self._symbol, mt5.TIMEFRAME_M5, 0, LOOKBACK_CANDLES)
        if rates is None or len(rates) < LOOKBACK_CANDLES:
            return

        highs = rates["high"].astype(float)
        lows = rates["low"].astype(float)
        closes = rates["close"].astype(float)
        opens = rates["open"].astype(float)
        volumes = rates["tick_volume"].astype(float)

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        atr_recent = np.mean(tr[-5:]) if len(tr) >= 5 else np.mean(tr)
        atr_baseline = np.mean(tr[-30:]) if len(tr) >= 30 else np.mean(tr)
        atr_ratio = atr_recent / max(atr_baseline, 1e-8)

        vol_recent = np.mean(volumes[-5:])
        vol_baseline = np.mean(volumes[-30:]) if len(volumes) >= 30 else np.mean(volumes)
        vol_ratio = vol_recent / max(vol_baseline, 1e-8)

        now = time.time()
        hour = datetime.now(timezone.utc).hour

        if self._phase == AMDPhase.UNKNOWN or self._phase == AMDPhase.DISTRIBUTION:
            if atr_ratio < ATR_COMPRESSION_RATIO:
                self._enter_accumulation(highs, lows, now)
            else:
                self._confidence = max(0, self._confidence - 0.02)

        elif self._phase == AMDPhase.ACCUMULATION:
            if atr_ratio < ATR_COMPRESSION_RATIO * 1.3:
                self._accumulation_bars += 1
                recent_high = np.max(highs[-MIN_RANGE_CANDLES:])
                recent_low = np.min(lows[-MIN_RANGE_CANDLES:])
                self._range_high = max(self._range_high, recent_high)
                self._range_low = min(self._range_low, recent_low)
            else:
                manip = self._detect_manipulation(
                    highs, lows, closes, opens, volumes, vol_baseline
                )
                if manip is not None:
                    self._enter_manipulation(manip, now)
                elif self._accumulation_bars < MIN_RANGE_CANDLES:
                    self._phase = AMDPhase.UNKNOWN
                    self._confidence = 0.0

        elif self._phase == AMDPhase.MANIPULATION:
            if now - self._phase_start_time > 600:
                self._phase = AMDPhase.UNKNOWN
                self._confidence = 0.0
                return

            dist = self._detect_distribution(closes, volumes, vol_baseline)
            if dist is not None:
                self._enter_distribution(dist, now)

    def _enter_accumulation(self, highs, lows, now):
        self._phase = AMDPhase.ACCUMULATION
        self._phase_start_time = now
        self._accumulation_bars = 0
        self._range_high = np.max(highs[-MIN_RANGE_CANDLES:])
        self._range_low = np.min(lows[-MIN_RANGE_CANDLES:])
        self._confidence = 0.3
        self._bias = None
        logger.debug(
            "[AMD] ACCUMULATION detected | range=%.2f-%.2f",
            self._range_low, self._range_high,
        )

    def _detect_manipulation(self, highs, lows, closes, opens, volumes, vol_base):
        last_high = highs[-1]
        last_low = lows[-1]
        last_close = closes[-1]
        last_open = opens[-1]
        last_vol = volumes[-1]

        candle_range = last_high - last_low
        if candle_range < self._pip_size:
            return None

        body = abs(last_close - last_open)
        wick_top = last_high - max(last_close, last_open)
        wick_bot = min(last_close, last_open) - last_low

        vol_spike = last_vol > vol_base * MANIPULATION_VOL_MULT

        if last_high > self._range_high and wick_top / candle_range > MANIPULATION_WICK_RATIO:
            if last_close < self._range_high and vol_spike:
                return "BEARISH"

        if last_low < self._range_low and wick_bot / candle_range > MANIPULATION_WICK_RATIO:
            if last_close > self._range_low and vol_spike:
                return "BULLISH"

        return None

    def _enter_manipulation(self, direction: str, now):
        self._phase = AMDPhase.MANIPULATION
        self._manipulation_direction = direction
        self._phase_start_time = now
        self._confidence = 0.6
        self._bias = "BUY" if direction == "BULLISH" else "SELL"
        logger.info(
            "[AMD] MANIPULATION detected | direction=%s | bias=%s | conf=%.2f",
            direction, self._bias, self._confidence,
        )

    def _detect_distribution(self, closes, volumes, vol_base):
        if len(closes) < DISTRIBUTION_MOMENTUM_BARS + 1:
            return None

        recent = closes[-DISTRIBUTION_MOMENTUM_BARS:]
        direction_consistent = False

        if self._manipulation_direction == "BULLISH":
            direction_consistent = all(
                recent[i] > recent[i - 1] for i in range(1, len(recent))
            )
        elif self._manipulation_direction == "BEARISH":
            direction_consistent = all(
                recent[i] < recent[i - 1] for i in range(1, len(recent))
            )

        if not direction_consistent:
            return None

        move = abs(recent[-1] - recent[0])
        range_size = self._range_high - self._range_low
        if range_size > 0 and move > range_size * 0.5:
            return self._manipulation_direction

        return None

    def _enter_distribution(self, direction: str, now):
        self._phase = AMDPhase.DISTRIBUTION
        self._phase_start_time = now
        self._confidence = 0.85
        self._bias = "BUY" if direction == "BULLISH" else "SELL"
        logger.info(
            "[AMD] DISTRIBUTION confirmed | bias=%s | conf=%.2f | "
            "range=%.2f-%.2f",
            self._bias, self._confidence, self._range_low, self._range_high,
        )

    def get_state(self) -> Dict:
        return {
            "phase": self._phase,
            "bias": self._bias,
            "confidence": round(self._confidence, 3),
            "range_high": round(self._range_high, 2),
            "range_low": round(self._range_low, 2),
            "accumulation_bars": self._accumulation_bars,
        }


amd_engine = AMDEngine()
