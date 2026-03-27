"""
quant/breakout_hunter.py -- v8.2 Breakout Hunter

Detects explosive directional moves and switches the system to aggressive
mode.  Detection is purely reactive: 3+ consecutive M15 candles in the
same direction with expanding range and a volume spike.

When active the risk_guard loosens cooldown, concurrent-position limits,
and lot sizing so the system can ride confirmed momentum instead of
sitting on the sidelines.
"""

import asyncio
import time
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

_SYMBOL = "XAUUSD"


class BreakoutHunter:
    def __init__(self):
        self.active = False
        self.direction = None
        self.activation_price = 0.0
        self.activation_time = 0.0
        self.consecutive_bars = 0
        self.last_check = 0.0
        self.check_interval = 30
        self._atr_at_activation = 0.0

    def detect(self):
        now = time.time()
        if now - self.last_check < self.check_interval:
            return
        self.last_check = now

        try:
            import config as _cfg
            _min_bars = getattr(_cfg, "BREAKOUT_MIN_CONSECUTIVE_BARS", 3)
            _vol_mult = getattr(_cfg, "BREAKOUT_VOLUME_SPIKE_MULT", 1.5)
        except Exception:
            _min_bars, _vol_mult = 3, 1.5

        try:
            rates = mt5.copy_rates_from_pos(_SYMBOL, mt5.TIMEFRAME_M15, 0, 25)
            if rates is None or len(rates) < 12:
                return

            closes = rates["close"].astype(float)
            opens = rates["open"].astype(float)
            highs = rates["high"].astype(float)
            lows = rates["low"].astype(float)
            volumes = rates["tick_volume"].astype(float)

            ranges = highs - lows
            avg_range = float(np.mean(ranges[-20:]))
            avg_volume = float(np.mean(volumes[-20:]))

            atr = float(np.mean(ranges[-14:])) if len(ranges) >= 14 else avg_range

            bullish_streak = 0
            bearish_streak = 0

            for i in range(-2, -2 - 6, -1):
                if abs(i) > len(closes):
                    break
                body = closes[i] - opens[i]
                if body > 0:
                    bullish_streak += 1
                    if bearish_streak:
                        bearish_streak = 0
                elif body < 0:
                    bearish_streak += 1
                    if bullish_streak:
                        bullish_streak = 0
                else:
                    break
                if ranges[i] < avg_range * 0.8:
                    break

            last_volume = float(volumes[-2])
            volume_spike = last_volume > avg_volume * _vol_mult

            _new_dir = None
            _streak = 0

            if bullish_streak >= _min_bars:
                _new_dir = "BUY"
                _streak = bullish_streak
            elif bearish_streak >= _min_bars:
                _new_dir = "SELL"
                _streak = bearish_streak

            if _new_dir and volume_spike:
                if not self.active or self.direction != _new_dir:
                    self.active = True
                    self.direction = _new_dir
                    self.activation_price = float(closes[-2])
                    self.activation_time = now
                    self.consecutive_bars = _streak
                    self._atr_at_activation = atr

                    logger.warning(
                        "[BreakoutHunter] BREAKOUT %s DETECTED! "
                        "%d consecutive bars, volume %.1fx avg, "
                        "price=%.2f, ATR=%.2f",
                        _new_dir, _streak,
                        last_volume / avg_volume,
                        self.activation_price, atr,
                    )

                    try:
                        from utils.notifier import notify
                        notify(
                            f"*BREAKOUT DETECTED*\n\n"
                            f"Direction: {_new_dir}\n"
                            f"Streak: {_streak} consecutive M15 bars\n"
                            f"Volume: {last_volume / avg_volume:.1f}x average\n"
                            f"Price: {self.activation_price:.2f}\n\n"
                            f"Switching to AGGRESSIVE mode"
                        )
                    except Exception:
                        pass

                    self._fire_signal()

                elif self.active and self.direction == _new_dir:
                    if _streak > self.consecutive_bars:
                        self.consecutive_bars = _streak
                        self._fire_signal()
            else:
                if self.active:
                    _elapsed = now - self.activation_time
                    _price_now = float(closes[-2])

                    _rev_threshold = max(self._atr_at_activation * 1.0, 8.0)
                    _reversed = False
                    if self.direction == "BUY" and _price_now < self.activation_price - _rev_threshold:
                        _reversed = True
                    elif self.direction == "SELL" and _price_now > self.activation_price + _rev_threshold:
                        _reversed = True

                    if _elapsed > 7200 or _reversed:
                        logger.info(
                            "[BreakoutHunter] Breakout ended. Active %.0fmin, "
                            "direction=%s, reversed=%s",
                            _elapsed / 60, self.direction, _reversed,
                        )
                        self.active = False
                        self.direction = None

        except Exception as e:
            logger.error("[BreakoutHunter] Detect error: %s", e)

    def _fire_signal(self):
        try:
            tick = mt5.symbol_info_tick(_SYMBOL)
            if tick is None:
                return

            if self.direction == "BUY":
                _entry = tick.ask
                _sl = _entry - 15.0
                _tp = _entry + 25.0
            else:
                _entry = tick.bid
                _sl = _entry + 15.0
                _tp = _entry - 25.0

            from ingestion.signal_queue import RawSignal, push

            _content = (
                f"XAUUSD {self.direction} @ {_entry:.2f}\n"
                f"SL: {_sl:.2f}\n"
                f"TP: {_tp:.2f}\n"
                f"[Breakout-Hunter] {self.consecutive_bars} consecutive "
                f"M15 bars with volume spike. BREAKOUT MODE ACTIVE."
            )

            raw = RawSignal(
                source="AUTO_BREAKOUT",
                content=_content,
                received_at=datetime.now(),
            )

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(push(raw))

            logger.info(
                "[BreakoutHunter] Signal fired: %s @ %.2f (streak=%d)",
                self.direction, _entry, self.consecutive_bars,
            )

        except Exception as e:
            logger.error("[BreakoutHunter] Fire error: %s", e)

    def is_breakout_active(self):
        """Check if we are in an active breakout. Runs detect() first."""
        self.detect()
        return self.active, self.direction

    async def run(self):
        logger.info("[BreakoutHunter] Engine started")
        while True:
            try:
                self.detect()
            except Exception as e:
                logger.error("[BreakoutHunter] Run error: %s", e)
            await asyncio.sleep(15)


breakout_hunter = BreakoutHunter()
