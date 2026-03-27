"""
v8.0: London Open Breakout Strategy

At 07:05 UTC every trading day:
1. Calculate Asian session range (00:00-07:00 UTC high/low)
2. Wait for price to break above high or below low
3. Enter in breakout direction with SL at opposite side
4. TP1 at 1.5x range width, TP2 at 2.5x
"""

import asyncio
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from utils.logger import get_logger
from database import db_manager

logger = get_logger(__name__)


class LondonSniper:
    def __init__(self):
        self.asia_high = None
        self.asia_low = None
        self.range_calculated = False
        self.today_traded = False
        self.last_calc_date = None

    def _calculate_asian_range(self):
        now = datetime.now(timezone.utc)
        today = now.date()

        if self.last_calc_date == today:
            return
        if now.hour < 7 or now.hour >= 8:
            return

        try:
            rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 50)
            if rates is None or len(rates) < 20:
                return

            asia_bars = []
            for bar in rates:
                bar_time = datetime.fromtimestamp(int(bar["time"]), tz=timezone.utc)
                bar_hour = bar_time.hour
                if bar_hour >= 22 or bar_hour < 7:
                    asia_bars.append(bar)

            if len(asia_bars) < 5:
                return

            self.asia_high = max(float(b["high"]) for b in asia_bars)
            self.asia_low = min(float(b["low"]) for b in asia_bars)
            self.range_calculated = True
            self.today_traded = False
            self.last_calc_date = today

            _range = self.asia_high - self.asia_low
            logger.info(
                "[LondonSniper] Asian range: %.2f - %.2f (%.1f pips)",
                self.asia_low, self.asia_high, _range * 10,
            )
        except Exception as e:
            logger.error("[LondonSniper] Range calc error: %s", e)

    def check_breakout(self):
        if not self.range_calculated or self.today_traded:
            return None

        now = datetime.now(timezone.utc)
        if now.hour < 7 or now.hour >= 9:
            return None

        try:
            tick = mt5.symbol_info_tick("XAUUSD")
            if tick is None:
                return None

            price = tick.ask
            _range = self.asia_high - self.asia_low

            if _range < 5.0 or _range > 40.0:
                return None

            if price > self.asia_high + 2.0:
                self.today_traded = True
                return {
                    "action": "BUY",
                    "entry": price,
                    "sl": self.asia_low - 3.0,
                    "tp1": price + (_range * 1.5),
                    "tp2": price + (_range * 2.5),
                    "asia_high": self.asia_high,
                    "asia_low": self.asia_low,
                    "range": _range,
                }

            if price < self.asia_low - 2.0:
                self.today_traded = True
                return {
                    "action": "SELL",
                    "entry": price,
                    "sl": self.asia_high + 3.0,
                    "tp1": price - (_range * 1.5),
                    "tp2": price - (_range * 2.5),
                    "asia_high": self.asia_high,
                    "asia_low": self.asia_low,
                    "range": _range,
                }

            return None
        except Exception as e:
            logger.error("[LondonSniper] Breakout check error: %s", e)
            return None

    async def run(self):
        logger.info("[LondonSniper] Engine started")
        while True:
            try:
                now = datetime.now(timezone.utc)

                if now.weekday() < 5 and 7 <= now.hour < 9:
                    self._calculate_asian_range()
                    breakout = self.check_breakout()

                    if breakout:
                        from ingestion.signal_queue import RawSignal, push

                        _content = (
                            "XAUUSD %s @ %.2f\n"
                            "SL: %.2f\n"
                            "TP: %.2f\n"
                            "[London-Breakout] Asian range %.2f-%.2f (%.0fp)"
                        ) % (
                            breakout["action"], breakout["entry"],
                            breakout["sl"], breakout["tp1"],
                            breakout["asia_low"], breakout["asia_high"],
                            breakout["range"],
                        )

                        raw = RawSignal(
                            source="AUTO_LONDON_BREAKOUT",
                            content=_content,
                            received_at=datetime.now(),
                        )
                        await push(raw)

                        logger.info(
                            "[LondonSniper] BREAKOUT %s! Price=%.2f Range=%.2f-%.2f",
                            breakout["action"], breakout["entry"],
                            breakout["asia_low"], breakout["asia_high"],
                        )

                        try:
                            from utils.notifier import notify
                            notify(
                                "\U0001f3af *LONDON BREAKOUT*\n\n"
                                "XAUUSD %s @ %.2f\n"
                                "Asian Range: %.2f - %.2f\n"
                                "SL: %.2f | TP1: %.2f\n"
                                "Range: %.0f pips"
                                % (
                                    breakout["action"], breakout["entry"],
                                    breakout["asia_low"], breakout["asia_high"],
                                    breakout["sl"], breakout["tp1"],
                                    breakout["range"],
                                )
                            )
                        except Exception:
                            pass

                        db_manager.log_audit("LONDON_BREAKOUT", {
                            "action": breakout["action"],
                            "entry": breakout["entry"],
                            "asia_high": breakout["asia_high"],
                            "asia_low": breakout["asia_low"],
                            "range": breakout["range"],
                        })

                if now.hour == 0 and self.last_calc_date != now.date():
                    self.range_calculated = False
                    self.today_traded = False

            except Exception as e:
                logger.error("[LondonSniper] Run error: %s", e)

            await asyncio.sleep(15)


london_sniper = LondonSniper()
