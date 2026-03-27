"""
v8.0 EDGE 2: Canary Engine - DXY weakness detector for XAUUSD BUY signals.

Research: When EURUSD rises >5 pips in 1 M1 bar (= DXY drops),
XAUUSD rises 61% of the time within 1 minute (115 events).
Only fires on DXY DOWN. DXY UP has no edge (48% = noise).
"""

import asyncio
import time
import MetaTrader5 as mt5
from datetime import datetime, timezone
from utils.logger import get_logger

logger = get_logger(__name__)


class CanaryEngine:
    def __init__(self):
        self.last_eurusd_close = None
        self.last_fire_time = 0
        self.min_cooldown = 300
        self.pip_threshold = 5
        self.fires_today = 0
        self.max_fires_per_day = 3
        self.last_reset_date = None
        self._last_bar_time = 0

    async def run(self):
        logger.info("[Canary] Engine started -- watching EURUSD for DXY weakness")

        while True:
            try:
                now = time.time()
                utc_now = datetime.now(timezone.utc)
                utc_hour = utc_now.hour

                if utc_now.date() != self.last_reset_date:
                    self.fires_today = 0
                    self.last_reset_date = utc_now.date()

                if utc_hour < 7 or utc_hour >= 20:
                    await asyncio.sleep(10)
                    continue

                if now - self.last_fire_time < self.min_cooldown:
                    await asyncio.sleep(5)
                    continue

                if self.fires_today >= self.max_fires_per_day:
                    await asyncio.sleep(60)
                    continue

                rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 3)
                if rates is None or len(rates) < 2:
                    await asyncio.sleep(5)
                    continue

                last_bar = rates[-2]
                bar_move = (float(last_bar["close"]) - float(last_bar["open"])) * 10000

                if bar_move >= self.pip_threshold:
                    _bar_time = int(last_bar['time'])
                    if _bar_time == self._last_bar_time:
                        await asyncio.sleep(5)
                        continue
                    self._last_bar_time = _bar_time
                    self.last_fire_time = now
                    self.fires_today += 1

                    gold_tick = mt5.symbol_info_tick("XAUUSD")
                    if gold_tick is None:
                        continue

                    _entry = gold_tick.ask
                    _sl = round(_entry - 8.0, 2)
                    _tp = round(_entry + 15.0, 2)

                    from ingestion.signal_queue import RawSignal, push

                    _content = (
                        "XAUUSD BUY @ %.2f\n"
                        "SL: %.2f\n"
                        "TP: %.2f\n"
                        "[Canary] DXY weakness: EURUSD +%.1f pips in 1 min. "
                        "Gold expected UP (61%% historical)."
                    ) % (_entry, _sl, _tp, bar_move)

                    raw = RawSignal(
                        source="AUTO_CANARY",
                        content=_content,
                        received_at=datetime.now(),
                    )
                    await push(raw)

                    logger.info(
                        "[Canary] DXY DOWN! EURUSD +%.1f pips. "
                        "Fired XAUUSD BUY @ %.2f (fire #%d today)",
                        bar_move, _entry, self.fires_today,
                    )

                    try:
                        from database import db_manager
                        db_manager.log_audit("CANARY_FIRE", {
                            "eurusd_move": round(bar_move, 1),
                            "gold_entry": _entry,
                            "sl": _sl, "tp": _tp,
                            "fire_num": self.fires_today,
                        })
                    except Exception:
                        pass

                    try:
                        from utils.notifier import notify
                        notify(
                            "CANARY ENGINE\n\n"
                            "DXY Weakness: EURUSD +%.1f pips\n"
                            "XAUUSD BUY @ %.2f\n"
                            "SL: %.2f | TP: %.2f\n"
                            "Historical: 61%% win rate on DXY-DOWN events"
                            % (bar_move, _entry, _sl, _tp)
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error("[Canary] Error: %s", e)

            await asyncio.sleep(5)


canary_engine = CanaryEngine()
