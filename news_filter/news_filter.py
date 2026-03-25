"""
news_filter/news_filter.py — ForexFactory High-Impact News Guard.  [PATCHED v1.1]

FIX vs v1.0 (Gemini rewrite):
  CRITICAL: The Gemini-rewritten version used datetime.fromisoformat() which CRASHES
  on ForexFactory's actual date format: "date": "01-13-2025", "time": "8:30am"
  (These are SEPARATE JSON keys, not an ISO 8601 string.)

  fromisoformat("01-13-2025") → ValueError: Invalid isoformat string

  FIXED: Parse the FF "date" and "time" fields separately using the correct format.
  Confirmed working format: strptime("01-13-2025 8:30am", "%m-%d-%Y %I:%M%p")
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class NewsFilter:
    def __init__(self):
        self._events: List[Dict] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(hours=4)

    async def refresh_if_needed(self):
        now = datetime.now()
        if self._last_fetch and (now - self._last_fetch) < self._fetch_interval:
            return
        await self._fetch_events()

    async def _fetch_events(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config.NEWS_API_URL,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={"User-Agent": "OmniSignal/1.0"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        self._events = data if isinstance(data, list) else []
                        self._last_fetch = datetime.now()
                        high = sum(1 for e in self._events if e.get("impact", "").lower() == "high")
                        logger.info(
                            f"[NewsFilter] Loaded {len(self._events)} events "
                            f"({high} High Impact)."
                        )
                    else:
                        logger.warning(f"[NewsFilter] Calendar fetch returned {resp.status}")
        except Exception as e:
            logger.warning(f"[NewsFilter] Fetch failed: {e} — using cached {len(self._events)} events.")

    def is_blocked(self, symbol: str) -> Tuple[bool, str]:
        """
        Returns (blocked, reason).

        FIX: Parses ForexFactory's actual JSON structure:
          { "date": "01-13-2025", "time": "8:30am", "country": "USD",
            "title": "CPI m/m", "impact": "High" }

        NOT an ISO timestamp — separate date + time string fields.
        """
        if not self._events:
            return False, ""

        affected_currencies = self._extract_currencies(symbol)
        if not affected_currencies:
            return False, ""

        now = datetime.now()
        before = timedelta(minutes=config.NEWS_BLOCK_BEFORE_MINS)
        after  = timedelta(minutes=config.NEWS_BLOCK_AFTER_MINS)

        for event in self._events:
            try:
                impact = event.get("impact", "").lower()
                if config.NEWS_HIGH_IMPACT_ONLY and impact != "high":
                    continue

                currency = event.get("country", "").upper()
                if currency not in affected_currencies:
                    continue

                event_time = self._parse_ff_datetime(
                    event.get("date", ""),
                    event.get("time", "")
                )
                if event_time is None:
                    continue

                window_start = event_time - before
                window_end   = event_time + after

                if window_start <= now <= window_end:
                    title = event.get("title", "Unknown Event")
                    reason = (
                        f"News block: '{title}' ({currency}) "
                        f"at {event_time.strftime('%H:%M')} — "
                        f"±{config.NEWS_BLOCK_BEFORE_MINS}min window"
                    )
                    logger.info(f"[NewsFilter] BLOCKED: {reason}")
                    return True, reason

            except Exception as e:
                logger.debug(f"[NewsFilter] Skipping malformed event: {e}")
                continue

        # Gold-specific medium-impact blocking
        is_gold = "XAU" in symbol.upper() or "GOLD" in symbol.upper()
        if is_gold and getattr(config, "NEWS_GOLD_MEDIUM_IMPACT", False):
            gold_before = timedelta(minutes=config.NEWS_GOLD_MEDIUM_BLOCK_BEFORE_MINS)
            gold_after = timedelta(minutes=config.NEWS_GOLD_MEDIUM_BLOCK_AFTER_MINS)
            sensitive = [kw.lower() for kw in config.NEWS_GOLD_SENSITIVE_EVENTS]

            for event in self._events:
                try:
                    impact = event.get("impact", "").lower()
                    if impact != "medium":
                        continue

                    currency = event.get("country", "").upper()
                    if currency != "USD":
                        continue

                    title_lower = event.get("title", "").lower()
                    if not any(kw in title_lower for kw in sensitive):
                        continue

                    event_time = self._parse_ff_datetime(
                        event.get("date", ""),
                        event.get("time", "")
                    )
                    if event_time is None:
                        continue

                    window_start = event_time - gold_before
                    window_end = event_time + gold_after

                    if window_start <= now <= window_end:
                        title = event.get("title", "Unknown Event")
                        reason = (
                            f"Gold medium-impact block: '{title}' ({currency}) "
                            f"at {event_time.strftime('%H:%M')} - "
                            f"\u00b1{config.NEWS_GOLD_MEDIUM_BLOCK_BEFORE_MINS}min window"
                        )
                        logger.info(f"[NewsFilter] GOLD MEDIUM BLOCKED: {reason}")
                        return True, reason
                except Exception as e:
                    logger.debug(f"[NewsFilter] Skipping gold event check: {e}")
                    continue

        return False, ""

    def _parse_ff_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        """
        FIXED: Parses ForexFactory's actual date/time format.

        ForexFactory JSON format:
          date: "01-13-2025"  → MM-DD-YYYY
          time: "8:30am"      → H:MMam/pm  (no space before am/pm)

        Combined: "01-13-2025 8:30am" → strptime format: "%m-%d-%Y %I:%M%p"

        Edge cases handled:
          - "All Day" events have empty time → skip
          - "Tentative" events → skip (time_str contains "Tentative")
        """
        if not date_str or not time_str:
            return None
        if "tentative" in time_str.lower() or "all day" in time_str.lower():
            return None

        combined = f"{date_str.strip()} {time_str.strip()}"
        # Normalize: "8:30am" and "08:30am" and "8:30AM" all handled
        combined = combined.lower().replace(" am", "am").replace(" pm", "pm")

        formats = [
            "%m-%d-%Y %I:%M%p",   # "01-13-2025 8:30am"
            "%m-%d-%Y %H:%M",     # "01-13-2025 08:30" (24h fallback)
            "%Y-%m-%dT%H:%M:%S",  # ISO fallback if FF ever changes
        ]
        for fmt in formats:
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue

        logger.debug(f"[NewsFilter] Could not parse datetime: '{combined}'")
        return None

    def _extract_currencies(self, symbol: str) -> List[str]:
        """Extract the two currency codes from a forex symbol."""
        symbol = symbol.upper()
        # Standard 6-char FX pair
        if len(symbol) == 6 and symbol.isalpha():
            return [symbol[:3], symbol[3:]]
        # Metals/Crypto that contain USD
        if "USD" in symbol:
            return ["USD"]
        # Fallback: first 3 chars
        return [symbol[:3]]

    def get_next_event(self, symbol: str) -> Optional[Dict]:
        """Returns the next upcoming high-impact event affecting this symbol (for dashboard)."""
        now = datetime.now()
        currencies = self._extract_currencies(symbol)
        upcoming = []

        for event in self._events:
            if event.get("impact", "").lower() != "high":
                continue
            if event.get("country", "").upper() not in currencies:
                continue
            event_time = self._parse_ff_datetime(event.get("date"), event.get("time"))
            if event_time and event_time > now:
                upcoming.append({**event, "_parsed_time": event_time})

        if not upcoming:
            return None
        return min(upcoming, key=lambda e: e["_parsed_time"])


    def get_upcoming_gold_events(self, hours: int = 24) -> List[Dict]:
        """Returns upcoming gold-sensitive events for the dashboard."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        results = []

        sensitive = [kw.lower() for kw in getattr(config, "NEWS_GOLD_SENSITIVE_EVENTS", [])]

        for event in self._events:
            try:
                impact = event.get("impact", "").lower()
                currency = event.get("country", "").upper()
                if currency != "USD":
                    continue

                title = event.get("title", "")
                title_lower = title.lower()

                is_relevant = (
                    impact == "high"
                    or (impact == "medium" and any(kw in title_lower for kw in sensitive))
                )
                if not is_relevant:
                    continue

                event_time = self._parse_ff_datetime(
                    event.get("date", ""),
                    event.get("time", "")
                )
                if event_time is None or event_time < now or event_time > cutoff:
                    continue

                results.append({
                    "time": event_time.strftime("%Y-%m-%d %H:%M"),
                    "title": title,
                    "impact": event.get("impact", ""),
                    "currency": currency,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["time"])
        return results

# Module-level singleton
news_filter = NewsFilter()
