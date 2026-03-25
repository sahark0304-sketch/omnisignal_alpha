"""
quant/latency_monitor.py — OmniSignal Alpha v2.0
Pillar 13: Latency & Network Jitter Management

Continuously pings the broker host. Three state tiers:
  NORMAL   → latency < LATENCY_WARN_MS        → full execution
  WARNING  → WARN_MS  ≤ latency < SAFETY_MS   → log warning, still execute
  SAFETY   → SAFETY_MS ≤ latency < CRITICAL_MS → skip new entries, manage only
  CRITICAL → latency ≥ CRITICAL_MS             → halt all new trades, alert

Also tracks consecutive failures (host unreachable) as a connectivity signal.
The latency state is accessible via module-level is_safe_to_trade() / get_state().
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Optional, Deque
from dataclasses import dataclass
import MetaTrader5 as mt5
import config
from utils.logger import get_logger
from utils.notifier import notify

logger = get_logger(__name__)


@dataclass
class LatencyState:
    latest_ms: float     = 0.0
    avg_ms: float        = 0.0
    min_ms: float        = 0.0
    max_ms: float        = 0.0
    jitter_ms: float     = 0.0      # std-dev of recent samples
    mode: str            = "INIT"   # NORMAL / WARNING / SAFETY / CRITICAL / UNREACHABLE
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    sample_count: int    = 0


# Module-level state — read by risk_guard and main.py
_state = LatencyState()
_samples: Deque[float] = deque(maxlen=config.LATENCY_SAMPLES)
_prev_mode: str = "INIT"


def get_state() -> LatencyState:
    """Thread-safe read of current latency state."""
    return _state


def is_safe_to_trade() -> bool:
    """
    Returns False when latency is in SAFETY or CRITICAL mode.
    risk_guard.validate() calls this before executing new entries.
    """
    return _state.mode in ("NORMAL", "WARNING", "INIT")


def is_critical() -> bool:
    return _state.mode in ("CRITICAL", "UNREACHABLE")


# ── PING IMPLEMENTATION ──────────────────────────────────────────────────────

def _mt5_ping_ms() -> Optional[float]:
    """Read broker ping from MT5 terminal_info().ping_last (microseconds -> ms)."""
    try:
        info = mt5.terminal_info()
        if info is not None and info.ping_last > 0:
            return info.ping_last / 1000.0
        return None
    except Exception:
        return None


def _compute_jitter(samples: Deque[float]) -> float:
    """Jitter = standard deviation of latency samples."""
    if len(samples) < 2:
        return 0.0
    arr = list(samples)
    mean = sum(arr) / len(arr)
    variance = sum((x - mean) ** 2 for x in arr) / len(arr)
    return variance ** 0.5


def _classify_mode(avg_ms: float, consecutive_failures: int) -> str:
    if consecutive_failures >= 3:
        return "UNREACHABLE"
    if avg_ms >= config.LATENCY_CRITICAL_MS:
        return "CRITICAL"
    if avg_ms >= config.LATENCY_SAFETY_MODE_MS:
        return "SAFETY"
    if avg_ms >= config.LATENCY_WARN_MS:
        return "WARNING"
    return "NORMAL"


# ── MONITORING LOOP ──────────────────────────────────────────────────────────

async def run_latency_monitor():
    """
    Async background task. Run alongside main trading tasks.
    Updates _state in place; other modules read it synchronously.
    """
    global _state, _samples, _prev_mode
    logger.info(f"[Latency] Monitor started → host={config.LATENCY_BROKER_HOST}")

    loop = asyncio.get_event_loop()

    while True:
        try:
            ms = await loop.run_in_executor(None, _mt5_ping_ms)

            if ms is None:
                _state.consecutive_failures += 1
                logger.warning(
                    f"[Latency] Ping failed ({_state.consecutive_failures} consecutive)"
                )
            else:
                _state.consecutive_failures = 0
                _samples.append(ms)
                _state.latest_ms  = ms
                _state.sample_count += 1

                if _samples:
                    slist = list(_samples)
                    _state.avg_ms    = sum(slist) / len(slist)
                    _state.min_ms    = min(slist)
                    _state.max_ms    = max(slist)
                    _state.jitter_ms = _compute_jitter(_samples)

            new_mode = _classify_mode(_state.avg_ms, _state.consecutive_failures)
            _state.mode       = new_mode
            _state.last_check = datetime.now()

            # Mode transition alerts
            if new_mode != _prev_mode:
                _on_mode_change(_prev_mode, new_mode)
                _prev_mode = new_mode
            else:
                logger.debug(
                    f"[Latency] {new_mode} | avg={_state.avg_ms:.0f}ms "
                    f"jitter={_state.jitter_ms:.0f}ms latest={_state.latest_ms:.0f}ms"
                )

        except Exception as e:
            logger.error(f"[Latency] Monitor error: {e}", exc_info=True)

        await asyncio.sleep(config.LATENCY_CHECK_INTERVAL_SECS)


def _on_mode_change(old: str, new: str):
    """Log and notify on latency mode transitions."""
    emoji_map = {
        "NORMAL": "✅", "WARNING": "⚠️", "SAFETY": "🟠", "CRITICAL": "🔴", "UNREACHABLE": "💀"
    }
    icon = emoji_map.get(new, "❓")
    msg = (
        f"{icon} *Latency Mode: {old} → {new}*\n"
        f"Avg: `{_state.avg_ms:.0f}ms` | Jitter: `{_state.jitter_ms:.0f}ms` | "
        f"Failures: `{_state.consecutive_failures}`"
    )
    logger.warning(f"[Latency] Mode change: {old} → {new} | {msg.replace('*','').replace('`','')}")

    if new in ("SAFETY", "CRITICAL", "UNREACHABLE"):
        notify(msg)
        # Log to main DB
        try:
            from database import db_manager
            db_manager.log_audit("LATENCY_MODE_CHANGE", {
                "old": old, "new": new,
                "avg_ms": _state.avg_ms, "jitter_ms": _state.jitter_ms,
                "failures": _state.consecutive_failures,
            })
        except Exception:
            pass

    if new == "NORMAL" and old in ("SAFETY", "CRITICAL", "UNREACHABLE"):
        notify(f"✅ *Latency Recovered* — back to NORMAL ({_state.avg_ms:.0f}ms avg)")
