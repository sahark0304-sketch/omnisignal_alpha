"""
quant/convergence_engine.py -- OmniSignal Alpha v4.2
Breakthrough #1: Alpha Convergence Engine

Fuses partial signals from 5 independent Alpha scanners into synthetic
triggers.  Fires BEFORE any individual scanner reaches its own threshold
when the composite directional pressure across multiple scanners is high.

Mathematical basis:
  composite = sum(w_i * pressure_i)  for scanners with consistent direction
  Fire when |composite| >= COMPOSITE_THRESHOLD and n_contributing >= 2

This creates genuinely NEW alpha by detecting multi-dimensional convergence
that no individual scanner can see.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional
import MetaTrader5 as mt5

from ingestion.signal_queue import push as queue_push, RawSignal
from utils.logger import get_logger

logger = get_logger(__name__)

# v4.2: AMD Cycle integration
try:
    from quant.amd_engine import amd_engine, AMDPhase
    _amd_available = True
except ImportError:
    _amd_available = False

COMPOSITE_THRESHOLD = 0.50
MIN_CONTRIBUTING    = 2
PRESSURE_FLOOR      = 0.30
COOLDOWN_SECS       = 45
DEDUP_WINDOW_SECS   = 120
MAX_SPREAD_PIPS     = 12.0
POLL_INTERVAL       = 4.0
MAX_SIGNALS_PER_HOUR = 15

SCANNER_WEIGHTS = {
    "TFI":       0.231,
    "CATCD":     0.169,
    "Momentum":  0.246,
    "MR":        0.100,
    "Liquidity": 0.115,
    "SMC":       0.138,
}


class ConvergenceEngine:
    def __init__(self, symbol: str = "XAUUSD"):
        self._symbol = symbol
        self._pip_size = 0.01
        self._last_signal_time: Optional[float] = None
        self._last_direction: Optional[str] = None
        self._last_direction_time: Optional[float] = None
        self._signals_generated = 0
        self._signal_timestamps: list = []
        self._global_bias = None
        self._bias_strength = 0.0
        self._bias_until = 0.0

    async def run(self):
        logger.info(
            f"[Convergence] Engine started for {self._symbol} "
            f"(poll {POLL_INTERVAL}s | threshold {COMPOSITE_THRESHOLD} | "
            f"min_contributors {MIN_CONTRIBUTING})"
        )
        while True:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[Convergence] Cycle error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    async def _scan_cycle(self):
        now = time.time()
        hour = datetime.now(timezone.utc).hour
        if hour < 7 or hour >= 21:
            return

        pressures = self._collect_pressures()
        if not pressures:
            return

        bullish_pressure = 0.0
        bearish_pressure = 0.0
        bull_contributors = []
        bear_contributors = []

        for name, (weight, pressure) in pressures.items():
            if pressure > PRESSURE_FLOOR:
                bullish_pressure += weight * pressure
                bull_contributors.append((name, pressure))
            elif pressure < -PRESSURE_FLOOR:
                bearish_pressure += weight * abs(pressure)
                bear_contributors.append((name, pressure))

        action = None
        composite = 0.0
        contributors = []

        if bullish_pressure >= COMPOSITE_THRESHOLD and len(bull_contributors) >= MIN_CONTRIBUTING:
            action = "BUY"
            composite = bullish_pressure
            contributors = bull_contributors
        elif bearish_pressure >= COMPOSITE_THRESHOLD and len(bear_contributors) >= MIN_CONTRIBUTING:
            action = "SELL"
            composite = bearish_pressure
            contributors = bear_contributors

        if bullish_pressure > 0.45 and len(bull_contributors) >= 2:
            self._global_bias = "BUY"
            self._bias_strength = bullish_pressure
            self._bias_until = now + 300
        elif bearish_pressure > 0.45 and len(bear_contributors) >= 2:
            self._global_bias = "SELL"
            self._bias_strength = bearish_pressure
            self._bias_until = now + 300
        elif now > self._bias_until:
            self._global_bias = None
            self._bias_strength = 0.0

        if action is None:
            return

        if self._last_signal_time and (now - self._last_signal_time) < COOLDOWN_SECS:
            return
        if (self._last_direction == action and self._last_direction_time
                and (now - self._last_direction_time) < DEDUP_WINDOW_SECS):
            return

        self._signal_timestamps = [t for t in self._signal_timestamps if now - t < 3600]
        if len(self._signal_timestamps) >= MAX_SIGNALS_PER_HOUR:
            return

        tick = mt5.symbol_info_tick(self._symbol)
        if tick is None:
            return
        spread_pips = (tick.ask - tick.bid) / self._pip_size
        if spread_pips > MAX_SPREAD_PIPS:
            return

        entry = tick.ask if action == "BUY" else tick.bid
        atr_est = abs(tick.ask - tick.bid) * 50
        if atr_est < 3.0:
            atr_est = 15.0

        sl_dist = atr_est * 0.4
        tp_dist = atr_est * 0.8

        if action == "BUY":
            sl = round(entry - sl_dist, 2)
            tp = round(entry + tp_dist, 2)
        else:
            sl = round(entry + sl_dist, 2)
            tp = round(entry - tp_dist, 2)

        scanner_info = ", ".join(f"{n}={p:+.2f}" for n, p in contributors)
        content = (
            f"{self._symbol} {action} @ {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP: {tp:.2f}\n"
            f"[Auto-Convergence] Multi-scanner fusion | composite={composite:.2f} "
            f"| {len(contributors)} scanners | {scanner_info}"
        )

        await queue_push(RawSignal(content=content, source="AUTO_CONVERGENCE"))
        self._signals_generated += 1
        self._last_signal_time = now
        self._last_direction = action
        self._last_direction_time = now
        self._signal_timestamps.append(now)

        logger.info(
            f"[Convergence] SYNTHETIC SIGNAL: {self._symbol} {action} @ {entry:.2f} "
            f"| composite={composite:.2f} | contributors: {scanner_info}"
        )

    def _collect_pressures(self):
        result = {}
        try:
            from quant.tick_flow import tfi_engine
            result["TFI"] = (SCANNER_WEIGHTS["TFI"], tfi_engine.pressure)
        except Exception:
            pass
        try:
            from quant.catcd_engine import catcd_engine
            result["CATCD"] = (SCANNER_WEIGHTS["CATCD"], catcd_engine.pressure)
        except Exception:
            pass
        try:
            from quant.momentum_scanner import momentum_scanner
            result["Momentum"] = (SCANNER_WEIGHTS["Momentum"], momentum_scanner.pressure)
        except Exception:
            pass
        try:
            from quant.mean_reversion_engine import mr_engine
            result["MR"] = (SCANNER_WEIGHTS["MR"], mr_engine.pressure)
        except Exception:
            pass
        try:
            from quant.liquidity_scanner import liquidity_scanner
            result["Liquidity"] = (SCANNER_WEIGHTS["Liquidity"], liquidity_scanner.pressure)
        except Exception:
            pass
        try:
            from quant.smc_scanner import smc_scanner
            result["SMC"] = (SCANNER_WEIGHTS["SMC"], smc_scanner.pressure)
        except Exception:
            pass
        return result

    def get_consensus_score(self) -> dict:
        """
        v4.2 Decision Consensus Layer.
        Returns a unified score (0-100) combining all intelligence sources.
        Used by risk_guard for lot sizing decisions.
        """
        pressures = self._collect_pressures()
        if not pressures:
            return {"score": 0, "direction": None, "components": {}}

        bull_sum = 0.0
        bear_sum = 0.0
        bull_count = 0
        bear_count = 0
        components = {}

        for name, (weight, pressure) in pressures.items():
            components[name] = round(pressure, 3)
            if pressure > 0.15:
                bull_sum += weight * pressure
                bull_count += 1
            elif pressure < -0.15:
                bear_sum += weight * abs(pressure)
                bear_count += 1

        direction = "BUY" if bull_sum > bear_sum else "SELL"
        dominant_sum = max(bull_sum, bear_sum)
        dominant_count = bull_count if bull_sum > bear_sum else bear_count

        raw_score = dominant_sum * 100
        agreement_bonus = min(dominant_count * 10, 30)
        raw_score += agreement_bonus

        amd_bonus = 0
        amd_phase = "NONE"
        if _amd_available:
            try:
                amd_state = amd_engine.get_state()
                amd_phase = amd_state["phase"]
                amd_bias = amd_state.get("bias")
                amd_conf = amd_state.get("confidence", 0)

                if amd_bias == direction and amd_conf > 0.5:
                    amd_bonus = int(amd_conf * 25)
                    raw_score += amd_bonus
                elif amd_bias and amd_bias != direction and amd_conf > 0.6:
                    raw_score -= 15
            except Exception:
                pass

        score = max(0, min(100, int(raw_score)))

        return {
            "score": score,
            "direction": direction,
            "dominant_pressure": round(dominant_sum, 3),
            "agreement_count": dominant_count,
            "amd_phase": amd_phase,
            "amd_bonus": amd_bonus,
            "components": components,
        }

    @property
    def global_bias(self):
        if time.time() > self._bias_until:
            return None
        return self._global_bias

    @property
    def bias_strength(self):
        if time.time() > self._bias_until:
            return 0.0
        return self._bias_strength


convergence_engine = ConvergenceEngine()
