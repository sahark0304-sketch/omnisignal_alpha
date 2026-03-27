"""
trade_manager/trade_manager.py — OmniSignal Alpha v4.6
Active Position Management Loop.

v4.6: Full feature restoration
  + Time-Based Stale Exit (close dead trades after 30min)
  + ATR-Adaptive Trailing Stop (1.5x M5 ATR floor)
  + Profit Guard (progressive profit lock with tick intensity regime)
  + CVD Exhaustion Exit (flow-based early exit on hidden selling)
  + Pyramid Runner (add 25% at TP1 if toxicity is low)
  + Friday Weekend Flattening (gap risk protection)
  + Trade Logger hooks (structured CLOSE events)
"""

import asyncio
import time
import numpy as np
import MetaTrader5 as _mt5
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Optional, Callable, Tuple
import config
from mt5_executor import mt5_executor
from database import db_manager
from utils.logger import get_logger, get_trade_logger
from utils.notifier import notify

logger = get_logger(__name__)
_trade_log = get_trade_logger()

# ── Per-ticket state tracking ──
_be_triggered: Set[int] = set()
_tp1_hit: Set[int] = set()
_tp2_hit: Set[int] = set()
_trailing: Dict[int, float] = {}
_original_lots: Dict[int, float] = {}
_pyramided: Set[int] = set()
_partialed_tickets: Set[int] = set()  # v8.0: MFE partial close tracking
_close_callback: Optional[Callable] = None

# ── Caches ──
_atr_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (atr_value, timestamp)
_tick_intensity_cache: Dict[str, Tuple[float, str, float]] = {}  # symbol -> (score, regime, timestamp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_m5_atr(symbol: str, period: int = 14) -> float:
    """Fetch M5 ATR. Cached for 30 seconds."""
    now = time.time()
    cached = _atr_cache.get(symbol)
    if cached and (now - cached[1]) < 30:
        return cached[0]
    try:
        rates = _mt5.copy_rates_from_pos(symbol, _mt5.TIMEFRAME_M5, 0, period + 1)
        if rates is None or len(rates) < period + 1:
            return 0.0
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
        atr = float(np.mean(tr[-period:]))
        _atr_cache[symbol] = (atr, now)
        return atr
    except Exception as e:
        logger.debug("[TM] ATR fetch error for %s: %s", symbol, e)
        return 0.0


def _measure_tick_intensity(symbol: str) -> Tuple[float, str]:
    """Returns (score, regime). Regimes: SLOW_CHOP, NORMAL, FAST_TREND. Cached 8s."""
    now = time.time()
    cached = _tick_intensity_cache.get(symbol)
    if cached and (now - cached[2]) < 8:
        return cached[0], cached[1]
    try:
        tick_from = datetime.now() - timedelta(seconds=60)
        ticks = _mt5.copy_ticks_from(symbol, tick_from, 5000, _mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) < 10:
            _tick_intensity_cache[symbol] = (1.0, "NORMAL", now)
            return 1.0, "NORMAL"
        prices = ticks["bid"].astype(float)
        times = ticks["time"].astype(float)
        duration = max(times[-1] - times[0], 1.0)
        changes_per_sec = len(prices) / duration
        displacement = abs(prices[-1] - prices[0])
        price_range = max(np.max(prices) - np.min(prices), 1e-8)
        directionality = displacement / price_range
        score = changes_per_sec * (1.0 + directionality)
        if changes_per_sec < 1.5 and directionality < 0.4:
            regime = "SLOW_CHOP"
        elif changes_per_sec > 4.0 and directionality > 0.6:
            regime = "FAST_TREND"
        else:
            regime = "NORMAL"
        _tick_intensity_cache[symbol] = (score, regime, now)
        return score, regime
    except Exception as e:
        logger.debug("[TM] Tick intensity error: %s", e)
        _tick_intensity_cache[symbol] = (1.0, "NORMAL", now)
        return 1.0, "NORMAL"


def _check_pyramid_conditions(symbol: str, action: str) -> bool:
    """Check if market conditions favor adding to a winner at TP1."""
    try:
        from quant.htf_filter import get_current_toxicity
        tox = get_current_toxicity(symbol)
        tox_score = tox.get("score", 0.5) if isinstance(tox, dict) else 0.5
        if tox_score >= 0.40:
            logger.debug("[TM] Pyramid blocked: toxicity %.2f >= 0.40", tox_score)
            return False
        if tox_score < 0.15:
            return True
    except Exception:
        pass
    try:
        from quant.convergence_engine import convergence_engine
        cs = convergence_engine.get_consensus_score()
        direction = cs.get("direction", "")
        if (action == "BUY" and direction == "BUY") or (action == "SELL" and direction == "SELL"):
            return True
    except Exception:
        pass
    return False

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_close_callback(callback: Callable):
    """
    Wire main.py's backfill_trade_label() here.
    Called from main.py startup() after trade_manager is imported.
    """
    global _close_callback
    _close_callback = callback
    logger.info("[TradeManager] Close callback registered for ML label backfill.")


async def run_management_loop():
    logger.info("[TradeManager] Management loop started.")
    while True:
        try:
            await _tick()
        except Exception as e:
            logger.error(f"[TradeManager] Loop error: {e}", exc_info=True)
        await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------

async def _tick():
    positions = mt5_executor.get_all_positions()
    live_tickets = {p["ticket"] for p in positions}

    # ── Friday Weekend Flattening ──
    try:
        _now_utc = datetime.now(timezone.utc)
        if _now_utc.weekday() == 4 and _now_utc.hour >= 20:
            if positions:
                for _fp in positions:
                    _ft = _fp["ticket"]
                    _fpnl = _fp.get("profit", 0)
                    _fentry = _fp["price_open"]
                    _fcur = _fp["price_current"]
                    _fpip = mt5_executor.get_pip_size(_fp["symbol"])
                    _fpips = abs(_fcur - _fentry) / _fpip
                    if _fpnl > 0 and _fpips > 10:
                        be_sl = _fentry + (2 * _fpip) if _fp["type"] == "BUY" else _fentry - (2 * _fpip)
                        mt5_executor.modify_sl(_ft, round(be_sl, 5))
                        logger.info("[TM] FRIDAY_FLATTEN: ticket=%s tightened SL to BE (profit +%.0fp)", _ft, _fpips)
                    else:
                        ok = mt5_executor.close_position(_ft)
                        if ok:
                            logger.info("[TM] FRIDAY_FLATTEN: closed ticket=%s at market (pnl=$%.2f)", _ft, _fpnl)
                    db_manager.log_audit("FRIDAY_FLATTEN", {
                        "ticket": _ft, "action": "BE_TIGHTEN" if (_fpnl > 0 and _fpips > 10) else "CLOSE",
                        "pnl": _fpnl, "pips": round(_fpips, 1),
                    })
                return
    except Exception as e:
        logger.debug("[TM] Friday flatten check error: %s", e)

    # ── Detect closed positions ──
    for ticket in list(_original_lots.keys()):
        if ticket not in live_tickets:
            logger.debug("[TM] Ticket %d no longer live — closing out.", ticket)
            await _handle_position_closed(ticket)

    if not positions:
        return

    db_open = {t["ticket"]: t for t in db_manager.get_open_trades()}

    for pos in positions:
        ticket = pos["ticket"]
        symbol = pos["symbol"]
        action = pos["type"]
        current = pos["price_current"]
        entry = pos["price_open"]
        current_sl = pos["sl"]
        pip_size = mt5_executor.get_pip_size(symbol)

        db_trade = db_open.get(ticket)
        if not db_trade:
            continue

        # == v7.0: Hard time kill at 60 minutes - trades >4hr lost -$699 ======
        try:
            _open_time_raw = pos.get("time") or db_trade.get("open_time")
            if _open_time_raw:
                import time as _time_mod
                if isinstance(_open_time_raw, (int, float)):
                    _age_mins = (_time_mod.time() - _open_time_raw) / 60
                else:
                    if isinstance(_open_time_raw, str):
                        for _tfmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                            try:
                                _open_time_raw = datetime.strptime(_open_time_raw, _tfmt)
                                break
                            except ValueError:
                                continue
                    if isinstance(_open_time_raw, datetime):
                        _age_mins = (datetime.now() - _open_time_raw).total_seconds() / 60
                    else:
                        _age_mins = 0
                _max_hold = getattr(config, "MAX_HOLD_MINUTES", 60)
                if _age_mins >= _max_hold:
                    logger.warning(
                        "[TM] TIME KILL: ticket=%d held %.0f min (max %d). Closing.",
                        ticket, _age_mins, _max_hold,
                    )
                    mt5_executor.close_position(ticket)
                    db_manager.log_audit("TIME_KILL", {
                        "ticket": ticket, "age_mins": round(_age_mins, 1),
                        "max_hold": _max_hold,
                    })
                    try:
                        notify(
                            f"TIME KILL: ticket {ticket} held {_age_mins:.0f}min (max {_max_hold})"
                        )
                    except Exception:
                        pass
                    continue
        except Exception as _tk_err:
            logger.debug("[TM] Time kill check error: %s", _tk_err)

        tp1 = db_trade.get("tp1_price")
        tp2 = db_trade.get("tp2_price")
        tp3 = db_trade.get("tp3_price")
        original_sl = db_trade.get("sl_price")

        if ticket not in _original_lots:
            _original_lots[ticket] = pos["volume"]

        original_lot = _original_lots[ticket]

        if action == "BUY":
            unrealized_pips = (current - entry) / pip_size
        else:
            unrealized_pips = (entry - current) / pip_size
        # == v8.0 EDGE 1: MFE Trailing Stop ======================================
        # v8.1: Regime-adaptive levels — wider trail in trending, tight in chop
        try:
            _mfe_pips = unrealized_pips
            _mfe_sl_changed = False

            _is_trending = False
            try:
                from quant.regime_detector import regime_detector, MarketRegime
                _rd_state = regime_detector.detect_regime(symbol)
                _is_trending = (_rd_state.regime == MarketRegime.TRENDING
                                and _rd_state.confidence >= 0.5)
            except Exception:
                pass

            # v8.2: Force trending MFE during active breakout
            try:
                from quant.breakout_hunter import breakout_hunter as _bh
                _bo_active, _ = _bh.is_breakout_active()
                if _bo_active:
                    _is_trending = True
            except Exception:
                pass

            if _is_trending:
                _mfe_be_trigger = 8;   _mfe_be_buf = 0.08
                _mfe_lock_trigger = 20; _mfe_lock_amt = 1.0
                _mfe_partial_trigger = 35; _mfe_partial_lock = 2.0
                _mfe_trail_dist = 1.5
            else:
                _mfe_be_trigger = 5;   _mfe_be_buf = 0.05
                _mfe_lock_trigger = 15; _mfe_lock_amt = 0.80
                _mfe_partial_trigger = 25; _mfe_partial_lock = 1.50
                _mfe_trail_dist = 1.0

            # TIER 1: Move SL to breakeven + tiny buffer
            if _mfe_pips >= _mfe_be_trigger and ticket not in _be_triggered:
                if action == "BUY":
                    _be = round(entry + _mfe_be_buf, 2)
                    if current_sl is None or current_sl < _be:
                        mt5_executor.modify_sl(ticket, _be)
                        _be_triggered.add(ticket)
                        _mfe_sl_changed = True
                        logger.info("[TM] MFE_BE: ticket=%d at +%.0f pips, SL -> BE %.2f", ticket, _mfe_pips, _be)
                else:
                    _be = round(entry - _mfe_be_buf, 2)
                    if current_sl is None or current_sl > _be:
                        mt5_executor.modify_sl(ticket, _be)
                        _be_triggered.add(ticket)
                        _mfe_sl_changed = True
                        logger.info("[TM] MFE_BE: ticket=%d at +%.0f pips, SL -> BE %.2f", ticket, _mfe_pips, _be)

            # TIER 2: Lock in profits
            if _mfe_pips >= _mfe_lock_trigger and not _mfe_sl_changed:
                if action == "BUY":
                    _lock = round(entry + _mfe_lock_amt, 2)
                    if current_sl is None or current_sl < _lock:
                        mt5_executor.modify_sl(ticket, _lock)
                        _mfe_sl_changed = True
                        logger.info("[TM] MFE_LOCK8: ticket=%d +%.0fp, SL -> +8p %.2f", ticket, _mfe_pips, _lock)
                else:
                    _lock = round(entry - _mfe_lock_amt, 2)
                    if current_sl is None or current_sl > _lock:
                        mt5_executor.modify_sl(ticket, _lock)
                        _mfe_sl_changed = True
                        logger.info("[TM] MFE_LOCK8: ticket=%d +%.0fp, SL -> +8p %.2f", ticket, _mfe_pips, _lock)

            # TIER 3: Lock deeper + partial close (once)
            if _mfe_pips >= _mfe_partial_trigger:
                if action == "BUY":
                    _lock25 = round(entry + _mfe_partial_lock, 2)
                    if current_sl is None or current_sl < _lock25:
                        mt5_executor.modify_sl(ticket, _lock25)
                        logger.info("[TM] MFE_LOCK15: ticket=%d +%.0fp, SL -> +15p %.2f", ticket, _mfe_pips, _lock25)
                else:
                    _lock25 = round(entry - _mfe_partial_lock, 2)
                    if current_sl is None or current_sl > _lock25:
                        mt5_executor.modify_sl(ticket, _lock25)
                        logger.info("[TM] MFE_LOCK15: ticket=%d +%.0fp, SL -> +15p %.2f", ticket, _mfe_pips, _lock25)

                if ticket not in _partialed_tickets:
                    _half = round(pos["volume"] / 2, 2)
                    if _half >= 0.01:
                        _ok = mt5_executor.close_partial(ticket, _half)
                        if _ok:
                            _partialed_tickets.add(ticket)
                            logger.info("[TM] MFE_PARTIAL: ticket=%d +%.0fp, closed 50%% (%.2f lots)", ticket, _mfe_pips, _half)
                            db_manager.log_audit("MFE_PARTIAL_CLOSE", {"ticket": ticket, "mfe_pips": round(_mfe_pips, 1), "lots_closed": _half})

            # TIER 4: Dynamic trail behind current price
            if _mfe_pips >= _mfe_partial_trigger:
                if action == "BUY":
                    _trail = round(current - _mfe_trail_dist, 2)
                    if current_sl is not None and _trail > current_sl + 0.3:
                        mt5_executor.modify_sl(ticket, _trail)
                else:
                    _trail = round(current + _mfe_trail_dist, 2)
                    if current_sl is not None and _trail < current_sl - 0.3:
                        mt5_executor.modify_sl(ticket, _trail)

        except Exception as _mfe_err:
            logger.debug("[TM] MFE trailing error: %s", _mfe_err)
        # == v7.1: Hybrid time-based exit - the $2,826 fix =======================
        try:
            _v71_open_time = pos.get("time") or db_trade.get("open_time")
            _v71_age = 0
            if _v71_open_time:
                import time as _t71
                if isinstance(_v71_open_time, (int, float)):
                    _v71_age = (_t71.time() - _v71_open_time) / 60
                else:
                    if isinstance(_v71_open_time, str):
                        for _tf in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                            try:
                                _v71_open_time = datetime.strptime(_v71_open_time, _tf)
                                break
                            except ValueError:
                                continue
                    if isinstance(_v71_open_time, datetime):
                        _v71_age = (datetime.now() - _v71_open_time).total_seconds() / 60

            _v71_pnl = pos.get("profit", 0)
            _v71_dir = 1 if action == "BUY" else -1

            # RULE 4: Early cut - if losing >60% of SL distance after 8 min
            if _v71_age >= 8 and _v71_pnl < 0 and original_sl and entry:
                _sl_dist = abs(entry - original_sl)
                _loss_dist = abs(current - entry)
                if _sl_dist > 0 and _loss_dist / _sl_dist > 0.6:
                    mt5_executor.close_position(ticket)
                    logger.warning(
                        "[TM] v7.1 EARLY_CUT: ticket=%d losing 60%%+ of SL at %.0fmin",
                        ticket, _v71_age,
                    )
                    db_manager.log_audit("V71_EARLY_CUT", {
                        "ticket": ticket, "age_mins": round(_v71_age, 1),
                        "loss_pct": round(_loss_dist / _sl_dist * 100, 0),
                    })
                    continue

            # RULE 1: Early BE - if profitable after 10 min, lock breakeven+2 pips
            if _v71_age >= 10 and _v71_pnl > 0 and ticket not in _be_triggered:
                _be_price = entry + (2 * pip_size * _v71_dir)
                if current_sl and abs(current_sl - _be_price) > pip_size:
                    success = mt5_executor.modify_sl(ticket, round(_be_price, 5))
                    if success:
                        _be_triggered.add(ticket)
                        logger.info(
                            "[TM] v7.1 EARLY_BE: ticket=%d at +%.0fmin, locking profit",
                            ticket, _v71_age,
                        )
                        db_manager.log_audit("V71_EARLY_BE", {
                            "ticket": ticket, "age_mins": round(_v71_age, 1),
                        })

            # RULE 2: At 20 min, if profitable, close 50%
            if 20 <= _v71_age < 22 and _v71_pnl > 0 and ticket not in _tp1_hit:
                _half = round(pos["volume"] / 2, 2)
                if _half >= 0.01:
                    mt5_executor.close_partial(ticket, _half)
                    _tp1_hit.add(ticket)
                    logger.info(
                        "[TM] v7.1 TIME_PARTIAL: ticket=%d closing 50%% at +%.0fmin pnl=$%.2f",
                        ticket, _v71_age, _v71_pnl,
                    )
                    db_manager.log_audit("V71_TIME_PARTIAL", {
                        "ticket": ticket, "age_mins": round(_v71_age, 1),
                        "closed_lots": _half, "pnl": round(_v71_pnl, 2),
                    })

        except Exception as _v71_err:
            logger.debug("[TM] v7.1 hybrid exit error: %s", _v71_err)

        # ── FEATURE 1: STALE EXIT ──
        try:
            stale_mins = getattr(config, "STALE_EXIT_MINUTES", 30)
            stale_min_pips = getattr(config, "STALE_EXIT_MIN_PIPS", 5.0)
            open_time_str = db_trade.get("open_time")
            if open_time_str and ticket in _be_triggered:
                if isinstance(open_time_str, str):
                    open_time = datetime.fromisoformat(open_time_str)
                else:
                    open_time = open_time_str
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
                minutes_open = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0
                if minutes_open >= stale_mins and abs(unrealized_pips) < stale_min_pips:
                    ok = mt5_executor.close_position(ticket)
                    if ok:
                        logger.warning(
                            "[TM] STALE_EXIT: ticket=%d closed after %.0fmin (%.1f pips)",
                            ticket, minutes_open, unrealized_pips,
                        )
                        _trade_log.info("STALE_EXIT | %s | %s | ticket=%d min=%.0f pips=%.1f",
                            symbol, action, ticket, minutes_open, unrealized_pips)
                        db_manager.log_audit("STALE_EXIT", {
                            "ticket": ticket, "minutes_open": round(minutes_open, 1),
                            "unrealized_pips": round(unrealized_pips, 1),
                        })
                        notify(f"⏰ *Stale Exit* | Ticket:`{ticket}` | {minutes_open:.0f}min | {unrealized_pips:.1f}p")
                        continue
        except Exception as e:
            logger.debug("[TM] Stale exit check error: %s", e)

        # ── BREAK-EVEN ──
        if ticket not in _be_triggered and tp1 and entry and current_sl:
            tp1_dist = abs(tp1 - entry)
            if tp1_dist > 0:
                be_trigger = (
                    entry + (tp1_dist * config.BE_TRIGGER_PCT) if action == "BUY"
                    else entry - (tp1_dist * config.BE_TRIGGER_PCT)
                )
                be_reached = (
                    (action == "BUY" and current >= be_trigger) or
                    (action == "SELL" and current <= be_trigger)
                )
                if be_reached and abs(current_sl - entry) > 0.00001:
                    success = mt5_executor.modify_sl(ticket, entry)
                    if success:
                        _be_triggered.add(ticket)
                        db_manager.update_trade_status(ticket, "BE_TRIGGERED")
                        db_manager.log_audit("BE_TRIGGERED", {"ticket": ticket, "entry": entry})
                        notify(f"🛡 *BE Set* | {symbol} {action} Ticket:`{ticket}` SL→`{entry}`")

        # ── FEATURE 3: PROFIT GUARD ──
        if ticket in _be_triggered and ticket not in _tp1_hit and tp1 and entry and original_sl:
            try:
                tp1_dist = abs(tp1 - entry)
                sl_dist = abs(original_sl - entry) if original_sl else tp1_dist
                if tp1_dist > 0:
                    progress = abs(current - entry) / tp1_dist
                    guard_trigger = min(sl_dist * 0.85, tp1_dist * 0.60) / tp1_dist
                    if progress >= guard_trigger and unrealized_pips > 0:
                        _, regime = _measure_tick_intensity(symbol)
                        lock_pct = {"SLOW_CHOP": 0.60, "FAST_TREND": 0.35}.get(regime, 0.50)

                        try:
                            from quant.htf_filter import get_current_toxicity
                            tox = get_current_toxicity(symbol)
                            tox_score = tox.get("score", 0) if isinstance(tox, dict) else 0
                            tox_dir = tox.get("direction", "") if isinstance(tox, dict) else ""
                            is_counter = (
                                (action == "BUY" and tox_dir == "SELL") or
                                (action == "SELL" and tox_dir == "BUY")
                            )
                            if tox_score > 0.5 and is_counter and unrealized_pips * pip_size > 0.5:
                                lock_pct = 0.70
                                logger.warning("[TM] TOXICITY_GUARD: ticket=%d tox=%.2f counter=%s lock=70%%", ticket, tox_score, tox_dir)
                                db_manager.log_audit("TOXICITY_GUARD", {"ticket": ticket, "tox_score": round(tox_score, 2)})
                        except Exception:
                            pass

                        locked_dist = abs(current - entry) * lock_pct
                        if action == "BUY":
                            guard_sl = round(entry + locked_dist, 5)
                            is_better = guard_sl > current_sl + (pip_size * 0.5)
                        else:
                            guard_sl = round(entry - locked_dist, 5)
                            is_better = guard_sl < current_sl - (pip_size * 0.5)

                        if is_better:
                            ok = mt5_executor.modify_sl(ticket, guard_sl)
                            if ok:
                                logger.info(
                                    "[TM] PROFIT_GUARD: ticket=%d progress=%.0f%% lock=%.0f%% SL→%.5f (%s)",
                                    ticket, progress * 100, lock_pct * 100, guard_sl, regime,
                                )
                                _trade_log.info("PROFIT_GUARD | %s | %s | ticket=%d SL=%.5f lock=%.0f%% regime=%s",
                                    symbol, action, ticket, guard_sl, lock_pct * 100, regime)
                                db_manager.log_audit("PROFIT_GUARD", {
                                    "ticket": ticket, "progress": round(progress, 2),
                                    "lock_pct": lock_pct, "guard_sl": guard_sl, "regime": regime,
                                })
            except Exception as e:
                logger.debug("[TM] Profit guard error: %s", e)
        # ── FEATURE 5: PYRAMID + TP1 PARTIAL CLOSE ──
        if ticket not in _tp1_hit and tp1:
            tp1_reached = (
                (action == "BUY" and current >= tp1) or
                (action == "SELL" and current <= tp1)
            )
            if tp1_reached:
                pyramid_done = False
                if getattr(config, "PYRAMID_ENABLED", False) and ticket not in _pyramided:
                    if _check_pyramid_conditions(symbol, action):
                        try:
                            add_lots = max(round(original_lot * config.PYRAMID_ADD_PCT, 2), 0.01)
                            new_sl = entry + (5 * pip_size) if action == "BUY" else entry - (5 * pip_size)
                            new_sl = round(new_sl, 5)
                            add_ticket = mt5_executor.place_raw_market_order(
                                symbol=symbol, action=action, lot_size=add_lots,
                                sl=new_sl, comment="OmniV2|Pyramid",
                            )
                            if add_ticket and add_ticket > 0:
                                mt5_executor.modify_sl(ticket, new_sl)
                                _pyramided.add(ticket)
                                _tp1_hit.add(ticket)
                                _trailing[ticket] = new_sl
                                pyramid_done = True
                                logger.info(
                                    "[TM] PYRAMID_ADD: parent=%d child=%d +%.2fL SL→%.5f",
                                    ticket, add_ticket, add_lots, new_sl,
                                )
                                _trade_log.info("PYRAMID | %s | %s | parent=%d add_lots=%.2f",
                                    symbol, action, ticket, add_lots)
                                db_manager.log_audit("PYRAMID_ADD", {
                                    "parent": ticket, "child": add_ticket,
                                    "add_lots": add_lots, "new_sl": new_sl,
                                })
                                notify(f"🔺 *Pyramid* | {symbol} {action} +{add_lots}L | SL→`{new_sl}`")
                        except Exception as e:
                            logger.warning("[TM] Pyramid failed for ticket %d: %s", ticket, e)

                if not pyramid_done:
                    close_lots = round(original_lot * config.TP1_CLOSE_PCT, 2)
                    if close_lots >= 0.01:
                        success = mt5_executor.close_partial(ticket, close_lots)
                        if success:
                            _tp1_hit.add(ticket)
                            db_manager.update_trade_status(ticket, "TP1_HIT")
                            db_manager.log_audit("TP1_HIT", {"ticket": ticket, "lots": close_lots})
                            notify(f"🎯 *TP1!* | {symbol} {action} Ticket:`{ticket}` @ `{current}`")
                            _trailing[ticket] = entry

        # ── TP2 PARTIAL CLOSE ──
        if ticket in _tp1_hit and ticket not in _tp2_hit and tp2:
            tp2_reached = (
                (action == "BUY" and current >= tp2) or
                (action == "SELL" and current <= tp2)
            )
            if tp2_reached:
                close_lots = min(
                    round(original_lot * config.TP2_CLOSE_PCT, 2),
                    pos["volume"],
                )
                if close_lots >= 0.01:
                    success = mt5_executor.close_partial(ticket, close_lots)
                    if success:
                        _tp2_hit.add(ticket)
                        db_manager.log_audit("TP2_HIT", {"ticket": ticket, "lots": close_lots})
                        notify(f"🎯🎯 *TP2!* | {symbol} {action} Ticket:`{ticket}` @ `{current}`")

        # ── FEATURE 4: CVD EXHAUSTION EXIT ──
        if ticket in _be_triggered and unrealized_pips > 0:
            try:
                from quant.flow_exit import check_flow_exit
                should_exit, cvd_reason = check_flow_exit(ticket, symbol, action, entry, current, pip_size)
                if should_exit:
                    ok = mt5_executor.close_position(ticket)
                    if ok:
                        logger.warning("[TM] CVD_EXHAUSTION_EXIT: ticket=%d %s", ticket, cvd_reason)
                        _trade_log.info("CVD_EXIT | %s | %s | ticket=%d unrealized=%.1fp",
                            symbol, action, ticket, unrealized_pips)
                        db_manager.log_audit("CVD_EXHAUSTION_EXIT", {
                            "ticket": ticket, "reason": cvd_reason,
                            "unrealized_pips": round(unrealized_pips, 1),
                        })
                        notify(f"🌊 *CVD Exit* | Ticket:`{ticket}` | {unrealized_pips:.1f}p | {cvd_reason[:60]}")
                        continue
            except Exception as e:
                logger.debug("[TM] CVD check error: %s", e)

        # ── FEATURE 2: ATR-ADAPTIVE TRAILING STOP ──
        if ticket in _tp1_hit and ticket in _trailing:
            try:
                m5_atr = _get_m5_atr(symbol)
                static_activation = config.TRAILING_STOP_ACTIVATION_PIPS * pip_size
                static_step = config.TRAILING_STOP_STEP_PIPS * pip_size
                if m5_atr > 0:
                    trail_activation = max(1.5 * m5_atr, static_activation)
                    trail_step = max(0.4 * m5_atr, static_step)
                else:
                    trail_activation = static_activation
                    trail_step = static_step

                if action == "BUY":
                    ideal_sl = current - trail_activation
                    if ideal_sl > _trailing[ticket] + trail_step:
                        success = mt5_executor.modify_sl(ticket, round(ideal_sl, 5))
                        if success:
                            _trailing[ticket] = ideal_sl
                else:
                    ideal_sl = current + trail_activation
                    if ideal_sl < _trailing[ticket] - trail_step:
                        success = mt5_executor.modify_sl(ticket, round(ideal_sl, 5))
                        if success:
                            _trailing[ticket] = ideal_sl
            except Exception as e:
                logger.debug("[TM] Trailing error: %s", e)

# ---------------------------------------------------------------------------
# Close handler
# ---------------------------------------------------------------------------

async def _handle_position_closed(ticket: int):
    """Fetch close data, update DB, trigger forensics and ML backfill."""
    if ticket not in _original_lots:
        return
    _original_lots.pop(ticket, None)

    try:
        close_price = 0.0
        pnl = 0.0
        tp1_hit = ticket in _tp1_hit
        symbol = "XAUUSD"
        action = "BUY"

        deals = _mt5.history_deals_get(position=ticket)
        if deals:
            close_deal = sorted(deals, key=lambda d: d.time)[-1]
            close_price = float(close_deal.price)
            pnl = sum(float(d.profit) for d in deals)
            symbol = close_deal.symbol
            action = "BUY" if close_deal.type == 1 else "SELL"

        db_manager.close_trade(ticket, close_price, pnl)

        # v5.0: Record close in ATO
        try:
            from quant.trade_orchestrator import trade_orchestrator as _ato
            _ato.record_close(ticket=ticket, pnl=pnl, source="")
        except Exception:
            pass
        logger.info("[TM] Closed ticket:%d @ %.5f PnL:$%.2f", ticket, close_price, pnl)
        get_trade_logger().info("CLOSE | ticket=%d | pnl=%.2f close_price=%.5f", ticket, pnl, close_price)

        try:
            from quant.breakout_guard import register_consecutive_win, register_consecutive_loss, register_trend_win
            _be_neutral_threshold = getattr(config, "BE_NEUTRAL_PNL_THRESHOLD", 3.0)
            if abs(pnl) < _be_neutral_threshold:
                logger.debug("[TM] BE-NEUTRAL: ticket=%d pnl=$%.2f— not counted for streaks", ticket, pnl)
            elif pnl > 0:
                register_consecutive_win(symbol)
                _, regime = _measure_tick_intensity(symbol)
                if regime == "FAST_TREND":
                    try:
                        db_trades = db_manager.get_open_trades()
                        db_t = next((t for t in db_trades if t.get("ticket") == ticket), None)
                        if db_t:
                            register_trend_win(symbol, db_t.get("action", "BUY"), regime)
                    except Exception:
                        pass
            else:
                register_consecutive_loss(symbol)
                try:
                    from quant.breakout_guard import register_session_loss
                    register_session_loss(symbol, pnl)
                except Exception:
                    pass
        except Exception:
            pass

        if config.NOTIFY_ON_TRADE_CLOSE and abs(pnl) > 0:
            emoji = "✅" if pnl >= 0 else "❌"
            notify(f"{emoji} *Trade Closed*\nTicket:`{ticket}` PnL:`${pnl:+.2f}`")

        if _close_callback is not None:
            pip_size = mt5_executor.get_pip_size(symbol)
            _close_callback(ticket, close_price, pnl, tp1_hit, pip_size)

        # v8.0: Smart re-entry after profitable TP1 hit
        if tp1_hit and pnl > 0:
            try:
                _re_action = db_trade.get("action", "")
                _re_entry = db_trade.get("entry_price", 0)
                _re_sl = db_trade.get("sl_price", 0)
                _re_tp1 = db_trade.get("tp1_price", 0)
                _re_tp2 = db_trade.get("tp2_price")

                if _re_action and _re_entry and _re_sl and _re_tp1:
                    _re_content = (
                        "XAUUSD %s @ %.2f\n"
                        "SL: %.2f\n"
                        "TP: %.2f\n"
                        "[Smart-ReEntry] TP1 hit, re-entering on trend continuation"
                    ) % (_re_action, close_price, _re_sl, _re_tp2 or _re_tp1)

                    async def _delayed_reentry():
                        import asyncio
                        await asyncio.sleep(60)
                        from ingestion.signal_queue import RawSignal, push
                        raw = RawSignal(
                            source="AUTO_REENTRY",
                            content=_re_content,
                            received_at=datetime.now(),
                        )
                        await push(raw)
                        logger.info(
                            "[TM] SMART RE-ENTRY pushed: %s %s after TP1 (pnl=$%.2f)",
                            symbol, _re_action, pnl,
                        )

                    import asyncio
                    asyncio.ensure_future(_delayed_reentry())
                    db_manager.log_audit("SMART_REENTRY_QUEUED", {
                        "ticket": ticket, "action": _re_action,
                        "close_price": close_price, "pnl": round(pnl, 2),
                    })
            except Exception as _re_err:
                logger.debug("[TM] Smart re-entry error: %s", _re_err)

    except Exception as e:
        logger.error("[TM] Close handler error for ticket %d: %s", ticket, e)
    finally:
        cleanup_closed_ticket(ticket)


def cleanup_closed_ticket(ticket: int):
    """Free all per-ticket in-memory state."""
    _be_triggered.discard(ticket)
    _tp1_hit.discard(ticket)
    _tp2_hit.discard(ticket)
    _trailing.pop(ticket, None)
    _original_lots.pop(ticket, None)
    _pyramided.discard(ticket)
    _partialed_tickets.discard(ticket)
    try:
        from quant.flow_exit import cleanup_ticket
        cleanup_ticket(ticket)
    except Exception:
        pass




