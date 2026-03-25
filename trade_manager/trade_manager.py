"""
trade_manager/trade_manager.py - OmniSignal Alpha v4.4
Active Position Management Loop.

Features:
  - Break-Even with commission cover (+2 pips offset)
  - TP1 partial close with optional pyramiding (Ghost Tag bypass)
  - TP2 partial close
  - v4.3 Profit Guard: progressive profit lock before TP1
  - v4.3.1 Golden Mean: SL-capped adaptive guard threshold
  - v4.3.1 Tick-intensity adaptive trailing
  - v4.3 Toxicity Guard: counter-trend protection
  - v4.3.2 Consecutive loss registration
  - v4.4 CVD Exhaustion exits
  - v4.4 Post-win trend bias registration
  - Trailing stop after TP1
  - ML close callback for label backfill
"""

import asyncio
import time
from typing import Dict, Set, Optional, Callable, Tuple
import numpy as np
import config
from mt5_executor import mt5_executor
from database import db_manager
from utils.logger import get_logger
from utils.notifier import notify
from quant.htf_filter import get_current_toxicity

logger = get_logger(__name__)

_be_triggered: Set[int]          = set()
_tp1_hit: Set[int]               = set()
_tp2_hit: Set[int]               = set()
_trailing: Dict[int, float]      = {}
_original_lots: Dict[int, float] = {}
_pyramided: Set[int]             = set()
_close_callback: Optional[Callable] = None
_tick_intensity_cache: Dict[str, Tuple[float, str, float]] = {}


def _measure_tick_intensity(symbol: str) -> Tuple[float, str]:
    """
    Measure real-time tick intensity over 60s window.
    Returns (intensity_score, regime_label).
    Regimes: SLOW_CHOP, NORMAL, FAST_TREND
    Cached for 8 seconds per symbol.
    """
    now = time.time()
    cached = _tick_intensity_cache.get(symbol)
    if cached and now - cached[2] < 8:
        return cached[0], cached[1]

    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta
        tick_from = datetime.now() - timedelta(seconds=60)
        ticks = mt5.copy_ticks_from(symbol, tick_from, 5000, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) < 20:
            _tick_intensity_cache[symbol] = (1.0, "NORMAL", now)
            return 1.0, "NORMAL"

        prices = (ticks["bid"] + ticks["ask"]) / 2.0
        diffs = np.abs(np.diff(prices))
        nonzero = diffs[diffs > 0]

        if len(nonzero) < 5:
            _tick_intensity_cache[symbol] = (0.3, "SLOW_CHOP", now)
            return 0.3, "SLOW_CHOP"

        tick_times = ticks["time"].astype(float)
        elapsed = max(tick_times[-1] - tick_times[0], 1.0)
        changes_per_sec = len(nonzero) / elapsed
        total_displacement = abs(float(prices[-1] - prices[0]))
        total_range = float(np.max(prices) - np.min(prices))
        directionality = total_displacement / max(total_range, 0.01)

        if changes_per_sec < 1.5 and directionality < 0.4:
            regime = "SLOW_CHOP"
            score = max(0.2, changes_per_sec / 3.0)
        elif changes_per_sec > 4.0 and directionality > 0.6:
            regime = "FAST_TREND"
            score = min(3.0, changes_per_sec / 2.0)
        else:
            regime = "NORMAL"
            score = 1.0

        _tick_intensity_cache[symbol] = (score, regime, now)
        return score, regime
    except Exception:
        _tick_intensity_cache[symbol] = (1.0, "NORMAL", now)
        return 1.0, "NORMAL"


def set_close_callback(callback: Callable):
    global _close_callback
    _close_callback = callback
    logger.info("[TradeManager] Close callback registered.")


async def run_management_loop():
    logger.info("[TradeManager] Management loop started.")
    while True:
        try:
            await _tick()
        except Exception as e:
            logger.error("[TradeManager] Loop error: %s", e, exc_info=True)
        await asyncio.sleep(5)


async def _tick():
    positions = mt5_executor.get_all_positions()
    live_tickets = {p["ticket"] for p in positions}
    for ticket in list(_original_lots.keys()):
        if ticket not in live_tickets:
            await _handle_position_closed(ticket)
    if not positions:
        return
    db_open = {t["ticket"]: t for t in db_manager.get_open_trades()}
    for pos in positions:
        ticket     = pos["ticket"]
        symbol     = pos["symbol"]
        action     = pos["type"]
        current    = pos["price_current"]
        entry      = pos["price_open"]
        current_sl = pos["sl"]
        db_trade = db_open.get(ticket)
        if not db_trade:
            continue
        tp1 = db_trade.get("tp1_price")
        tp2 = db_trade.get("tp2_price")
        tp3 = db_trade.get("tp3_price")
        if ticket not in _original_lots:
            _original_lots[ticket] = pos["volume"]
        original_lot = _original_lots[ticket]

        pip_size = mt5_executor.get_pip_size(symbol)

        # -- BREAK-EVEN (+2 pips commission cover) --
        if ticket not in _be_triggered and tp1 and entry and current_sl:
            tp1_dist = abs(tp1 - entry)
            if tp1_dist > 0:
                be_trigger = (entry + tp1_dist * config.BE_TRIGGER_PCT if action == "BUY"
                              else entry - tp1_dist * config.BE_TRIGGER_PCT)
                be_reached = ((action == "BUY" and current >= be_trigger)
                              or (action == "SELL" and current <= be_trigger))
                if be_reached and abs(current_sl - entry) > 0.00001:
                    be_offset = 2.0 * pip_size
                    be_sl = round(entry + be_offset, 5) if action == "BUY" else round(entry - be_offset, 5)
                    ok = mt5_executor.modify_sl(ticket, be_sl)
                    if ok:
                        _be_triggered.add(ticket)
                        db_manager.update_trade_status(ticket, "BE_TRIGGERED")
                        db_manager.log_audit("BE_TRIGGERED", {
                            "ticket": ticket, "entry": entry, "be_sl": be_sl,
                        })
                        notify(
                            "\U0001f6e1\ufe0f *Break-Even Locked*\n\n"
                            "%s %s\nSL moved to `%s`\nTicket: `%s`"
                            % (symbol, action, be_sl, ticket)
                        )

        # -- TP1: PYRAMID or PARTIAL CLOSE --
        if ticket not in _tp1_hit and tp1:
            tp1_reached = ((action == "BUY" and current >= tp1)
                           or (action == "SELL" and current <= tp1))
            if tp1_reached:
                should_pyramid = _check_pyramid_conditions(symbol, action)
                pyr_on = getattr(config, "PYRAMID_ENABLED", True)
                if should_pyramid and ticket not in _pyramided and pyr_on:
                    add_pct = getattr(config, "PYRAMID_ADD_PCT", 0.25)
                    add_lots = max(round(original_lot * add_pct, 2), 0.01)
                    lock_pips = 5.0
                    lock = lock_pips * pip_size
                    new_sl = round(entry + lock, 5) if action == "BUY" else round(entry - lock, 5)
                    add_ticket = mt5_executor.place_raw_market_order(
                        symbol=symbol, action=action, lot_size=add_lots, sl=new_sl)
                    if add_ticket and add_ticket > 0:
                        _pyramided.add(ticket)
                        mt5_executor.modify_sl(ticket, new_sl)
                        _tp1_hit.add(ticket)
                        _trailing[ticket] = new_sl
                        db_manager.update_trade_status(ticket, "PYRAMID_TP1")
                        db_manager.log_audit("PYRAMID_ADD", {
                            "parent_ticket": ticket, "add_ticket": add_ticket,
                            "add_lots": add_lots, "new_sl": new_sl,
                        })
                        logger.info(
                            "[TM] PYRAMID %s %s +%sL parent=%s add=%s SL->%s",
                            symbol, action, add_lots, ticket, add_ticket, new_sl,
                        )
                    else:
                        _do_tp1_close(ticket, symbol, action, current, entry, original_lot, True)
                else:
                    _do_tp1_close(ticket, symbol, action, current, entry, original_lot)

        # -- TP2 PARTIAL CLOSE --
        if ticket in _tp1_hit and ticket not in _tp2_hit and tp2:
            tp2_reached = ((action == "BUY" and current >= tp2)
                           or (action == "SELL" and current <= tp2))
            if tp2_reached:
                close_lots = min(round(original_lot * config.TP2_CLOSE_PCT, 2), pos["volume"])
                if close_lots >= 0.01:
                    ok = mt5_executor.close_partial(ticket, close_lots)
                    if ok:
                        _tp2_hit.add(ticket)
                        db_manager.log_audit("TP2_HIT", {"ticket": ticket, "lots": close_lots})

        # -- v4.3.1 ADAPTIVE PROFIT GUARD (before TP1, Golden Mean) --
        if ticket in _be_triggered and ticket not in _tp1_hit:
            if action == "BUY":
                unrealized_pips = (current - entry) / pip_size
            else:
                unrealized_pips = (entry - current) / pip_size

            unrealized_pnl = unrealized_pips * pip_size * pos["volume"] * 100

            if tp1 and entry:
                tp1_dist = abs(tp1 - entry)
                sl_dist = abs(db_trade.get("sl_price", current_sl) - entry)

                guard_trigger_dist = min(sl_dist * 0.85, tp1_dist * 0.60)
                guard_threshold = guard_trigger_dist / tp1_dist if tp1_dist > 0 else 0.60
                progress = abs(current - entry) / tp1_dist if tp1_dist > 0 else 0

                if progress >= guard_threshold:
                    _, tick_regime = _measure_tick_intensity(symbol)
                    if tick_regime == "SLOW_CHOP":
                        lock_pct = 0.60
                    elif tick_regime == "FAST_TREND":
                        lock_pct = 0.35
                    else:
                        lock_pct = 0.50

                    locked_dist = abs(current - entry) * lock_pct
                    if action == "BUY":
                        guard_sl = round(entry + locked_dist, 2)
                    else:
                        guard_sl = round(entry - locked_dist, 2)

                    current_sl_ok = (
                        (action == "BUY" and guard_sl > current_sl + pip_size) or
                        (action == "SELL" and guard_sl < current_sl - pip_size)
                    )
                    if current_sl_ok:
                        success = mt5_executor.modify_sl(ticket, guard_sl)
                        if success:
                            logger.info(
                                "[TM] PROFIT GUARD: %s %s ticket:%d SL->%.2f "
                                "(locking %d%% at %.0f%% to TP1, regime=%s)",
                                symbol, action, ticket, guard_sl,
                                int(lock_pct * 100), progress * 100, tick_regime,
                            )
                            db_manager.log_audit("PROFIT_GUARD", {
                                "ticket": ticket, "guard_sl": guard_sl,
                                "progress_pct": round(progress * 100, 1),
                                "unrealized_pnl": round(unrealized_pnl, 2),
                                "tick_regime": tick_regime,
                                "lock_pct": lock_pct,
                                "guard_threshold": round(guard_threshold, 3),
                            })

            # Toxicity Guard: tighten if counter-trend detected while in profit
            if unrealized_pnl > 50:
                try:
                    tox = get_current_toxicity(symbol)
                    tox_score = tox.get("score", 0)
                    tox_bias = tox.get("bias")
                    counter = (
                        (action == "BUY" and tox_bias == "BEARISH" and tox_score > 0.5) or
                        (action == "SELL" and tox_bias == "BULLISH" and tox_score > 0.5)
                    )
                    if counter:
                        lock_pct = 0.70
                        locked_dist = abs(current - entry) * lock_pct
                        if action == "BUY":
                            tox_sl = round(entry + locked_dist, 2)
                        else:
                            tox_sl = round(entry - locked_dist, 2)
                        tox_sl_better = (
                            (action == "BUY" and tox_sl > current_sl + pip_size) or
                            (action == "SELL" and tox_sl < current_sl - pip_size)
                        )
                        if tox_sl_better:
                            success = mt5_executor.modify_sl(ticket, tox_sl)
                            if success:
                                logger.warning(
                                    "[TM] TOXICITY GUARD: %s %s ticket:%d "
                                    "SL->%.2f (locking 70%% tox=%.2f)",
                                    symbol, action, ticket, tox_sl, tox_score,
                                )
                                db_manager.log_audit("TOXICITY_GUARD", {
                                    "ticket": ticket, "tox_sl": tox_sl,
                                    "tox_score": round(tox_score, 3),
                                    "unrealized_pnl": round(unrealized_pnl, 2),
                                })
                except Exception:
                    pass

        # -- v4.4: CVD EXHAUSTION EXIT --
        if ticket in _be_triggered:
            try:
                from quant.flow_exit import check_flow_exit
                should_exit, exit_reason = check_flow_exit(
                    ticket, symbol, action, entry, current, pip_size,
                )
                if should_exit:
                    ok = mt5_executor.close_partial(ticket, pos["volume"])
                    if ok:
                        logger.warning(
                            "[TM] CVD EXIT: %s %s ticket:%d closed at %.2f | %s",
                            symbol, action, ticket, current, exit_reason,
                        )
                        db_manager.log_audit("CVD_EXHAUSTION_EXIT", {
                            "ticket": ticket, "symbol": symbol, "action": action,
                            "close_price": current, "reason": exit_reason,
                        })
                        notify(
                            "\U0001f4a8 *CVD Exhaustion Exit*\n\n"
                            "%s %s\nTicket: `%s`\nClosed @ `%.2f`\n\n_%s_"
                            % (symbol, action, ticket, current, exit_reason)
                        )
            except Exception as e:
                logger.debug("[TM] Flow exit check error: %s", e)

        # -- TRAILING STOP (after TP1) --
        if ticket in _tp1_hit and ticket in _trailing:
            trail_step       = config.TRAILING_STOP_STEP_PIPS * pip_size
            trail_activation = config.TRAILING_STOP_ACTIVATION_PIPS * pip_size
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


def _do_tp1_close(ticket, symbol, action, current, entry, original_lot, pyr_fail=False):
    close_lots = round(original_lot * config.TP1_CLOSE_PCT, 2)
    if close_lots >= 0.01:
        ok = mt5_executor.close_partial(ticket, close_lots)
        if ok:
            _tp1_hit.add(ticket)
            db_manager.update_trade_status(ticket, "TP1_HIT")
            extra = {"ticket": ticket, "lots": close_lots}
            if pyr_fail:
                extra["pyramid_failed"] = True
            db_manager.log_audit("TP1_HIT", extra)
            _trailing[ticket] = entry


def _check_pyramid_conditions(symbol, action):
    try:
        tox = get_current_toxicity(symbol)
        score = tox.get("score", 0.0)
        if score >= 0.40:
            return False
        from quant.convergence_engine import convergence_engine
        bias = convergence_engine.global_bias
        if bias and bias == action:
            return True
        if score < 0.15:
            return True
    except Exception:
        pass
    return False


def _collect_forensics(ticket, close_price, pnl, symbol, action, tp1_hit):
    """v4.4: Collect full universe snapshot and generate trade autopsy."""
    import json
    from datetime import datetime

    pip_size = 0.1

    # Get trade details from DB
    with db_manager.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE ticket=?", (ticket,)
        ).fetchone()
    if not row:
        return
    trade = dict(row)

    signal_id = trade.get("signal_id")
    entry_price = trade.get("entry_price", 0)
    lot_size = trade.get("lot_size", 0)
    open_time = trade.get("open_time")
    sl_price = trade.get("sl_price", 0)

    # Duration
    duration_secs = 0
    if open_time:
        try:
            if isinstance(open_time, str):
                ot = datetime.fromisoformat(open_time)
            else:
                ot = open_time
            duration_secs = int((datetime.now() - ot).total_seconds())
        except Exception:
            pass

    # MFE/MAE from deal history
    mfe_pips = 0.0
    mae_pips = 0.0
    try:
        import MetaTrader5 as mt5
        deals = mt5.history_deals_get(position=ticket)
        if deals and entry_price:
            prices = [float(d.price) for d in deals if d.price > 0]
            if prices:
                if action == "BUY":
                    mfe_pips = (max(prices) - entry_price) / pip_size
                    mae_pips = (entry_price - min(prices)) / pip_size
                else:
                    mfe_pips = (entry_price - min(prices)) / pip_size
                    mae_pips = (max(prices) - entry_price) / pip_size
    except Exception:
        pass

    mfe_dollars = mfe_pips * pip_size * lot_size * 100
    mae_dollars = mae_pips * pip_size * lot_size * 100

    # Tick regime
    _, tick_regime = _measure_tick_intensity(symbol)

    # Toxicity
    try:
        from quant.htf_filter import get_current_toxicity, check_htf_trend_gate
        tox = get_current_toxicity(symbol)
        tox_score = tox.get("score", 0)
    except Exception:
        tox_score = 0

    # HTF bias
    m15_bias = "N/A"
    m5_bias = "N/A"
    try:
        from quant.htf_filter import check_htf_trend_gate
        import MetaTrader5 as mt5
        import numpy as np

        def _ema(data, period):
            k = 2.0 / (period + 1)
            ema = np.empty(len(data))
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = data[i] * k + ema[i - 1] * (1 - k)
            return ema

        m15_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 60)
        if m15_rates is not None and len(m15_rates) >= 55:
            closes = m15_rates['close'].astype(float)
            e20 = _ema(closes, 20)[-1]
            e50 = _ema(closes, 50)[-1]
            m15_bias = "BULLISH" if e20 > e50 else "BEARISH"

        m5_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60)
        if m5_rates is not None and len(m5_rates) >= 55:
            closes = m5_rates['close'].astype(float)
            e20 = _ema(closes, 20)[-1]
            e50 = _ema(closes, 50)[-1]
            m5_bias = "BULLISH" if e20 > e50 else "BEARISH"
    except Exception:
        pass

    # DXY z-score
    dxy_z = 0.0
    try:
        from quant.catcd_engine import catcd_engine
        dxy_z = catcd_engine._last_pressure
    except Exception:
        pass

    # Consensus
    consensus = 0.0
    try:
        from quant.convergence_engine import convergence_engine
        consensus = convergence_engine.bias_strength
    except Exception:
        pass

    # AI confidence from signal
    ai_conf = 0
    try:
        with db_manager.get_connection() as conn:
            sig = conn.execute(
                "SELECT ai_confidence, source FROM signals WHERE id=?", (signal_id,)
            ).fetchone()
            if sig:
                ai_conf = sig["ai_confidence"] or 0
    except Exception:
        pass

    # Exit trigger
    exit_trigger = "UNKNOWN"
    try:
        with db_manager.get_connection() as conn:
            audit = conn.execute(
                "SELECT event_type FROM audit_log WHERE details LIKE ? "
                "ORDER BY ts DESC LIMIT 1",
                (f'%"ticket": {ticket}%',)
            ).fetchone()
            if audit:
                et = audit["event_type"]
                if et in ("CVD_EXHAUSTION_EXIT", "PROFIT_GUARD", "TOXICITY_GUARD",
                          "TP1_HIT", "TP2_HIT", "PYRAMID_ADD", "BE_TRIGGERED"):
                    exit_trigger = et
                elif tp1_hit:
                    exit_trigger = "TP1_TRAILING"
                elif pnl < 0:
                    exit_trigger = "SL_HIT"
                else:
                    exit_trigger = "TRAILING_STOP"
            else:
                exit_trigger = "SL_HIT" if pnl < 0 else "TP_OR_TRAILING"
    except Exception:
        pass

    # Slippage
    slippage = 0.0
    try:
        with db_manager.get_connection() as conn:
            sig_row = conn.execute(
                "SELECT parsed_json FROM signals WHERE id=?", (signal_id,)
            ).fetchone()
            if sig_row and sig_row["parsed_json"]:
                parsed = json.loads(sig_row["parsed_json"])
                expected = parsed.get("entry_price", entry_price)
                if expected and entry_price:
                    slippage = abs(entry_price - float(expected)) / pip_size
    except Exception:
        pass

    # Source
    source = trade.get("source", "")
    if not source:
        try:
            with db_manager.get_connection() as conn:
                s = conn.execute(
                    "SELECT source FROM signals WHERE id=?", (signal_id,)
                ).fetchone()
                if s:
                    source = s["source"]
        except Exception:
            pass

    # Generate lesson learned
    lesson = _generate_lesson(pnl, mfe_pips, mae_pips, tick_regime, exit_trigger,
                              source, action, tox_score, m15_bias, m5_bias)

    # Save forensic snapshot
    forensic_data = {
        "ticket": ticket,
        "signal_id": signal_id,
        "m15_bias": m15_bias,
        "m5_bias": m5_bias,
        "tick_regime": tick_regime,
        "toxicity_score": round(tox_score, 3),
        "consensus_score": round(consensus, 3),
        "ai_confidence": ai_conf,
        "active_scanners": "",
        "dxy_z_score": round(dxy_z, 3),
        "entry_spread_pips": 0,
        "expected_entry": entry_price,
        "actual_entry": entry_price,
        "slippage_pips": round(slippage, 2),
        "mfe_pips": round(mfe_pips, 1),
        "mae_pips": round(mae_pips, 1),
        "mfe_dollars": round(mfe_dollars, 2),
        "mae_dollars": round(mae_dollars, 2),
        "duration_secs": duration_secs,
        "exit_trigger": exit_trigger,
        "exit_price": close_price,
        "pnl": round(pnl, 2),
        "lesson_learned": lesson,
        "weight_adjustments": "",
        "source": source,
        "action": action,
        "lot_size": lot_size,
    }
    db_manager.insert_forensic(forensic_data)

    # Format and send Telegram autopsy
    duration_str = f"{duration_secs // 60}m {duration_secs % 60}s" if duration_secs else "N/A"
    emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
    autopsy = (
        f"{emoji} *TRADE AUTOPSY: Ticket {ticket}*\n\n"
        f"Result: `${pnl:+.2f}` | Duration: `{duration_str}`\n"
        f"Regime: `{tick_regime}` | Toxicity: `{tox_score:.2f}`\n"
        f"HTF: M15=`{m15_bias}` M5=`{m5_bias}`\n\n"
        f"*Entry:* `{action} @ {entry_price}` ({source})\n"
        f"*Exit:* `{exit_trigger} @ {close_price}`\n"
        f"MFE: `+{mfe_pips:.0f}p ($+{mfe_dollars:.0f})` | "
        f"MAE: `-{mae_pips:.0f}p ($-{mae_dollars:.0f})`\n\n"
        f"*AI Lesson:* _{lesson}_"
    )
    try:
        from utils.notifier import send_autopsy
        send_autopsy(autopsy)
    except Exception:
        pass

    logger.info("[TM] FORENSIC: ticket:%s exit:%s MFE:+%.0fp MAE:-%.0fp lesson:%s",
                ticket, exit_trigger, mfe_pips, mae_pips, lesson[:80])


def _generate_lesson(pnl, mfe_pips, mae_pips, regime, exit_trigger,
                     source, action, tox_score, m15_bias, m5_bias):
    """Generate an explainable AI lesson string from trade outcome."""
    if pnl > 0:
        if mfe_pips > 30 and exit_trigger in ("CVD_EXHAUSTION_EXIT",):
            return (f"CVD exit protected ${pnl:.0f} gain. MFE +{mfe_pips:.0f}p. "
                    f"Flow reversal detected early. Retaining {source} confidence.")
        elif mfe_pips > 20 and exit_trigger == "TP1_TRAILING":
            return (f"TP1 partial + trailing captured ${pnl:.0f}. "
                    f"Regime {regime} favored runners. Increasing trailing aggression.")
        elif exit_trigger == "PROFIT_GUARD":
            return (f"Profit Guard locked ${pnl:.0f} in {regime} regime. "
                    f"Guard threshold correctly calibrated for this volatility.")
        else:
            return (f"Clean win ${pnl:.0f}. MFE +{mfe_pips:.0f}p. "
                    f"Source {source} performing. Maintaining current parameters.")
    else:
        htf_aligned = (
            (action == "BUY" and m15_bias == "BULLISH") or
            (action == "SELL" and m15_bias == "BEARISH")
        )
        if mfe_pips > 15 and exit_trigger == "SL_HIT":
            return (f"MFE was +{mfe_pips:.0f}p but reversed to SL. "
                    f"Lost ${abs(pnl):.0f}. Trailing was too loose for {regime}. "
                    f"Consider tightening activation for {source}.")
        elif not htf_aligned:
            return (f"Counter-HTF entry: {action} vs M15={m15_bias}. "
                    f"Lost ${abs(pnl):.0f}. HTF gate should have blocked. "
                    f"Tightening HTF threshold for {source}.")
        elif tox_score > 0.4:
            return (f"Entered during elevated toxicity ({tox_score:.2f}). "
                    f"Lost ${abs(pnl):.0f}. Toxicity dampener insufficient. "
                    f"Consider raising dampener for {regime}.")
        elif regime == "SLOW_CHOP":
            return (f"Lost ${abs(pnl):.0f} in SLOW_CHOP. MAE -{mae_pips:.0f}p. "
                    f"Ranging market ate the SL. Reduce {source} weight in chop.")
        else:
            return (f"Standard loss ${abs(pnl):.0f}. MAE -{mae_pips:.0f}p in {regime}. "
                    f"No structural flaw detected. Accepting variance.")


async def _handle_position_closed(ticket):
    try:
        import MetaTrader5 as mt5
        close_price = 0.0
        pnl = 0.0
        tp1_hit = ticket in _tp1_hit
        deals = mt5.history_deals_get(position=ticket)
        if deals:
            close_deal = sorted(deals, key=lambda d: d.time)[-1]
            close_price = float(close_deal.price)
            pnl = sum(float(d.profit) for d in deals)
        db_manager.close_trade(ticket, close_price, pnl)
        logger.info("[TM] Closed ticket:%s @ %.5f PnL:%.2f", ticket, close_price, pnl)

        try:
            from quant.breakout_guard import register_consecutive_loss, register_consecutive_win
            trade_symbol = "XAUUSD"
            trade_action = ""
            with db_manager.get_connection() as conn:
                row = conn.execute(
                    "SELECT symbol, action FROM trades WHERE ticket=?", (ticket,)
                ).fetchone()
                if row:
                    trade_symbol = row[0]
                    trade_action = row[1] or ""
            if pnl > 0:
                register_consecutive_win(trade_symbol)
                try:
                    _, tick_regime = _measure_tick_intensity(trade_symbol)
                    from quant.breakout_guard import register_trend_win
                    register_trend_win(trade_symbol, trade_action, tick_regime)
                except Exception:
                    pass
            elif pnl < -1:
                register_consecutive_loss(trade_symbol)
                try:
                    from quant.breakout_guard import register_sl_hit
                    register_sl_hit(trade_action, close_price, trade_symbol)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("[TM] Consecutive tracking error: %s", e)

        if config.NOTIFY_ON_TRADE_CLOSE and abs(pnl) > 0:
            tag = "WIN" if pnl >= 0 else "LOSS"
            emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
            notify(
                "%s *Trade Closed*\n\nTicket: `%s`\nP&L: `$%+.2f`\n\n_%s executed cleanly._"
                % (emoji, ticket, pnl, tag)
            )
        if _close_callback is not None:
            db_open = db_manager.get_open_trades()
            db_trade = next((t for t in db_open if t.get("ticket") == ticket), None)
            pip_sz = 0.1
            try:
                if db_trade:
                    pip_sz = mt5_executor.get_pip_size(db_trade.get("symbol", "XAUUSD"))
            except Exception:
                pass
            _close_callback(ticket, close_price, pnl, tp1_hit, pip_sz)

        # v4.4: Forensic Black-Box snapshot
        try:
            _collect_forensics(ticket, close_price, pnl, trade_symbol, trade_action, tp1_hit)
        except Exception as fe:
            logger.debug("[TM] Forensic collection error: %s", fe)
    except Exception as e:
        logger.error("[TM] Close handler error ticket %s: %s", ticket, e)
    finally:
        cleanup_closed_ticket(ticket)


def cleanup_closed_ticket(ticket):
    _be_triggered.discard(ticket)
    _tp1_hit.discard(ticket)
    _tp2_hit.discard(ticket)
    _trailing.pop(ticket, None)
    _original_lots.pop(ticket, None)
    _pyramided.discard(ticket)
    try:
        from quant.flow_exit import cleanup_ticket
        cleanup_ticket(ticket)
    except Exception:
        pass