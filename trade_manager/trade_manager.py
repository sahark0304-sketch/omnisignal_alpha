"""
trade_manager/trade_manager.py — OmniSignal Alpha v3.0
Active Position Management Loop.

v3.0 CHANGES:
  + set_close_callback() — wires main.py's backfill_trade_label() to be called
    every time a position closes, enabling ML label backfill (Pillar 4)
  + cleanup_closed_ticket() now calls _close_callback with trade outcome
  + All original BE/TP/trailing logic preserved from v1.1 (no regressions)
"""

import asyncio
from typing import Dict, Set, Optional, Callable
import config
from mt5_executor import mt5_executor
from database import db_manager
from utils.logger import get_logger, get_trade_logger
from utils.notifier import notify

logger = get_logger(__name__)

# Per-ticket state tracking (unchanged from v1.1)
_be_triggered: Set[int]          = set()
_tp1_hit: Set[int]               = set()
_tp2_hit: Set[int]               = set()
_trailing: Dict[int, float]      = {}
_original_lots: Dict[int, float] = {}

# v3.0 NEW: callback for ML label backfilling
# Signature: (ticket, close_price, pnl, tp1_hit, pip_size) → None
_close_callback: Optional[Callable] = None


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


async def _tick():  # test
    positions    = mt5_executor.get_all_positions()
    live_tickets = {p["ticket"] for p in positions}

    # v4.4-audit: Friday Weekend Flattening (gap risk protection)
    try:
        from datetime import datetime, timezone
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

    # Detect positions that closed since last tick
    for ticket in list(_original_lots.keys()):
        if ticket not in live_tickets:
            logger.debug(f"[TradeManager] Ticket {ticket} no longer live — closing out.")
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

        # ── BREAK-EVEN ─────────────────────────────────────────────────────
        if ticket not in _be_triggered and tp1 and entry and current_sl:
            tp1_dist = abs(tp1 - entry)
            if tp1_dist > 0:
                be_trigger = (
                    entry + (tp1_dist * config.BE_TRIGGER_PCT) if action == "BUY"
                    else entry - (tp1_dist * config.BE_TRIGGER_PCT)
                )
                be_reached = (
                    (action == "BUY"  and current >= be_trigger) or
                    (action == "SELL" and current <= be_trigger)
                )
                if be_reached and abs(current_sl - entry) > 0.00001:
                    success = mt5_executor.modify_sl(ticket, entry)
                    if success:
                        _be_triggered.add(ticket)
                        db_manager.update_trade_status(ticket, "BE_TRIGGERED")
                        db_manager.log_audit("BE_TRIGGERED", {"ticket": ticket, "entry": entry})
                        notify(f"🔒 *BE Set* | {symbol} {action} Ticket:`{ticket}` SL→`{entry}`")

        # ── TP1 PARTIAL CLOSE ───────────────────────────────────────────────
        if ticket not in _tp1_hit and tp1:
            tp1_reached = (
                (action == "BUY"  and current >= tp1) or
                (action == "SELL" and current <= tp1)
            )
            if tp1_reached:
                close_lots = round(original_lot * config.TP1_CLOSE_PCT, 2)
                if close_lots >= 0.01:
                    success = mt5_executor.close_partial(ticket, close_lots)
                    if success:
                        _tp1_hit.add(ticket)
                        db_manager.update_trade_status(ticket, "TP1_HIT")
                        db_manager.log_audit("TP1_HIT", {"ticket": ticket, "lots": close_lots})
                        notify(f"🎯 *TP1!* | {symbol} {action} Ticket:`{ticket}` @ `{current}`")
                        _trailing[ticket] = entry

        # ── TP2 PARTIAL CLOSE ───────────────────────────────────────────────
        if ticket in _tp1_hit and ticket not in _tp2_hit and tp2:
            tp2_reached = (
                (action == "BUY"  and current >= tp2) or
                (action == "SELL" and current <= tp2)
            )
            if tp2_reached:
                close_lots = min(
                    round(original_lot * config.TP2_CLOSE_PCT, 2),
                    pos["volume"]
                )
                if close_lots >= 0.01:
                    success = mt5_executor.close_partial(ticket, close_lots)
                    if success:
                        _tp2_hit.add(ticket)
                        db_manager.log_audit("TP2_HIT", {"ticket": ticket, "lots": close_lots})
                        notify(f"🎯🎯 *TP2!* | {symbol} {action} Ticket:`{ticket}` @ `{current}`")

        # ── TRAILING STOP ───────────────────────────────────────────────────
        if ticket in _tp1_hit and ticket in _trailing:
            pip_size         = mt5_executor.get_pip_size(symbol)
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


async def _handle_position_closed(ticket: int):
    """
    Called when a position disappears from MT5.
    Fetches final deal data, updates DB, and triggers ML label backfill.
    """
    try:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        # Fetch close data from MT5 deal history
        close_price = 0.0
        pnl         = 0.0
        tp1_hit     = ticket in _tp1_hit  # Was TP1 reached before close?

        deals = mt5.history_deals_get(position=ticket)
        if deals:
            # The last deal in the sequence is the close
            close_deal = sorted(deals, key=lambda d: d.time)[-1]
            close_price = float(close_deal.price)
            pnl         = sum(float(d.profit) for d in deals)

        db_manager.close_trade(ticket, close_price, pnl)
        logger.info(f"[TradeManager] Closed ticket:{ticket} @ {close_price:.5f} PnL:${pnl:.2f}")
        get_trade_logger().info("CLOSE | ticket=%d | pnl=%.2f close_price=%.5f", ticket, pnl, close_price)

        if config.NOTIFY_ON_TRADE_CLOSE and abs(pnl) > 0:
            emoji = "✅" if pnl >= 0 else "❌"
            notify(f"{emoji} *Trade Closed*\nTicket:`{ticket}` PnL:`${pnl:+.2f}`")

        # Trigger ML label backfill (v3.0 new)
        if _close_callback is not None:
            db_open = db_manager.get_open_trades()
            db_trade = next((t for t in db_open if t.get("ticket") == ticket), None)
            pip_size = 0.1  # XAUUSD default; try to get from position symbol
            try:
                # Try to reconstruct symbol from DB
                if db_trade:
                    pip_size = mt5_executor.get_pip_size(db_trade.get("symbol", "XAUUSD"))
            except Exception:
                pass
            _close_callback(ticket, close_price, pnl, tp1_hit, pip_size)

    except Exception as e:
        logger.error(f"[TradeManager] Close handler error for ticket {ticket}: {e}")
    finally:
        cleanup_closed_ticket(ticket)


def cleanup_closed_ticket(ticket: int):
    """Free all per-ticket in-memory state."""
    _be_triggered.discard(ticket)
    _tp1_hit.discard(ticket)
    _tp2_hit.discard(ticket)
    _trailing.pop(ticket, None)
    _original_lots.pop(ticket, None)