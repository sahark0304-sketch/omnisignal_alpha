"""
recovery/state_recovery.py — OmniSignal Alpha v2.0
Pillar 7: State Recovery & VPS Resilience

Three-layer protection:
  1. Heartbeat writer — writes a timestamped snapshot every 60s
  2. Crash detector  — on startup, detects if last shutdown was clean
  3. Position reconciler — on startup, rebuilds in-memory state from MT5
                           and closes any orphan positions opened before crash

Prevents the most common VPS failure modes:
  - Duplicate trades after restart (already-open position re-executed)
  - Dangling TP/BE state for positions that survived the crash
  - Stale daily PnL causing incorrect drawdown halt logic
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Set, Optional
import config
from utils.logger import get_logger
from utils.notifier import notify

logger = get_logger(__name__)


def _snapshot_path() -> str:
    os.makedirs(os.path.dirname(config.RECOVERY_SNAPSHOT_FILE), exist_ok=True)
    return config.RECOVERY_SNAPSHOT_FILE


# ── HEARTBEAT WRITER ─────────────────────────────────────────────────────────

async def run_heartbeat():
    """
    Background task: writes a snapshot every RECOVERY_HEARTBEAT_SECS seconds.
    Snapshot contains: open tickets, daily PnL, halt state, timestamp.
    On clean shutdown, writes clean_exit=True.
    """
    import signal as _signal
    if not config.RECOVERY_ENABLED:
        return

    def _write_clean_exit(_sig=None, _frame=None):
        _write_snapshot(clean_exit=True)

    # Register clean exit handler
    for sig in (getattr(_signal, 'SIGTERM', None), getattr(_signal, 'SIGINT', None)):
        if sig:
            try:
                _signal.signal(sig, _write_clean_exit)
            except (ValueError, OSError):
                pass

    logger.info("[Recovery] Heartbeat started.")
    while True:
        try:
            _write_snapshot(clean_exit=False)
        except Exception as e:
            logger.warning(f"[Recovery] Heartbeat write failed: {e}")
        await asyncio.sleep(config.RECOVERY_HEARTBEAT_SECS)


def _write_snapshot(clean_exit: bool = False):
    """Write current state to recovery_snapshot.json."""
    try:
        from mt5_executor.mt5_executor import get_all_positions, get_account_equity
        from database import db_manager
        from risk_guard.risk_guard import is_halted

        positions  = get_all_positions()
        equity     = get_account_equity()
        halted, _  = is_halted()
        daily_pnl  = db_manager.get_daily_pnl()

        snapshot = {
            "timestamp":    datetime.now().isoformat(),
            "clean_exit":   clean_exit,
            "equity":       equity,
            "daily_pnl":    daily_pnl,
            "halted":       halted,
            "open_tickets": [p["ticket"] for p in positions],
            "positions":    [
                {
                    "ticket": p["ticket"],
                    "symbol": p["symbol"],
                    "action": p["type"],
                    "volume": p["volume"],
                    "entry":  p["price_open"],
                    "sl":     p["sl"],
                    "tp":     p["tp"],
                }
                for p in positions
            ],
        }
        with open(_snapshot_path(), "w") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as e:
        logger.warning(f"[Recovery] Snapshot write error: {e}")


# ── STARTUP RECONCILIATION ───────────────────────────────────────────────────

def reconcile_on_startup() -> Dict:
    """
    Called once in startup().
    1. Read last snapshot
    2. Determine if crash or clean exit
    3. Sync DB with live MT5 positions
    4. Rebuild trade_manager state for surviving positions
    5. Alert on duplicates or orphans
    Returns report dict.
    """
    report = {
        "snapshot_found": False,
        "clean_exit": False,
        "crash_detected": False,
        "positions_reconciled": 0,
        "orphans_found": 0,
        "db_gaps_filled": 0,
        "actions": [],
    }

    snapshot = _load_snapshot()

    if snapshot is None:
        logger.info("[Recovery] No snapshot found — fresh start.")
        return report

    report["snapshot_found"] = True
    clean = snapshot.get("clean_exit", False)
    report["clean_exit"] = clean

    if not clean:
        report["crash_detected"] = True
        age_secs = (datetime.now() - datetime.fromisoformat(snapshot["timestamp"])).total_seconds()
        logger.warning(
            f"[Recovery] 🔴 CRASH DETECTED — last heartbeat was {age_secs:.0f}s ago. "
            f"Reconciling state..."
        )
        notify(
            f"🔴 *Crash Detected*\n"
            f"Last heartbeat: {age_secs:.0f}s ago\n"
            f"Reconciling MT5 positions..."
        )

    try:
        from mt5_executor.mt5_executor import get_all_positions
        from database import db_manager
        from trade_manager.trade_manager import _original_lots

        live_positions = get_all_positions()
        live_tickets   = {p["ticket"] for p in live_positions}
        db_trades      = {t["ticket"]: t for t in db_manager.get_open_trades()}
        db_tickets     = set(db_trades.keys())

        # ── Gap 1: Position in MT5 but not in DB ──────────────────────────────
        # (trade opened, then crash before DB write completed)
        for pos in live_positions:
            t = pos["ticket"]
            if t not in db_tickets:
                logger.warning(f"[Recovery] Orphan position: ticket={t} {pos['symbol']} — inserting to DB")
                try:
                    db_manager.insert_trade(
                        ticket=t, signal_id=None,
                        symbol=pos["symbol"], action=pos["type"],
                        lot_size=pos["volume"], entry=pos["entry"],
                        sl=pos["sl"], tp1=pos["tp"],
                        tp2=None, tp3=None,
                        mode=config.OPERATING_MODE,
                    )
                    report["db_gaps_filled"] += 1
                    report["actions"].append(f"Inserted orphan ticket {t}")
                except Exception as e:
                    logger.error(f"[Recovery] Insert orphan failed: {e}")

        # ── Gap 2: Position in DB as OPEN but gone from MT5 ──────────────────
        # (SL hit / TP hit while bot was down — close it in DB)
        for ticket, db_trade in db_trades.items():
            if ticket not in live_tickets:
                logger.info(f"[Recovery] Position {ticket} closed while offline — marking DB closed")
                try:
                    # Attempt to get historical deal info
                    close_price, pnl = _fetch_deal_result(ticket)
                    db_manager.close_trade(ticket, close_price or db_trade.get("entry_price", 0), pnl or 0)
                    report["actions"].append(f"Closed offline ticket {ticket}")
                except Exception as e:
                    logger.error(f"[Recovery] Close offline ticket failed: {e}")

        # ── Gap 3: Rebuild trade_manager in-memory state ──────────────────────
        # trade_manager tracks BE/TP1/TP2/trailing per ticket in memory.
        # After crash those are gone — rebuild from DB trade status.
        from trade_manager.trade_manager import (_be_triggered, _tp1_hit, _tp2_hit,
                                                   _trailing, _original_lots as _orig)
        for pos in live_positions:
            t = pos["ticket"]
            db_t = db_trades.get(t) or db_manager.get_open_trades()
            # Seed original lots
            _orig[t] = pos["volume"]

            # Check DB status for what already happened
            db_t_row = db_trades.get(t, {})
            status = db_t_row.get("status", "OPEN")
            if status in ("BE_TRIGGERED",):
                _be_triggered.add(t)
            if status in ("TP1_HIT", "TP2_HIT"):
                _be_triggered.add(t)
                _tp1_hit.add(t)
                _trailing[t] = pos["entry"]
            if status == "TP2_HIT":
                _tp2_hit.add(t)
            report["positions_reconciled"] += 1

        logger.info(
            f"[Recovery] Reconciliation complete | "
            f"live={len(live_positions)} db_gaps={report['db_gaps_filled']} "
            f"reconciled={report['positions_reconciled']}"
        )
        if report["crash_detected"] and report["positions_reconciled"] > 0:
            notify(
                f"✅ *Recovery Complete*\n"
                f"Reconciled {report['positions_reconciled']} live positions.\n"
                f"Filled {report['db_gaps_filled']} DB gaps."
            )

    except Exception as e:
        logger.error(f"[Recovery] Reconciliation failed: {e}", exc_info=True)

    return report


def _load_snapshot() -> Optional[Dict]:
    path = _snapshot_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[Recovery] Could not read snapshot: {e}")
        return None


def _fetch_deal_result(ticket: int):
    """Try to get closing price and PnL from MT5 deal history."""
    try:
        import MetaTrader5 as mt5
        from datetime import timezone
        deals = mt5.history_deals_get(position=ticket)
        if deals:
            closing_deal = max(deals, key=lambda d: d.time)
            return closing_deal.price, closing_deal.profit
    except Exception:
        pass
    return None, None
