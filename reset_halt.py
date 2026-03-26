"""
reset_halt.py - Force-reset the velocity breaker halt state.

Usage: python reset_halt.py

Clears the HALT flag and resets opening_equity to current MT5 equity,
effectively zeroing the daily DD calculation and resuming trading.
"""
import sqlite3
import sys
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "omnisignal.db")


def reset():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Clear halt flags
    c.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('halt', '0')")
    c.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('halt_reason', '')")
    print("[OK] Halt flags cleared.")

    # 2. Reset opening equity to current MT5 equity
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            print("[WARN] MT5 not available. Halt flags cleared but equity not updated.")
        else:
            acct = mt5.account_info()
            if acct:
                equity = acct.equity
                c.execute(
                    "INSERT OR REPLACE INTO system_state (key, value) VALUES ('opening_equity', ?)",
                    (str(equity),),
                )
                print("[OK] Opening equity reset to $%.2f" % equity)
            else:
                print("[WARN] Could not get MT5 account info.")
            mt5.shutdown()
    except ImportError:
        print("[WARN] MetaTrader5 not installed. Halt flags cleared but equity not updated.")

    # 3. Log the manual reset in audit_log
    c.execute(
        "INSERT INTO audit_log (event_type, details, ts) VALUES (?, ?, ?)",
        ("MANUAL_HALT_RESET", json.dumps({"reason": "CTO force-reset via reset_halt.py"}),
         datetime.now().isoformat()),
    )

    conn.commit()

    # 4. Verify
    c.execute("SELECT key, value FROM system_state WHERE key IN ('halt', 'halt_reason', 'opening_equity')")
    for row in c.fetchall():
        print("  %s = %s" % (row[0], row[1]))

    conn.close()
    print("")
    print("[DONE] Halt state cleared. Restart the bot to resume trading.")


if __name__ == "__main__":
    reset()
