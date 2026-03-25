"""Source performance and system stats from omnisignal.db."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "omnisignal.db"


def fmt_num(x, decimals=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{decimals}f}"
    return str(x)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"Database: {DB_PATH}")
    print()

    # 1 & 2: Per-source stats (closed trades only) + Bayesian WR
    print("=" * 72)
    print("1-2. PER-SOURCE PERFORMANCE (status=CLOSED, win = pnl > 0)")
    print("     Bayesian WR% = (wins + 3) / (total + 6) * 100")
    print("=" * 72)
    rows = cur.execute(
        """
        SELECT
            COALESCE(s.source, '(no signal)') AS source_name,
            COUNT(*) AS total_trades,
            SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(COALESCE(t.pnl, 0)) AS total_pnl
        FROM trades t
        LEFT JOIN signals s ON t.signal_id = s.id
        WHERE t.status = 'CLOSED'
        GROUP BY s.source
        ORDER BY total_trades DESC, source_name
        """
    ).fetchall()

    hdr = f"{'Source':<42} {'N':>5} {'Wins':>5} {'PnL':>12} {'Raw WR%':>9} {'Bayes WR%':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        n = r["total_trades"]
        w = r["wins"]
        pnl = r["total_pnl"]
        raw_wr = (100.0 * w / n) if n else 0.0
        bayes_wr = (100.0 * (w + 3) / (n + 6)) if (n + 6) else 0.0
        print(
            f"{r['source_name']:<42} {n:5d} {w:5d} {pnl:12.2f} {raw_wr:9.2f} {bayes_wr:10.2f}"
        )
    print()

    # 3: AUTO_CATCD trades
    print("=" * 72)
    print("3. ALL TRADES LINKED TO SOURCE = AUTO_CATCD")
    print("=" * 72)
    auto_rows = cur.execute(
        """
        SELECT t.ticket, t.action, t.pnl, t.status, t.open_time, t.close_time
        FROM trades t
        INNER JOIN signals s ON t.signal_id = s.id
        WHERE s.source = 'AUTO_CATCD'
        ORDER BY COALESCE(t.close_time, t.open_time) DESC
        """
    ).fetchall()
    if not auto_rows:
        print("(no rows)")
    else:
        for r in auto_rows:
            print(
                f"  ticket={r['ticket']}  action={r['action']}  pnl={fmt_num(r['pnl'])}  "
                f"status={r['status']}  open={r['open_time']}  close={r['close_time']}"
            )
    print(f"  (count: {len(auto_rows)})")
    print()

    # 4: System-level (closed only)
    print("=" * 72)
    print("4. SYSTEM-LEVEL STATS (status=CLOSED)")
    print("=" * 72)
    sys_row = cur.execute(
        """
        SELECT
            COUNT(*) AS total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN pnl = 0 OR pnl IS NULL THEN 1 ELSE 0 END) AS breakeven_or_null,
            SUM(COALESCE(pnl, 0)) AS total_pnl
        FROM trades
        WHERE status = 'CLOSED'
        """
    ).fetchone()
    tot = sys_row["total_trades"] or 0
    wins = sys_row["wins"] or 0
    losses = sys_row["losses"] or 0
    be = sys_row["breakeven_or_null"] or 0
    tp = sys_row["total_pnl"] or 0.0
    owr = (100.0 * wins / tot) if tot else 0.0
    print(f"  Total closed trades: {tot}")
    print(f"  Wins (pnl > 0):      {wins}")
    print(f"  Losses (pnl < 0):    {losses}")
    print(f"  Breakeven / null:    {be}")
    print(f"  Total PnL:           {tp:.2f}")
    print(f"  Overall WR%:         {owr:.2f}")
    print()

    # Other statuses (context)
    other = cur.execute(
        "SELECT status, COUNT(*) FROM trades GROUP BY status ORDER BY status"
    ).fetchall()
    print("  Trade counts by status (all):")
    for st, cnt in other:
        print(f"    {st}: {cnt}")
    print()

    # 5: Recent 10 closed
    print("=" * 72)
    print("5. MOST RECENT 10 CLOSED TRADES (by close_time)")
    print("=" * 72)
    recent = cur.execute(
        """
        SELECT
            COALESCE(s.source, '(no signal)') AS source,
            t.action,
            t.pnl,
            t.ticket,
            t.close_time
        FROM trades t
        LEFT JOIN signals s ON t.signal_id = s.id
        WHERE t.status = 'CLOSED'
        ORDER BY t.close_time DESC
        LIMIT 10
        """
    ).fetchall()
    for r in recent:
        print(
            f"  close={r['close_time']}  ticket={r['ticket']}  source={r['source']}  "
            f"action={r['action']}  pnl={fmt_num(r['pnl'])}"
        )

    conn.close()


if __name__ == "__main__":
    main()
