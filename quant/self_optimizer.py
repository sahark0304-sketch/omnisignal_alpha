"""
quant/self_optimizer.py - OmniSignal v8.0 Self-Optimizer

Runs nightly at 03:00 UTC. Analyzes last 7 days of trades.
Outputs parameter adjustment recommendations and a Telegram report.

Phase 1: Reporting only (no auto-apply).
Phase 2 (future): Auto-apply safe adjustments after validation period.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = "data/omnisignal.db"
ADJUSTMENTS_FILE = "data/daily_adjustments.json"
MIN_SAMPLE = 3


class SelfOptimizer:

    def run_nightly(self):
        """Analyze last 7 days and output parameter adjustments."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")

        adj = {"generated_at": datetime.now().isoformat(),
               "period": "7 days ending %s" % datetime.now().date()}

        # ── 1. SESSION HOUR ANALYSIS ────────────────────────────────────
        rows = conn.execute("""
            SELECT CAST(strftime('%%H', open_time) AS INTEGER) AS hour,
                   COUNT(*)                                     AS trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)    AS wins,
                   SUM(pnl)                                     AS total_pnl
            FROM trades
            WHERE status = 'CLOSED' AND open_time > ?
            GROUP BY hour ORDER BY hour
        """, (cutoff,)).fetchall()

        profitable_hours, losing_hours, hour_detail = [], [], []
        for r in rows:
            h, n, w, pnl = r["hour"], r["trades"], r["wins"], r["total_pnl"]
            if n < MIN_SAMPLE:
                continue
            wr = w / n
            hour_detail.append({"hour": h, "trades": n, "wr": round(wr, 2),
                                "pnl": round(pnl, 2)})
            if wr >= 0.45 and pnl > 0:
                profitable_hours.append(h)
            elif wr < 0.35 or pnl < -20:
                losing_hours.append(h)

        adj["profitable_hours"] = sorted(profitable_hours)
        adj["losing_hours"]     = sorted(losing_hours)
        adj["hour_detail"]      = hour_detail

        # ── 2. SOURCE PERFORMANCE (join trades -> signals) ──────────────
        src_rows = conn.execute("""
            SELECT s.source,
                   COUNT(*)                                     AS trades,
                   SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END)  AS wins,
                   SUM(t.pnl)                                   AS total_pnl,
                   AVG(t.pnl)                                   AS avg_pnl
            FROM trades t
            JOIN signals s ON t.signal_id = s.id
            WHERE t.status = 'CLOSED' AND t.open_time > ?
            GROUP BY s.source
            ORDER BY total_pnl DESC
        """, (cutoff,)).fetchall()

        top_sources, worst_sources, all_sources = [], [], []
        for r in src_rows:
            n = r["trades"]
            if n < MIN_SAMPLE:
                continue
            wr = r["wins"] / n
            entry = {"source": r["source"], "trades": n,
                     "wr": round(wr, 2), "pnl": round(r["total_pnl"], 2),
                     "avg_pnl": round(r["avg_pnl"], 2)}
            all_sources.append(entry)
            if wr >= 0.55 and r["total_pnl"] > 0:
                top_sources.append(entry)
            elif wr < 0.30 and r["total_pnl"] < -20:
                worst_sources.append(entry)

        adj["top_sources"]   = top_sources
        adj["worst_sources"] = worst_sources
        adj["all_sources"]   = all_sources

        # ── 3. DIRECTION BIAS ──────────────────────────────────────────
        dir_rows = conn.execute("""
            SELECT action,
                   COUNT(*)                                     AS trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)    AS wins,
                   SUM(pnl)                                     AS total_pnl
            FROM trades
            WHERE status = 'CLOSED' AND open_time > ?
            GROUP BY action
        """, (cutoff,)).fetchall()

        for d in dir_rows:
            if d["trades"] >= 5:
                wr = d["wins"] / d["trades"]
                adj["%s_wr_7d" % d["action"]]  = round(wr, 2)
                adj["%s_pnl_7d" % d["action"]] = round(d["total_pnl"], 2)
                adj["%s_trades_7d" % d["action"]] = d["trades"]

        # ── 4. LOT SIZE TIER ANALYSIS ──────────────────────────────────
        lot_rows = conn.execute("""
            SELECT CASE
                     WHEN lot_size <= 0.01 THEN '0.01'
                     WHEN lot_size <= 0.02 THEN '0.02'
                     WHEN lot_size <= 0.03 THEN '0.03'
                     ELSE '0.03+'
                   END                                          AS lot_tier,
                   COUNT(*)                                     AS trades,
                   SUM(pnl)                                     AS total_pnl,
                   AVG(pnl)                                     AS avg_pnl
            FROM trades
            WHERE status = 'CLOSED' AND open_time > ?
            GROUP BY lot_tier
        """, (cutoff,)).fetchall()

        lot_detail = []
        best_lot_tier, best_lot_pnl = None, -99999
        for r in lot_rows:
            lot_detail.append({"tier": r["lot_tier"], "trades": r["trades"],
                               "pnl": round(r["total_pnl"], 2),
                               "avg": round(r["avg_pnl"], 2)})
            if r["trades"] >= MIN_SAMPLE and r["total_pnl"] > best_lot_pnl:
                best_lot_pnl = r["total_pnl"]
                best_lot_tier = r["lot_tier"]
        adj["lot_detail"]    = lot_detail
        adj["best_lot_tier"] = best_lot_tier

        # ── 5. MFE ANALYSIS (losers that were profitable) ─────────────
        losers = conn.execute("""
            SELECT COUNT(*) AS n, SUM(pnl) AS loss
            FROM trades
            WHERE status = 'CLOSED' AND pnl < 0 AND open_time > ?
        """, (cutoff,)).fetchone()
        adj["total_losers_7d"] = losers["n"]
        adj["total_loss_7d"]   = round(losers["loss"] or 0, 2)

        # ── 6. WHAT-IF LEDGER ANALYSIS ─────────────────────────────────
        try:
            wif_rows = conn.execute("""
                SELECT reject_reason,
                       COUNT(*)                                          AS rejected,
                       SUM(CASE WHEN would_hit_tp = 1 THEN 1 ELSE 0 END) AS would_win,
                       SUM(CASE WHEN would_hit_sl = 1 THEN 1 ELSE 0 END) AS would_lose,
                       SUM(virtual_pnl)                                   AS virtual_pnl
                FROM what_if_ledger
                WHERE rejected_at > ? AND would_hit_tp IS NOT NULL
                GROUP BY reject_reason
                ORDER BY virtual_pnl DESC
            """, (cutoff,)).fetchall()

            costly_filters, valuable_filters = [], []
            for w in wif_rows:
                if w["rejected"] < MIN_SAMPLE:
                    continue
                vpnl = w["virtual_pnl"] or 0
                if vpnl > 0:
                    costly_filters.append({
                        "filter": w["reject_reason"], "rejected": w["rejected"],
                        "would_win": w["would_win"], "would_lose": w["would_lose"],
                        "missed_pnl": round(vpnl, 2),
                    })
                else:
                    valuable_filters.append({
                        "filter": w["reject_reason"],
                        "saved_pnl": round(abs(vpnl), 2),
                    })
            adj["costly_filters"]   = costly_filters
            adj["valuable_filters"] = valuable_filters
        except Exception:
            adj["costly_filters"]   = []
            adj["valuable_filters"] = []

        # ── 7. CONFIDENCE LEVEL PERFORMANCE ────────────────────────────
        conf_rows = conn.execute("""
            SELECT s.ai_confidence AS conf,
                   COUNT(*)                                     AS trades,
                   SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END)  AS wins,
                   SUM(t.pnl)                                   AS total_pnl
            FROM trades t
            JOIN signals s ON t.signal_id = s.id
            WHERE t.status = 'CLOSED' AND t.open_time > ?
              AND s.ai_confidence IS NOT NULL
            GROUP BY s.ai_confidence
            ORDER BY s.ai_confidence
        """, (cutoff,)).fetchall()

        conf_detail = []
        for c in conf_rows:
            if c["trades"] >= 2:
                conf_detail.append({
                    "confidence": c["conf"], "trades": c["trades"],
                    "wr": round(c["wins"] / c["trades"], 2),
                    "pnl": round(c["total_pnl"], 2),
                })
        adj["confidence_detail"] = conf_detail

        # ── 8. STREAK ANALYSIS (consecutive W/L) ──────────────────────
        recent = conn.execute("""
            SELECT pnl FROM trades
            WHERE status = 'CLOSED' AND open_time > ?
            ORDER BY close_time DESC LIMIT 20
        """, (cutoff,)).fetchall()

        streak = 0
        streak_dir = None
        for r in recent:
            d = "W" if r["pnl"] > 0 else "L"
            if streak_dir is None:
                streak_dir = d
                streak = 1
            elif d == streak_dir:
                streak += 1
            else:
                break
        adj["current_streak"] = "%s%d" % (streak_dir or "?", streak)

        # ── 9. DAILY PnL TREND ─────────────────────────────────────────
        daily = conn.execute("""
            SELECT DATE(close_time) AS day,
                   COUNT(*)         AS trades,
                   SUM(pnl)         AS pnl
            FROM trades
            WHERE status = 'CLOSED' AND close_time > ?
            GROUP BY day ORDER BY day
        """, (cutoff,)).fetchall()
        adj["daily_pnl"] = [{"date": d["day"], "trades": d["trades"],
                             "pnl": round(d["pnl"], 2)} for d in daily]

        conn.close()

        # ── SAVE ──────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(ADJUSTMENTS_FILE), exist_ok=True)
        with open(ADJUSTMENTS_FILE, "w") as f:
            json.dump(adj, f, indent=2)

        # ── TELEGRAM REPORT ───────────────────────────────────────────
        try:
            from utils.notifier import notify
            lines = [
                "NIGHTLY OPTIMIZER REPORT",
                "Period: %s" % adj["period"],
                "",
                "DIRECTION: BUY WR=%s PnL=$%s | SELL WR=%s PnL=$%s" % (
                    adj.get("BUY_wr_7d", "?"), adj.get("BUY_pnl_7d", "?"),
                    adj.get("SELL_wr_7d", "?"), adj.get("SELL_pnl_7d", "?")),
                "Streak: %s" % adj["current_streak"],
                "Losers: %d ($%.2f)" % (adj["total_losers_7d"], adj["total_loss_7d"]),
                "Best lot tier: %s" % adj["best_lot_tier"],
                "",
            ]
            if adj["losing_hours"]:
                lines.append("LOSING HOURS: %s" % adj["losing_hours"])
            if adj["profitable_hours"]:
                lines.append("PROFITABLE HOURS: %s" % adj["profitable_hours"])
            if top_sources:
                lines.append("")
                lines.append("TOP SOURCES:")
                for s in top_sources[:5]:
                    lines.append("  %s: %d trades, WR=%d%%, $%+.2f" % (
                        s["source"][:25], s["trades"], s["wr"]*100, s["pnl"]))
            if worst_sources:
                lines.append("WORST SOURCES:")
                for s in worst_sources[:3]:
                    lines.append("  %s: WR=%d%%, $%+.2f" % (
                        s["source"][:25], s["wr"]*100, s["pnl"]))
            if adj.get("costly_filters"):
                lines.append("")
                lines.append("FILTERS COSTING MONEY (what-if):")
                for cf in adj["costly_filters"][:3]:
                    lines.append("  %s: missed $%.2f (%d rejected)" % (
                        cf["filter"][:25], cf["missed_pnl"], cf["rejected"]))
            if conf_detail:
                lines.append("")
                lines.append("CONFIDENCE PERFORMANCE:")
                for c in conf_detail:
                    lines.append("  Conf %s: %d trades, WR=%d%%, $%+.2f" % (
                        c["confidence"], c["trades"], c["wr"]*100, c["pnl"]))
            if adj["daily_pnl"]:
                lines.append("")
                lines.append("DAILY PnL:")
                for d in adj["daily_pnl"][-7:]:
                    lines.append("  %s: %d trades, $%+.2f" % (
                        d["date"], d["trades"], d["pnl"]))

            notify("\n".join(lines))
        except Exception as _ne:
            logger.debug("[Optimizer] Notify error: %s", _ne)

        logger.info(
            "[Optimizer] Nightly complete: %d profitable hours, %d losing, "
            "%d top sources, %d worst, streak=%s",
            len(profitable_hours), len(losing_hours),
            len(top_sources), len(worst_sources), adj["current_streak"],
        )
        return adj

    def load_adjustments(self):
        """Load latest adjustments for use during trading."""
        if os.path.exists(ADJUSTMENTS_FILE):
            try:
                with open(ADJUSTMENTS_FILE) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


self_optimizer = SelfOptimizer()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    result = self_optimizer.run_nightly()
    print(json.dumps(result, indent=2))
