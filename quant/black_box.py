"""
quant/black_box.py — OmniSignal Alpha v2.0
Pillar 12: The Black Box — Complete Decision Audit Trail

Records the full reasoning chain for every signal processed:
  Raw Message → AI Classification → AI Parse → Confluence Check →
  Risk Validation → Exposure Check → Execution Result

Stored in a separate SQLite database (black_box.db) so it never
competes with the main trading DB. Queryable for retrospective analysis.
"""

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any
import config
from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def _conn():
    conn = sqlite3.connect(config.BLACK_BOX_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"[BlackBox] DB error: {e}")
        raise
    finally:
        conn.close()


def init_black_box():
    """Create black_box.db schema. Call once at startup."""
    import os
    os.makedirs(os.path.dirname(config.BLACK_BOX_DB_PATH), exist_ok=True)

    create_sql = """
        CREATE TABLE IF NOT EXISTS decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TIMESTAMP NOT NULL,
            source          TEXT NOT NULL,

            -- Stage 1: Raw input
            raw_message     TEXT,

            -- Stage 2: AI output
            ai_category     TEXT,           -- SIGNAL/NOISE/CANCEL/UPDATE/CLOSE
            ai_symbol       TEXT,
            ai_action       TEXT,
            ai_entry        REAL,
            ai_sl           REAL,
            ai_tp1          REAL,
            ai_confidence   INTEGER,
            ai_reasoning    TEXT,

            -- Stage 3: Confluence
            confluence_score    INTEGER,
            confluence_max      INTEGER,
            confluence_passed   INTEGER,
            confluence_details  TEXT,       -- JSON: {ema:T, rsi:F, ...}
            atr_value           REAL,

            -- Stage 4: Risk check
            risk_check_passed   INTEGER,
            risk_reject_reason  TEXT,

            -- Stage 5: Sizing
            lot_size            REAL,
            risk_pct            REAL,
            sizing_method       TEXT,
            alpha_tier          TEXT,
            alpha_multiplier    REAL,

            -- Stage 6: Exposure
            exposure_passed     INTEGER,
            exposure_reason     TEXT,

            -- Stage 7: Execution
            final_decision      TEXT,       -- EXECUTED/REJECTED/MUTED/EXPIRED
            trade_ticket        INTEGER,
            execution_price     REAL,
            execution_latency_ms REAL,
            fill_pct            REAL        -- 1.0 = 100% fill
        );
    """

    # Columns beyond id/ts/source — migrate older DBs via ALTER TABLE
    expected_columns = [
        ("raw_message", "TEXT"),
        ("ai_category", "TEXT"),
        ("ai_symbol", "TEXT"),
        ("ai_action", "TEXT"),
        ("ai_entry", "REAL"),
        ("ai_sl", "REAL"),
        ("ai_tp1", "REAL"),
        ("ai_confidence", "INTEGER"),
        ("ai_reasoning", "TEXT"),
        ("confluence_score", "INTEGER"),
        ("confluence_max", "INTEGER"),
        ("confluence_passed", "INTEGER"),
        ("confluence_details", "TEXT"),
        ("atr_value", "REAL"),
        ("risk_check_passed", "INTEGER"),
        ("risk_reject_reason", "TEXT"),
        ("lot_size", "REAL"),
        ("risk_pct", "REAL"),
        ("sizing_method", "TEXT"),
        ("alpha_tier", "TEXT"),
        ("alpha_multiplier", "REAL"),
        ("exposure_passed", "INTEGER"),
        ("exposure_reason", "TEXT"),
        ("final_decision", "TEXT"),
        ("trade_ticket", "INTEGER"),
        ("execution_price", "REAL"),
        ("execution_latency_ms", "REAL"),
        ("fill_pct", "REAL"),
    ]

    index_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_bb_ts     ON decisions(ts DESC)",
        "CREATE INDEX IF NOT EXISTS idx_bb_source ON decisions(source)",
        "CREATE INDEX IF NOT EXISTS idx_bb_symbol ON decisions(ai_symbol)",
        "CREATE INDEX IF NOT EXISTS idx_bb_decision ON decisions(final_decision)",
    ]

    with _conn() as conn:
        conn.execute(create_sql)

        info = conn.execute("PRAGMA table_info(decisions)").fetchall()
        existing = {row[1].lower() for row in info}

        for col_name, col_type in expected_columns:
            if col_name.lower() not in existing:
                conn.execute(
                    f"ALTER TABLE decisions ADD COLUMN {col_name} {col_type}"
                )
                existing.add(col_name.lower())

        for sql in index_sqls:
            try:
                conn.execute(sql)
            except sqlite3.Error as e:
                logger.warning(f"[BlackBox] Index creation skipped: {e}")

    logger.info("[BlackBox] Initialized.")


class DecisionTrace:
    """
    Builder pattern for assembling a decision record across pipeline stages.
    Commit to DB at the end of the pipeline with .save().
    """
    def __init__(self, source: str, raw_message: str):
        self.data: Dict[str, Any] = {
            "ts": datetime.now().isoformat(),
            "source": source,
            "raw_message": raw_message[:2000] if raw_message else "",
        }

    # ── Stage setters ─────────────────────────────────────────────────────────

    def set_ai_gate(self, category: str):
        self.data["ai_category"] = category

    def set_ai_parse(self, signal):
        """Accept a ParsedSignal dataclass."""
        self.data.update({
            "ai_symbol":     signal.symbol if signal else None,
            "ai_action":     signal.action if signal else None,
            "ai_entry":      signal.entry_price if signal else None,
            "ai_sl":         signal.stop_loss if signal else None,
            "ai_tp1":        signal.tp1 if signal else None,
            "ai_confidence": signal.confidence if signal else 0,
            "ai_reasoning":  signal.ai_reasoning if signal else "",
        })

    def set_confluence(self, result):
        """Accept a ConfluenceResult dataclass."""
        self.data.update({
            "confluence_score":   result.score,
            "confluence_max":     result.max_score,
            "confluence_passed":  int(result.passed),
            "confluence_details": json.dumps(result.checks),
            "atr_value":          result.atr,
        })

    def set_risk(self, passed: bool, reason: str = ""):
        self.data.update({
            "risk_check_passed": int(passed),
            "risk_reject_reason": reason,
        })

    def set_sizing(self, sizing_result, alpha_tier: str = "UNRATED"):
        self.data.update({
            "lot_size":         sizing_result.lot_size,
            "risk_pct":         sizing_result.risk_pct,
            "sizing_method":    sizing_result.method,
            "alpha_tier":       alpha_tier,
            "alpha_multiplier": sizing_result.alpha_multiplier,
        })

    def set_exposure(self, passed: bool, reason: str = ""):
        self.data.update({
            "exposure_passed": int(passed),
            "exposure_reason": reason,
        })

    def set_execution(self, decision: str, ticket: int = None,
                      exec_price: float = None, latency_ms: float = None,
                      fill_pct: float = None):
        self.data.update({
            "final_decision":        decision,
            "trade_ticket":          ticket,
            "execution_price":       exec_price,
            "execution_latency_ms":  latency_ms,
            "fill_pct":              fill_pct,
        })

    def save(self) -> int:
        """Insert to black_box.db and return rowid."""
        if not config.BLACK_BOX_ENABLED:
            return -1
        try:
            cols   = ", ".join(self.data.keys())
            params = ", ".join(["?"] * len(self.data))
            vals   = list(self.data.values())
            with _conn() as conn:
                cur = conn.execute(
                    f"INSERT INTO decisions ({cols}) VALUES ({params})", vals
                )
                return cur.lastrowid
        except Exception as e:
            logger.error(f"[BlackBox] Save failed: {e}")
            return -1


# ── Query helpers for dashboard & self-correction ────────────────────────────

def query_decisions(
    limit: int = 100,
    source: str = None,
    symbol: str = None,
    decision: str = None,
) -> list:
    """Fetch decision records for dashboard or self-correction analysis."""
    if not config.BLACK_BOX_ENABLED:
        return []
    try:
        clauses, params = [], []
        if source:
            clauses.append("source = ?");   params.append(source)
        if symbol:
            clauses.append("ai_symbol = ?"); params.append(symbol.upper())
        if decision:
            clauses.append("final_decision = ?"); params.append(decision)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with _conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM decisions {where} ORDER BY ts DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[BlackBox] Query failed: {e}")
        return []


def get_rejection_breakdown(limit: int = 500) -> Dict[str, int]:
    """Returns a dict of reject_reason → count for the last N decisions."""
    try:
        with _conn() as conn:
            rows = conn.execute("""
                SELECT COALESCE(risk_reject_reason, exposure_reason, 'Unknown') AS reason,
                       COUNT(*) AS cnt
                FROM decisions
                WHERE final_decision = 'REJECTED'
                ORDER BY ts DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return {r["reason"]: r["cnt"] for r in rows}
    except Exception:
        return {}


def get_noise_rate(hours: int = 24) -> float:
    """Fraction of messages classified as NOISE in the last N hours."""
    try:
        with _conn() as conn:
            total  = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE ts > datetime('now', ? || ' hours')",
                (f"-{hours}",)
            ).fetchone()[0]
            noise  = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE ai_category='NOISE' "
                "AND ts > datetime('now', ? || ' hours')", (f"-{hours}",)
            ).fetchone()[0]
        return noise / max(total, 1)
    except Exception:
        return 0.0
