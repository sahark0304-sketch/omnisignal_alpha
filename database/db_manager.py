"""
database/db_manager.py — OmniSignal Alpha v2.0
Extended schema supporting all 14 pillars.

New tables vs v1.1:
  - system_state          (v1.1: cross-process halt state)
  - seen_messages         (v1.1: dedup fingerprints)
  - source_profiles       (v2: alpha ranker cache)
  - latency_log           (v2: Pillar 13 latency history)
  - equity_snapshots      (v2: fine-grained equity curve for dashboard)
"""

import sqlite3
import json
import hashlib
from datetime import datetime, date
from contextlib import contextmanager
from typing import Optional, Dict, List
import config
from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection():
    import os
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent reads (dashboard process)
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"[DB] Rolled back: {e}")
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS signals (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source              TEXT NOT NULL,
            raw_text            TEXT,
            parsed_json         TEXT,
            ai_confidence       INTEGER,
            is_high_conviction  INTEGER DEFAULT 0,
            status              TEXT DEFAULT 'PENDING',
            reject_reason       TEXT,
            trade_ticket        INTEGER,
            session             TEXT,
            hurst_at_signal     REAL
        );

        CREATE TABLE IF NOT EXISTS trades (
            ticket          INTEGER PRIMARY KEY,
            signal_id       INTEGER,
            symbol          TEXT NOT NULL,
            action          TEXT NOT NULL,
            lot_size        REAL,
            entry_price     REAL,
            sl_price        REAL,
            tp1_price       REAL,
            tp2_price       REAL,
            tp3_price       REAL,
            open_time       TIMESTAMP,
            close_time      TIMESTAMP,
            close_price     REAL,
            pnl             REAL,
            status          TEXT DEFAULT 'OPEN',
            mode            TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        );

        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snap_date       DATE UNIQUE,
            opening_equity  REAL,
            closing_equity  REAL,
            daily_pnl       REAL,
            trade_count     INTEGER,
            win_count       INTEGER,
            loss_count      INTEGER,
            halt_triggered  INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type      TEXT,
            details         TEXT
        );

        CREATE TABLE IF NOT EXISTS system_state (
            key             TEXT PRIMARY KEY,
            value           TEXT,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS seen_messages (
            fingerprint     TEXT PRIMARY KEY,
            source          TEXT,
            first_seen      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            count           INTEGER DEFAULT 1
        );

        -- Pillar 13: latency history
        CREATE TABLE IF NOT EXISTS latency_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            avg_ms      REAL,
            jitter_ms   REAL,
            mode        TEXT
        );

        -- Pillar 10: fine-grained equity snapshots for curve
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            equity      REAL,
            balance     REAL,
            open_trades INTEGER,
            daily_pnl   REAL
        );

        CREATE INDEX IF NOT EXISTS idx_signals_received ON signals(received_at DESC);
        CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_close     ON trades(close_time DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_ts         ON audit_log(ts DESC);
        CREATE INDEX IF NOT EXISTS idx_equity_ts        ON equity_snapshots(ts DESC);

        CREATE TABLE IF NOT EXISTS trade_forensics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket              INTEGER NOT NULL,
            signal_id           INTEGER,
            ts                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            -- Entry state snapshot
            m15_bias            TEXT,
            m5_bias             TEXT,
            tick_regime         TEXT,
            toxicity_score      REAL,
            consensus_score     REAL,
            ai_confidence       INTEGER,
            active_scanners     TEXT,
            dxy_z_score         REAL,
            entry_spread_pips   REAL,
            -- Execution quality
            expected_entry      REAL,
            actual_entry        REAL,
            slippage_pips       REAL,
            -- Trade journey
            mfe_pips            REAL,
            mae_pips            REAL,
            mfe_dollars         REAL,
            mae_dollars         REAL,
            duration_secs       INTEGER,
            -- Exit
            exit_trigger        TEXT,
            exit_price          REAL,
            pnl                 REAL,
            -- AI learning
            lesson_learned      TEXT,
            weight_adjustments  TEXT,
            source              TEXT,
            action              TEXT,
            lot_size            REAL
        );
        CREATE INDEX IF NOT EXISTS idx_forensics_ticket ON trade_forensics(ticket);
        """)
    logger.info("[DB] v2.0 schema initialized.")


# ── SYSTEM STATE ──────────────────────────────────────────────────────────────

def set_system_state(key: str, value: str):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO system_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (key, value))

def get_system_state(key: str) -> Optional[str]:
    with get_connection() as conn:
        r = conn.execute("SELECT value FROM system_state WHERE key=?", (key,)).fetchone()
    return r[0] if r else None


# ── DEDUPLICATION ─────────────────────────────────────────────────────────────

def _fingerprint(source: str, content: str) -> str:
    norm = " ".join(content.lower().split())
    return hashlib.sha256(f"{source}:{norm}".encode()).hexdigest()[:32]

def is_duplicate_message(source: str, content: str, window_minutes: int = None) -> bool:
    win = window_minutes or config.SIGNAL_EXPIRY_MINUTES
    fp  = _fingerprint(source, content)
    with get_connection() as conn:
        r = conn.execute("""
            SELECT 1 FROM seen_messages
            WHERE fingerprint=? AND first_seen > datetime('now', ? || ' minutes')
        """, (fp, f"-{win}")).fetchone()
    return r is not None

def mark_message_seen(source: str, content: str):
    fp = _fingerprint(source, content)
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO seen_messages (fingerprint, source)
            VALUES (?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET count=count+1
        """, (fp, source))


# ── SIGNALS ───────────────────────────────────────────────────────────────────

def insert_signal(source, raw_text, parsed, confidence, high_conviction=False,
                  session: str = None, hurst: float = None) -> int:
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO signals (source, raw_text, parsed_json, ai_confidence,
                                 is_high_conviction, session, hurst_at_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source, raw_text, json.dumps(parsed), confidence,
              int(high_conviction), session, hurst))
        return cur.lastrowid

def update_signal_status(signal_id, status, reject_reason=None, trade_ticket=None):
    with get_connection() as conn:
        conn.execute("""
            UPDATE signals SET status=?, reject_reason=?, trade_ticket=? WHERE id=?
        """, (status, reject_reason, trade_ticket, signal_id))

def update_signal_conviction(signal_id: int, is_high_conviction: bool):
    with get_connection() as conn:
        conn.execute("UPDATE signals SET is_high_conviction=? WHERE id=?",
                     (int(is_high_conviction), signal_id))


# ── TRADES ────────────────────────────────────────────────────────────────────

def insert_trade(ticket, signal_id, symbol, action, lot_size, entry,
                 sl, tp1, tp2, tp3, mode):
    with get_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO trades
            (ticket, signal_id, symbol, action, lot_size, entry_price,
             sl_price, tp1_price, tp2_price, tp3_price, open_time, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticket, signal_id, symbol, action, lot_size, entry,
              sl, tp1, tp2, tp3, datetime.now(), mode))

def close_trade(ticket, close_price, pnl):
    with get_connection() as conn:
        conn.execute("""
            UPDATE trades SET status='CLOSED', close_time=?, close_price=?, pnl=?
            WHERE ticket=?
        """, (datetime.now(), close_price, pnl, ticket))

def update_trade_status(ticket, status):
    with get_connection() as conn:
        conn.execute("UPDATE trades SET status=? WHERE ticket=?", (status, ticket))

def get_open_trades() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status NOT IN ('CLOSED')"
        ).fetchall()
    return [dict(r) for r in rows]

def get_closed_trades(limit: int = 500) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT t.*, s.source, s.raw_text
            FROM trades t
            LEFT JOIN signals s ON t.signal_id = s.id
            WHERE t.status = 'CLOSED'
            ORDER BY t.close_time DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── ANALYTICS ─────────────────────────────────────────────────────────────────

def get_daily_pnl(today: date = None) -> float:
    if today is None:
        today = date.today()
    with get_connection() as conn:
        r = conn.execute("""
            SELECT COALESCE(SUM(pnl), 0)
            FROM trades WHERE DATE(open_time)=? AND status='CLOSED'
        """, (today.isoformat(),)).fetchone()
    return float(r[0]) if r else 0.0

def get_equity_curve(limit: int = 500) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT ts, equity, daily_pnl FROM equity_snapshots
            ORDER BY ts DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in reversed(rows)]

def insert_equity_snapshot(equity: float, balance: float,
                            open_trades: int, daily_pnl: float):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO equity_snapshots (equity, balance, open_trades, daily_pnl)
            VALUES (?, ?, ?, ?)
        """, (equity, balance, open_trades, daily_pnl))

def get_source_performance() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT s.source,
                   COUNT(t.ticket)                              AS total,
                   SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   ROUND(SUM(t.pnl), 2)                        AS net_pnl,
                   ROUND(AVG(t.pnl), 2)                        AS avg_pnl,
                   MIN(t.pnl)                                   AS worst_trade,
                   MAX(t.pnl)                                   AS best_trade
            FROM trades t
            JOIN signals s ON t.signal_id = s.id
            WHERE t.status='CLOSED'
            GROUP BY s.source
            ORDER BY net_pnl DESC
        """).fetchall()
    return [dict(r) for r in rows]

def log_audit(event_type: str, details: Dict):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_log (event_type, details) VALUES (?, ?)",
            (event_type, json.dumps(details, default=str))
        )

def log_latency(avg_ms: float, jitter_ms: float, mode: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO latency_log (avg_ms, jitter_ms, mode) VALUES (?, ?, ?)",
            (avg_ms, jitter_ms, mode)
        )



def _get_daily_key() -> str:
    return date.today().isoformat()


def get_opening_equity(day_key: str = None):
    if day_key is None:
        day_key = _get_daily_key()
    with get_connection() as conn:
        row = conn.execute(
            'SELECT opening_equity FROM daily_snapshots WHERE snap_date = ?',
            (day_key,)
        ).fetchone()
    return row[0] if row else None


def set_opening_equity(equity: float, day_key: str = None):
    if day_key is None:
        day_key = _get_daily_key()
    with get_connection() as conn:
        conn.execute(
            'INSERT INTO daily_snapshots (snap_date, opening_equity) VALUES (?, ?) '
            'ON CONFLICT(snap_date) DO UPDATE SET opening_equity = excluded.opening_equity',
            (day_key, equity)
        )


def get_high_water_mark():
    val = get_system_state('high_water_mark')
    return float(val) if val else None


def update_high_water_mark(equity: float):
    set_system_state('high_water_mark', str(equity))


def backfill_feature_label(
    feature_row_id: int,
    signal_id: int,
    trade_ticket: int,
    tp1_hit: bool,
    sl_hit: bool,
    pips_pnl: float,
    bars_held: int,
    pnl_r: float,
):
    with get_connection() as conn:
        conn.execute(
            'UPDATE signals SET status = ? WHERE id = ?',
            ('TP1_HIT' if tp1_hit else ('SL_HIT' if sl_hit else 'CLOSED'), signal_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  WIN MODEL TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_win_model_training_data(limit: int = 10000) -> List[Dict]:
    """
    Return closed trades joined with their market_features snapshot.
    Each row contains the feature columns the RF model needs plus
    pnl / source / session for labelling and analysis.
    Falls back to trades+signals join when no market_features rows exist.
    """
    try:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT mf.*, t.pnl, t.entry_price, t.close_price,
                       t.lot_size, t.action, t.symbol,
                       s.source, s.ai_confidence, s.session AS sig_session
                FROM market_features mf
                JOIN trades t   ON mf.trade_ticket = t.ticket
                JOIN signals s  ON mf.signal_id    = s.id
                WHERE t.status = 'CLOSED' AND t.pnl IS NOT NULL
                ORDER BY t.close_time DESC
                LIMIT ?
            """, (limit,)).fetchall()
            if rows:
                result = []
                for r in rows:
                    d = {k: r[k] for k in r.keys()}
                    if not d.get("session"):
                        d["session"] = d.pop("sig_session", None)
                    else:
                        d.pop("sig_session", None)
                    result.append(d)
                return result

        # Fallback: no market_features linked yet, use trades+signals
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT t.*, s.source, s.ai_confidence,
                       COALESCE(t.session, s.session) AS session
                FROM trades t
                LEFT JOIN signals s ON t.signal_id = s.id
                WHERE t.status = 'CLOSED' AND t.pnl IS NOT NULL
                ORDER BY t.close_time DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [{k: r[k] for k in r.keys()} for r in rows]

    except Exception as e:
        logger.error(f"[DB] get_win_model_training_data failed: {e}")
        return []


def get_source_win_rate(source: str) -> Dict:
    """
    Return {"wins": int, "total": int} for a given signal source.
    Used by win_model for Bayesian-smoothed source win rate feature.
    """
    try:
        with get_connection() as conn:
            r = conn.execute("""
                SELECT COUNT(*)                              AS total,
                       SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS wins
                FROM trades t
                JOIN signals s ON t.signal_id = s.id
                WHERE t.status = 'CLOSED' AND s.source = ?
            """, (source,)).fetchone()
        if r:
            return {"total": int(r["total"]), "wins": int(r["wins"] or 0)}
        return {"total": 0, "wins": 0}
    except Exception:
        return {"total": 0, "wins": 0}


def get_last_trade_close_time(symbol: str):
    """Return the close_time of the most recent closed trade on this symbol."""
    try:
        with get_connection() as conn:
            r = conn.execute(
                "SELECT close_time FROM trades WHERE symbol=? AND status='CLOSED' "
                "ORDER BY close_time DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return r[0] if r else None
    except Exception:
        return None


def insert_forensic(data: Dict):
    """Insert a per-trade forensic snapshot."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO trade_forensics (
                ticket, signal_id, m15_bias, m5_bias, tick_regime,
                toxicity_score, consensus_score, ai_confidence,
                active_scanners, dxy_z_score, entry_spread_pips,
                expected_entry, actual_entry, slippage_pips,
                mfe_pips, mae_pips, mfe_dollars, mae_dollars,
                duration_secs, exit_trigger, exit_price, pnl,
                lesson_learned, weight_adjustments,
                source, action, lot_size
            ) VALUES (
                :ticket, :signal_id, :m15_bias, :m5_bias, :tick_regime,
                :toxicity_score, :consensus_score, :ai_confidence,
                :active_scanners, :dxy_z_score, :entry_spread_pips,
                :expected_entry, :actual_entry, :slippage_pips,
                :mfe_pips, :mae_pips, :mfe_dollars, :mae_dollars,
                :duration_secs, :exit_trigger, :exit_price, :pnl,
                :lesson_learned, :weight_adjustments,
                :source, :action, :lot_size
            )
        """, data)


def get_forensic(ticket: int) -> Optional[Dict]:
    """Get forensic record for a specific trade ticket."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trade_forensics WHERE ticket=? ORDER BY id DESC LIMIT 1",
            (ticket,)
        ).fetchone()
    return dict(row) if row else None

