'''
OmniSignal Alpha - Performance Analyzer
Standalone script that reads omnisignal.db and prints a terminal performance report.
'''

import json
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

W = 72
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'data' / 'omnisignal.db'


def section_header(title):
    print('\n' + '=' * W)
    print(title.center(W))
    print('=' * W)


def account_summary(conn):
    section_header('OVERALL ACCOUNT SUMMARY')

    closed = conn.execute(
        "SELECT * FROM trades WHERE status = 'CLOSED' AND pnl IS NOT NULL"
    ).fetchall()

    open_count = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status NOT IN ('CLOSED')"
    ).fetchone()[0]

    if not closed:
        print('No data yet')
        return

    total = len(closed)
    wins = [t for t in closed if t['pnl'] > 0]
    losses = [t for t in closed if t['pnl'] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total * 100) if total else 0
    net_pnl = sum(t['pnl'] for t in closed)
    avg_pnl = net_pnl / total if total else 0
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    best = max(closed, key=lambda t: t['pnl'])
    worst = min(closed, key=lambda t: t['pnl'])
    symbols = sorted({t['symbol'] for t in closed if t['symbol']})

    pf_str = f'{profit_factor:.2f}' if profit_factor != float('inf') else 'inf'

    rows = [
        ('Total Closed Trades:', f'{total}'),
        ('Open Positions:', f'{open_count}'),
        ('Wins / Losses:', f'{win_count} / {loss_count}'),
        ('Win Rate:', f'{win_rate:.0f}%'),
        ('Net PnL:', '${:+,.2f}'.format(net_pnl)),
        ('Avg PnL per trade:', '${:+,.2f}'.format(avg_pnl)),
        ('Profit Factor:', pf_str),
        ('Best Trade:', '${:+,.2f}  ({})'.format(best['pnl'], best['symbol'])),
        ('Worst Trade:', '${:+,.2f}  ({})'.format(worst['pnl'], worst['symbol'])),
        ('Symbols Traded:', ', '.join(symbols)),
    ]
    for label, value in rows:
        print(f'  {label:<35} {value:>10}')

def supplier_leaderboard(conn):
    section_header('SUPPLIER LEADERBOARD & AUTO-BLACKLIST')

    rows = conn.execute(
        "SELECT s.source, t.pnl "
        "FROM trades t LEFT JOIN signals s ON s.id = t.signal_id "
        "WHERE t.status = 'CLOSED' AND t.pnl IS NOT NULL"
    ).fetchall()

    if not rows:
        print('No data yet')
        return

    sources = {}
    for r in rows:
        src = r['source'] or 'UNKNOWN'
        sources.setdefault(src, []).append(r['pnl'])

    stats = []
    for src, pnls in sources.items():
        total = len(pnls)
        w = sum(1 for p in pnls if p > 0)
        wr = (w / total * 100) if total else 0
        net = sum(pnls)
        stats.append((src, total, wr, net))

    stats.sort(key=lambda x: x[3], reverse=True)

    blacklisted = []
    print(f'  {"Source":<30} {"Trades":>6} {"WR%":>6} {"Net PnL":>12}')
    print('  ' + '-' * (W - 4))

    for src, total, wr, net in stats:
        flag = ''
        if total >= 3 and (wr < 30 or net < -50):
            flag = ' [BLACKLISTED]'
            blacklisted.append(src)
        print(f'  {src:<30} {total:>6} {wr:>5.0f}% {"${:+,.2f}".format(net):>12}{flag}')

    if blacklisted:
        print(f'\n  --- {len(blacklisted)} source(s) flagged for blacklist ---')
        quoted = ', '.join(f'"{s}"' for s in blacklisted)
        print(f'\n  Add to config.py:\n')
        print(f'    BLACKLISTED_SOURCES = [{quoted}]')
        print(f'\n  Add to risk_guard.validate():\n')
        print('    if signal.raw_source in config.BLACKLISTED_SOURCES:')
        print('        return _reject("Source blacklisted", "BLACKLIST")')
    else:
        print('\n  No sources flagged for blacklist.')

def sl_cap_savings(conn):
    section_header('SL-CAP SAVINGS CALCULATOR')

    rows = conn.execute(
        "SELECT details FROM audit_log WHERE event_type = 'SL_CAPPED'"
    ).fetchall()

    if not rows:
        print('No SL-cap events recorded yet.')
        return

    total_saved = 0.0
    count = 0

    for r in rows:
        try:
            d = json.loads(r['details'])
            old_pips = float(d.get('old_pips', 0))
            new_pips = float(d.get('new_pips', 0))
            saved = old_pips - new_pips
            if saved > 0:
                total_saved += saved
                count += 1
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if count == 0:
        print('No positive SL-cap savings found.')
        return

    avg_saved = total_saved / count
    capital_saved = total_saved * 1.0

    print(f'  {"SL Override Events:":<35} {count:>10}')
    print(f'  {"Total Pips Saved:":<35} {total_saved:>10.1f}')
    print(f'  {"Avg Pips Saved per Cap:":<35} {avg_saved:>10.1f}')
    print(f'  {"Est. Capital Saved (0.01 lot):":<35} {"${:+,.2f}".format(capital_saved):>10}')


def _derive_session(open_time_str):
    if not open_time_str:
        return 'UNKNOWN'
    try:
        dt = datetime.fromisoformat(open_time_str)
        h = dt.hour
    except (ValueError, TypeError):
        return 'UNKNOWN'
    if 12 <= h < 14:
        return 'OVERLAP'
    if 7 <= h < 12:
        return 'LONDON'
    if 14 <= h < 21:
        return 'NY'
    if h >= 22 or h < 7:
        return 'ASIA'
    return 'NY'

def session_analysis(conn):
    section_header('TIME-OF-DAY SESSION ANALYSIS')

    rows = conn.execute(
        "SELECT session, open_time, pnl FROM trades "
        "WHERE status = 'CLOSED' AND pnl IS NOT NULL"
    ).fetchall()

    if not rows:
        print('No data yet')
        return

    buckets = {}
    for r in rows:
        sess = r['session'] if r['session'] else _derive_session(r['open_time'])
        buckets.setdefault(sess, []).append(r['pnl'])

    stats = []
    for sess, pnls in buckets.items():
        total = len(pnls)
        w = sum(1 for p in pnls if p > 0)
        wr = (w / total * 100) if total else 0
        net = sum(pnls)
        stats.append((sess, total, w, wr, net))

    best_sess = max(stats, key=lambda x: x[4])

    print(f'  {"Session":<15} {"Trades":>7} {"Wins":>6} {"WR%":>6} {"Net PnL":>12}')
    print('  ' + '-' * (W - 4))

    for sess, total, w, wr, net in stats:
        tag = ' ** BEST **' if sess == best_sess[0] else ''
        print(f'  {sess:<15} {total:>7} {w:>6} {wr:>5.0f}% {"${:+,.2f}".format(net):>12}{tag}')

    print(f'\n  Most profitable session: {best_sess[0]} '
          f'({"${:+,.2f}".format(best_sess[4])})')

def ml_progress(conn):
    section_header('ML MODEL PROGRESS')

    row = conn.execute(
        'SELECT COUNT(*) FROM market_features '
        'WHERE trade_ticket IS NOT NULL AND label_pips_pnl IS NOT NULL'
    ).fetchone()

    labeled = row[0] if row else 0
    min_required = 50
    pct = min(labeled / min_required * 100, 100)
    status = 'READY' if labeled >= min_required else 'OBSERVATION'

    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = '#' * filled + '.' * (bar_len - filled)

    print(f'  {"Status:":<25} {status}')
    print(f'  {"Labeled Samples:":<25} {labeled} / {min_required}')
    print(f'  Progress: [{bar}] {pct:.0f}%')

    if labeled < min_required:
        remaining = min_required - labeled
        print(f'  {remaining} more labeled trade(s) needed before training.')
    else:
        print('  Sufficient data for model training.')
        model_path = ROOT / 'data' / 'win_model.pkl'
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    state = pickle.load(f)
                acc = getattr(state, 'accuracy_cv', None)
                n_samp = getattr(state, 'n_samples', None)
                n_pos = getattr(state, 'n_positive', None)
                importances = getattr(state, 'feature_importances', None)
                print()
                if acc is not None:
                    print(f'  {"CV Accuracy:":<25} {acc:.2%}')
                if n_samp is not None:
                    print(f'  {"Training Samples:":<25} {n_samp}')
                if n_pos is not None:
                    print(f'  {"Positive Samples:":<25} {n_pos}')
                if importances and isinstance(importances, dict):
                    top3 = sorted(importances.items(),
                                  key=lambda x: x[1], reverse=True)[:3]
                    print('\n  Top 3 Features:')
                    for rank, (feat, imp) in enumerate(top3, 1):
                        print(f'    {rank}. {feat:<30} {imp:.4f}')
            except Exception as e:
                print(f'  Could not load model: {e}')
        else:
            print('  Model file not found (data/win_model.pkl).')

def main():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print('*' * W)
    print('OMNISIGNAL ALPHA - PERFORMANCE REPORT'.center(W))
    print(now.center(W))
    print('*' * W)

    if not DB_PATH.exists():
        print(f'\nDatabase not found: {DB_PATH}')
        print('Run the bot first to generate trading data.')
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row

    try:
        account_summary(conn)
        supplier_leaderboard(conn)
        sl_cap_savings(conn)
        session_analysis(conn)
        ml_progress(conn)
    finally:
        conn.close()

    print('\n' + '=' * W)
    print('Report generated. All data from omnisignal.db'.center(W))
    print('=' * W)


if __name__ == '__main__':
    main()
