"""
validate/02_run_validation.py — OmniSignal Alpha Scanner Validation Suite
==========================================================================
STEP 2: Backtest each scanner INDEPENDENTLY on 6 months of XAUUSD data.

Run after downloading data:
    python validate/02_run_validation.py

This does NOT use the live scanners. It extracts the CORE DETECTION LOGIC
from each scanner into pure functions that take numpy arrays, then simulates
entries and exits with realistic spread, SL/TP, and partial close rules.

Output:
    data/validation/scanner_report.txt     (terminal-readable report)
    data/validation/trades_*.csv           (per-scanner trade logs)

Scanners tested:
    1. LIQUIDITY — M1 range sweep + wick rejection + volume spike
    2. MOMENTUM  — M1 EMA(20) pullback in slope direction
    3. SMC       — M15 BOS + M5 OB/FVG entry with rejection candle
    4. BASELINE  — Simple EMA(20)/EMA(50) crossover (benchmark)

Scanners NOT testable from bars alone (need tick data):
    - TFI (tick flow imbalance) — requires real-time bid/ask tick stream
    - CATCD (correlation decay) — requires synchronized XAUUSD + DXY ticks
    - MR (mean reversion) — requires tick-level VWAP computation

These will be validated separately using recorded tick data once available.
"""

import os
import sys
import csv
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join("data", "validation")
PIP_SIZE = 0.10       # XAUUSD
SPREAD_PIPS = 3.0     # Realistic average spread
COMMISSION_PER_LOT = 7.0
DEFAULT_LOTS = 0.01   # Standardized for comparison
PIP_VALUE = 1.0       # $1 per pip per 0.01 lot (XAUUSD approximate)


# =====================================================================
#  DATA STRUCTURES
# =====================================================================

@dataclass
class Bar:
    time: str
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int = 0

@dataclass
class Signal:
    bar_index: int
    action: str       # BUY or SELL
    entry: float
    sl: float
    tp: float
    scanner: str
    reason: str = ""

@dataclass
class Trade:
    signal_idx: int
    scanner: str
    action: str
    entry: float
    sl: float
    tp: float
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_bar: int = 0
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    bars_held: int = 0


# =====================================================================
#  DATA LOADING
# =====================================================================

def load_bars(filename: str) -> List[Bar]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        print(f"          Run 01_download_history.py first.")
        return []
    bars = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(Bar(
                time=row["time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=int(row["tick_volume"]),
                spread=int(row.get("spread", 0)),
            ))
    return bars


def compute_ema(closes: np.ndarray, period: int) -> np.ndarray:
    ema = np.empty_like(closes)
    alpha = 2.0 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.zeros(n)
    if n < period:
        atr[:] = np.mean(tr)
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]
    return atr


# =====================================================================
#  TRADE SIMULATOR (shared by all scanners)
# =====================================================================

def simulate_trades(signals: List[Signal], bars: List[Bar], max_bars_held: int = 120) -> List[Trade]:
    """
    Simulate trades from signals using bar-by-bar replay.
    One trade at a time (no overlapping). Realistic spread applied.
    """
    trades = []
    in_trade = False
    current_trade = None

    for sig in sorted(signals, key=lambda s: s.bar_index):
        if in_trade:
            continue
        if sig.bar_index + 1 >= len(bars):
            continue

        # Entry at next bar open + spread
        entry_bar = bars[sig.bar_index + 1]
        spread_price = SPREAD_PIPS * PIP_SIZE
        if sig.action == "BUY":
            fill = entry_bar.open + spread_price
        else:
            fill = entry_bar.open - spread_price

        trade = Trade(
            signal_idx=sig.bar_index,
            scanner=sig.scanner,
            action=sig.action,
            entry=fill,
            sl=sig.sl,
            tp=sig.tp,
        )
        in_trade = True

        # Walk forward bar-by-bar
        for i in range(sig.bar_index + 2, min(sig.bar_index + max_bars_held + 2, len(bars))):
            bar = bars[i]
            trade.bars_held = i - sig.bar_index - 1

            if trade.action == "BUY":
                # SL hit
                if bar.low <= trade.sl:
                    trade.exit_price = trade.sl
                    trade.exit_reason = "SL_HIT"
                    trade.exit_bar = i
                    trade.pnl_pips = (trade.sl - trade.entry) / PIP_SIZE
                    break
                # TP hit
                if bar.high >= trade.tp:
                    trade.exit_price = trade.tp
                    trade.exit_reason = "TP_HIT"
                    trade.exit_bar = i
                    trade.pnl_pips = (trade.tp - trade.entry) / PIP_SIZE
                    break
            else:  # SELL
                if bar.high >= trade.sl:
                    trade.exit_price = trade.sl
                    trade.exit_reason = "SL_HIT"
                    trade.exit_bar = i
                    trade.pnl_pips = (trade.entry - trade.sl) / PIP_SIZE
                    break
                if bar.low <= trade.tp:
                    trade.exit_price = trade.tp
                    trade.exit_reason = "TP_HIT"
                    trade.exit_bar = i
                    trade.pnl_pips = (trade.entry - trade.tp) / PIP_SIZE
                    break
        else:
            # Time exit — close at last bar's close
            last_bar = bars[min(sig.bar_index + max_bars_held + 1, len(bars) - 1)]
            trade.exit_price = last_bar.close
            trade.exit_reason = "TIME_EXIT"
            trade.exit_bar = min(sig.bar_index + max_bars_held + 1, len(bars) - 1)
            if trade.action == "BUY":
                trade.pnl_pips = (last_bar.close - trade.entry) / PIP_SIZE
            else:
                trade.pnl_pips = (trade.entry - last_bar.close) / PIP_SIZE

        # Apply commission
        trade.pnl_pips -= (COMMISSION_PER_LOT * DEFAULT_LOTS) / PIP_VALUE / DEFAULT_LOTS
        trade.pnl_usd = trade.pnl_pips * PIP_VALUE * (DEFAULT_LOTS / 0.01)
        trades.append(trade)
        in_trade = False

    return trades


# =====================================================================
#  SCANNER 1: LIQUIDITY SWEEP
# =====================================================================

def scan_liquidity(bars_m1: List[Bar], range_lookback: int = 50,
                   min_sweep_pips: float = 7.0, min_wick_pct: float = 0.60,
                   min_vol_mult: float = 1.5) -> List[Signal]:
    """
    Detects M1 liquidity sweeps: price pierces range boundary with volume
    spike, then rejects with a long wick.
    """
    signals = []
    closes = np.array([b.close for b in bars_m1])
    highs = np.array([b.high for b in bars_m1])
    lows = np.array([b.low for b in bars_m1])
    opens = np.array([b.open for b in bars_m1])
    volumes = np.array([b.tick_volume for b in bars_m1], dtype=float)
    atr = compute_atr(highs, lows, closes, 14)

    for i in range(range_lookback + 1, len(bars_m1) - 1):
        history = slice(i - range_lookback, i)
        range_high = float(np.max(highs[history]))
        range_low = float(np.min(lows[history]))
        vol_avg = float(np.mean(volumes[i - 20:i]))

        if vol_avg <= 0:
            continue

        c = bars_m1[i]
        candle_range = c.high - c.low
        if candle_range < PIP_SIZE * 0.5:
            continue

        vol_ratio = c.tick_volume / vol_avg
        if vol_ratio < min_vol_mult:
            continue

        min_pierce = min_sweep_pips * PIP_SIZE

        # Bullish sweep: wick below range low, close back above
        if c.low < range_low - min_pierce and c.close > range_low:
            lower_wick = min(c.open, c.close) - c.low
            if candle_range > 0 and lower_wick / candle_range >= min_wick_pct:
                entry = closes[i]
                sl_dist = max(1.5 * atr[i], 25 * PIP_SIZE)
                sl = round(entry - min(sl_dist, 35 * PIP_SIZE), 2)
                tp = round(range_high, 2)
                if abs(tp - entry) > abs(entry - sl) * 0.5:  # Minimum R:R 0.5
                    signals.append(Signal(
                        bar_index=i, action="BUY", entry=entry, sl=sl, tp=tp,
                        scanner="LIQUIDITY", reason=f"bullish sweep vol={vol_ratio:.1f}x"
                    ))

        # Bearish sweep: wick above range high, close back below
        if c.high > range_high + min_pierce and c.close < range_high:
            upper_wick = c.high - max(c.open, c.close)
            if candle_range > 0 and upper_wick / candle_range >= min_wick_pct:
                entry = closes[i]
                sl_dist = max(1.5 * atr[i], 25 * PIP_SIZE)
                sl = round(entry + min(sl_dist, 35 * PIP_SIZE), 2)
                tp = round(range_low, 2)
                if abs(entry - tp) > abs(sl - entry) * 0.5:
                    signals.append(Signal(
                        bar_index=i, action="SELL", entry=entry, sl=sl, tp=tp,
                        scanner="LIQUIDITY", reason=f"bearish sweep vol={vol_ratio:.1f}x"
                    ))

    return signals


# =====================================================================
#  SCANNER 2: MOMENTUM PULLBACK
# =====================================================================

def scan_momentum(bars_m1: List[Bar], ema_period: int = 20,
                  min_slope: float = 0.8) -> List[Signal]:
    """
    Detects M1 EMA(20) momentum pullbacks: strong slope, price touches
    EMA and bounces in trend direction.
    """
    signals = []
    closes = np.array([b.close for b in bars_m1])
    highs = np.array([b.high for b in bars_m1])
    lows = np.array([b.low for b in bars_m1])
    atr = compute_atr(highs, lows, closes, 14)
    ema = compute_ema(closes, ema_period)

    for i in range(max(ema_period + 5, 30), len(bars_m1) - 1):
        # Compute slope over last 5 bars (pips per bar)
        slope = (ema[i] - ema[i - 5]) / (5 * PIP_SIZE)

        if abs(slope) < min_slope:
            continue

        bullish = slope > 0

        # Check last 3 candles for pullback to EMA
        for offset in range(0, 3):
            idx = i - offset
            c = bars_m1[idx]

            if bullish:
                touched = c.low <= ema[idx]
                bounced = c.close > ema[idx]
                if touched and bounced:
                    entry = closes[i]
                    sl_dist = max(1.0 * atr[i], 30 * PIP_SIZE)
                    sl = round(entry - min(sl_dist, 2.0 * atr[i]), 2)
                    tp = round(entry + sl_dist * 1.5, 2)
                    signals.append(Signal(
                        bar_index=i, action="BUY", entry=entry, sl=sl, tp=tp,
                        scanner="MOMENTUM", reason=f"pullback slope={slope:.1f}"
                    ))
                    break
            else:
                touched = c.high >= ema[idx]
                bounced = c.close < ema[idx]
                if touched and bounced:
                    entry = closes[i]
                    sl_dist = max(1.0 * atr[i], 30 * PIP_SIZE)
                    sl = round(entry + min(sl_dist, 2.0 * atr[i]), 2)
                    tp = round(entry - sl_dist * 1.5, 2)
                    signals.append(Signal(
                        bar_index=i, action="SELL", entry=entry, sl=sl, tp=tp,
                        scanner="MOMENTUM", reason=f"pullback slope={slope:.1f}"
                    ))
                    break

    return signals


# =====================================================================
#  SCANNER 3: SMC (Break of Structure + Order Block)
# =====================================================================

def scan_smc(bars_m15: List[Bar], bars_m5: List[Bar],
             swing_lookback: int = 5, ob_lookback: int = 30,
             min_impulse_pips: float = 5.0, min_wick_pct: float = 0.40) -> List[Signal]:
    """
    M15: detect Break of Structure (BOS) for directional bias.
    M5: detect Order Block entry with rejection candle.
    """
    signals = []
    if len(bars_m15) < 60 or len(bars_m5) < 30:
        return signals

    h15 = np.array([b.high for b in bars_m15])
    l15 = np.array([b.low for b in bars_m15])
    c15 = np.array([b.close for b in bars_m15])
    o15 = np.array([b.open for b in bars_m15])
    atr15 = compute_atr(h15, l15, c15, 14)

    # Find swing highs and lows on M15
    def find_swings(highs, lows, lookback):
        swing_h, swing_l = [], []
        for i in range(lookback, len(highs) - lookback):
            window_h = highs[i - lookback:i + lookback + 1]
            window_l = lows[i - lookback:i + lookback + 1]
            if highs[i] == np.max(window_h):
                swing_h.append((i, float(highs[i])))
            if lows[i] == np.min(window_l):
                swing_l.append((i, float(lows[i])))
        return swing_h, swing_l

    swing_highs, swing_lows = find_swings(h15, l15, swing_lookback)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return signals

    # Build M5 time index for alignment
    m5_times = {b.time: idx for idx, b in enumerate(bars_m5)}

    # Scan M15 for BOS events
    for i in range(30, len(bars_m15)):
        # Find most recent swing high and low before bar i
        recent_sh = [s for s in swing_highs if s[0] < i]
        recent_sl = [s for s in swing_lows if s[0] < i]
        if len(recent_sh) < 1 or len(recent_sl) < 1:
            continue

        last_sh = recent_sh[-1]
        last_sl = recent_sl[-1]

        bias = None
        # BOS bullish: close above last swing high
        if c15[i] > last_sh[1] and c15[i - 1] <= last_sh[1]:
            bias = "BULLISH"
        # BOS bearish: close below last swing low
        elif c15[i] < last_sl[1] and c15[i - 1] >= last_sl[1]:
            bias = "BEARISH"

        if bias is None:
            continue

        # Look for Order Block in M15 history (last ob_lookback bars before BOS)
        for j in range(max(0, i - ob_lookback), i - 2):
            if bias == "BULLISH":
                # Bullish OB: bearish candle before impulse up
                if c15[j] < o15[j]:  # bearish candle
                    impulse_high = float(np.max(h15[j + 1:min(j + 4, i)]))
                    impulse_pips = (impulse_high - h15[j]) / PIP_SIZE
                    if impulse_pips >= min_impulse_pips:
                        ob_top = float(h15[j])
                        ob_bot = float(l15[j])
                        # Check if current price is near the OB (within 1 ATR)
                        if c15[i] <= ob_top + atr15[i] and c15[i] >= ob_bot - atr15[i] * 0.3:
                            entry = float(c15[i])
                            sl = round(ob_bot - atr15[i] * 0.3, 2)
                            sl_dist = abs(entry - sl)
                            if sl_dist > PIP_SIZE * 5:
                                tp = round(entry + sl_dist * 2.0, 2)
                                signals.append(Signal(
                                    bar_index=i, action="BUY", entry=entry, sl=sl, tp=tp,
                                    scanner="SMC", reason=f"BOS_BULL + OB impulse={impulse_pips:.0f}p"
                                ))
                                break

            elif bias == "BEARISH":
                if c15[j] > o15[j]:  # bullish candle
                    impulse_low = float(np.min(l15[j + 1:min(j + 4, i)]))
                    impulse_pips = (l15[j] - impulse_low) / PIP_SIZE
                    if impulse_pips >= min_impulse_pips:
                        ob_top = float(h15[j])
                        ob_bot = float(l15[j])
                        if c15[i] >= ob_bot - atr15[i] and c15[i] <= ob_top + atr15[i] * 0.3:
                            entry = float(c15[i])
                            sl = round(ob_top + atr15[i] * 0.3, 2)
                            sl_dist = abs(sl - entry)
                            if sl_dist > PIP_SIZE * 5:
                                tp = round(entry - sl_dist * 2.0, 2)
                                signals.append(Signal(
                                    bar_index=i, action="SELL", entry=entry, sl=sl, tp=tp,
                                    scanner="SMC", reason=f"BOS_BEAR + OB impulse={impulse_pips:.0f}p"
                                ))
                                break

    return signals


# =====================================================================
#  BASELINE: Simple EMA Cross (benchmark)
# =====================================================================

def scan_baseline(bars_m15: List[Bar]) -> List[Signal]:
    """
    Simple EMA(20)/EMA(50) crossover on M15. This is the benchmark.
    If a scanner can't beat THIS, it has no edge.
    """
    signals = []
    closes = np.array([b.close for b in bars_m15])
    highs = np.array([b.high for b in bars_m15])
    lows = np.array([b.low for b in bars_m15])
    atr = compute_atr(highs, lows, closes, 14)
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)

    for i in range(51, len(bars_m15) - 1):
        # Bullish cross
        if ema20[i] > ema50[i] and ema20[i - 1] <= ema50[i - 1]:
            entry = closes[i]
            sl = round(entry - 2.0 * atr[i], 2)
            tp = round(entry + 3.0 * atr[i], 2)
            signals.append(Signal(
                bar_index=i, action="BUY", entry=entry, sl=sl, tp=tp,
                scanner="BASELINE", reason="EMA20 cross above EMA50"
            ))
        # Bearish cross
        elif ema20[i] < ema50[i] and ema20[i - 1] >= ema50[i - 1]:
            entry = closes[i]
            sl = round(entry + 2.0 * atr[i], 2)
            tp = round(entry - 3.0 * atr[i], 2)
            signals.append(Signal(
                bar_index=i, action="SELL", entry=entry, sl=sl, tp=tp,
                scanner="BASELINE", reason="EMA20 cross below EMA50"
            ))

    return signals


# =====================================================================
#  REPORT GENERATOR
# =====================================================================

def generate_report(scanner_name: str, trades: List[Trade], signals: List[Signal]) -> Dict:
    if not trades:
        return {
            "scanner": scanner_name,
            "signals": len(signals),
            "trades": 0,
            "verdict": "NO DATA",
        }

    pnls = [t.pnl_pips for t in trades]
    wins = [t for t in trades if t.pnl_pips > 0]
    losses = [t for t in trades if t.pnl_pips <= 0]

    total = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total if total > 0 else 0
    net_pips = sum(pnls)
    avg_win = np.mean([t.pnl_pips for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl_pips for t in losses])) if losses else 0
    profit_factor = sum(t.pnl_pips for t in wins) / abs(sum(t.pnl_pips for t in losses)) if losses and sum(t.pnl_pips for t in losses) != 0 else 0
    expectancy = net_pips / total if total > 0 else 0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    # Max drawdown in pips
    cumsum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumsum)
    drawdown = peak - cumsum
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Sharpe (annualized, assuming ~250 trading days)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(250)
    else:
        sharpe = 0

    # Consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Exit breakdown
    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # VERDICT
    if total < 30:
        verdict = "INSUFFICIENT DATA"
    elif expectancy > 2.0 and win_rate > 0.40 and profit_factor > 1.3:
        verdict = "PASS — ENABLE"
    elif expectancy > 0 and profit_factor > 1.0:
        verdict = "MARGINAL — NEEDS TUNING"
    else:
        verdict = "FAIL — DISABLE"

    return {
        "scanner": scanner_name,
        "signals": len(signals),
        "trades": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate": win_rate,
        "net_pips": net_pips,
        "expectancy_pips": expectancy,
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "avg_rr": avg_rr,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_dd_pips": max_dd,
        "max_consec_losses": max_consec_loss,
        "exit_breakdown": exit_counts,
        "avg_bars_held": np.mean([t.bars_held for t in trades]),
        "verdict": verdict,
    }


def print_report(results: List[Dict]):
    W = 72
    print()
    print("*" * W)
    print("OMNISIGNAL ALPHA — SCANNER VALIDATION REPORT".center(W))
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(W))
    print("*" * W)

    for r in results:
        print()
        print("=" * W)
        verdict_icon = {"PASS — ENABLE": "✅", "MARGINAL — NEEDS TUNING": "⚠️",
                        "FAIL — DISABLE": "❌", "INSUFFICIENT DATA": "❓",
                        "NO DATA": "⛔"}.get(r["verdict"], "?")
        print(f"  {verdict_icon} {r['scanner']}: {r['verdict']}")
        print("=" * W)

        if r["trades"] == 0:
            print(f"  Signals generated: {r['signals']}")
            print(f"  No trades executed.")
            continue

        rows = [
            ("Signals Generated:", f"{r['signals']}"),
            ("Trades Executed:", f"{r['trades']}"),
            ("Wins / Losses:", f"{r['wins']} / {r['losses']}"),
            ("Win Rate:", f"{r['win_rate']:.1%}"),
            ("Net P&L (pips):", f"{r['net_pips']:+.1f}"),
            ("Expectancy (pips/trade):", f"{r['expectancy_pips']:+.1f}"),
            ("Avg Win:", f"{r['avg_win_pips']:.1f} pips"),
            ("Avg Loss:", f"{r['avg_loss_pips']:.1f} pips"),
            ("Avg R:R:", f"{r['avg_rr']:.2f}"),
            ("Profit Factor:", f"{r['profit_factor']:.2f}"),
            ("Sharpe Ratio:", f"{r['sharpe']:.2f}"),
            ("Max Drawdown:", f"{r['max_dd_pips']:.1f} pips"),
            ("Max Consec Losses:", f"{r['max_consec_losses']}"),
            ("Avg Bars Held:", f"{r['avg_bars_held']:.0f}"),
        ]
        for label, value in rows:
            print(f"  {label:<30} {value:>15}")

        if r.get("exit_breakdown"):
            print(f"\n  Exit Breakdown:")
            for reason, count in sorted(r["exit_breakdown"].items()):
                pct = count / r["trades"] * 100
                print(f"    {reason:<20} {count:>5} ({pct:.0f}%)")

    # Summary table
    print()
    print("=" * W)
    print("  SUMMARY COMPARISON".center(W))
    print("=" * W)
    print(f"  {'Scanner':<15} {'Trades':>7} {'WR%':>7} {'Expect':>8} {'PF':>7} {'Sharpe':>8} {'Verdict':<25}")
    print("  " + "-" * (W - 4))
    for r in results:
        if r["trades"] == 0:
            print(f"  {r['scanner']:<15} {'—':>7} {'—':>7} {'—':>8} {'—':>7} {'—':>8} {r['verdict']:<25}")
        else:
            print(f"  {r['scanner']:<15} {r['trades']:>7} {r['win_rate']:>6.1%} {r['expectancy_pips']:>+7.1f} {r['profit_factor']:>7.2f} {r['sharpe']:>8.2f} {r['verdict']:<25}")

    # Action items
    print()
    print("=" * W)
    print("  ACTION ITEMS".center(W))
    print("=" * W)
    passed = [r for r in results if "PASS" in r["verdict"]]
    marginal = [r for r in results if "MARGINAL" in r["verdict"]]
    failed = [r for r in results if "FAIL" in r["verdict"]]

    if passed:
        print(f"\n  RE-ENABLE these scanners:")
        for r in passed:
            print(f"    ✅ {r['scanner']} — {r['expectancy_pips']:+.1f} pips/trade, PF={r['profit_factor']:.2f}")
    if marginal:
        print(f"\n  TUNE these scanners (adjust SL/TP/thresholds, then re-test):")
        for r in marginal:
            print(f"    ⚠️ {r['scanner']} — {r['expectancy_pips']:+.1f} pips/trade, PF={r['profit_factor']:.2f}")
    if failed:
        print(f"\n  DISABLE these scanners (negative edge confirmed):")
        for r in failed:
            print(f"    ❌ {r['scanner']} — {r['expectancy_pips']:+.1f} pips/trade, PF={r['profit_factor']:.2f}")

    not_tested = ["TFI (needs tick data)", "CATCD (needs DXY ticks)", "MR (needs tick VWAP)"]
    print(f"\n  NOT YET TESTED (need tick-level recording):")
    for s in not_tested:
        print(f"    ❓ {s}")

    print()
    print("=" * W)

    return results


def save_trades_csv(trades: List[Trade], scanner_name: str):
    path = os.path.join(DATA_DIR, f"trades_{scanner_name.lower()}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scanner", "action", "entry", "sl", "tp", "exit_price",
                          "exit_reason", "pnl_pips", "pnl_usd", "bars_held"])
        for t in trades:
            writer.writerow([
                t.scanner, t.action, t.entry, t.sl, t.tp, t.exit_price,
                t.exit_reason, round(t.pnl_pips, 1), round(t.pnl_usd, 2), t.bars_held
            ])
    print(f"  Trades saved: {path}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("Loading historical data...")
    bars_m1 = load_bars("XAUUSD_M1.csv")
    bars_m5 = load_bars("XAUUSD_M5.csv")
    bars_m15 = load_bars("XAUUSD_M15.csv")

    if not bars_m1 and not bars_m15:
        print("\n[ERROR] No data files found.")
        print("Run first: python validate/01_download_history.py")
        sys.exit(1)

    print(f"  M1:  {len(bars_m1):,} bars")
    print(f"  M5:  {len(bars_m5):,} bars")
    print(f"  M15: {len(bars_m15):,} bars")

    all_results = []

    # ── SCANNER 1: LIQUIDITY ────────────────────────────────────────
    if bars_m1:
        print("\n[1/4] Scanning: LIQUIDITY (M1 range sweeps)...")
        sigs = scan_liquidity(bars_m1)
        print(f"  Signals found: {len(sigs)}")
        # Deduplicate: no signal within 120 bars of previous
        deduped = []
        last_idx = -999
        for s in sigs:
            if s.bar_index - last_idx > 120:
                deduped.append(s)
                last_idx = s.bar_index
        print(f"  After dedup (120-bar gap): {len(deduped)}")
        trades = simulate_trades(deduped, bars_m1, max_bars_held=120)
        all_results.append(generate_report("LIQUIDITY", trades, deduped))
        save_trades_csv(trades, "LIQUIDITY")
    else:
        all_results.append(generate_report("LIQUIDITY", [], []))

    # ── SCANNER 2: MOMENTUM ─────────────────────────────────────────
    if bars_m1:
        print("\n[2/4] Scanning: MOMENTUM (M1 EMA pullbacks)...")
        sigs = scan_momentum(bars_m1)
        print(f"  Signals found: {len(sigs)}")
        deduped = []
        last_idx = -999
        last_dir = None
        for s in sigs:
            if s.bar_index - last_idx > 120 or s.action != last_dir:
                deduped.append(s)
                last_idx = s.bar_index
                last_dir = s.action
        print(f"  After dedup: {len(deduped)}")
        trades = simulate_trades(deduped, bars_m1, max_bars_held=120)
        all_results.append(generate_report("MOMENTUM", trades, deduped))
        save_trades_csv(trades, "MOMENTUM")
    else:
        all_results.append(generate_report("MOMENTUM", [], []))

    # ── SCANNER 3: SMC ──────────────────────────────────────────────
    if bars_m15 and bars_m5:
        print("\n[3/4] Scanning: SMC (M15 BOS + OB entry)...")
        sigs = scan_smc(bars_m15, bars_m5)
        print(f"  Signals found: {len(sigs)}")
        deduped = []
        last_idx = -999
        for s in sigs:
            if s.bar_index - last_idx > 8:  # M15 = 8 bars = 2 hours
                deduped.append(s)
                last_idx = s.bar_index
        print(f"  After dedup: {len(deduped)}")
        trades = simulate_trades(deduped, bars_m15, max_bars_held=40)
        all_results.append(generate_report("SMC", trades, deduped))
        save_trades_csv(trades, "SMC")
    else:
        all_results.append(generate_report("SMC", [], []))

    # ── BASELINE: EMA Cross ─────────────────────────────────────────
    if bars_m15:
        print("\n[4/4] Scanning: BASELINE (EMA20/50 cross)...")
        sigs = scan_baseline(bars_m15)
        print(f"  Signals found: {len(sigs)}")
        trades = simulate_trades(sigs, bars_m15, max_bars_held=96)
        all_results.append(generate_report("BASELINE", trades, sigs))
        save_trades_csv(trades, "BASELINE")
    else:
        all_results.append(generate_report("BASELINE", [], []))

    # ── PRINT REPORT ────────────────────────────────────────────────
    print_report(all_results)

    # Save report as JSON
    report_path = os.path.join(DATA_DIR, "scanner_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull report saved: {report_path}")

    # Save text report
    import io
    from contextlib import redirect_stdout
    text_path = os.path.join(DATA_DIR, "scanner_report.txt")
    with open(text_path, "w") as f:
        with redirect_stdout(f):
            print_report(all_results)
    print(f"Text report saved: {text_path}")


if __name__ == "__main__":
    main()
