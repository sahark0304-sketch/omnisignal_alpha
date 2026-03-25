"""
backtest/backtest_engine.py — OmniSignal Alpha v2.0
Pillar 9: Backtesting Architecture

Decoupled design: the same ParsedSignal + risk_guard logic runs
against historical OHLCV data instead of live MT5.

Usage:
    engine = BacktestEngine(initial_capital=10_000)
    engine.load_ohlcv("EURUSD", "data/historical/EURUSD_H1.csv")
    results = engine.run(signals=my_parsed_signals)
    report  = results.summary()

Historical data format (CSV):
    time,open,high,low,close,tick_volume
    2024-01-02 00:00:00,1.10500,1.10780,...

Signal feed: list of ParsedSignal objects with timestamps attached.
            Can be replayed from black_box decisions or from a CSV signal log.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OHLCVBar:
    time:   datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class BacktestTrade:
    signal_text: str
    symbol: str
    action: str
    entry:  float
    sl:     float
    tp1:    Optional[float]
    tp2:    Optional[float]
    tp3:    Optional[float]
    lots:   float
    open_bar:  int           # Index into bars list
    close_bar: Optional[int] = None
    close_price: float = 0.0
    pnl:    float = 0.0
    pnl_r:  float = 0.0      # PnL in R (multiples of risk)
    exit_reason: str = ""    # SL_HIT / TP1 / TP2 / TP3 / EOD / END_OF_DATA
    # Partial close tracking
    tp1_closed: bool = False
    tp2_closed: bool = False


@dataclass
class BacktestResult:
    trades:         List[BacktestTrade]
    initial_capital: float
    commission_per_lot: float

    @property
    def closed(self) -> List[BacktestTrade]:
        return [t for t in self.trades if t.close_bar is not None]

    def summary(self) -> Dict:
        c = self.closed
        if not c:
            return {"error": "No closed trades"}

        wins   = [t for t in c if t.pnl > 0]
        losses = [t for t in c if t.pnl <= 0]
        pnls   = [t.pnl for t in c]
        r_vals = [t.pnl_r for t in c]

        equity_curve = self._equity_curve()
        dd           = self._max_drawdown(equity_curve)
        sharpe       = self._sharpe(pnls)
        expectancy   = sum(pnls) / len(pnls) if pnls else 0

        return {
            "total_trades":    len(c),
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate":        len(wins) / max(len(c), 1),
            "net_pnl":         sum(pnls),
            "final_equity":    self.initial_capital + sum(pnls),
            "max_drawdown":    dd,
            "sharpe_ratio":    sharpe,
            "expectancy_usd":  expectancy,
            "expectancy_r":    sum(r_vals) / max(len(r_vals), 1),
            "avg_win_usd":     sum(t.pnl for t in wins) / max(len(wins), 1),
            "avg_loss_usd":    sum(abs(t.pnl) for t in losses) / max(len(losses), 1),
            "profit_factor":   (
                sum(t.pnl for t in wins) /
                max(sum(abs(t.pnl) for t in losses), 0.01)
            ),
            "by_exit":         self._by_exit(),
        }

    def _equity_curve(self) -> List[float]:
        eq = self.initial_capital
        curve = [eq]
        for t in sorted(self.closed, key=lambda x: x.close_bar):
            eq += t.pnl
            curve.append(eq)
        return curve

    def _max_drawdown(self, curve: List[float]) -> float:
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _sharpe(self, pnls: List[float], risk_free: float = 0.0) -> float:
        if len(pnls) < 2:
            return 0.0
        import statistics
        mean = statistics.mean(pnls)
        std  = statistics.stdev(pnls)
        return (mean - risk_free) / std if std > 0 else 0.0

    def _by_exit(self) -> Dict[str, int]:
        counts = {}
        for t in self.closed:
            counts[t.exit_reason] = counts.get(t.exit_reason, 0) + 1
        return counts


class BacktestEngine:
    """
    Event-driven bar-by-bar backtester.
    Decoupled from MT5 — uses CSV OHLCV data.
    Applies same TP1/TP2/BE/Trailing logic as live trade_manager.
    """

    def __init__(
        self,
        initial_capital: float = config.BACKTEST_INITIAL_CAPITAL,
        commission_per_lot: float = config.BACKTEST_COMMISSION_PER_LOT,
        spread_pips: float = 1.5,
    ):
        self.initial_capital    = initial_capital
        self.commission         = commission_per_lot
        self.spread_pips        = spread_pips
        self._bars: Dict[str, List[OHLCVBar]] = {}

    def load_ohlcv_csv(self, symbol: str, path: str):
        """Load historical OHLCV from CSV. Columns: time,open,high,low,close,tick_volume"""
        bars = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    bars.append(OHLCVBar(
                        time=datetime.fromisoformat(row["time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("tick_volume", row.get("volume", 0))),
                    ))
                except (KeyError, ValueError) as e:
                    logger.debug(f"[Backtest] Skipping row: {e}")
        self._bars[symbol] = sorted(bars, key=lambda b: b.time)
        logger.info(f"[Backtest] Loaded {len(bars)} bars for {symbol}")

    def run(self, signals: List) -> BacktestResult:
        """
        signals: list of dicts with keys:
          signal_time (datetime), symbol, action, entry_price, stop_loss, tp1, tp2, tp3, lots
        """
        trades  = []
        equity  = self.initial_capital
        open_trades: List[BacktestTrade] = []

        for sig_dict in signals:
            symbol     = sig_dict["symbol"]
            bars       = self._bars.get(symbol, [])
            if not bars:
                logger.warning(f"[Backtest] No bars for {symbol} — signal skipped")
                continue

            sig_time   = sig_dict.get("signal_time", bars[0].time)
            entry      = sig_dict.get("entry_price")
            sl         = sig_dict["stop_loss"]
            lots       = sig_dict.get("lots", 0.1)
            action     = sig_dict["action"]

            # Find bar index at signal time
            start_idx  = self._find_bar(bars, sig_time)
            if start_idx is None:
                continue

            # Fill: use open of next bar + spread
            fill_bar = bars[start_idx]
            pip_size = _get_pip_size_static(symbol)
            spread_price = self.spread_pips * pip_size
            if action == "BUY":
                fill_price = fill_bar.open + spread_price
            else:
                fill_price = fill_bar.open - spread_price

            if entry is None:
                entry = fill_price

            trade = BacktestTrade(
                signal_text = sig_dict.get("raw_text", ""),
                symbol      = symbol,
                action      = action,
                entry       = fill_price,
                sl          = sl,
                tp1         = sig_dict.get("tp1"),
                tp2         = sig_dict.get("tp2"),
                tp3         = sig_dict.get("tp3"),
                lots        = lots,
                open_bar    = start_idx,
            )
            open_trades.append(trade)
            trades.append(trade)

        # Simulate bar-by-bar across all open trades
        for trade in open_trades:
            symbol = trade.symbol
            bars   = self._bars.get(symbol, [])
            pip_size = _get_pip_size_static(symbol)
            pip_val  = 10.0  # Simplified — use 10 $/pip/lot for all pairs

            be_triggered = False
            trailing_sl  = None
            current_sl   = trade.sl

            for i in range(trade.open_bar + 1, len(bars)):
                bar = bars[i]
                pnl_mult = 1.0 if trade.action == "BUY" else -1.0

                # ── TP1 partial close ──────────────────────────────────────────
                if trade.tp1 and not trade.tp1_closed:
                    tp1_hit = (trade.action == "BUY" and bar.high >= trade.tp1) or \
                              (trade.action == "SELL" and bar.low <= trade.tp1)
                    if tp1_hit:
                        tp1_pnl = pnl_mult * abs(trade.tp1 - trade.entry) / pip_size * pip_val * (trade.lots * config.TP1_CLOSE_PCT)
                        tp1_pnl -= self.commission * (trade.lots * config.TP1_CLOSE_PCT)
                        trade.pnl += tp1_pnl
                        trade.tp1_closed = True
                        # BE trigger
                        be_triggered = True
                        current_sl = trade.entry
                        trailing_sl = trade.entry

                # ── TP2 partial close ──────────────────────────────────────────
                if trade.tp2 and trade.tp1_closed and not trade.tp2_closed:
                    tp2_hit = (trade.action == "BUY" and bar.high >= trade.tp2) or \
                              (trade.action == "SELL" and bar.low <= trade.tp2)
                    if tp2_hit:
                        tp2_pnl = pnl_mult * abs(trade.tp2 - trade.entry) / pip_size * pip_val * (trade.lots * config.TP2_CLOSE_PCT)
                        tp2_pnl -= self.commission * (trade.lots * config.TP2_CLOSE_PCT)
                        trade.pnl += tp2_pnl
                        trade.tp2_closed = True

                # ── Trailing stop ──────────────────────────────────────────────
                if trade.tp1_closed and trailing_sl is not None:
                    trail_step_price = config.TRAILING_STOP_STEP_PIPS * pip_size
                    trail_act_price  = config.TRAILING_STOP_ACTIVATION_PIPS * pip_size
                    if trade.action == "BUY":
                        ideal = bar.close - trail_act_price
                        if ideal > trailing_sl + trail_step_price:
                            trailing_sl = ideal
                            current_sl  = trailing_sl
                    else:
                        ideal = bar.close + trail_act_price
                        if ideal < trailing_sl - trail_step_price:
                            trailing_sl = ideal
                            current_sl  = trailing_sl

                # ── SL hit ────────────────────────────────────────────────────
                sl_hit = (trade.action == "BUY" and bar.low <= current_sl) or \
                         (trade.action == "SELL" and bar.high >= current_sl)
                if sl_hit:
                    remaining_lots = trade.lots * (1 - config.TP1_CLOSE_PCT * int(trade.tp1_closed)
                                                     - config.TP2_CLOSE_PCT * int(trade.tp2_closed))
                    sl_pnl = pnl_mult * abs(current_sl - trade.entry) / pip_size * pip_val * remaining_lots * -1
                    sl_pnl -= self.commission * remaining_lots
                    trade.pnl += sl_pnl
                    trade.close_bar = i
                    trade.close_price = current_sl
                    trade.exit_reason = "SL_HIT"
                    trade.pnl_r = trade.pnl / max(abs(trade.entry - trade.sl) / pip_size * pip_val * trade.lots, 1)
                    break

                # ── TP3 full close ─────────────────────────────────────────────
                if trade.tp3:
                    tp3_hit = (trade.action == "BUY" and bar.high >= trade.tp3) or \
                              (trade.action == "SELL" and bar.low <= trade.tp3)
                    if tp3_hit:
                        remaining_lots = trade.lots * (1 - config.TP1_CLOSE_PCT * int(trade.tp1_closed)
                                                         - config.TP2_CLOSE_PCT * int(trade.tp2_closed))
                        tp3_pnl = pnl_mult * abs(trade.tp3 - trade.entry) / pip_size * pip_val * remaining_lots
                        tp3_pnl -= self.commission * remaining_lots
                        trade.pnl += tp3_pnl
                        trade.close_bar = i
                        trade.close_price = trade.tp3
                        trade.exit_reason = "TP3"
                        trade.pnl_r = trade.pnl / max(abs(trade.entry - trade.sl) / pip_size * pip_val * trade.lots, 1)
                        break
            else:
                # End of data — close at last price
                if trade.close_bar is None:
                    last_bar = bars[-1]
                    remaining_lots = trade.lots
                    eod_pnl = pnl_mult * abs(last_bar.close - trade.entry) / pip_size * pip_val * remaining_lots
                    trade.pnl += eod_pnl
                    trade.close_bar = len(bars) - 1
                    trade.close_price = last_bar.close
                    trade.exit_reason = "END_OF_DATA"

        return BacktestResult(
            trades=trades,
            initial_capital=self.initial_capital,
            commission_per_lot=self.commission,
        )

    def _find_bar(self, bars: List[OHLCVBar], target_time: datetime) -> Optional[int]:
        """Binary search for the first bar at or after target_time."""
        lo, hi = 0, len(bars) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if bars[mid].time < target_time:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo if lo < len(bars) else None

    def export_trades_csv(self, result: BacktestResult, path: str):
        """Export backtest trades to CSV for external analysis."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "symbol","action","entry","sl","tp1","tp2","tp3",
                "lots","pnl","pnl_r","exit_reason","tp1_closed","tp2_closed"
            ])
            for t in result.closed:
                writer.writerow([
                    t.symbol, t.action, t.entry, t.sl, t.tp1, t.tp2, t.tp3,
                    t.lots, round(t.pnl, 2), round(t.pnl_r, 3),
                    t.exit_reason, t.tp1_closed, t.tp2_closed
                ])
        logger.info(f"[Backtest] Exported {len(result.closed)} trades to {path}")


def _get_pip_size_static(symbol: str) -> float:
    """Static pip size without MT5 — for backtesting only."""
    s = symbol.upper()
    if any(x in s for x in ("JPY", "HUF", "KRW")):
        return 0.01
    if "XAU" in s:
        return 0.10
    if "XAG" in s:
        return 0.001
    if any(x in s for x in ("US30", "NAS", "SPX", "DAX", "FTSE")):
        return 1.0
    return 0.0001


def load_signals_from_blackbox(symbol: str = None, source: str = None) -> List[Dict]:
    """
    Reconstruct historical signal feed from black_box.db for replay backtesting.
    Filters to EXECUTED trades only (signals that made it through risk).
    """
    from quant.black_box import query_decisions
    decisions = query_decisions(limit=1000, symbol=symbol, source=source, decision="EXECUTED")
    signals = []
    for d in decisions:
        if not d.get("ai_sl") or not d.get("ai_symbol"):
            continue
        signals.append({
            "signal_time": datetime.fromisoformat(d["ts"]) if d.get("ts") else datetime.now(),
            "symbol":      d["ai_symbol"],
            "action":      d["ai_action"],
            "entry_price": d.get("ai_entry"),
            "stop_loss":   d["ai_sl"],
            "tp1":         d.get("ai_tp1"),
            "lots":        d.get("lot_size", 0.01),
            "raw_text":    d.get("raw_message", ""),
        })
    return signals
