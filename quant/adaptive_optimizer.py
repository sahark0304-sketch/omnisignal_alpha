"""
quant/adaptive_optimizer.py -- Adaptive Parameter Optimization Engine.

Tracks MAE/MFE for closed trades segmented by source, then every N trades
runs a grid-search solver to find the (SL_ATR_mult, FE_time, FE_mfe_pips)
tuple that maximizes the Sharpe Ratio of the hypothetical equity curve.
Live parameters are updated via EMA blending for smooth adaptation.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
import threading

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeOutcome:
    ticket: int
    source: str
    action: str
    entry_price: float
    close_price: float
    pnl: float
    mae_pips: float
    mfe_pips: float
    duration_secs: float
    atr_at_entry: float
    sl_distance_pips: float


_DEFAULT_PARAMS: Dict[str, Dict] = {
    "AUTO_TFI":      {"sl_atr_mult": 2.5, "fe_time_secs": 120, "fe_min_mfe_pips": 4.0},
    "AUTO_PULLBACK": {"sl_atr_mult": 3.0, "fe_time_secs": 300, "fe_min_mfe_pips": 5.0},
    "AUTO_SCANNER":  {"sl_atr_mult": 2.0, "fe_time_secs": 180, "fe_min_mfe_pips": 4.0},
}

_SL_MULT_GRID = np.arange(1.5, 5.0, 0.5)
_FE_TIME_GRID = np.arange(60, 480, 60)
_FE_MFE_GRID = np.arange(2.0, 8.0, 1.0)

EMA_ALPHA = 0.20
OPTIMIZE_EVERY = 10
MIN_TRADES_FOR_OPT = 15
BUFFER_SIZE = 50


class AdaptiveOptimizer:

    def __init__(self):
        self._outcomes: Dict[str, Deque[TradeOutcome]] = defaultdict(
            lambda: deque(maxlen=BUFFER_SIZE)
        )
        self._trade_count = 0
        self._lock = threading.Lock()

        self._live_params: Dict[str, Dict] = {
            src: dict(params) for src, params in _DEFAULT_PARAMS.items()
        }
        self._optimization_count = 0

    def record_trade(self, outcome: TradeOutcome):
        """Called on every trade close. Triggers optimization every N trades."""
        with self._lock:
            self._outcomes[outcome.source].append(outcome)
            self._trade_count += 1

            logger.debug(
                f"[Optimizer] Recorded #{outcome.ticket} ({outcome.source}) "
                f"MAE={outcome.mae_pips:.1f}p MFE={outcome.mfe_pips:.1f}p "
                f"PnL={outcome.pnl:.2f} dur={outcome.duration_secs:.0f}s"
            )

            if self._trade_count % OPTIMIZE_EVERY == 0:
                self._run_optimization()

    def get_params(self, source: str) -> Dict:
        """Return current optimized parameters for a source."""
        with self._lock:
            return dict(self._live_params.get(source, {}))

    def get_all_params(self) -> Dict[str, Dict]:
        with self._lock:
            return {s: dict(p) for s, p in self._live_params.items()}

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_trades_recorded": self._trade_count,
                "optimizations_run": self._optimization_count,
                "buffer_sizes": {
                    src: len(buf) for src, buf in self._outcomes.items()
                },
                "live_params": self.get_all_params(),
            }

    def _run_optimization(self):
        self._optimization_count += 1
        logger.info(
            f"[Optimizer] Running optimization #{self._optimization_count} "
            f"(total trades: {self._trade_count})"
        )

        for source, buffer in self._outcomes.items():
            trades = list(buffer)
            if len(trades) < MIN_TRADES_FOR_OPT:
                logger.debug(
                    f"[Optimizer] {source}: only {len(trades)} trades, "
                    f"need {MIN_TRADES_FOR_OPT}. Skipping."
                )
                continue

            optimal = self._find_optimal_params(trades)
            if optimal is None:
                continue

            old = self._live_params.get(source, dict(_DEFAULT_PARAMS.get(source, {})))
            new_blended = self._ema_blend(old, optimal)
            self._live_params[source] = new_blended

            logger.info(
                f"[Optimizer] {source} updated: "
                f"SL_mult={old.get('sl_atr_mult', 0):.1f}->{new_blended['sl_atr_mult']:.2f} | "
                f"FE_time={old.get('fe_time_secs', 0):.0f}->{new_blended['fe_time_secs']:.0f}s | "
                f"FE_mfe={old.get('fe_min_mfe_pips', 0):.1f}->{new_blended['fe_min_mfe_pips']:.1f}p | "
                f"Sharpe={optimal.get('sharpe', 0):.3f}"
            )

    def _find_optimal_params(self, trades: List[TradeOutcome]) -> Optional[Dict]:
        best_sharpe = -np.inf
        best_params = None

        for sl_mult in _SL_MULT_GRID:
            for fe_time in _FE_TIME_GRID:
                for fe_mfe in _FE_MFE_GRID:
                    sharpe = self._simulate_sharpe(
                        trades, float(sl_mult), float(fe_time), float(fe_mfe)
                    )
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            "sl_atr_mult": float(sl_mult),
                            "fe_time_secs": float(fe_time),
                            "fe_min_mfe_pips": float(fe_mfe),
                            "sharpe": sharpe,
                        }

        if best_params and best_sharpe > -5.0:
            return best_params
        return None

    def _simulate_sharpe(
        self,
        trades: List[TradeOutcome],
        sl_mult: float,
        fe_time: float,
        fe_mfe: float,
    ) -> float:
        pnls = []
        pip_size = 0.1

        for t in trades:
            atr_pips = t.atr_at_entry / pip_size if t.atr_at_entry > 0 else 15.0
            sim_sl_pips = sl_mult * atr_pips

            if t.mae_pips >= sim_sl_pips:
                pnls.append(-sim_sl_pips)
            elif t.duration_secs >= fe_time and t.mfe_pips < fe_mfe:
                fe_exit_loss = min(t.mae_pips, fe_mfe * 0.5)
                pnls.append(-fe_exit_loss)
            else:
                actual_pnl_pips = t.pnl / pip_size
                pnls.append(actual_pnl_pips)

        if len(pnls) < 5:
            return -np.inf

        arr = np.array(pnls, dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-9:
            return 0.0
        return float(mean / std)

    @staticmethod
    def _ema_blend(old: Dict, new: Dict) -> Dict:
        blended = {}
        for key in ("sl_atr_mult", "fe_time_secs", "fe_min_mfe_pips"):
            old_val = old.get(key, new.get(key, 0))
            new_val = new.get(key, old_val)
            blended[key] = round((1 - EMA_ALPHA) * old_val + EMA_ALPHA * new_val, 2)
        return blended


adaptive_optimizer = AdaptiveOptimizer()