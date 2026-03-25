"""
Regime-Conditional Bayesian Kelly Sizing Engine
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass

import config

logger = logging.getLogger("OmniSignal")


KAPPA: float = getattr(config, "KELLY_FRACTION", 0.30)
MAX_RISK_PCT: float = getattr(config, "KELLY_MAX_RISK_PCT", 2.0)
BASE_RISK_PCT: float = getattr(config, "RISK_PER_TRADE_PCT", 1.5)
MIN_TRADES_FOR_KELLY: int = getattr(config, "KELLY_MIN_TRADES", 30)
MAX_DD_PCT: float = getattr(config, "MAX_DRAWDOWN_LIMIT_PCT", 8.0)
DAMPENER_GAMMA: float = 1.5


@dataclass
class _RegimePosterior:
    """Tracks realized win/loss statistics per regime."""
    alpha: float = 5.0
    beta: float = 4.0
    sum_win_pips: float = 0.0
    sum_loss_pips: float = 0.0
    n_win: int = 0
    n_loss: int = 0

    @property
    def total_trades(self) -> int:
        return self.n_win + self.n_loss

    @property
    def posterior_win_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def avg_win_loss_ratio(self) -> float:
        """b = average win size / average loss size (pips)."""
        avg_w = self.sum_win_pips / max(self.n_win, 1)
        avg_l = self.sum_loss_pips / max(self.n_loss, 1)
        if avg_l <= 0:
            return 1.5
        return max(0.2, avg_w / avg_l)

    def update(self, pnl_pips: float) -> None:
        if pnl_pips >= 0:
            self.alpha += 1
            self.n_win += 1
            self.sum_win_pips += pnl_pips
        else:
            self.beta += 1
            self.n_loss += 1
            self.sum_loss_pips += abs(pnl_pips)


class KellyEngine:
    """Regime-conditional Bayesian Kelly sizer with concave DD dampening."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._posteriors: dict[str, _RegimePosterior] = defaultdict(_RegimePosterior)
        self._global_posterior = _RegimePosterior()

    def record_outcome(self, regime: str, pnl_pips: float) -> None:
        """Called after every trade close to update posteriors."""
        regime_key = (regime or "UNKNOWN").upper()
        with self._lock:
            self._posteriors[regime_key].update(pnl_pips)
            self._global_posterior.update(pnl_pips)

    def compute_multiplier(self, p_win, regime: str, current_equity: float, hwm=None) -> float:
        """Returns a sizing multiplier in [0.10, max_mult]."""
        if hwm is None:
            try:
                from database import db_manager
                hwm = db_manager.get_high_water_mark() or current_equity
            except Exception:
                hwm = current_equity

        regime_key = (regime or "UNKNOWN").upper()
        with self._lock:
            posterior = self._posteriors.get(regime_key, self._global_posterior)
            total = max(posterior.total_trades, self._global_posterior.total_trades)
            b = (posterior.avg_win_loss_ratio
                 if posterior.total_trades >= 5
                 else self._global_posterior.avg_win_loss_ratio)

        if p_win is not None and p_win > 0:
            p = max(0.01, min(p_win, 0.99))
        else:
            p = max(0.01, min(posterior.posterior_win_rate, 0.99))

        q = 1.0 - p
        f_raw = (p * b - q) / b if b > 0 else 0.0

        if f_raw <= 0:
            kelly_risk_pct = 0.0
        else:
            kelly_risk_pct = f_raw * KAPPA * 100.0

        dd_dampener = self._compute_dd_dampener(current_equity, hwm)

        if total < MIN_TRADES_FOR_KELLY:
            blend_weight = total / MIN_TRADES_FOR_KELLY
            kelly_mult_raw = (blend_weight * (kelly_risk_pct / BASE_RISK_PCT)
                              + (1.0 - blend_weight) * 1.0)
        else:
            kelly_mult_raw = (kelly_risk_pct / BASE_RISK_PCT
                              if BASE_RISK_PCT > 0 else 1.0)

        kelly_mult = kelly_mult_raw * dd_dampener

        max_mult = MAX_RISK_PCT / BASE_RISK_PCT if BASE_RISK_PCT > 0 else 1.5
        kelly_mult = max(0.10, min(kelly_mult, max_mult))

        logger.debug(
            f"[Kelly] p={p:.2f} b={b:.2f} f*={f_raw:.4f} kappa={KAPPA:.2f} "
            f"risk%={kelly_risk_pct:.3f} dd_damp={dd_dampener:.3f} "
            f"mult={kelly_mult:.3f} regime={regime_key} "
            f"trades={total}"
        )

        return round(kelly_mult, 4)

    def get_diagnostics(self) -> dict:
        """Return a snapshot for the dashboard / logging."""
        with self._lock:
            diag = {
                "global_trades": self._global_posterior.total_trades,
                "global_posterior_wr": round(self._global_posterior.posterior_win_rate, 3),
                "global_wl_ratio": round(self._global_posterior.avg_win_loss_ratio, 3),
                "regimes": {},
            }
            for regime, post in self._posteriors.items():
                diag["regimes"][regime] = {
                    "trades": post.total_trades,
                    "posterior_wr": round(post.posterior_win_rate, 3),
                    "wl_ratio": round(post.avg_win_loss_ratio, 3),
                }
            return diag

    @staticmethod
    def _compute_dd_dampener(equity: float, hwm: float) -> float:
        """Concave DD dampener: D = ((equity - floor) / (hwm - floor))^gamma."""
        if hwm <= 0:
            return 1.0
        floor = hwm * (1.0 - MAX_DD_PCT / 100.0)
        if equity <= floor:
            return 0.0
        if equity >= hwm:
            return 1.0
        span = hwm - floor
        if span <= 0:
            return 1.0
        ratio = (equity - floor) / span
        return ratio ** DAMPENER_GAMMA


kelly_engine = KellyEngine()
