"""
quant/alpha_ranker.py  --  OmniSignal Alpha v3.1
Pillar 6: Signal Source Profiling  --  The Alpha Ranker

Architecture:
  Uses Bayesian Beta-Binomial shrinkage to prevent small-sample sources from
  receiving extreme tier assignments.  Raw win rate is replaced with a posterior
  estimate that pulls toward a neutral prior (default 50%) and converges on the
  true rate as sample size grows.

Tier system (based on Bayesian posterior WR):
  S   >=65%  ->  1.50x lots   (elite)
  A   >=55%  ->  1.20x lots   (strong)
  B   >=45%  ->  1.00x lots   (baseline)
  C   >=20%  ->  0.70x lots   (underperforming)
  F   < 20%  ->  0.25x lots   (weak -- reduced, never muted)
  TOXIC      ->  0.00x lots   (hard block -- only with statistically significant toxicity)

TOXIC requires BOTH:
  - posterior WR < ALPHA_TOXIC_WR_THRESHOLD  (default 10%)
  - total trades >= ALPHA_TOXIC_MIN_TRADES   (default 15)

Refreshes from DB every 15 minutes.  Symbol-specific WR used when >= 5 trades.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SourceProfile:
    source: str
    total: int            = 0
    wins: int             = 0
    total_pnl: float      = 0.0
    raw_wr: float         = 0.0
    bayesian_wr: float    = 0.5
    tier: str             = "UNRATED"
    multiplier: float     = 1.0
    is_toxic: bool        = False
    symbol_stats: Dict    = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.bayesian_wr

    @property
    def expectancy(self) -> float:
        return self.total_pnl / max(self.total, 1)


def _bayesian_wr(wins: int, total: int) -> float:
    """Beta-Binomial posterior mean with conjugate prior."""
    alpha = config.ALPHA_BAYESIAN_PRIOR_ALPHA
    beta  = config.ALPHA_BAYESIAN_PRIOR_BETA
    return (wins + alpha) / (total + alpha + beta)


class AlphaRanker:
    def __init__(self):
        self._profiles: Dict[str, SourceProfile] = {}
        self._last_refresh: Optional[datetime] = None

    def get_multiplier(self, source: str, symbol: str = "") -> Tuple[float, str]:
        """Returns (lot_multiplier, tier).  Fails safe to 1.0 if insufficient data."""
        if not config.ALPHA_RANKER_ENABLED:
            return 1.0, "DISABLED"
        self._refresh_if_needed()

        p = self._profiles.get(source)
        if p is None or p.total < config.ALPHA_MIN_TRADES:
            return 1.0, "UNRATED"

        if p.is_toxic:
            return 0.0, "TOXIC"

        if p.tier == "VETOED":
            return 0.0, "VETOED"

        if symbol and symbol in p.symbol_stats:
            ss = p.symbol_stats[symbol]
            if ss.get("total", 0) >= 5:
                bwr = _bayesian_wr(ss["wins"], ss["total"])
                tier, mult = self._classify(bwr)
                logger.debug(
                    f"[Alpha] {source}@{symbol} sym_bwr={bwr:.1%} "
                    f"(raw={ss['wins']}/{ss['total']}) -> {tier} {mult}x"
                )
                return mult, tier

        return p.multiplier, p.tier

    def get_profile(self, source: str) -> Optional[SourceProfile]:
        """Get the full profile for a source (used by AI override for logging)."""
        self._refresh_if_needed()
        return self._profiles.get(source)

    def get_all_profiles(self) -> List[SourceProfile]:
        self._refresh_if_needed()
        return sorted(self._profiles.values(), key=lambda p: -p.total_pnl)

    def _refresh_if_needed(self):
        now = datetime.now()
        if self._last_refresh and (now - self._last_refresh).total_seconds() < 900:
            return
        self._load_from_db()
        self._last_refresh = now

    def _load_from_db(self):
        try:
            from database import db_manager
            cutoff = (datetime.now() - timedelta(days=config.ALPHA_ROLLING_DAYS)).isoformat()
            with db_manager.get_connection() as conn:
                rows = conn.execute("""
                    SELECT s.source, t.symbol,
                           COUNT(t.ticket)                              AS total,
                           SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) AS wins,
                           COALESCE(SUM(t.pnl), 0)                     AS total_pnl
                    FROM trades t
                    JOIN signals s ON t.signal_id = s.id
                    WHERE t.status='CLOSED' AND t.close_time >= ?
                    GROUP BY s.source, t.symbol
                """, (cutoff,)).fetchall()

            src_agg: Dict[str, Dict] = {}
            for r in rows:
                src = r["source"]
                if src not in src_agg:
                    src_agg[src] = {"total": 0, "wins": 0, "pnl": 0.0, "symbols": {}}
                src_agg[src]["total"] += r["total"]
                src_agg[src]["wins"]  += r["wins"]
                src_agg[src]["pnl"]   += r["total_pnl"]
                src_agg[src]["symbols"][r["symbol"]] = {
                    "total": r["total"], "wins": r["wins"]
                }

            new_profiles = {}
            for src, agg in src_agg.items():
                raw_wr = agg["wins"] / max(agg["total"], 1)
                bwr = _bayesian_wr(agg["wins"], agg["total"])

                if agg["total"] >= config.ALPHA_MIN_TRADES:
                    tier, mult = self._classify(bwr)
                else:
                    tier, mult = "UNRATED", 1.0

                is_toxic = (
                    agg["total"] >= config.ALPHA_TOXIC_MIN_TRADES
                    and bwr < config.ALPHA_TOXIC_WR_THRESHOLD
                )

                # v4.4: Hard veto -- block sources with <35% Bayesian WR AND negative PnL
                is_vetoed = (
                    not is_toxic
                    and agg["total"] >= config.ALPHA_VETO_MIN_TRADES
                    and bwr < config.ALPHA_VETO_WR
                    and agg["pnl"] < 0
                )
                if is_vetoed:
                    tier = "VETOED"
                    mult = 0.0
                    logger.warning(
                        "[Alpha] VETO: %s | bayes_wr=%.1f%% pnl=$%.2f over %d trades | "
                        "HARD BLOCK -- negative expectancy confirmed",
                        src, bwr * 100, agg["pnl"], agg["total"],
                    )

                new_profiles[src] = SourceProfile(
                    source=src, total=agg["total"], wins=agg["wins"],
                    total_pnl=agg["pnl"], raw_wr=raw_wr, bayesian_wr=bwr,
                    tier=tier, multiplier=mult,
                    is_toxic=is_toxic, symbol_stats=agg["symbols"],
                )
                logger.debug(
                    f"[Alpha] {src}: raw_wr={raw_wr:.1%} bayes_wr={bwr:.1%} "
                    f"({agg['wins']}/{agg['total']}) -> {tier} {mult}x "
                    f"toxic={is_toxic}"
                )

            self._profiles = new_profiles
            logger.info(f"[Alpha] Loaded {len(new_profiles)} source profiles.")
        except Exception as e:
            logger.warning(f"[Alpha] DB load failed: {e}")

    @staticmethod
    def _classify(win_rate: float) -> Tuple[str, float]:
        if win_rate >= config.ALPHA_S_TIER_WR: return "S", 1.5
        if win_rate >= config.ALPHA_A_TIER_WR: return "A", 1.2
        if win_rate >= config.ALPHA_B_TIER_WR: return "B", 1.0
        if win_rate >= config.ALPHA_C_TIER_WR: return "C", 0.7
        return "F", 0.25


alpha_ranker = AlphaRanker()
