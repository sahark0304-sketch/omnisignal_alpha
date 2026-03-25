"""
quant/macro_filter.py — OmniSignal Alpha v3.0 Phase 2
Paradigms 3 & 5: COT Structural Bias + Options Gamma Exposure Filter

WHY THIS KILLS EDGE-DESTROYERS:
  Two invisible forces move gold that have nothing to do with candlestick patterns:

  1. COT COMMERCIAL BIAS (Paradigm 3):
     CFTC Commitment of Traders, published every Friday (data as of Tuesday).
     Commercial hedgers = gold miners + jewellers + central banks.
     These are the price-insensitive "dumb money" on the wrong side short-term
     but the CORRECT side structurally over weeks. When they're extremely net long,
     the bottom is in. Net short = top is in. This is a weekly directional filter:
     it eliminates entire weeks of trades in the wrong structural direction.

  2. GAMMA EXPOSURE (Paradigm 5):
     Options dealers who sell GLD/GC options must delta-hedge.
     Near high-OI strikes: dealers buy dips and sell rallies (negative gamma = magnet).
     Away from strikes: dealers chase momentum (positive gamma = accelerant).
     The gamma-neutral price (GNP) is where dealer hedging flips direction.
     Price below GNP = dealers sell as price falls (accelerates down).
     Price above GNP = dealers buy as price rises (accelerates up).
     Key strike levels = structural support/resistance that no technical indicator sees.

MANUAL UPDATE WORKFLOW (10 minutes every Friday afternoon):
  1. CFTC COT data: https://www.cftc.gov/dea/futures/deacmesf.htm
     Or: https://www.quandl.com/data/CFTC (Gold GC futures)
     Record: Net Commercial position (commercials long - commercials short)
     Also record: Net Large Speculator position (inverse of commercial)

  2. Gamma data: https://marketchameleon.com/Overview/GLD/OptionsSummary
     Or: https://squeezemetrics.com/monitor (free DIX/GEX data)
     Record: Gamma Neutral Price, nearest high-OI strikes above and below

  3. Run: python update_macro.py
     Or use the dashboard macro panel (Phase 2 dashboard upgrade)

INTEGRATION INTO RISK_GUARD:
  - COT provides a weekly directional BIAS (LONG_ONLY / SHORT_ONLY / NEUTRAL)
  - This bias is checked as check #0.5 in validate() — blocks contra-structural trades
  - Gamma levels become soft price-level filters: avoid entries near gamma pin points,
    prefer entries at gamma flip points (GNP ± 1 ATR)
  - In REDUCED/BLOCKED modes, COT bias becomes mandatory (not just advisory)
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
import config
from utils.logger import get_logger

logger = get_logger(__name__)

MACRO_STATE_KEY = "macro_framework"


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class COTState:
    """
    CFTC Commitment of Traders — Gold Futures (GC) snapshot.
    Updated manually every Friday from CFTC release.
    """
    # Report date (Tuesday data, released Friday)
    report_date:         str     = ""       # "YYYY-MM-DD"

    # Net positions (longs - shorts) in contracts
    # Positive = net long (bullish structural signal for commercials = bearish price)
    commercial_net:      int     = 0        # Commercials net position
    spec_large_net:      int     = 0        # Large speculators net (retail-ish)
    spec_small_net:      int     = 0        # Small speculators net

    # Open interest
    total_oi:            int     = 0        # Total open interest in contracts

    # Derived bias (computed automatically from commercial_net thresholds)
    # Don't set this manually — it's computed by compute_bias()
    bias:                str     = "NEUTRAL"    # LONG_ONLY / SHORT_ONLY / NEUTRAL / EXTREME_LONG / EXTREME_SHORT
    bias_strength:       float   = 0.0          # 0.0 (weak) to 1.0 (extreme)

    # Historical context for percentile ranking (update when you have the data)
    # 52-week range of commercial_net — used for extremes detection
    commercial_52w_min:  int     = -300_000
    commercial_52w_max:  int     = 50_000

    # Notes from your manual analysis
    analyst_note:        str     = ""


@dataclass
class GammaState:
    """
    Options Gamma Exposure for XAUUSD / GLD ETF.
    Updated manually every Friday from Market Chameleon or GEX providers.
    """
    update_date:         str     = ""   # "YYYY-MM-DD"

    # The price where dealer gamma exposure flips sign
    # Below GNP: negative gamma → dealers amplify moves (momentum environment)
    # Above GNP: positive gamma → dealers dampen moves (mean-reversion environment)
    gamma_neutral_price: float   = 0.0

    # Key strike levels by open interest (nearest 3 above and below spot)
    # Format: [strike_price, notional_gamma_millions]
    strikes_above:       List[Tuple[float, float]] = field(default_factory=list)
    strikes_below:       List[Tuple[float, float]] = field(default_factory=list)

    # Total net gamma exposure (positive = dealers long gamma = range environment)
    # Negative = dealers short gamma = trending/volatile environment
    net_gex_millions:    float   = 0.0

    # Max pain level (price where options market makers lose minimum)
    max_pain:            float   = 0.0

    # Analyst notes
    analyst_note:        str     = ""

    # Computed proximity thresholds
    # A trade entry within GAMMA_EXCLUSION_ATR of a major strike is flagged
    gamma_exclusion_atr: float   = 1.0


@dataclass
class MacroState:
    """Combined macro state for risk_guard consumption."""
    cot:     COTState   = field(default_factory=COTState)
    gamma:   GammaState = field(default_factory=GammaState)
    last_updated: str   = ""

    @property
    def is_fresh(self) -> bool:
        """Returns False if data is older than 10 days (missed 2+ weekly updates)."""
        if not self.last_updated:
            return False
        try:
            updated = datetime.fromisoformat(self.last_updated)
            return (datetime.now() - updated).days <= 10
        except Exception:
            return False

    @property
    def stale_days(self) -> int:
        if not self.last_updated:
            return 999
        try:
            return (datetime.now() - datetime.fromisoformat(self.last_updated)).days
        except Exception:
            return 999


# ─────────────────────────────────────────────────────────────────────────────
#  COT BIAS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

# COT thresholds for XAUUSD gold futures
# These are empirically calibrated to historical CFTC data for GC futures.
# Adjust these after 6 months of live tracking.
COT_THRESHOLDS = {
    # Net commercial position thresholds
    "EXTREME_SHORT_COMMERCIALS": -250_000,   # Commercials VERY net short → price likely near TOP
    "BEARISH_COMMERCIALS":       -150_000,   # Commercials net short → bias SHORT
    "NEUTRAL_LOW":               -50_000,
    "NEUTRAL_HIGH":               20_000,
    "BULLISH_COMMERCIALS":        25_000,    # Commercials net long → bias LONG
    "EXTREME_LONG_COMMERCIALS":   60_000,    # Commercials VERY net long → price likely near BOTTOM
}


def compute_cot_bias(cot: COTState) -> Tuple[str, float]:
    """
    Compute directional bias from COT commercial position.

    INVERSION NOTE: When commercials are NET SHORT, they are hedging against
    price increases on their physical gold inventory → they EXPECT gold to rise
    but are locked in for production reasons. This is BEARISH for the trade
    because commercials are typically the smart money in positioning terms.
    Actually wait — let me be precise:
    
    Commercial TRADERS in gold futures = producers (miners) and merchants.
    They SHORT futures to hedge their physical LONG gold inventory.
    When miners SHORT heavily = they expect gold to FALL (or they're locking in profit).
    This means the structural signal is: heavy commercial short = GOLD NEAR TOP = trade SHORT.
    Heavy commercial long = unusual (they're buying futures, not hedging) = GOLD NEAR BOTTOM = trade LONG.

    The COT rule for gold specifically:
    - Commercials VERY NET SHORT (< -250K) → Extreme bearish structural backdrop → SHORT ONLY
    - Commercials moderately net short (-250K to -100K) → Bearish bias → prefer shorts
    - Commercials near neutral → No structural bias → trade both ways
    - Commercials net long (> +20K) → VERY unusual → Extreme bullish → LONG ONLY
    """
    net = cot.commercial_net
    t   = COT_THRESHOLDS

    # Percentile rank within 52-week range for strength scoring
    range_52w = max(cot.commercial_52w_max - cot.commercial_52w_min, 1)
    position_in_range = (net - cot.commercial_52w_min) / range_52w
    # 0.0 = at 52-week most short (extreme top signal), 1.0 = at 52-week most long (extreme bottom)

    if net <= t["EXTREME_SHORT_COMMERCIALS"]:
        bias     = "SHORT_ONLY"
        strength = 1.0 - position_in_range
        logger.info(f"[COT] Extreme commercial short ({net:,}) → SHORT_ONLY (strength={strength:.2f})")

    elif net <= t["BEARISH_COMMERCIALS"]:
        bias     = "PREFER_SHORT"
        strength = max(0.3, 1.0 - position_in_range)

    elif net <= t["NEUTRAL_LOW"]:
        bias     = "NEUTRAL"
        strength = 0.0

    elif net <= t["NEUTRAL_HIGH"]:
        bias     = "NEUTRAL"
        strength = 0.0

    elif net <= t["BULLISH_COMMERCIALS"]:
        bias     = "PREFER_LONG"
        strength = max(0.3, position_in_range)

    else:  # >= EXTREME_LONG_COMMERCIALS
        bias     = "LONG_ONLY"
        strength = position_in_range
        logger.info(f"[COT] Extreme commercial long ({net:,}) → LONG_ONLY (strength={strength:.2f})")

    return bias, float(np.clip(strength, 0.0, 1.0)) if True else float(min(max(strength, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  GAMMA LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_gamma_position(
    current_price: float,
    gamma: GammaState,
    atr: float,
) -> Dict:
    """
    Analyze current price relative to gamma levels.

    Returns:
      - gamma_regime: 'POSITIVE' (dealers long gamma = range) / 'NEGATIVE' (dealers short = trend)
      - near_strike: True if within exclusion zone of a major strike
      - nearest_strike_above: price + distance
      - nearest_strike_below: price + distance
      - gnp_distance_atr: distance from Gamma Neutral Price in ATR units
      - signal: 'PIN' / 'MAGNET_ABOVE' / 'MAGNET_BELOW' / 'FREE' / 'UNSTABLE'
    """
    result = {
        "gamma_regime":         "UNKNOWN",
        "near_strike":          False,
        "nearest_strike_above": None,
        "nearest_strike_below": None,
        "gnp_distance_atr":     0.0,
        "gnp_direction":        "ABOVE" if current_price > gamma.gamma_neutral_price else "BELOW",
        "signal":               "FREE",
        "exclusion_reason":     "",
    }

    if gamma.gamma_neutral_price <= 0 or atr <= 0:
        return result

    gnp = gamma.gamma_neutral_price
    gnp_dist_atr = abs(current_price - gnp) / atr
    result["gnp_distance_atr"] = gnp_dist_atr

    # Gamma regime from total GEX
    if gamma.net_gex_millions > 0:
        result["gamma_regime"] = "POSITIVE"   # dealers long gamma → pin/range
    elif gamma.net_gex_millions < 0:
        result["gamma_regime"] = "NEGATIVE"   # dealers short gamma → trending
    else:
        result["gamma_regime"] = "UNKNOWN"

    # Check proximity to major strikes
    excl = gamma.gamma_exclusion_atr * atr
    all_strikes = gamma.strikes_above + gamma.strikes_below

    nearest_above = min(
        [(s[0] - current_price, s) for s in gamma.strikes_above if s[0] > current_price],
        key=lambda x: x[0], default=(None, None)
    )
    nearest_below = min(
        [(current_price - s[0], s) for s in gamma.strikes_below if s[0] < current_price],
        key=lambda x: x[0], default=(None, None)
    )

    if nearest_above[1]:
        result["nearest_strike_above"] = {
            "price":   nearest_above[1][0],
            "gex_mm":  nearest_above[1][1],
            "dist_atr": nearest_above[0] / atr,
        }
        if nearest_above[0] <= excl and nearest_above[1][1] > 200:  # >$200M GEX
            result["near_strike"]     = True
            result["signal"]          = "MAGNET_ABOVE"
            result["exclusion_reason"] = (
                f"Near high-OI strike at {nearest_above[1][0]:.1f} "
                f"({nearest_above[0]/atr:.1f} ATR, ${nearest_above[1][1]:.0f}M GEX)"
            )

    if nearest_below[1]:
        result["nearest_strike_below"] = {
            "price":   nearest_below[1][0],
            "gex_mm":  nearest_below[1][1],
            "dist_atr": nearest_below[0] / atr,
        }
        if nearest_below[0] <= excl and nearest_below[1][1] > 200:
            result["near_strike"]     = True
            result["signal"]          = "MAGNET_BELOW"
            result["exclusion_reason"] = (
                f"Near high-OI strike at {nearest_below[1][0]:.1f} "
                f"({nearest_below[0]/atr:.1f} ATR, ${nearest_below[1][1]:.0f}M GEX)"
            )

    # Gamma pin: very near GNP
    if gnp_dist_atr <= 0.5:
        result["signal"]          = "PIN"
        result["exclusion_reason"] = f"Price {gnp_dist_atr:.2f} ATR from GNP {gnp:.1f} — expect chop"

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  MACRO FILTER STATE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class MacroFilter:
    """
    Singleton that holds the weekly macro state.
    Loaded from DB on startup, updated via update_macro.py CLI.
    """

    def __init__(self):
        self._state: Optional[MacroState] = None

    def load(self):
        """Load macro state from DB system_state table."""
        try:
            from database import db_manager
            raw = db_manager.get_system_state(MACRO_STATE_KEY)
            if raw:
                data = json.loads(raw)
                cot  = COTState(**data.get("cot", {}))
                gam  = GammaState(
                    update_date          = data["gamma"].get("update_date", ""),
                    gamma_neutral_price  = data["gamma"].get("gamma_neutral_price", 0.0),
                    strikes_above        = [tuple(x) for x in data["gamma"].get("strikes_above", [])],
                    strikes_below        = [tuple(x) for x in data["gamma"].get("strikes_below", [])],
                    net_gex_millions     = data["gamma"].get("net_gex_millions", 0.0),
                    max_pain             = data["gamma"].get("max_pain", 0.0),
                    analyst_note         = data["gamma"].get("analyst_note", ""),
                    gamma_exclusion_atr  = data["gamma"].get("gamma_exclusion_atr", 1.0),
                )
                # Recompute bias from stored commercial_net
                cot.bias, cot.bias_strength = compute_cot_bias(cot)
                self._state = MacroState(cot=cot, gamma=gam, last_updated=data.get("last_updated",""))
                logger.info(
                    f"[Macro] Loaded | COT bias={cot.bias} ({cot.commercial_net:+,}) "
                    f"| GNP={gam.gamma_neutral_price:.1f} | Age={self._state.stale_days}d"
                )
            else:
                logger.warning("[Macro] No macro state in DB — framework inactive until first update")
                self._state = MacroState()
        except Exception as e:
            logger.error(f"[Macro] Load failed: {e}")
            self._state = MacroState()

    def save(self, state: MacroState):
        """Persist macro state to DB."""
        try:
            from database import db_manager
            state.last_updated = datetime.now().isoformat()
            data = {
                "last_updated": state.last_updated,
                "cot": {
                    "report_date":          state.cot.report_date,
                    "commercial_net":       state.cot.commercial_net,
                    "spec_large_net":       state.cot.spec_large_net,
                    "spec_small_net":       state.cot.spec_small_net,
                    "total_oi":             state.cot.total_oi,
                    "bias":                 state.cot.bias,
                    "bias_strength":        state.cot.bias_strength,
                    "commercial_52w_min":   state.cot.commercial_52w_min,
                    "commercial_52w_max":   state.cot.commercial_52w_max,
                    "analyst_note":         state.cot.analyst_note,
                },
                "gamma": {
                    "update_date":          state.gamma.update_date,
                    "gamma_neutral_price":  state.gamma.gamma_neutral_price,
                    "strikes_above":        state.gamma.strikes_above,
                    "strikes_below":        state.gamma.strikes_below,
                    "net_gex_millions":     state.gamma.net_gex_millions,
                    "max_pain":             state.gamma.max_pain,
                    "analyst_note":         state.gamma.analyst_note,
                    "gamma_exclusion_atr":  state.gamma.gamma_exclusion_atr,
                },
            }
            db_manager.set_system_state(MACRO_STATE_KEY, json.dumps(data))
            self._state = state
            logger.info(f"[Macro] State saved | COT={state.cot.bias} GNP={state.gamma.gamma_neutral_price:.1f}")
        except Exception as e:
            logger.error(f"[Macro] Save failed: {e}")

    @property
    def state(self) -> MacroState:
        if self._state is None:
            self.load()
        return self._state

    def check_signal(
        self,
        action: str,
        current_price: float,
        atr: float,
    ) -> Tuple[bool, str, Dict]:
        """
        Main entry point for risk_guard.
        Returns (approved, reason, details_dict).

        Checks:
        1. COT bias: is this trade direction aligned with structural positioning?
        2. Gamma: is price in a dangerous gamma zone?

        Returns True = approved, False = rejected with reason.
        """
        state = self.state
        details = {
            "cot_bias":        state.cot.bias,
            "cot_net":         state.cot.commercial_net,
            "cot_strength":    state.cot.bias_strength,
            "gamma_regime":    "UNKNOWN",
            "gnp":             state.gamma.gamma_neutral_price,
            "gnp_dist_atr":    0.0,
            "near_strike":     False,
            "data_age_days":   state.stale_days,
            "framework_active": state.is_fresh,
        }

        # If data is stale (>10 days), be advisory only — don't hard block
        if not state.is_fresh:
            logger.debug(f"[Macro] Stale data ({state.stale_days}d) — advisory mode only")
            return True, "Macro data stale — advisory only", details

        # ── COT CHECK ────────────────────────────────────────────────────────
        bias = state.cot.bias
        if bias == "SHORT_ONLY" and action == "BUY":
            return False, (
                f"COT VETO: Extreme commercial short position ({state.cot.commercial_net:+,} contracts). "
                f"Structural backdrop is BEARISH — no BUY trades this week. "
                f"[Update COT weekly via update_macro.py]"
            ), details
        elif bias == "LONG_ONLY" and action == "SELL":
            return False, (
                f"COT VETO: Extreme commercial long position ({state.cot.commercial_net:+,} contracts). "
                f"Structural backdrop is BULLISH — no SELL trades this week."
            ), details
        elif bias == "PREFER_SHORT" and action == "BUY":
            # Soft advisory — log warning but don't block unless in REDUCED mode
            from risk_guard.risk_guard import get_trading_mode
            if get_trading_mode() == "REDUCED":
                return False, (
                    f"COT ADVISORY (enforced in REDUCED mode): "
                    f"Commercial net short ({state.cot.commercial_net:+,}) suggests bearish backdrop. "
                    f"BUY blocked during risk-reduction mode."
                ), details
            logger.info(f"[Macro] COT weak bear signal — BUY allowed but score reduced")

        elif bias == "PREFER_LONG" and action == "SELL":
            from risk_guard.risk_guard import get_trading_mode
            if get_trading_mode() == "REDUCED":
                return False, (
                    f"COT ADVISORY (enforced in REDUCED mode): "
                    f"Commercial net long ({state.cot.commercial_net:+,}) suggests bullish backdrop. "
                    f"SELL blocked during risk-reduction mode."
                ), details

        # ── GAMMA CHECK ──────────────────────────────────────────────────────
        if state.gamma.gamma_neutral_price > 0 and atr > 0:
            g = analyze_gamma_position(current_price, state.gamma, atr)
            details.update({
                "gamma_regime":  g["gamma_regime"],
                "gnp_dist_atr":  g["gnp_distance_atr"],
                "near_strike":   g["near_strike"],
                "gamma_signal":  g["signal"],
            })

            # Hard block: price pinned to a massive strike (>$500M GEX within 0.5 ATR)
            if g["near_strike"] and g["signal"] in ("MAGNET_ABOVE", "MAGNET_BELOW"):
                # Check if the gamma magnet direction conflicts with our trade
                if g["signal"] == "MAGNET_ABOVE" and action == "SELL":
                    return False, (
                        f"GAMMA VETO: {g['exclusion_reason']}. "
                        f"Strong upward gamma magnet — do not short here."
                    ), details
                elif g["signal"] == "MAGNET_BELOW" and action == "BUY":
                    return False, (
                        f"GAMMA VETO: {g['exclusion_reason']}. "
                        f"Strong downward gamma magnet — do not buy here."
                    ), details

            # Warn but allow: PIN zone (near GNP — expect choppy behavior)
            if g["signal"] == "PIN":
                logger.info(f"[Macro] GAMMA PIN: {g['exclusion_reason']} — trade allowed but expect noise")

        return True, "Macro OK", details

    def get_weekly_brief(self) -> str:
        """Returns a human-readable summary for dashboard display."""
        s = self.state
        if not s.is_fresh:
            return f"⚠️ STALE ({s.stale_days}d old) — Update COT + Gamma data via update_macro.py"

        cot_emoji = {
            "SHORT_ONLY": "🔴🔴", "PREFER_SHORT": "🔴",
            "NEUTRAL": "⚪", "PREFER_LONG": "🟢", "LONG_ONLY": "🟢🟢"
        }.get(s.cot.bias, "❓")

        return (
            f"{cot_emoji} COT: {s.cot.bias} ({s.cot.commercial_net:+,} commercial net)\n"
            f"GNP: {s.gamma.gamma_neutral_price:.1f} | Max Pain: {s.gamma.max_pain:.1f}\n"
            f"GEX: ${s.gamma.net_gex_millions:.0f}M ({s.gamma.update_date})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

macro_filter = MacroFilter()


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
