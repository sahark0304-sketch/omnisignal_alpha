"""
quant/exposure_guard.py — OmniSignal Alpha v2.0
Pillar 11: Cross-Pair Correlation & Portfolio Overexposure Guard

Three protection layers:
  1. Correlated symbol block — only 1 trade per correlation group at a time
  2. Currency exposure cap — max % equity at risk per base/quote currency
  3. Symbol concentration — max 1 open trade per symbol (configurable)

Uses live MT5 positions + DB to calculate actual dollar risk per currency.
"""

from typing import Dict, List, Optional, Tuple
import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Currency extraction lookup
_SYMBOL_CURRENCIES: Dict[str, List[str]] = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "AUDUSD": ["AUD", "USD"], "NZDUSD": ["NZD", "USD"],
    "USDJPY": ["USD", "JPY"], "USDCHF": ["USD", "CHF"],
    "USDCAD": ["USD", "CAD"], "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"], "GBPJPY": ["GBP", "JPY"],
    "EURAUD": ["EUR", "AUD"], "GBPAUD": ["GBP", "AUD"],
    "XAUUSD": ["XAU", "USD"], "XAGUSD": ["XAG", "USD"],
    "XTIUSD": ["XTI", "USD"], "US30":   ["USD"],
    "NAS100": ["USD"], "BTCUSDT": ["BTC", "USD"],
    "ETHUSDT": ["ETH", "USD"],
}


def _extract_currencies(symbol: str) -> List[str]:
    sym = symbol.upper()
    if sym in _SYMBOL_CURRENCIES:
        return _SYMBOL_CURRENCIES[sym]
    # Generic 6-char FX
    if len(sym) == 6 and sym.isalpha():
        return [sym[:3], sym[3:]]
    if "USD" in sym:
        return ["USD"]
    return [sym[:3]]


def check_exposure(
    new_symbol: str,
    new_action: str,
    new_risk_amount: float,
    account_equity: float,
) -> Tuple[bool, str]:
    """
    Check whether adding a new position would breach any exposure rule.
    Returns (approved: bool, reason: str).

    Called from risk_guard.validate() AFTER lot sizing is known.
    """
    try:
        from mt5_executor.mt5_executor import get_all_positions, get_pip_size, get_pip_value_per_lot
        from database import db_manager

        live_positions = get_all_positions()
        db_trades      = {t["ticket"]: t for t in db_manager.get_open_trades()}
        open_symbols   = {p["symbol"] for p in live_positions}

        # ── Layer 1: Correlated symbol block ────────────────────────────────
        for group in config.CORRELATION_GROUPS:
            if new_symbol in group:
                for open_sym in open_symbols:
                    if open_sym in group and open_sym != new_symbol:
                        return False, f"Correlated position open: {open_sym} in group with {new_symbol}"

        # ── Layer 2: Symbol concentration ────────────────────────────────────
        sym_count = sum(1 for p in live_positions if p["symbol"] == new_symbol)
        if sym_count >= config.MAX_CONCURRENT_PER_SYMBOL:
            return False, f"Max {config.MAX_CONCURRENT_PER_SYMBOL} concurrent positions on {new_symbol}"

        # ── Layer 2b: Aggregate notional cap ─────────────────────────────────
        _max_total = getattr(config, "MAX_TOTAL_LOTS", 0.50)
        _total_open_lots = sum(float(p["volume"]) for p in live_positions)
        if _total_open_lots >= _max_total:
            return False, (
                f"MAX_NOTIONAL_EXPOSURE: total open {_total_open_lots:.2f}L "
                f">= cap {_max_total:.2f}L"
            )

        # ── Layer 3: Currency exposure cap ────────────────────────────────────
        new_currencies = _extract_currencies(new_symbol)
        max_currency_risk = account_equity * (config.MAX_CURRENCY_EXPOSURE_PCT / 100.0)

        # Compute current risk per currency from open positions
        currency_risk: Dict[str, float] = {}
        for pos in live_positions:
            db_t = db_trades.get(pos["ticket"])
            if db_t and db_t.get("sl_price"):
                pip_s = get_pip_size(pos["symbol"])
                pip_v = get_pip_value_per_lot(pos["symbol"])
                sl_dist_pips = abs(pos["price_open"] - db_t["sl_price"]) / pip_s
                pos_risk = pos["volume"] * sl_dist_pips * pip_v
                for ccy in _extract_currencies(pos["symbol"]):
                    currency_risk[ccy] = currency_risk.get(ccy, 0.0) + pos_risk

        for ccy in new_currencies:
            existing_risk = currency_risk.get(ccy, 0.0)
            projected     = existing_risk + new_risk_amount
            if projected > max_currency_risk:
                return False, (
                    f"Currency exposure breach: {ccy} would be "
                    f"${projected:.0f} / limit ${max_currency_risk:.0f} "
                    f"({projected/account_equity*100:.1f}% equity)"
                )

        return True, "Exposure OK"

    except Exception as e:
        # Fail open — don't block on exposure check errors
        logger.warning(f"[Exposure] Check failed (pass-through): {e}")
        return True, "Exposure check unavailable — pass-through"


def get_portfolio_heatmap(account_equity: float) -> Dict[str, float]:
    """
    Returns currency → risk_pct for dashboard heatmap display.
    """
    try:
        from mt5_executor.mt5_executor import get_all_positions, get_pip_size, get_pip_value_per_lot
        from database import db_manager

        positions = get_all_positions()
        db_trades = {t["ticket"]: t for t in db_manager.get_open_trades()}
        heat: Dict[str, float] = {}

        for pos in positions:
            db_t = db_trades.get(pos["ticket"])
            if db_t and db_t.get("sl_price"):
                pip_s = get_pip_size(pos["symbol"])
                pip_v = get_pip_value_per_lot(pos["symbol"])
                sl_pips = abs(pos["price_open"] - db_t["sl_price"]) / pip_s
                risk = pos["volume"] * sl_pips * pip_v
                for ccy in _extract_currencies(pos["symbol"]):
                    heat[ccy] = heat.get(ccy, 0.0) + risk

        return {ccy: (risk / account_equity * 100.0) for ccy, risk in heat.items()}
    except Exception:
        return {}
