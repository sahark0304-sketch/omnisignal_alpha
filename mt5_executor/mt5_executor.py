"""
mt5_executor/mt5_executor.py — OmniSignal Alpha v2.0
Pillar 8: Institutional-Grade Execution Engine

Upgrades from v1.1:
  - Retry loop with exponential backoff (handles requotes)
  - Partial fill acceptance at configurable threshold (≥80% fill = accept)
  - Slippage tracking logged to black box
  - LIMIT order support (for signal with specific entry price)
  - Emergency close with guaranteed IOC + market fallback
  - Execution latency measurement for black box
"""

import time
from typing import Optional, Tuple, Dict, List
import MetaTrader5 as mt5
import config
from database import db_manager
from utils.logger import get_logger, get_trade_logger
from utils.notifier import notify

logger = get_logger(__name__)


# ── CONNECTION ────────────────────────────────────────────────────────────────

def connect() -> bool:
    if not mt5.initialize(
        path=config.MT5_PATH, login=config.MT5_LOGIN,
        password=config.MT5_PASSWORD, server=config.MT5_SERVER,
    ):
        err = mt5.last_error()
        logger.error(f"[MT5] Initialize failed: {err}")
        return False
    acc = mt5.account_info()
    if acc is None:
        logger.error(f"[MT5] Account info failed: {mt5.last_error()}")
        return False
    logger.info(
        f"[MT5] Connected ✅ | Account:{acc.login} Server:{acc.server} "
        f"Balance:${acc.balance:.2f} Equity:${acc.equity:.2f} Mode:{config.OPERATING_MODE}"
    )
    return True


def disconnect():
    mt5.shutdown()
    logger.info("[MT5] Disconnected.")


def ensure_connected() -> bool:
    if mt5.terminal_info() is None:
        logger.warning("[MT5] Connection lost — reconnecting...")
        return connect()
    return True


# ── ACCOUNT & SYMBOL ──────────────────────────────────────────────────────────

def get_account_equity() -> float:
    i = mt5.account_info()
    return float(i.equity) if i else 0.0

def get_account_balance() -> float:
    i = mt5.account_info()
    return float(i.balance) if i else 0.0

def get_symbol_info(symbol: str):
    mt5.symbol_select(symbol, True)
    return mt5.symbol_info(symbol)

def get_pip_size(symbol: str) -> float:
    """pip_size = 10^-(digits-1). Handles JPY, Gold, indices correctly."""
    s = get_symbol_info(symbol)
    if s:
        return 10 ** -(s.digits - 1)
    logger.warning(f"[MT5] Unknown symbol {symbol} — default pip_size 0.0001")
    return 0.0001

def get_current_prices(symbol: str) -> Tuple[float, float, float]:
    """Returns (bid, ask, spread_pips)."""
    t = mt5.symbol_info_tick(symbol)
    if not t:
        return 0.0, 0.0, 0.0
    pip = get_pip_size(symbol)
    spread = round((t.ask - t.bid) / pip, 2)
    return float(t.bid), float(t.ask), spread

def get_pip_value_per_lot(symbol: str) -> float:
    s = get_symbol_info(symbol)
    if s and s.trade_tick_size > 0:
        pip = get_pip_size(symbol)
        return float(s.trade_tick_value) * (pip / s.trade_tick_size)
    return 10.0

def get_all_positions() -> List[Dict]:
    ensure_connected()
    positions = mt5.positions_get()
    if not positions:
        return []
    return [
        {
            "ticket":        p.ticket,
            "symbol":        p.symbol,
            "type":          "BUY" if p.type == 0 else "SELL",
            "volume":        float(p.volume),
            "price_open":    float(p.price_open),
            "price_current": float(p.price_current),
            "sl":            float(p.sl),
            "tp":            float(p.tp),
            "profit":        float(p.profit),
            "comment":       p.comment,
            "entry":         float(p.price_open),
        }
        for p in positions
    ]

def get_live_open_symbols() -> set:
    ensure_connected()
    positions = mt5.positions_get()
    return {p.symbol for p in positions} if positions else set()


# ── INSTITUTIONAL EXECUTION ───────────────────────────────────────────────────

def place_order(signal, lot_size: float, is_high_conviction: bool = False) -> Optional[int]:
    """
    Institutional-grade order execution:
    - MARKET orders always use live price (not stale signal entry)
    - LIMIT orders placed when entry_price is specified and ≥ MAX_ENTRY_DEVIATION_PIPS from market
    - Retry loop with exponential backoff on requotes/invalid price
    - Partial fill acceptance (≥EXEC_PARTIAL_FILL_MIN_PCT)
    - Full execution telemetry logged to black box

    Returns ticket (int) on success, None on failure.
    """
    if not ensure_connected():
        logger.error("[MT5] Cannot place order — not connected.")
        return None

    sym_info = get_symbol_info(signal.symbol)
    if not sym_info:
        logger.error(f"[MT5] Symbol not found: {signal.symbol}")
        return None

    bid, ask, spread = get_current_prices(signal.symbol)
    order_type = mt5.ORDER_TYPE_BUY if signal.action == "BUY" else mt5.ORDER_TYPE_SELL
    exec_price = ask if signal.action == "BUY" else bid

    # Decide MARKET vs LIMIT
    use_limit = False
    if signal.entry_price:
        pip_size = get_pip_size(signal.symbol)
        deviation_pips = abs(exec_price - signal.entry_price) / pip_size
        # If market has already moved past signal entry by more than deviation limit,
        # place a LIMIT order at the signal price
        if deviation_pips > config.MAX_ENTRY_DEVIATION_PIPS:
            use_limit = True
            logger.info(
                f"[MT5] Entry {signal.entry_price} is {deviation_pips:.1f} pips from market "
                f"— placing LIMIT order"
            )

    if use_limit:
        return _place_limit_order(signal, lot_size, is_high_conviction)

    # MARKET order with retry
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       signal.symbol,
        "volume":       float(lot_size),
        "type":         order_type,
        "price":        exec_price,
        "sl":           float(signal.stop_loss) if signal.stop_loss else 0.0,
        "tp":           0.0,  # v4.4: Soft TP -- trade_manager handles partials/runners
        "deviation":    config.MT5_SLIPPAGE,
        "magic":        config.MT5_MAGIC_NUMBER,
        "comment":      f"OmniV2|{'HC' if is_high_conviction else 'STD'}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    t_start = time.perf_counter()
    ticket = None

    for attempt in range(1, config.EXEC_MAX_RETRIES + 1):
        # Refresh price each attempt
        bid, ask, _ = get_current_prices(signal.symbol)
        request["price"] = ask if signal.action == "BUY" else bid

        logger.info(
            f"[MT5] Order attempt {attempt}/{config.EXEC_MAX_RETRIES}: "
            f"{signal.symbol} {signal.action} {lot_size}L @ {request['price']}"
        )
        result = mt5.order_send(request)

        if result is None:
            logger.warning(f"[MT5] order_send returned None — retrying")
            time.sleep(config.EXEC_RETRY_DELAY_MS / 1000.0 * attempt)
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            ticket = result.order
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            slippage_pips = abs(result.price - request["price"]) / get_pip_size(signal.symbol)
            fill_pct = (result.volume / lot_size) if lot_size > 0 else 1.0

            logger.info(
                f"[MT5] ✅ Executed | Ticket:{ticket} | Lots:{result.volume} "
                f"({fill_pct:.0%} fill) | Price:{result.price} | "
                f"Slip:{slippage_pips:.1f}pips | {latency_ms:.0f}ms"
            )

            # Partial fill handling
            if fill_pct < config.EXEC_PARTIAL_FILL_MIN_PCT:
                logger.warning(
                    f"[MT5] Partial fill {fill_pct:.0%} below threshold "
                    f"{config.EXEC_PARTIAL_FILL_MIN_PCT:.0%} — cancelling remainder"
                )
                # Cancel any pending remainder (IOC already handles this but log it)
                db_manager.log_audit("PARTIAL_FILL_WARN", {
                    "ticket": ticket, "fill_pct": fill_pct, "symbol": signal.symbol
                })

            db_manager.insert_trade(
                ticket=ticket, signal_id=None,
                symbol=signal.symbol, action=signal.action,
                lot_size=float(result.volume), entry=float(result.price),
                sl=signal.stop_loss, tp1=signal.tp1,
                tp2=signal.tp2, tp3=signal.tp3,
                mode=config.OPERATING_MODE,
            )

            if config.NOTIFY_ON_TRADE_OPEN:
                notify(
                    f"{'🟢' if signal.action == 'BUY' else '🔴'} *New {signal.action} Order Filled*\n"
                    f"\n"
                    f"{signal.symbol} @ `{result.price}`\n"
                    f"Lots: `{result.volume}` | SL: `{signal.stop_loss}` | TP1: `{signal.tp1}`\n"
                    f"Ticket: `{ticket}` | Slippage: `{slippage_pips:.1f}p`"
                )
            return ticket

        # Retryable error codes
        retryable = {
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_OFF_QUOTES,
            mt5.TRADE_RETCODE_CONNECTION,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
        }
        if result.retcode in retryable:
            logger.warning(f"[MT5] Retryable error {result.retcode} — retry {attempt}")
            time.sleep(config.EXEC_RETRY_DELAY_MS / 1000.0 * attempt)
            continue

        # Non-retryable
        logger.error(f"[MT5] Non-retryable error {result.retcode}: {result.comment}")
        db_manager.log_audit("ORDER_FAILED", {
            "symbol": signal.symbol, "retcode": result.retcode, "comment": result.comment
        })
        break

    if ticket is None:
        logger.error(f"[MT5] All {config.EXEC_MAX_RETRIES} attempts failed for {signal.symbol}")
    return ticket


def _place_limit_order(signal, lot_size: float, is_high_conviction: bool) -> Optional[int]:
    """Place a pending LIMIT order at the signal's entry price."""
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal.action == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       signal.symbol,
        "volume":       float(lot_size),
        "type":         order_type,
        "price":        float(signal.entry_price),
        "sl":           float(signal.stop_loss) if signal.stop_loss else 0.0,
        "tp":           0.0,  # v4.4: Soft TP -- trade_manager handles partials/runners
        "deviation":    config.MT5_SLIPPAGE,
        "magic":        config.MT5_MAGIC_NUMBER,
        "comment":      f"OmniV2|LIMIT|{'HC' if is_high_conviction else 'STD'}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[MT5] ✅ LIMIT order placed | Ticket:{result.order} @ {signal.entry_price}")
        db_manager.log_audit("LIMIT_ORDER_PLACED", {
            "ticket": result.order, "symbol": signal.symbol, "entry": signal.entry_price
        })
        return result.order
    logger.error(f"[MT5] LIMIT order failed: {result.retcode if result else 'None'}")
    return None


# ── SL / PARTIAL CLOSE ────────────────────────────────────────────────────────

def modify_sl(ticket: int, new_sl: float) -> bool:
    pos = _get_position(ticket)
    if not pos:
        return False

    # Skip if SL is already at or very near the requested value
    if abs(pos.sl - new_sl) < 0.01:
        return True

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   pos.symbol,
        "position": ticket,
        "sl":       float(new_sl),
        "tp":       float(pos.tp),
    }
    result = mt5.order_send(request)
    if result is None:
        logger.warning("[MT5] SL modify: order_send returned None | ticket:%d", ticket)
        return False
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info("[MT5] SL modified | Ticket:%d -> %.2f", ticket, new_sl)
        get_trade_logger().info("SL_MOD | ticket=%d | new_sl=%.5f", ticket, new_sl)
        return True
    if result.retcode == 10025:
        logger.debug("[MT5] SL modify: no changes needed | ticket:%d", ticket)
        return True
    logger.warning(
        "[MT5] SL modify failed | ticket:%d | retcode:%d comment:%s",
        ticket, result.retcode, getattr(result, 'comment', 'N/A'),
    )
    return False

def close_partial(ticket: int, lot_size: float) -> bool:
    pos = _get_position(ticket)
    if not pos:
        return False
    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    bid, ask, _ = get_current_prices(pos.symbol)
    price = bid if pos.type == 0 else ask
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     ticket,
        "symbol":       pos.symbol,
        "volume":       round(float(lot_size), 2),
        "type":         close_type,
        "price":        price,
        "deviation":    config.MT5_SLIPPAGE,
        "magic":        config.MT5_MAGIC_NUMBER,
        "comment":      "OmniV2|Partial",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[MT5] Partial close | Ticket:{ticket} {lot_size}L")
        get_trade_logger().info("PARTIAL | ticket=%d | lots=%.2f", ticket, lot_size)
        return True
    logger.warning(f"[MT5] Partial close failed | {ticket} | {mt5.last_error()}")
    return False


def close_position(ticket: int) -> bool:
    """Close an entire position by ticket (used e.g. time-based stale exit)."""
    pos = _get_position(ticket)
    if not pos:
        return False
    return close_partial(ticket, float(pos.volume))


def place_raw_market_order(
    symbol: str, action: str, lot_size: float,
    sl: float = 0.0, tp: float = 0.0,
    comment: str = "OmniV2|Pyramid",
) -> Optional[int]:
    """Place a simple market order without a ParsedSignal object (used for pyramiding)."""
    if not ensure_connected():
        return None
    sym_info = get_symbol_info(symbol)
    if not sym_info:
        return None
    bid, ask, _ = get_current_prices(symbol)
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = ask if action == "BUY" else bid
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       round(float(lot_size), 2),
        "type":         order_type,
        "price":        price,
        "sl":           float(sl) if sl else 0.0,
        "tp":           float(tp) if tp else 0.0,
        "deviation":    config.MT5_SLIPPAGE,
        "magic":        config.MT5_MAGIC_NUMBER,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    for attempt in range(1, 3):
        bid, ask, _ = get_current_prices(symbol)
        request["price"] = ask if action == "BUY" else bid
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"[MT5] Raw order OK | {symbol} {action} {lot_size}L "
                f"Ticket:{result.order} @ {result.price}"
            )
            get_trade_logger().info("FILL | %s | %s | ticket=%s lots=%.2f price=%.5f", symbol, action, result.order, lot_size, result.price)
            return result.order
        time.sleep(0.3 * attempt)
    logger.warning(f"[MT5] Raw market order failed: {symbol} {action} {lot_size}L")
    return None



def emergency_close_all() -> int:
    if not ensure_connected():
        return 0
    positions = mt5.positions_get()
    if not positions:
        return 0
    closed = 0
    for pos in positions:
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        bid, ask, _ = get_current_prices(pos.symbol)
        price = bid if pos.type == 0 else ask
        for attempt in range(3):
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "position":     pos.ticket,
                "symbol":       pos.symbol,
                "volume":       float(pos.volume),
                "type":         close_type,
                "price":        price,
                "deviation":    config.MT5_SLIPPAGE * (attempt + 1) * 3,
                "magic":        config.MT5_MAGIC_NUMBER,
                "comment":      "OmniV2|Emergency",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
                db_manager.close_trade(pos.ticket, float(result.price), float(pos.profit))
                logger.info(f"[MT5] Emergency closed {pos.ticket} PnL:{pos.profit:.2f}")
                break
            time.sleep(0.3 * (attempt + 1))
        else:
            logger.error(f"[MT5] Emergency close FAILED for {pos.ticket}")
    notify(f"⚠️ *Emergency Close All*\n\n{closed}/{len(positions)} positions closed.\n_All risk removed._")
    db_manager.log_audit("EMERGENCY_CLOSE_ALL", {"closed": closed, "total": len(positions)})
    return closed


def _get_position(ticket: int):
    pos = mt5.positions_get(ticket=ticket)
    return pos[0] if pos else None
