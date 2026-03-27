"""
main.py — OmniSignal Alpha v6.3.1
RESTORED March 17 Architecture + Bug 1 Fix (Lot Ceiling)

v6.3.1: This is the March 17 system that produced +$310/day with the
following fixes:
  + Bug 1 FIX: Post-risk boosts cannot exceed 2x of risk_guard output
    (was: dampened 0.02 × amp × convex = 0.08 — ceiling violated)
  + Bug 2 FIX: created_at column in market_features (run migration)
  + Bug 3 FIX: reset_halt.py writes to daily_snapshots (run reset_halt)
  + Anti-hedge in risk_guard.validate() prevents BUY+SELL same symbol
  + Dampener floor in risk_guard prevents lots crushed to 0.01
"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from utils.logger import get_logger
from database import db_manager
from ingestion.signal_queue import pull, done, RawSignal
from ingestion.telegram_listener import run_telegram_listener, disconnect as tg_disconnect
from ingestion.discord_listener import run_discord_listener
from ai_engine.ai_engine import (
    parse_text_signal, analyze_chart_image,
    ParsedSignal, load_prompt_corrections,
)
from ai_engine.consensus_engine import consensus_engine
from risk_guard import risk_guard
from mt5_executor import mt5_executor
from trade_manager import trade_manager
from quant.black_box import init_black_box, DecisionTrace
from quant.latency_monitor import run_latency_monitor
from quant.self_correction import self_correction, post_trade_forensic
from quant.confluence_engine import check_confluence, get_current_session
from quant.liquidity_scanner import liquidity_scanner
from quant.momentum_scanner import momentum_scanner
from quant.tick_flow import tick_flow_engine
from quant.catcd_engine import catcd_engine
from quant.mean_reversion_engine import mr_engine
from quant.convergence_engine import convergence_engine
from quant.breakout_guard import breakout_guard
from quant.htf_filter import register_execution, get_current_toxicity, toxicity_monitor
from quant.convexity_engine import compute_convexity_boost, compute_institutional_scaling
from quant.kelly_engine import kelly_engine
from quant.retry_queue import retry_queue
from quant.signal_amplifier import signal_amplifier
from quant.macro_collector import update_macro_automatically
from quant.win_model import win_model
from quant.feature_engineering import compute_features, compute_dynamic_sl_tp
from quant.regime_detector import regime_detector
from quant.macro_filter import macro_filter
from recovery.state_recovery import reconcile_on_startup, run_heartbeat
from quant.smc_scanner import smc_scanner
from quant.shadow_ledger import shadow_ledger
from quant.breakout_hunter import breakout_hunter
from quant.prop_firm_finisher import prop_firm_finisher
import config

try:
    from quant.trade_orchestrator import trade_orchestrator, run_orchestrator_monitor
    _ato_available = True
except ImportError:
    _ato_available = False

logger = get_logger("main")


async def _supervise(name: str, coro_fn, restart_delay: int = 15):
    while True:
        try:
            await coro_fn()
        except asyncio.CancelledError:
            logger.info(f"[Supervisor] {name} cancelled.")
            break
        except Exception as e:
            logger.error(f"[Supervisor] Task '{name}' CRASHED: {e}. Restarting in {restart_delay}s...", exc_info=True)
            try:
                from utils.notifier import notify
                notify(f"🔴 *OmniSignal TASK CRASH*\n\nTask: {name}\nError: {str(e)[:200]}\n\nAuto-restarting in {restart_delay}s...")
            except Exception:
                pass
            await asyncio.sleep(restart_delay)


_pending_labels: dict = {}


async def equity_snapshot_loop():
    while True:
        try:
            loop   = asyncio.get_event_loop()
            equity = await loop.run_in_executor(None, mt5_executor.get_account_equity)
            balance= await loop.run_in_executor(None, mt5_executor.get_account_balance)
            open_cnt = len(await loop.run_in_executor(None, mt5_executor.get_all_positions))
            daily_pnl = db_manager.get_daily_pnl()
            db_manager.insert_equity_snapshot(equity, balance, open_cnt, daily_pnl)
            db_manager.update_high_water_mark(equity)
        except Exception as e:
            logger.debug(f"[Main] Equity snapshot error: {e}")
        await asyncio.sleep(300)


async def _ensure_opening_equity_set():
    today_key = db_manager._get_daily_key()
    existing  = db_manager.get_opening_equity(today_key)
    if existing is None:
        loop   = asyncio.get_event_loop()
        equity = await loop.run_in_executor(None, mt5_executor.get_account_equity)
        db_manager.set_opening_equity(equity, today_key)
        logger.info(f"[Main] Opening equity recorded: ${equity:,.2f} ({today_key})")
    else:
        logger.info(f"[Main] Opening equity already set: ${existing:,.2f} ({today_key})")


def register_pending_label(
    trade_ticket: int, feature_row_id: int, signal_id: int,
    entry: float, tp1: float, sl: float, symbol: str = "", action: str = "",
):
    if feature_row_id > 0:
        _pending_labels[trade_ticket] = {
            "feature_row_id": feature_row_id, "signal_id": signal_id,
            "entry": entry, "tp1": tp1, "sl": sl,
            "symbol": symbol, "action": action,
            "open_bar_ts": datetime.now(),
        }


def backfill_trade_label(trade_ticket: int, close_price: float, pnl: float, tp1_hit: bool, pip_size: float):
    pending = _pending_labels.pop(trade_ticket, None)
    if pending is None:
        return
    feature_row_id = pending["feature_row_id"]
    signal_id      = pending["signal_id"]
    entry          = pending["entry"] or close_price
    tp1            = pending["tp1"]
    sl             = pending["sl"] or close_price
    open_ts        = pending["open_bar_ts"]
    sl_hit     = not tp1_hit
    pips_pnl   = (close_price - entry) / pip_size if pip_size > 0 else 0.0
    bars_held  = int((datetime.now() - open_ts).total_seconds() / (15 * 60))
    risk_pips  = abs(entry - sl) / pip_size if (sl and pip_size > 0) else 1.0
    pnl_r      = pips_pnl / risk_pips if risk_pips > 0 else 0.0
    db_manager.backfill_feature_label(
        feature_row_id=feature_row_id, signal_id=signal_id,
        trade_ticket=trade_ticket, tp1_hit=tp1_hit, sl_hit=sl_hit,
        pips_pnl=pips_pnl, bars_held=bars_held, pnl_r=pnl_r,
    )
    logger.debug(
        f"[Main] Label backfilled | Ticket:{trade_ticket} "
        f"TP1={tp1_hit} R={pnl_r:.2f} pips={pips_pnl:.1f}"
    )
    symbol = pending.get("symbol", "")
    action = pending.get("action", "")
    if symbol and action:
        asyncio.create_task(post_trade_forensic(
            trade_ticket, pnl, tp1_hit, entry, close_price, symbol, action,
        ))


async def handle_management_action(signal: ParsedSignal):
    from mt5_executor.mt5_executor import get_all_positions, modify_sl, close_partial
    positions = get_all_positions()
    matching  = [p for p in positions if (not signal.symbol or p["symbol"] == signal.symbol)]
    if not matching:
        logger.info(f"[Main] MGMT {signal.action} — no matching positions for {signal.symbol}")
        return
    if signal.action == "CLOSE":
        for pos in matching:
            close_partial(pos["ticket"], pos["volume"])
            logger.info(f"[Main] CLOSE: ticket {pos['ticket']}")
    elif signal.action == "UPDATE" and signal.stop_loss:
        for pos in matching:
            modify_sl(pos["ticket"], signal.stop_loss)
            logger.info(f"[Main] UPDATE SL: ticket {pos['ticket']} → {signal.stop_loss}")
    elif signal.action == "CANCEL":
        import MetaTrader5 as mt5
        orders = mt5.orders_get(symbol=signal.symbol)
        if orders:
            for order in orders:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
                logger.info(f"[Main] CANCEL pending order {order.ticket}")
    db_manager.log_audit(f"MGMT_{signal.action}", {
        "symbol": signal.symbol, "source": signal.raw_source
    })


# ─────────────────────────────────────────────────────────────────────────────
#  CORE SIGNAL PROCESSOR — Same as March 17 + Bug 1 lot ceiling fix
# ─────────────────────────────────────────────────────────────────────────────

async def process_signal(raw: RawSignal):

    if datetime.now() - raw.received_at > timedelta(minutes=config.SIGNAL_EXPIRY_MINUTES):
        return

    if raw.content and db_manager.is_duplicate_message(raw.source, raw.content):
        return
    if raw.content:
        db_manager.mark_message_seen(raw.source, raw.content)

    logger.info(f"[Main] Processing: '{raw.content[:80]}' from {raw.source}")

    trace = DecisionTrace(source=raw.source, raw_message=raw.content or "")

    signal: ParsedSignal = await parse_text_signal(raw.content, raw.source)
    trace.set_ai_gate(signal.action if signal else "NOISE")

    if signal is None:
        trace.set_execution("NOISE")
        trace.save()
        return

    if signal.is_management_action:
        await handle_management_action(signal)
        trace.set_execution(f"MGMT_{signal.action}")
        trace.save()
        return

    if raw.image_bytes:
        signal = await analyze_chart_image(raw.image_bytes, signal)

    trace.set_ai_parse(signal)

    session = get_current_session()

    signal_id = db_manager.insert_signal(
        source=raw.source, raw_text=raw.content,
        parsed={
            "symbol": signal.symbol, "action": signal.action,
            "entry": signal.entry_price, "sl": signal.stop_loss,
            "tp1": signal.tp1, "tp2": signal.tp2, "tp3": signal.tp3,
        },
        confidence=signal.confidence,
        session=session,
    )

    if not signal.is_valid:
        db_manager.update_signal_status(signal_id, "REJECTED", signal.reject_reason)
        trace.set_execution("REJECTED")
        trace.save()
        return

    entry_for_amp = signal.entry_price or 0.0
    signal_amplifier.register_signal(
        source=raw.source, action=signal.action,
        symbol=signal.symbol, entry=entry_for_amp,
        metadata={"confidence": signal.confidence, "signal_id": signal_id},
    )

    is_high_conviction = consensus_engine.add_and_check(signal)
    if is_high_conviction:
        db_manager.update_signal_conviction(signal_id, True)

    loop = asyncio.get_event_loop()
    bid, ask, spread = await loop.run_in_executor(
        None, lambda: mt5_executor.get_current_prices(signal.symbol)
    )
    if bid == 0 and ask == 0:
        db_manager.update_signal_status(signal_id, "REJECTED", "MT5 price unavailable")
        trace.set_execution("REJECTED")
        trace.save()
        return

    equity       = await loop.run_in_executor(None, mt5_executor.get_account_equity)
    pip_size     = await loop.run_in_executor(None, lambda: mt5_executor.get_pip_size(signal.symbol))
    pip_value    = await loop.run_in_executor(None, lambda: mt5_executor.get_pip_value_per_lot(signal.symbol))

    entry_price = signal.entry_price or (ask if signal.action == "BUY" else bid)
    confluence = await check_confluence(
        symbol             = signal.symbol,
        action             = signal.action,
        entry_price        = entry_price,
        timeframe          = config.CONFLUENCE_TIMEFRAME,
        current_spread_pips = spread,
        signal_id          = signal_id,
    )
    db_manager.set_system_state(
        f"hurst_{signal.symbol}", str(round(confluence.hurst_50, 4))
    )

    approved, reason, order = await risk_guard.validate(
        signal             = signal,
        current_bid        = bid,
        current_ask        = ask,
        current_spread_pips = spread,
        account_equity     = equity,
        is_high_conviction = is_high_conviction,
        pip_size           = pip_size,
        pip_value_per_lot  = pip_value,
        confluence_result  = confluence,
        trace              = trace,
    )

    if not approved:
        db_manager.update_signal_status(signal_id, "REJECTED", reason)
        trace.set_execution("REJECTED")
        trace.save()
        if _ato_available:
            _stage = reason.split(":")[0][:30] if ":" in reason else reason[:30]
            trade_orchestrator.record_rejection(
                stage=_stage, source=raw.source,
                symbol=signal.symbol, action=signal.action,
                confidence=signal.confidence,
            )
        return

    # ── Signal Amplifier + Institutional Scaling ──────────────────────────────
    # v6.3.1 Bug 1 FIX: Store risk_guard output as ceiling.
    # Post-risk boosts can amplify up to 2x of this ceiling, not unlimited.
    _riskguard_lot_ceiling = order.lot_size
    _dd_mode = getattr(order, 'dd_mode', 'NORMAL')
    _dampeners_active = _dd_mode == "REDUCED" or _riskguard_lot_ceiling < 0.04

    amp_boost, amp_n, amp_sources = signal_amplifier.get_confluence_boost(
        action=signal.action, symbol=signal.symbol
    )

    has_convergence = signal.raw_source == "AUTO_CONVERGENCE" or amp_n >= 2
    consensus_score = 0
    try:
        from quant.convergence_engine import convergence_engine
        cs = convergence_engine.get_consensus_score()
        consensus_score = cs.get("score", 0)
    except Exception:
        pass

    convex_boost, convex_reason = compute_institutional_scaling(
        signal_confidence=signal.confidence,
        current_equity=equity,
        source=signal.raw_source,
        has_convergence=has_convergence,
        consensus_score=consensus_score,
    )

    # v6.3.1 Bug 1 FIX: Block ALL boosts when dampeners are active
    if _dampeners_active:
        logger.info(
            "[Main] BOOST BLOCKED: dampeners active (lots=%.2f dd=%s) "
            "— amp=%.1fx convex=%.2fx suppressed",
            _riskguard_lot_ceiling, _dd_mode, amp_boost, convex_boost,
        )
    else:
        combined_boost = amp_boost * convex_boost
        if combined_boost > 1.0:
            # Cap at 2x of risk_guard output — prevents ceiling violation
            _boosted = round(min(order.lot_size * combined_boost, _riskguard_lot_ceiling * 2.0), 2)
            order.lot_size = _boosted
            order.alpha_multiplier *= combined_boost
            logger.info(
                "[Main] v4.2 Scaling: amp=%.1fx convex=%.2fx combined=%.2fx "
                "consensus=%d | lots=%.2f (cap=%.2f)",
                amp_boost, convex_boost, combined_boost,
                consensus_score, order.lot_size, _riskguard_lot_ceiling * 2.0,
            )

    # ── PRE-REGISTER DEDUP ────────────────────────────────────────────────────
    register_execution(signal.symbol, signal.action)

    # ── Execute ───────────────────────────────────────────────────────────────
    t_exec = time.perf_counter()
    ticket = mt5_executor.place_order(
        signal             = order.signal,
        lot_size           = order.lot_size,
        is_high_conviction = order.is_high_conviction,
    )
    exec_latency_ms = (time.perf_counter() - t_exec) * 1000.0

    if ticket:
        db_manager.update_signal_status(signal_id, "EXECUTED", trade_ticket=ticket)
        with db_manager.get_connection() as conn:
            conn.execute(
                "UPDATE trades SET signal_id=? WHERE ticket=?",
                (signal_id, ticket)
            )

        register_pending_label(
            trade_ticket   = ticket,
            feature_row_id = confluence.feature_row_id,
            signal_id      = signal_id,
            entry          = entry_price,
            tp1            = signal.tp1,
            sl             = signal.stop_loss,
            symbol         = signal.symbol,
            action         = signal.action,
        )

        if confluence.feature_row_id > 0:
            with db_manager.get_connection() as conn:
                conn.execute(
                    "UPDATE market_features SET trade_ticket=? WHERE id=?",
                    (ticket, confluence.feature_row_id)
                )

        _, exec_ask, _ = mt5_executor.get_current_prices(signal.symbol)
        trace.set_execution(
            "EXECUTED", ticket=ticket,
            exec_price=exec_ask,
            latency_ms=exec_latency_ms,
            fill_pct=1.0,
        )
        logger.info(
            f"[Main] ✅ Ticket:{ticket} | {signal.symbol} {signal.action} "
            f"{order.lot_size}L | {exec_latency_ms:.0f}ms | "
            f"H={confluence.hurst_50:.3f} Session:{confluence.session} "
            f"DD:{order.dd_mode}"
        )

        from utils.notifier import notify
        tox = get_current_toxicity(signal.symbol)
        tox_tag = f" | Tox:{tox['score']:.0%}" if tox.get("score", 0) > 0.3 else ""
        buy_sell_icon = "\U0001f7e2" if signal.action == "BUY" else "\U0001f534"
        notify(
            f"{buy_sell_icon} *New {signal.action} Trade Opened*\n"
            f"\n"
            f"{signal.symbol} @ `{entry_price:.2f}`\n"
            f"Lots: `{order.lot_size}` | Confidence: `{signal.confidence}/10`\n"
            f"SL: `{signal.stop_loss:.2f}` | TP1: `{signal.tp1:.2f}`\n"
            f"\n"
            f"Source: `{signal.raw_source}` | Tier: `{order.alpha_tier}`{tox_tag}"
        )

        if _ato_available:
            trade_orchestrator.record_execution(
                source=raw.source, symbol=signal.symbol,
                action=signal.action, lot_size=order.lot_size, ticket=ticket,
            )
    else:
        db_manager.update_signal_status(signal_id, "REJECTED", "MT5 order_send failed")
        trace.set_execution("REJECTED")

    trace.save()


async def queue_consumer():
    logger.info("[Main] Queue consumer started.")
    while True:
        try:
            raw: RawSignal = await pull()
            await process_signal(raw)
        except Exception as e:
            logger.error(f"[Main] Consumer error: {e}", exc_info=True)
            try:
                from utils.notifier import notify
                notify(f"🔴 *OmniSignal ERROR*\n\nConsumer crashed: {str(e)[:200]}")
            except Exception:
                pass
        finally:
            done()


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP — Same as March 17
# ─────────────────────────────────────────────────────────────────────────────

async def daily_sentiment_loop():
    from utils.notifier import notify
    import MetaTrader5 as mt5

    while True:
        try:
            await asyncio.sleep(14400)
            equity = mt5_executor.get_account_equity()
            balance = mt5_executor.get_account_balance()
            opening_eq = db_manager.get_opening_equity() or balance
            daily_pnl = equity - opening_eq
            daily_pct = (daily_pnl / opening_eq * 100) if opening_eq > 0 else 0

            import sqlite3
            db = sqlite3.connect("data/omnisignal.db")
            db.row_factory = sqlite3.Row
            from datetime import date
            today = date.today().isoformat()
            stats = db.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                       SUM(pnl) as total_pnl, MAX(pnl) as best, MIN(pnl) as worst
                FROM trades
                WHERE CAST(open_time AS TEXT) LIKE ? AND status = 'CLOSED'
            """, (today + "%",)).fetchone()
            open_count = len(db_manager.get_open_trades())
            total = stats["total"] or 0
            wins = stats["wins"] or 0
            losses = stats["losses"] or 0
            wr = (wins / total * 100) if total > 0 else 0
            best = stats["best"] or 0
            worst = stats["worst"] or 0
            net = stats["total_pnl"] or 0
            if daily_pnl > 0:
                mood = "\U0001f7e2 BULLISH"
            elif daily_pnl < -10:
                mood = "\U0001f534 BEARISH"
            else:
                mood = "\U0001f7e1 NEUTRAL"
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            notify(
                f"\U0001f4ca *Daily Summary*\n"
                f"_{now.strftime('%Y-%m-%d %H:%M UTC')}_\n\n"
                f"{mood}\n\n"
                f"\U0001f4b0 *Account*\n"
                f"Equity: `${equity:,.2f}`\n"
                f"Daily P&L: `${daily_pnl:+.2f}` ({daily_pct:+.1f}%)\n\n"
                f"\U0001f4c8 *Trades*\n"
                f"Total: `{total}` ({wins}W / {losses}L)\n"
                f"Win Rate: `{wr:.0f}%`\n"
                f"Best: `${best:+.2f}` | Worst: `${worst:+.2f}`\n"
                f"Net: `${net:+.2f}`\n\n"
                f"Open Positions: `{open_count}`"
            )
            db.close()
        except Exception as e:
            logger.debug(f"[Main] Sentiment error: {e}")


async def startup():
    import os
    os.makedirs("data", exist_ok=True)

    logger.info("=" * 65)
    logger.info("  OmniSignal Alpha v6.3.1 — RESTORED March 17 + Bug Fixes")
    logger.info(f"  Mode: {config.OPERATING_MODE} | Phase: {config.PROP_FIRM_PHASE}")
    logger.info(f"  Daily DD limit:  {config.DAILY_DRAWDOWN_LIMIT_PCT}% of opening equity")
    logger.info(f"  Max DD limit:    {config.MAX_DRAWDOWN_LIMIT_PCT}% from HWM")
    logger.info(f"  Initial balance: ${config.INITIAL_ACCOUNT_BALANCE:,.0f}")
    logger.info(f"  ML features:     {'ON' if config.ML_FEATURE_RECORDING_ENABLED else 'OFF'}")
    logger.info(f"  Anti-Hedge:      {'ON' if getattr(config, 'ANTI_HEDGE_ENABLED', False) else 'OFF'}")
    logger.info(f"  Dampener Floor:  {getattr(config, 'DAMPENER_LOT_FLOOR', 0.03)}")
    logger.info(f"  Scanners:        {'ENABLED' if getattr(config, 'SCANNER_SIGNALS_ENABLED', True) else 'DISABLED'}")
    logger.info("=" * 65)

    db_manager.init_db()
    init_black_box()
    risk_guard.sync_halt_from_db()
    load_prompt_corrections()

    if not mt5_executor.connect():
        logger.critical("[Main] MT5 connection failed. Exiting.")
        try:
            from utils.notifier import notify
            notify("🆘 *OmniSignal CRITICAL*\n\nMT5 connection FAILED. Bot has exited.")
        except Exception:
            pass
        sys.exit(1)

    await _ensure_opening_equity_set()

    if db_manager.get_high_water_mark() is None:
        eq = mt5_executor.get_account_equity()
        db_manager.update_high_water_mark(max(eq, config.INITIAL_ACCOUNT_BALANCE))
        logger.info(f"[Main] HWM initialized at ${max(eq, config.INITIAL_ACCOUNT_BALANCE):,.2f}")

    if config.RECOVERY_ENABLED:
        report = reconcile_on_startup()
        if report.get("crash_detected"):
            logger.warning(f"[Main] Recovery: {report}")

    trade_manager.set_close_callback(backfill_trade_label)

    try:
        await update_macro_automatically()
    except Exception as e:
        logger.warning(f"[Main] Macro data collection skipped: {e}")

    # ALL tasks — same as March 17
    tasks = [
        asyncio.create_task(run_telegram_listener(),                                        name="telegram"),
        asyncio.create_task(run_discord_listener(),                                         name="discord"),
        asyncio.create_task(queue_consumer(),                                               name="queue"),
        asyncio.create_task(trade_manager.run_management_loop(),                            name="trade_mgr"),
        asyncio.create_task(run_heartbeat(),                                                name="heartbeat"),
        asyncio.create_task(equity_snapshot_loop(),                                         name="equity_snap"),
        asyncio.create_task(risk_guard.continuous_equity_monitor(),                          name="equity_monitor"),
        asyncio.create_task(risk_guard.daily_reset_watcher(),                               name="daily_reset"),
        asyncio.create_task(_supervise("liquidity_scanner",  liquidity_scanner.run),         name="liquidity_scanner"),
        asyncio.create_task(_supervise("momentum_scanner",   momentum_scanner.run),          name="momentum_scanner"),
        asyncio.create_task(_supervise("tfi_engine",         tick_flow_engine.run),           name="tfi_engine"),
        asyncio.create_task(_supervise("catcd",              catcd_engine.run),               name="catcd"),
        asyncio.create_task(_supervise("mr_engine",          mr_engine.run),                  name="mr_engine"),
        asyncio.create_task(_supervise("convergence",        convergence_engine.run),          name="convergence"),
        asyncio.create_task(_supervise("breakout_guard",     breakout_guard.run),              name="breakout_guard"),
        asyncio.create_task(_supervise("smc_scanner",        smc_scanner.run),                 name="smc_scanner"),
        asyncio.create_task(_supervise("shadow_ledger",      shadow_ledger.run_monitor),       name="shadow_ledger"),
        asyncio.create_task(_supervise("toxicity_monitor",  toxicity_monitor.run),            name="toxicity_monitor"),
        asyncio.create_task(_supervise("retry_queue",        retry_queue.run),                name="retry_queue"),
        asyncio.create_task(daily_sentiment_loop(),                                         name="sentiment"),
        asyncio.create_task(nightly_optimization_loop(),                                      name="nightly_ml"),
        asyncio.create_task(macro_update_loop(),                                              name="macro_update"),
    ]

    if config.LATENCY_ENABLED:
        tasks.append(asyncio.create_task(run_latency_monitor(), name="latency"))

    if config.SELF_CORRECTION_ENABLED:
        tasks.append(asyncio.create_task(
            self_correction.run_review_loop(), name="self_correction"
        ))

    from quant.amd_engine import amd_engine
    tasks.append(asyncio.create_task(amd_engine.run()))

    if config.ML_DOLLAR_BARS_ENABLED:
        from quant.dollar_bar_engine import get_engine as get_dollar_engine
        dollar_engine = get_dollar_engine("XAUUSD")
        tasks.append(asyncio.create_task(
            _supervise("dollar_bars", dollar_engine.run), name="dollar_bars"
        ))

    # v8.0: London Open Breakout scanner
    try:
        from quant.london_sniper import london_sniper
        tasks.append(asyncio.create_task(
            _supervise("london_sniper", london_sniper.run), name="london_sniper"
        ))
    except Exception as _ls_err:
        logger.warning("[Main] London Sniper not loaded: %s", _ls_err)

    # v8.0: DXY Canary Engine
    try:
        from quant.canary_engine import canary_engine
        tasks.append(asyncio.create_task(
            _supervise("canary_engine", canary_engine.run), name="canary_engine"
        ))
    except Exception as _ce_err:
        logger.warning("[Main] Canary Engine not loaded: %s", _ce_err)

    # v8.2: Breakout Hunter
    try:
        tasks.append(asyncio.create_task(
            _supervise("breakout_hunter", breakout_hunter.run), name="breakout_hunter"
        ))
    except Exception as _bh_err:
        logger.warning("[Main] Breakout Hunter not loaded: %s", _bh_err)

    if _ato_available and getattr(config, "ATO_ENABLED", True):
        tasks.append(asyncio.create_task(run_orchestrator_monitor(), name="ato_monitor"))


    # v7.1: What-if ledger backfill loop
    async def what_if_backfill_loop():
        """Fill in future prices for rejected signals."""
        while True:
            try:
                await asyncio.sleep(300)
                try:
                    import MetaTrader5 as _wif_mt5
                    from datetime import datetime as _wif_dt
                    with db_manager.get_connection() as _wif_conn:
                        _wif_rows = _wif_conn.execute(
                            "SELECT id, symbol, rejected_at, entry_price, sl, tp1, action "
                            "FROM what_if_ledger "
                            "WHERE price_15min IS NULL "
                            "AND rejected_at < datetime('now', '-16 minutes') "
                            "LIMIT 20"
                        ).fetchall()
                        for _wif_row in _wif_rows:
                            try:
                                _wid = _wif_row[0]
                                _wsym = _wif_row[1]
                                _wrej = _wif_row[2]
                                _wentry = _wif_row[3]
                                _wsl = _wif_row[4]
                                _wtp1 = _wif_row[5]
                                _wact = _wif_row[6]
                                _wrates = _wif_mt5.copy_rates_from(
                                    _wsym, _wif_mt5.TIMEFRAME_M1,
                                    _wif_dt.fromisoformat(_wrej), 65
                                )
                                if _wrates is not None and len(_wrates) >= 15:
                                    _wp5 = float(_wrates[4]["close"]) if len(_wrates) > 4 else None
                                    _wp15 = float(_wrates[14]["close"]) if len(_wrates) > 14 else None
                                    _wp30 = float(_wrates[29]["close"]) if len(_wrates) > 29 else None
                                    _wp60 = float(_wrates[59]["close"]) if len(_wrates) > 59 else None
                                    _wtp = 0
                                    _wslh = 0
                                    _wvpnl = 0.0
                                    if _wtp1 and _wsl and _wentry:
                                        for _wr in _wrates[:60]:
                                            _wc = float(_wr["close"])
                                            if _wact == "BUY":
                                                if _wc >= _wtp1: _wtp = 1; break
                                                if _wc <= _wsl: _wslh = 1; break
                                            else:
                                                if _wc <= _wtp1: _wtp = 1; break
                                                if _wc >= _wsl: _wslh = 1; break
                                        if _wtp: _wvpnl = abs(_wtp1 - _wentry) * 100
                                        elif _wslh: _wvpnl = -abs(_wsl - _wentry) * 100
                                    _wif_conn.execute(
                                        "UPDATE what_if_ledger SET "
                                        "price_5min=?, price_15min=?, price_30min=?, price_60min=?, "
                                        "would_hit_tp=?, would_hit_sl=?, virtual_pnl=? WHERE id=?",
                                        (_wp5, _wp15, _wp30, _wp60, _wtp, _wslh, _wvpnl, _wid)
                                    )
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception as _wif_err:
                logger.debug("[Main] What-if backfill error: %s", _wif_err)

    tasks.append(asyncio.create_task(what_if_backfill_loop(), name="what_if"))

    logger.info(f"[Main] {len(tasks)} tasks launched. System is live.")
    try:
        from utils.notifier import notify
        _eq = mt5_executor.get_account_equity()
        notify(
            f"🟢 *OmniSignal Alpha v6.3.1 ONLINE*\n"
            f"\n"
            f"Mode: {config.OPERATING_MODE} | Phase: {config.PROP_FIRM_PHASE}\n"
            f"Equity: ${_eq:,.2f}\n"
            f"Daily DD Limit: {config.DAILY_DRAWDOWN_LIMIT_PCT}%\n"
            f"Anti-Hedge: ON\n"
            f"Dampener Floor: {getattr(config, 'DAMPENER_LOT_FLOOR', 0.03)}\n"
            f"Scanners: ALL ACTIVE\n"
            f"Tasks: {len(tasks)}\n"
            f"\n"
            f"All systems nominal. Trading is live."
        )
    except Exception:
        pass
    await asyncio.gather(*tasks)


async def nightly_optimization_loop():
    while True:
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            if 3 <= now.hour < 4:
                logger.info("[Main] Starting nightly ML optimization...")
                report = await win_model.nightly_optimization()
                if report:
                    logger.info(f"[Main] Nightly optimization complete.")

                # v8.0: Self-Optimizer nightly analysis
                try:
                    from quant.self_optimizer import self_optimizer
                    _so_adj = self_optimizer.run_nightly()
                    logger.info("[Main] Self-Optimizer: streak=%s, %d top sources, %d losing hours",
                                _so_adj.get("current_streak", "?"),
                                len(_so_adj.get("top_sources", [])),
                                len(_so_adj.get("losing_hours", [])))
                except Exception as _so_err:
                    logger.error("[Main] Self-Optimizer error: %s", _so_err)
                await asyncio.sleep(3600)
            else:
                await asyncio.sleep(300)
        except Exception as e:
            logger.error(f"[Main] Nightly optimization error: {e}", exc_info=True)
            await asyncio.sleep(600)


async def macro_update_loop():
    while True:
        try:
            await asyncio.sleep(14400)
            await update_macro_automatically()
            logger.info("[Main] Macro data refreshed.")
        except Exception as e:
            logger.warning(f"[Main] Macro update error: {e}")
            await asyncio.sleep(3600)


async def _graceful_shutdown(loop: asyncio.AbstractEventLoop):
    logger.info("[Main] Graceful shutdown initiated...")
    try:
        await tg_disconnect()
    except Exception as e:
        logger.warning(f"[Main] Telegram disconnect error: {e}")
    mt5_executor.disconnect()
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("[Main] All tasks cancelled. Shutdown complete.")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_stop(sig, frame):
        logger.info(f"[Main] Received signal {sig}. Scheduling shutdown...")
        loop.create_task(_graceful_shutdown(loop))
        loop.call_later(5, loop.stop)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        loop.run_until_complete(startup())
    except KeyboardInterrupt:
        logger.info("[Main] KeyboardInterrupt caught.")
        loop.run_until_complete(_graceful_shutdown(loop))
    finally:
        mt5_executor.disconnect()
        loop.close()
        logger.info("[Main] Event loop closed.")




