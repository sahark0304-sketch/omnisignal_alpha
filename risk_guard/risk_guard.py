"""
risk_guard/risk_guard.py — OmniSignal Alpha v2.0
Unified risk validation gate — all pillar checks in sequence.

Validation sequence:
  0. Global halt check (DB-persisted)
  1. Daily drawdown limit
  2. Latency safety mode (Pillar 13)
  3. Concurrent trade limit
  4. Signal expiry (belt-and-suspenders)
  5. Entry deviation
  6. Spread check
  7. News filter (Pillar 4)
  8. Confluence engine (Pillar 2 & 3)
  9. Stop loss required
 10. Dynamic position sizing: ATR + Kelly + Alpha Ranker (Pillars 5, 6)
 11. Currency exposure guard (Pillar 11)
 12. Final approval → Black Box commit (Pillar 12)
"""

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

import numpy as np

import config
from ai_engine.ai_engine import ParsedSignal
from news_filter.news_filter import news_filter
from database import db_manager
from quant.breakout_guard import is_counter_trend_blocked, is_direction_blocked, is_trend_bias_blocked
from quant.htf_filter import (
    check_toxicity_gate, check_execution_dedup,
    compute_sl_floor_with_rr, register_execution,
    invert_signal_levels, get_sizing_coefficient,
    check_htf_trend_gate,
)
from mt5_executor import mt5_executor
from utils.logger import get_logger, get_trade_logger
from utils.notifier import notify

logger = get_logger(__name__)

# In-memory halt mirror (canonical value always in DB)
TRADING_HALTED: bool = False
HALT_REASON: str = ""


@dataclass
class SizedOrder:
    signal: ParsedSignal
    lot_size: float
    is_high_conviction: bool  = False
    risk_amount: float        = 0.0
    risk_pct: float           = 0.0
    sizing_method: str        = "BASE"
    alpha_tier: str           = "UNRATED"
    alpha_multiplier: float   = 1.0
    dd_mode: str              = "NORMAL"


# ── HALT STATE ────────────────────────────────────────────────────────────────

def halt_trading(reason: str):
    global TRADING_HALTED, HALT_REASON
    if TRADING_HALTED:
        return
    TRADING_HALTED = True
    HALT_REASON = reason
    db_manager.set_system_state("halt", "1")
    db_manager.set_system_state("halt_reason", reason)
    logger.critical(f"[RiskGuard] HALTED: {reason}")
    notify(
        f"⚠️ *Trading Paused*\n"
        f"\n"
        f"{reason}\n"
        f"\n"
        f"_Trading will auto-resume at midnight UTC._\n"
        f"_Your open positions are still being managed._"
    )
    db_manager.log_audit("HALT", {"reason": reason})


def resume_trading():
    global TRADING_HALTED, HALT_REASON
    was_halted = TRADING_HALTED
    TRADING_HALTED = False
    HALT_REASON = ""
    db_manager.set_system_state("halt", "0")
    db_manager.set_system_state("halt_reason", "")
    logger.info("[RiskGuard] Trading resumed.")
    if was_halted:
        notify(
            "✅ *Trading Resumed*\n"
            "\n"
            "_New trading day started. All systems active._"
        )
    db_manager.log_audit("RESUME", {})


def is_halted() -> Tuple[bool, str]:
    try:
        val = db_manager.get_system_state("halt")
        reason = db_manager.get_system_state("halt_reason") or ""
        return (val == "1", reason)
    except Exception:
        return TRADING_HALTED, HALT_REASON


def sync_halt_from_db():
    global TRADING_HALTED, HALT_REASON
    halted, reason = is_halted()
    TRADING_HALTED = halted
    HALT_REASON = reason
    if halted:
        logger.warning(f"[RiskGuard] Startup: halt active from DB — {reason}")


# ── MAIN VALIDATION ───────────────────────────────────────────────────────────

async def validate(
    signal: ParsedSignal,
    current_bid: float,
    current_ask: float,
    current_spread_pips: float,
    account_equity: float,
    is_high_conviction: bool = False,
    pip_size: float = 0.0001,
    pip_value_per_lot: float = 10.0,
    confluence_result=None,  # Pre-computed ConfluenceResult from main.py
    trace=None,             # Optional[DecisionTrace] from black_box
) -> Tuple[bool, str, Optional[SizedOrder]]:

    def _reject(reason: str, stage: str = "") -> Tuple[bool, str, None]:
        db_manager.log_audit("RISK_REJECTED", {
            "symbol": signal.symbol, "action": signal.action,
            "source": signal.raw_source, "reason": reason, "stage": stage
        })
        logger.info(f"[RiskGuard] ❌ {signal.symbol} {signal.action} — {reason}")
        get_trade_logger().info("REJECT | %s | %s | stage=%s reason=%s", signal.symbol, signal.action, stage, reason[:120])
        if trace:
            trace.set_risk(False, reason)
            trace.set_execution("REJECTED")
        return False, reason, None

    # ── 0. Global halt ────────────────────────────────────────────────────────
    halted, halt_reason = is_halted()
    if halted:
        return _reject(f"Trading halted: {halt_reason}", "HALT")

    # ── 1. Daily drawdown ─────────────────────────────────────────────────────
    opening_equity = db_manager.get_opening_equity()
    if opening_equity and opening_equity > 0:
        daily_dd_pct = ((opening_equity - account_equity) / opening_equity) * 100.0
        if daily_dd_pct >= config.DAILY_DRAWDOWN_LIMIT_PCT:
            halt_trading(
                f"Daily DD {daily_dd_pct:.2f}% >= {config.DAILY_DRAWDOWN_LIMIT_PCT}% "
                f"(opening=${opening_equity:,.2f} current=${account_equity:,.2f})"
            )
            return _reject(HALT_REASON, "DRAWDOWN")
        dd_fraction = daily_dd_pct / config.DAILY_DRAWDOWN_LIMIT_PCT
        if dd_fraction >= config.DD_BLOCK_THRESHOLD_PCT:
            return _reject(
                f"DD at {daily_dd_pct:.1f}% — {dd_fraction:.0%} of daily limit. "
                f"Blocking new entries.", "DD_BLOCK"
            )

    # ── TELEGRAM FAST-TRACK (v4.2: Expert Pipeline + Sniper Entry) ──────────
    is_telegram = not signal.raw_source.startswith("AUTO_")
    if is_telegram and signal.raw_source != "UNKNOWN":
        from quant.alpha_ranker import alpha_ranker
        alpha_mult, alpha_tier = alpha_ranker.get_multiplier(signal.raw_source, signal.symbol)
        is_fast_track = alpha_tier in ("S", "A", "UNRATED")
        if is_fast_track:
            logger.info(
                "[RiskGuard] FAST-TRACK: %s (Tier %s) — minimal filters",
                signal.raw_source, alpha_tier,
            )
            if not signal.stop_loss:
                return _reject("No stop-loss on fast-track signal", "NO_SL")

            # v4.2: Sniper Entry — check if AMD/consensus aligns with signal
            sniper_boost = 1.0
            sniper_tag = ""
            try:
                from quant.convergence_engine import convergence_engine
                from quant.amd_engine import amd_engine
                cs = convergence_engine.get_consensus_score()
                amd_state = amd_engine.get_state()
                consensus_aligns = (
                    cs.get("direction") == signal.action
                    and cs.get("score", 0) >= 40
                )
                amd_aligns = (
                    amd_state.get("bias") == signal.action
                    and amd_state.get("confidence", 0) > 0.4
                )
                if consensus_aligns and amd_aligns:
                    sniper_boost = 1.60
                    sniper_tag = (
                        f" | SNIPER ENTRY: consensus={cs['score']}"
                        f" amd={amd_state['phase']}"
                    )
                    logger.info(
                        "[RiskGuard] SNIPER ENTRY on %s: consensus=%d + AMD=%s",
                        signal.raw_source, cs["score"], amd_state["phase"],
                    )
                elif consensus_aligns or amd_aligns:
                    sniper_boost = 1.25
                    sniper_tag = " | partial alignment"
            except Exception:
                pass

            from quant.volatility_sizing import calculate_lot_size, get_source_stats
            equity = mt5_executor.get_account_equity() or 10000
            entry = signal.entry_price or (
                current_ask if signal.action == "BUY" else current_bid
            )
            source_stats = get_source_stats(signal.raw_source)
            sizing = calculate_lot_size(
                equity=equity,
                entry=entry,
                stop_loss=signal.stop_loss,
                pip_size=pip_size,
                pip_value_per_lot=pip_value_per_lot,
                source_stats=source_stats,
                alpha_multiplier=alpha_mult,
            )
            # v4.4: S-Tier gets 1.60x lot boost
            vip_tier_boost = 1.60 if alpha_tier == "S" else 1.0
            lot_size = max(
                round(sizing.lot_size * sniper_boost * vip_tier_boost, 2), config.VIP_MIN_LOT
            )

            order = SizedOrder(
                signal=signal,
                lot_size=lot_size,
                is_high_conviction=True,
                risk_amount=sizing.risk_amount,
                risk_pct=sizing.risk_pct,
                sizing_method=f"FAST_TRACK_{sizing.method}",
                alpha_tier=alpha_tier,
                alpha_multiplier=alpha_mult,
            )
            if trace:
                trace.set_risk_check("FAST_TRACK_APPROVED")
            db_manager.log_audit("FAST_TRACK", {
                "source": signal.raw_source,
                "tier": alpha_tier,
                "lot": lot_size,
                "sniper_boost": sniper_boost,
            })

            # v4.4-audit: Safety checks that MUST apply even to fast-track
            # Spread guard
            try:
                import MetaTrader5 as _mt5
                _sym_info = _mt5.symbol_info(signal.symbol)
                if _sym_info:
                    _ft_spread = _sym_info.spread * pip_size / pip_size
                    if _ft_spread > config.MT5_MAX_SPREAD_PIPS:
                        return _reject(
                            f"FAST-TRACK SPREAD VETO: {_ft_spread:.1f}p > "
                            f"max {config.MT5_MAX_SPREAD_PIPS}p",
                            "FT_SPREAD",
                        )
            except Exception:
                pass

            # News filter
            await news_filter.refresh_if_needed()
            _ft_blocked, _ft_news_reason = news_filter.is_blocked(signal.symbol)
            if _ft_blocked:
                return _reject(f"FAST-TRACK NEWS VETO: {_ft_news_reason}", "FT_NEWS")

            # Concurrent trade limit
            _ft_open = db_manager.get_open_trades()
            _ft_abs_limit = config.MAX_CONCURRENT_TRADES + config.VIP_OVERFLOW_SLOTS
            if len(_ft_open) >= _ft_abs_limit:
                return _reject(
                    f"FAST-TRACK CAPACITY VETO: {len(_ft_open)}/{_ft_abs_limit} slots full",
                    "FT_MAX_TRADES",
                )

            # Exposure guard
            from quant.exposure_guard import check_exposure
            _ft_exp_ok, _ft_exp_reason = check_exposure(
                new_symbol=signal.symbol,
                new_action=signal.action,
                new_risk_amount=sizing.risk_amount,
                account_equity=account_equity,
            )
            if not _ft_exp_ok:
                return _reject(f"FAST-TRACK EXPOSURE VETO: {_ft_exp_reason}", "FT_EXPOSURE")

            logger.info(
                "[RiskGuard] FAST-TRACK APPROVED: %s | lots=%.2f | tier=%s%s",
                signal.raw_source, lot_size, alpha_tier, sniper_tag,
            )
            return True, "Approved", order

    # ── 2. Latency safety mode (Pillar 13) ────────────────────────────────────
    if config.LATENCY_ENABLED:
        from quant.latency_monitor import is_safe_to_trade, get_state as lat_state
        if not is_safe_to_trade():
            mode = lat_state().mode
            return _reject(f"Latency safety mode active: {mode}", "LATENCY")

    # ── 3. Concurrent trade limit + VIP Overflow ────────────────────────────
    open_trades = db_manager.get_open_trades()
    n_open = len(open_trades)
    base_limit = config.MAX_CONCURRENT_TRADES
    abs_limit = base_limit + config.VIP_OVERFLOW_SLOTS

    if n_open >= base_limit:
        is_vip = (
            signal.confidence >= 9
            or signal.raw_source == "AUTO_CONVERGENCE"
        )
        if not is_vip:
            return _reject(
                f"Max concurrent trades ({base_limit}) reached "
                f"(VIP slots reserved for conf>=9 / convergence)",
                "MAX_TRADES",
            )
        if n_open >= abs_limit:
            return _reject(
                f"Absolute max capacity ({abs_limit}) reached — "
                f"even VIP overflow is full",
                "MAX_TRADES_ABS",
            )
        logger.info(
            f"[RiskGuard] VIP OVERFLOW: {signal.symbol} {signal.action} "
            f"using overflow slot ({n_open+1}/{abs_limit}) | "
            f"AI:{signal.confidence}/10 source:{signal.raw_source}"
        )

    # v4.3: Per-symbol position limit with pyramid bypass for profitable positions
    try:
        from mt5_executor import mt5_executor as _mt5
        live_positions = _mt5.get_all_positions()
        same_symbol_same_dir = [
            p for p in live_positions
            if p["symbol"] == signal.symbol
            and p["type"] == signal.action
            and "Pyramid" not in (p.get("comment") or "")
        ]
        max_per_sym = config.MAX_CONCURRENT_PER_SYMBOL
        if len(same_symbol_same_dir) >= max_per_sym:
            # Check if existing position is in profit — allow pyramid add
            existing = same_symbol_same_dir[0]
            existing_profit = existing.get("profit", 0)
            if existing_profit > 0 and signal.confidence >= 8:
                logger.info(
                    "[RiskGuard] PYRAMID BYPASS: %s %s has +$%.2f profit, "
                    "allowing add (conf=%d)",
                    signal.symbol, signal.action, existing_profit, signal.confidence,
                )
                db_manager.log_audit("PYRAMID_BYPASS", {
                    "symbol": signal.symbol, "action": signal.action,
                    "existing_profit": round(existing_profit, 2),
                    "confidence": signal.confidence,
                })
            else:
                return _reject(
                    f"MT5_POSITION_DEDUP: Already {len(same_symbol_same_dir)} live "
                    f"{signal.symbol} {signal.action} entries (max {max_per_sym})"
                    f"{' (position in loss, no pyramid)' if existing_profit <= 0 else ''}",
                    "MT5_POSITION_DEDUP",
                )
    except Exception as e:
        logger.warning("[RiskGuard] MT5 position dedup check failed: %s", e)

    # v3.7 SESSION BLACKOUT
    from datetime import datetime, timezone
    utc_hour = datetime.now(timezone.utc).hour
    # v4.4: Spread & Slippage Guard -- block entries when spread is abnormally wide
    try:
        import MetaTrader5 as _mt5
        _sym_info = _mt5.symbol_info(signal.symbol)
        if _sym_info:
            current_spread = _sym_info.spread * pip_size
            current_spread_pips_raw = current_spread / pip_size
            _rates = _mt5.copy_rates_from_pos(signal.symbol, _mt5.TIMEFRAME_M1, 0, 60)
            if _rates is not None and len(_rates) >= 30:
                _spreads = (_rates['high'] - _rates['low']).astype(float)
                _avg_spread = float(np.mean(_spreads[-30:]))
                _avg_spread_pips = _avg_spread / pip_size
                if _avg_spread_pips > 0 and current_spread_pips_raw > _avg_spread_pips * 1.5:
                    return _reject(
                        f"WIDENED SPREAD: current={current_spread_pips_raw:.1f}p > "
                        f"1.5x avg={_avg_spread_pips:.1f}p | "
                        f"Execution quality compromised",
                        "REJECTED_WIDENED_SPREAD",
                    )
    except Exception:
        pass

    # alpha_tier only exists if Telegram fast-track branch ran; avoid NameError on AUTO_ paths
    try:
        _blackout_alpha_tier = alpha_tier
    except NameError:
        _blackout_alpha_tier = None

    # v4.4: Dual Time Blackout (rollover + news windows)
    _blackout_windows = [(12, 13), (14, 15)]
    for _bw_start, _bw_end in _blackout_windows:
        if _bw_start <= utc_hour < _bw_end:
            is_vip_signal = (
                signal.confidence >= 10
                or signal.raw_source == "AUTO_CONVERGENCE"
                or (not signal.raw_source.startswith("AUTO_") and _blackout_alpha_tier == "S")
            )
            if not is_vip_signal:
                return _reject(
                    f"Session blackout {_bw_start}:00-{_bw_end}:00 UTC "
                    f"- only Convergence/S-Tier/10-conf signals allowed",
                    "SESSION_BLACKOUT",
                )
            break

    # ── 4. Entry deviation ────────────────────────────────────────────────────
    if signal.entry_price:
        market_price = current_ask if signal.action == "BUY" else current_bid
        dev_pips = abs(market_price - signal.entry_price) / pip_size
        if dev_pips > config.MAX_ENTRY_DEVIATION_PIPS:
            return _reject(
                f"Entry deviation {dev_pips:.1f}p > {config.MAX_ENTRY_DEVIATION_PIPS}p", "DEVIATION"
            )

    # ── 5. Spread check ───────────────────────────────────────────────────────
    if current_spread_pips > config.MT5_MAX_SPREAD_PIPS:
        return _reject(
            f"Spread {current_spread_pips:.1f}p > {config.MT5_MAX_SPREAD_PIPS}p", "SPREAD"
        )

    # ── 5b. Global Bias Gate (Master Direction Switch) ──────────────────────
    try:
        from quant.convergence_engine import convergence_engine as _conv
        _bias = _conv.global_bias
        _bstr = _conv.bias_strength
        _kill_thresh = getattr(config, "GLOBAL_BIAS_KILL_THRESHOLD", 0.45)
        if _bias and signal.action != _bias and _bstr >= _kill_thresh:
            if signal.raw_source.startswith("AUTO_") and signal.raw_source != "AUTO_CONVERGENCE":
                return _reject(
                    f"Global bias is {_bias} (strength {_conv.bias_strength:.2f}) "
                    f"— blocking counter-trend {signal.action} from {signal.raw_source}",
                    "GLOBAL_BIAS",
                )
            elif _bstr >= 0.60:
                return _reject(
                    f"Strong global bias {_bias} (strength {_bstr:.2f}) "
                    f"- hard-blocking ALL counter-trend {signal.action}",
                    "GLOBAL_BIAS_HARD",
                )
    except Exception:
        pass

    # v4.4: Multi-Timeframe Trend Gate
    htf_ok, htf_reason = check_htf_trend_gate(signal.symbol, signal.action, pip_size)
    if not htf_ok:
        return _reject(htf_reason, "HTF_TREND_GATE")

    # ── 5c. Breakout Kill Switch ──────────────────────────────────────────────
    bo_blocked, bo_reason = is_counter_trend_blocked(signal.action, signal.raw_source)
    if bo_blocked:
        return _reject(bo_reason, "BREAKOUT_BLOCK")

    # ── 5c. Directional Circuit Breaker (Anti-Martingale) ─────────────────────
    dir_blocked, dir_reason = is_direction_blocked(signal.action, signal.symbol)
    if dir_blocked:
        return _reject(dir_reason, "DIRECTION_BLOCK")

    # ── 5d. Order Flow Toxicity Gate + Signal Inversion Engine ──────────────────────
    tox_ok, tox_reason, inversion_action = check_toxicity_gate(
        signal.symbol, signal.action, pip_size
    )
    if not tox_ok:
        if inversion_action:
            original_action = signal.action
            entry_now = signal.entry_price or (current_ask if signal.action == "BUY" else current_bid)
            new_sl, new_tp, _ = invert_signal_levels(
                entry_now, signal.stop_loss or entry_now,
                signal.tp1, signal.action, inversion_action, pip_size,
            )
            signal.action = inversion_action
            signal.stop_loss = new_sl
            signal.tp1 = new_tp
            signal.tp2 = None
            signal.tp3 = None
            logger.warning(
                f"[RiskGuard] SIGNAL INVERTED: {signal.symbol} "
                f"{original_action}->{inversion_action} | "
                f"New SL:{new_sl:.2f} TP:{new_tp:.2f} | {tox_reason}"
            )
            db_manager.log_audit("SIGNAL_INVERSION", {
                "symbol": signal.symbol,
                "original_action": original_action,
                "inverted_action": inversion_action,
                "new_sl": new_sl, "new_tp": new_tp,
                "source": signal.raw_source,
                "reason": tox_reason,
            })
            signal._was_inverted = True
        else:
            return _reject(tox_reason, "TOXICITY")

    # v4.4-audit: Session Risk Budget Enforcement
    try:
        from datetime import datetime, timezone
        _utc_now = datetime.now(timezone.utc)
        _utc_h = _utc_now.hour
        _current_session = None
        for _sess_name, (_s_start, _s_end) in config.SESSION_HOURS_UTC.items():
            if _s_start < _s_end:
                if _s_start <= _utc_h < _s_end:
                    _current_session = _sess_name
                    break
            else:
                if _utc_h >= _s_start or _utc_h < _s_end:
                    _current_session = _sess_name
                    break
        if _current_session:
            _sess_budget_pct = config.SESSION_RISK_BUDGET_PCT.get(_current_session, 1.0)
            _sess_dd_limit = config.DAILY_DRAWDOWN_LIMIT_PCT * _sess_budget_pct
            _opening = db_manager.get_opening_equity()
            if _opening and _opening > 0:
                _sess_dd = (((_opening - account_equity) / _opening) * 100.0)
                if _sess_dd > 0 and _sess_dd >= _sess_dd_limit:
                    return _reject(
                        f"SESSION BUDGET EXHAUSTED: {_current_session} "
                        f"DD={_sess_dd:.1f}% >= session limit {_sess_dd_limit:.1f}% "
                        f"(budget={_sess_budget_pct:.0%} of daily {config.DAILY_DRAWDOWN_LIMIT_PCT}%)",
                        "SESSION_BUDGET",
                    )
    except Exception:
        pass

    # ── 5e. Execution Dedup (prevent duplicate trades < 120s apart) ───────
    dup_ok, dup_reason = check_execution_dedup(signal.symbol, signal.action)
    if not dup_ok:
        return _reject(dup_reason, "EXEC_DEDUP")

    # ── 6. News filter (Pillar 4) ─────────────────────────────────────────────
    await news_filter.refresh_if_needed()
    blocked, news_reason = news_filter.is_blocked(signal.symbol)
    if blocked:
        return _reject(news_reason, "NEWS")

    # ── 7. Confluence check (Pillars 2 & 3) ───────────────────
    if confluence_result is not None:
        confluence = confluence_result
    elif config.CONFLUENCE_ENABLED:
        from quant.confluence_engine import check_confluence
        entry = signal.entry_price or (current_ask if signal.action == 'BUY' else current_bid)
        confluence = await check_confluence(signal.symbol, signal.action, entry)
    else:
        confluence = None

    if confluence is not None:
        if trace:
            trace.set_confluence(confluence)
        atr_value = confluence.atr
    else:
        atr_value = 0.0

    # ── 8. Stop Loss required ─────────────────────────────────────────────────
    if not signal.stop_loss:
        return _reject("No Stop Loss — prop firm safety requires SL on all entries", "NO_SL")

    # ── 9. Dynamic sizing: Kelly + ATR + Alpha Ranker (Pillars 5 & 6) ────────
    from quant.alpha_ranker import alpha_ranker
    from quant.volatility_sizing import calculate_lot_size, get_source_stats

    alpha_mult, alpha_tier = alpha_ranker.get_multiplier(signal.raw_source, signal.symbol)

    # AI Confidence Override: high-conviction AI signals can rehabilitate low-tier sources.
    # TOXIC sources (statistically significant toxicity) are NEVER overridden.
    ai_override_applied = False
    if alpha_mult == 0.0 and alpha_tier == "TOXIC":
        profile = alpha_ranker.get_profile(signal.raw_source)
        return _reject(
            f"Source '{signal.raw_source}' is TOXIC "
            f"({profile.raw_wr:.0%} WR over {profile.total} trades)", "TOXIC"
        )
    elif alpha_mult < 0.5 and signal.confidence >= config.AI_OVERRIDE_MIN_CONFIDENCE:
        if signal.confidence >= 10:
            override_floor = config.AI_OVERRIDE_FLOOR_CONF_10
        else:
            override_floor = config.AI_OVERRIDE_FLOOR_CONF_9
        if alpha_mult < override_floor:
            old_mult = alpha_mult
            alpha_mult = override_floor
            ai_override_applied = True
            logger.warning(
                f"[RiskGuard] AI OVERRIDE: {signal.symbol} {signal.action} | "
                f"AI:{signal.confidence}/10 overrides {alpha_tier}-tier "
                f"(mult {old_mult:.2f}x -> {alpha_mult:.2f}x) | "
                f"Source: {signal.raw_source}"
            )
            db_manager.log_audit("AI_OVERRIDE", {
                "symbol": signal.symbol, "action": signal.action,
                "source": signal.raw_source, "ai_confidence": signal.confidence,
                "old_tier": alpha_tier, "old_mult": old_mult,
                "override_mult": alpha_mult,
            })

    source_stats = get_source_stats(signal.raw_source)
    entry = signal.entry_price or (current_ask if signal.action == "BUY" else current_bid)

    # ATR SL floor with R:R Preservation (v3.8: hard cap rejects impossible scalps)
    if signal.stop_loss and signal.entry_price:
        adj_sl, adj_tp1, adj_tp2, adj_tp3 = compute_sl_floor_with_rr(
            signal.symbol, entry, signal.stop_loss,
            signal.tp1, signal.tp2, signal.tp3,
            signal.action, pip_size,
        )
        if adj_sl is None:
            return _reject(
                'REJECT_ATR_TOO_WIDE: SL too tight for current volatility '
                '(ATR floor requires >1.5x widening). Alpha is dead.',
                'ATR_TOO_WIDE',
            )
        if adj_sl != signal.stop_loss:
            signal.stop_loss = adj_sl
            if adj_tp1 is not None:
                signal.tp1 = adj_tp1
            if adj_tp2 is not None:
                signal.tp2 = adj_tp2
            if adj_tp3 is not None:
                signal.tp3 = adj_tp3

    # ── Volatility-Adaptive TP Expansion ─────────────────────────────────────
    if confluence is not None and signal.tp1 and entry:
        vol_z = getattr(confluence, 'vol_z', 0.0)
        if vol_z > 1.5:
            vol_mult = min(1.0 + (vol_z - 1.5) * 0.3, 1.8)
            orig_tp1_dist = abs(signal.tp1 - entry)
            if signal.action == "BUY":
                signal.tp1 = round(entry + orig_tp1_dist * vol_mult, 2)
                if signal.tp2:
                    signal.tp2 = round(entry + abs(signal.tp2 - entry) * vol_mult, 2)
            else:
                signal.tp1 = round(entry - orig_tp1_dist * vol_mult, 2)
                if signal.tp2:
                    signal.tp2 = round(entry - abs(entry - signal.tp2) * vol_mult, 2)
            logger.info(
                f"[RiskGuard] VOL TP EXPANSION: vol_z={vol_z:.2f} mult={vol_mult:.2f}x "
                f"TP1 widened by {(vol_mult-1)*100:.0f}%"
            )

    sizing = calculate_lot_size(
        equity          = account_equity,
        entry           = entry,
        stop_loss       = signal.stop_loss,
        pip_size        = pip_size,
        pip_value_per_lot = pip_value_per_lot,
        atr_value       = atr_value,
        source_stats    = source_stats,
        alpha_multiplier = alpha_mult,
    )
    # Toxicity-adaptive sizing: boost inversions, dampen uncertain regimes
    was_inverted = getattr(signal, "_was_inverted", False)
    tox_coeff, tox_sizing_reason = get_sizing_coefficient(
        signal.symbol, signal.action, was_inverted
    )
    if tox_coeff != 1.0:
        sizing.lot_size = round(sizing.lot_size * tox_coeff, 2)
        logger.info(
            f"[RiskGuard] {tox_sizing_reason} | "
            f"lots adjusted to {sizing.lot_size}"
        )


    # v3.7 VIP Lot Floor
    vip_min = getattr(config, "VIP_MIN_LOT", 0.05)
    is_vip_signal = (
        signal.confidence >= 10
        or signal.raw_source == "AUTO_CONVERGENCE"
    )
    if is_vip_signal and sizing.lot_size < vip_min:
        logger.info(
            f"[RiskGuard] VIP LOT FLOOR: {sizing.lot_size} -> {vip_min} "
            f"(VIP signal: conf={signal.confidence} src={signal.raw_source})"
        )
        sizing.lot_size = vip_min

    if sizing.lot_size <= 0:
        return _reject("Sizing returned 0 lots", "SIZING")

    if trace:
        trace.set_sizing(sizing, alpha_tier)

    # ── 10. Currency exposure guard (Pillar 11) ───────────────────────────────
    from quant.exposure_guard import check_exposure
    exp_ok, exp_reason = check_exposure(
        new_symbol=signal.symbol,
        new_action=signal.action,
        new_risk_amount=sizing.risk_amount,
        account_equity=account_equity,
    )
    if trace:
        trace.set_exposure(exp_ok, exp_reason)
    if not exp_ok:
        return _reject(exp_reason, "EXPOSURE")

    if trace:
        trace.set_risk(True)

    # v4.3.2: Consensus boost removed — it double-dipped with institutional scaling,
    # causing 0.37L trades that wiped session profits on reversal
    lot_size = round(sizing.lot_size, 2)

    # v4.3.2: Hard lot ceiling — prevents any single trade from being so large
    # that one loss wipes the day's profits (learned from $413 blowback after $324 win)
    from quant.convexity_engine import HARD_LOT_CEILING
    if lot_size > HARD_LOT_CEILING:
        logger.info(
            "[RiskGuard] LOT CEILING: %.2f -> %.2f (hard cap)",
            lot_size, HARD_LOT_CEILING,
        )
        lot_size = HARD_LOT_CEILING

    order = SizedOrder(
        signal           = signal,
        lot_size         = lot_size,
        is_high_conviction = is_high_conviction,
        risk_amount      = sizing.risk_amount,
        risk_pct         = sizing.risk_pct,
        sizing_method    = sizing.method,
        alpha_tier       = alpha_tier,
        alpha_multiplier = alpha_mult,
    )

    logger.info(
        f"[RiskGuard] ✅ APPROVED {signal.symbol} {signal.action} @ {entry} | "
        f"Lots:{sizing.lot_size} ({sizing.risk_pct:.2f}%) Method:{sizing.method} "
        f"Tier:{alpha_tier} Mult:{alpha_mult:.2f}x AI:{signal.confidence}/10×"
    )
    get_trade_logger().info("APPROVE | %s | %s | lots=%.2f tier=%s method=%s", signal.symbol, signal.action, sizing.lot_size, alpha_tier, sizing.method)
    db_manager.log_audit("RISK_APPROVED", {
        "symbol": signal.symbol, "action": signal.action, "lot": sizing.lot_size,
        "risk_pct": sizing.risk_pct, "method": sizing.method, "tier": alpha_tier,
        "source": signal.raw_source, "hc": is_high_conviction,
    })
    return True, "Approved", order


async def continuous_equity_monitor():
    from mt5_executor import mt5_executor
    while True:
        try:
            equity = mt5_executor.get_account_equity()
            if equity and equity > 0:
                opening = db_manager.get_opening_equity()
                if opening and opening > 0:
                    dd_pct = (opening - equity) / opening * 100
                    if dd_pct >= config.DAILY_DRAWDOWN_LIMIT_PCT:
                        halt_trading(f"Daily DD {dd_pct:.2f}% >= {config.DAILY_DRAWDOWN_LIMIT_PCT}%")
        except Exception as e:
            logger.error(f"[RiskGuard] Equity monitor error: {e}")
        await asyncio.sleep(30)


async def daily_reset_watcher():
    from datetime import datetime, timezone
    last_day = datetime.now(timezone.utc).date()
    while True:
        try:
            now = datetime.now(timezone.utc).date()
            if now != last_day:
                last_day = now
                resume_trading()
                logger.info("[RiskGuard] Daily reset — halt cleared for new day.")
        except Exception as e:
            logger.error(f"[RiskGuard] Daily reset error: {e}")
        await asyncio.sleep(60)
