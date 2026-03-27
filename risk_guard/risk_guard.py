"""
risk_guard/risk_guard.py — OmniSignal Alpha v6.3.1
RESTORED March 17 system + 3 bug fixes + dampener floor

v6.3.1 changes vs original:
  + Anti-hedge gate (prevents BUY+SELL same symbol)
  + Dampener lot floor (prevents stacking dampeners crushing to 0.01)
  + Bug 1 fix: lot ceiling preserved in main.py (post-risk boosts respect dampeners)
  + resume_trading() flushes velocity history
  + Monkey-patch in daily_reset_watcher REMOVED (was fragile)
  
  SCANNERS ARE FULLY ENABLED — no scanner gate
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

import numpy as np

import config
from ai_engine.ai_engine import ParsedSignal
from news_filter.news_filter import news_filter
from database import db_manager
from quant.breakout_guard import is_counter_trend_blocked, is_direction_blocked, is_trend_bias_blocked

try:
    from quant.breakout_hunter import breakout_hunter as _breakout_hunter
    _breakout_hunter_available = True
except ImportError:
    _breakout_hunter_available = False
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

try:
    from quant.trade_orchestrator import (
        trade_orchestrator, get_scaled_cooldown,
        get_lot_size_multiplier, get_tp_expansion,
    )
    _ato_available = True
except ImportError:
    _ato_available = False

_equity_history: deque = deque(maxlen=300)
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
    global TRADING_HALTED, HALT_REASON, _velocity_halt_time
    was_halted = TRADING_HALTED
    TRADING_HALTED = False
    HALT_REASON = ""
    _velocity_halt_time = 0.0
    _equity_history.clear()          # v6.3.1: flush stale velocity readings
    db_manager.set_system_state("halt", "0")
    db_manager.set_system_state("halt_reason", "")
    logger.info("[RiskGuard] Trading resumed. Velocity history flushed.")
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
    if halted and "Daily DD" in reason:
        try:
            from mt5_executor import mt5_executor as _mt5_sync
            equity = _mt5_sync.get_account_equity()
            opening = db_manager.get_opening_equity()
            if opening and opening > 0 and equity and equity > 0:
                actual_dd = ((opening - equity) / opening) * 100.0
                if actual_dd < config.DAILY_DRAWDOWN_LIMIT_PCT:
                    logger.info(
                        "[RiskGuard] Startup: stale DD halt cleared — "
                        "actual DD=%.2f%% < limit=%.1f%%",
                        actual_dd, config.DAILY_DRAWDOWN_LIMIT_PCT,
                    )
                    resume_trading()
                    return
        except Exception as e:
            logger.warning("[RiskGuard] Startup halt validation failed: %s", e)
    TRADING_HALTED = halted
    HALT_REASON = reason
    if halted:
        logger.warning("[RiskGuard] Startup: halt active from DB — %s", reason)


def get_trading_mode() -> str:
    halted, _ = is_halted()
    if halted:
        return "HALTED"
    try:
        opening = db_manager.get_opening_equity()
        if opening and opening > 0:
            from mt5_executor import mt5_executor
            equity = mt5_executor.get_account_equity()
            dd_pct = ((opening - equity) / opening) * 100.0
            dd_frac = dd_pct / config.DAILY_DRAWDOWN_LIMIT_PCT if config.DAILY_DRAWDOWN_LIMIT_PCT > 0 else 0
            if dd_frac >= config.DD_BLOCK_THRESHOLD_PCT:
                return "BLOCKED"
            if dd_frac >= config.DD_REDUCED_MODE_THRESHOLD_PCT:
                return "REDUCED"
    except Exception:
        pass
    return "NORMAL"


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
    confluence_result=None,
    trace=None,
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
        if _ato_available:
            try:
                trade_orchestrator.record_rejection(
                    stage=stage, source=signal.raw_source,
                    symbol=signal.symbol, action=signal.action,
                    confidence=signal.confidence,
                )
            except Exception:
                pass
        try:
            from quant.shadow_ledger import shadow_ledger as _sl
            _sl.track_rejection(signal, stage, reason, current_bid, current_ask)
        except Exception:
            pass
        # v7.1: Log to what-if ledger for continuous optimization
        try:
            if hasattr(signal, 'symbol') and signal.symbol:
                _wif_entry = signal.entry_price or current_bid
                db_manager.log_rejected_signal_price(
                    signal_id=0, symbol=signal.symbol,
                    action=signal.action, entry_price=_wif_entry,
                    sl=signal.stop_loss, tp1=signal.tp1,
                    reject_reason=reason[:200],
                )
        except Exception:
            pass
        return False, reason, None

    # ── 0. Global halt ────────────────────────────────────────────────────────
    halted, halt_reason = is_halted()
    if halted:
        return _reject(f"Trading halted: {halt_reason}", "HALT")

    # == v8.0: Tier S Channel Sniper =============================================
    _tier_s_channels = getattr(config, "TIER_S_CHANNELS", set())
    _is_tier_s = False
    try:
        if signal.raw_source.startswith("telegram:"):
            _channel_id = int(signal.raw_source.split(":")[1])
            _is_tier_s = _channel_id in _tier_s_channels
    except Exception:
        pass

    if _is_tier_s:
        logger.info(
            "[RiskGuard] TIER S SIGNAL from %s -- priority execution",
            signal.raw_source,
        )
        from datetime import datetime as _dt_ts, timezone as _tz_ts
        _ts_hour = _dt_ts.now(_tz_ts.utc).hour
        if _ts_hour in {0, 1, 2, 3, 4, 5, 6, 21, 22, 23}:
            return _reject("TIER_S blocked in blackout hours", "TIER_S_BLACKOUT")

        if not signal.stop_loss:
            return _reject("TIER_S: No stop-loss", "TIER_S_NO_SL")

        _ts_lots = getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03)

        _ts_order = SizedOrder(
            signal=signal, lot_size=_ts_lots, is_high_conviction=True,
            risk_amount=0, risk_pct=0,
            sizing_method="TIER_S_SNIPER",
            alpha_tier="S", alpha_multiplier=1.0,
        )

        if trace:
            trace.set_risk(True, "TIER_S_APPROVED")

        db_manager.log_audit("TIER_S_SNIPER", {
            "source": signal.raw_source, "action": signal.action,
            "lot_size": _ts_lots, "symbol": signal.symbol,
        })

        try:
            from utils.notifier import notify as _ts_notify
            _ts_notify(
                "\U0001f525 *TIER S SIGNAL*\n"
                + signal.symbol + " " + signal.action
                + " | Lots: " + str(_ts_lots)
                + " | Source: " + signal.raw_source
            )
        except Exception:
            pass

        return True, "TIER_S_APPROVED", _ts_order

    # == v7.0: Hard session blackout - Asia is a -$585 death zone ===========
    from datetime import datetime as _dt_v7, timezone as _tz_v7
    _utc_hour = _dt_v7.now(_tz_v7.utc).hour
    _blocked_hours = {0, 1, 2, 3, 4, 5, 6, 21, 22, 23}
    if _utc_hour in _blocked_hours:
        if not (signal.raw_source == "AUTO_CONVERGENCE" and signal.confidence >= 9):
            return _reject(
                f"SESSION_BLACKOUT: Hour {_utc_hour} UTC is blocked. "
                f"Only trade 07:00-20:00 UTC.",
                "SESSION_BLACKOUT_V7",
            )

    # == v8.2: Breakout Mode Detection ==========================================
    _breakout_active, _breakout_dir = False, None
    _in_breakout_trade = False
    if _breakout_hunter_available:
        try:
            _breakout_active, _breakout_dir = _breakout_hunter.is_breakout_active()
            _in_breakout_trade = (
                _breakout_active
                and signal.action == _breakout_dir
            )
            if _in_breakout_trade:
                logger.info(
                    "[RiskGuard] BREAKOUT MODE ACTIVE: %s %s (streak=%d)",
                    signal.symbol, _breakout_dir,
                    _breakout_hunter.consecutive_bars,
                )
        except Exception:
            pass

    # == v7.0: 5-minute cooldown - rapid-fire trades lose -$561 ===============
    try:
        _last_trade_time = db_manager.get_last_trade_time()
        if _last_trade_time:
            if isinstance(_last_trade_time, str):
                for _tfmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        _last_trade_time = _dt_v7.strptime(_last_trade_time, _tfmt)
                        break
                    except ValueError:
                        continue
            if isinstance(_last_trade_time, _dt_v7):
                _now_v7 = _dt_v7.now()
                _mins_since = (_now_v7 - _last_trade_time).total_seconds() / 60
                _min_gap = getattr(config, "MIN_TRADE_GAP_MINUTES", 5)
                if _in_breakout_trade:
                    _min_gap = getattr(config, "BREAKOUT_COOLDOWN_MINUTES", 1)
                if _mins_since < _min_gap:
                    return _reject(
                        f"COOLDOWN: Last trade {_mins_since:.1f}min ago. "
                        f"Minimum gap is {_min_gap}min.",
                        "TRADE_COOLDOWN",
                    )
    except Exception:
        pass

    # -- 0.1 v6.3.1: Anti-Hedge with Smart Override (deferred close) ---
    _ah_conflict = None
    if getattr(config, "ANTI_HEDGE_ENABLED", False):
        try:
            from mt5_executor import mt5_executor as _mt5_ah
            for _p in (_mt5_ah.get_all_positions() or []):
                if _p["symbol"] == signal.symbol and _p["type"] != signal.action:
                    _ah_profit = _p.get("profit", 0)
                    _ah_in_loss = _ah_profit < 0

                    # Tiered override: losing position = lower threshold
                    _can_override = False
                    if _ah_in_loss and signal.confidence >= 7:
                        _can_override = True
                    elif signal.confidence >= 9 and is_high_conviction:
                        _can_override = True

                    if _can_override:
                        _ah_conflict = _p
                        logger.info(
                            "[RiskGuard] ANTI-HEDGE: Flagged %s %s "
                            "(ticket %d, PnL=$%.2f) for closure if %s passes all gates",
                            signal.symbol, _p["type"], _p["ticket"],
                            _ah_profit, signal.action,
                        )
                        break
                    else:
                        return _reject(
                            "ANTI-HEDGE: " + signal.symbol + " has open " + _p["type"] +
                            " (ticket " + str(_p["ticket"]) + "). Cannot open " + signal.action + ".",
                            "ANTI_HEDGE",
                        )
        except Exception as _ah_err:
            logger.debug("[RiskGuard] Anti-hedge check error: %s", _ah_err)

    # ── 0.2 v6.4: Solo Scanner Quality Gates ─────────────────────────────────
    # Solo pullback gate — requires consensus to trade
    if getattr(config, 'SOLO_PULLBACK_REQUIRE_CONSENSUS', True):
        if signal.raw_source == 'AUTO_PULLBACK' and not is_high_conviction:
            return _reject(
                'SOLO_PULLBACK: Requires consensus (2+ sources agreeing). '
                'Solo momentum signals are statistically unprofitable.',
                'SOLO_PULLBACK',
            )

    # CATCD solo confidence floor
    if getattr(config, 'CATCD_MIN_SOLO_CONFIDENCE', 8) > 0:
        if (signal.raw_source == 'AUTO_CATCD'
            and not is_high_conviction
            and signal.confidence < getattr(config, 'CATCD_MIN_SOLO_CONFIDENCE', 8)):
            return _reject(
                'LOW_CONF_CATCD: Solo CATCD requires confidence >= '
                + str(config.CATCD_MIN_SOLO_CONFIDENCE) + ' (got ' + str(signal.confidence) + ')',
                'LOW_CONF_CATCD',
            )

    # Catch-all: any solo scanner (not convergence) needs consensus
    _allowed_solo_scanners = {'AUTO_CONVERGENCE', 'AUTO_LONDON_BREAKOUT', 'AUTO_REENTRY', 'AUTO_CANARY', 'AUTO_BREAKOUT'}
    if (signal.raw_source.startswith('AUTO_')
        and signal.raw_source not in _allowed_solo_scanners
        and not is_high_conviction):
        return _reject(
            'SOLO_SCANNER: ' + signal.raw_source + ' requires consensus. '
            'Only AUTO_CONVERGENCE can trade independently.',
            'SOLO_SCANNER',
        )

    # ── 0.5. Prop-Firm Finisher — Coast Mode gate ────────────────────────────
    if getattr(config, "PROP_FIRM_PHASE", "") == "CHALLENGE":
        try:
            from quant.prop_firm_finisher import prop_firm_finisher
            from quant.convergence_engine import convergence_engine as _conv_pf
            _consensus_pf = 0
            try:
                _cs_pf = _conv_pf.get_consensus_score()
                _consensus_pf = _cs_pf.get("score", 0)
            except Exception:
                pass
            _coast = prop_firm_finisher.check_override(
                signal_source      = signal.raw_source,
                signal_confidence  = signal.confidence,
                signal_action      = signal.action,
                convergence_score  = _consensus_pf,
                account_equity     = account_equity,
            )
            if _coast.active and _coast.lot_multiplier == 0.0:
                return _reject(_coast.reason, "COAST_MODE")
        except Exception:
            pass

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

    order = None  # set by fast-track; scanner path creates it later

    # ── TELEGRAM FAST-TRACK ───────────────────────────────────────────────────
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

            # v7.0: Confidence 10 nerf - data shows conf 10 loses -$1,195
            # Confidence 8 is best performer at +$562
            _sizing_conf = signal.confidence
            if _sizing_conf >= 10:
                sniper_boost = min(sniper_boost, 1.00)
                logger.info(
                    "[RiskGuard] CONF NERF: AI confidence 10 -> sniper_boost capped at 1.0 "
                    "(conf 10 has -$1,195 track record)"
                )
            elif _sizing_conf == 9:
                sniper_boost = min(sniper_boost, 1.15)
            # conf 8 keeps full sniper_boost (best performer)

            # re-wrap to skip original except
            try:
                _noop = 0
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
            vip_tier_boost = 1.60 if alpha_tier == "S" else 1.0
            lot_size = max(
                round(sizing.lot_size * sniper_boost * vip_tier_boost, 2), config.VIP_MIN_LOT
            )
            # v7.0: Hard lot cap on fast-track too
            lot_size = min(lot_size, getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03))

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
                trace.set_risk(True, "FAST_TRACK_APPROVED")
            db_manager.log_audit("FAST_TRACK", {
                "source": signal.raw_source,
                "tier": alpha_tier,
                "lot": lot_size,
                "sniper_boost": sniper_boost,
            })

            # Safety checks on fast-track
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

            await news_filter.refresh_if_needed()
            _ft_blocked, _ft_news_reason = news_filter.is_blocked(signal.symbol)
            if _ft_blocked:
                return _reject(f"FAST-TRACK NEWS VETO: {_ft_news_reason}", "FT_NEWS")

            _ft_open = db_manager.get_open_trades()
            _ft_abs_limit = config.MAX_CONCURRENT_TRADES + config.VIP_OVERFLOW_SLOTS
            if len(_ft_open) >= _ft_abs_limit:
                return _reject(
                    f"FAST-TRACK CAPACITY VETO: {len(_ft_open)}/{_ft_abs_limit} slots full",
                    "FT_MAX_TRADES",
                )

            from quant.exposure_guard import check_exposure
            _ft_exp_ok, _ft_exp_reason = check_exposure(
                new_symbol=signal.symbol,
                new_action=signal.action,
                new_risk_amount=sizing.risk_amount,
                account_equity=account_equity,
            )
            if not _ft_exp_ok:
                return _reject(f"FAST-TRACK EXPOSURE VETO: {_ft_exp_reason}", "FT_EXPOSURE")

            # v8.1: Post-loss reduction applies to fast-track too
            try:
                _ft_last = db_manager.get_last_closed_trade(signal.symbol)
                if _ft_last:
                    _ft_lpnl = _ft_last.get("pnl", 0)
                    _ft_lct = _ft_last.get("close_time")
                    if _ft_lpnl and _ft_lpnl < 0 and _ft_lct:
                        from datetime import datetime as _dt_ft
                        if isinstance(_ft_lct, str):
                            for _ftfmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                                try:
                                    _ft_lct = _dt_ft.strptime(_ft_lct, _ftfmt)
                                    break
                                except ValueError:
                                    continue
                        if isinstance(_ft_lct, _dt_ft):
                            _ft_mins = (_dt_ft.now() - _ft_lct).total_seconds() / 60
                            if _ft_mins < 15:
                                lot_size = min(lot_size, 0.01)
                                order.lot_size = 0.01
                                logger.warning(
                                    "[RiskGuard] FT POST-LOSS: Last trade lost $%.2f "
                                    "%.0fmin ago. Fast-track lots capped at 0.01.",
                                    _ft_lpnl, _ft_mins,
                                )
            except Exception:
                pass

            logger.info(
                "[RiskGuard] FAST-TRACK APPROVED: %s | lots=%.2f | tier=%s%s",
                signal.raw_source, lot_size, alpha_tier, sniper_tag,
            )
            # v7.1: Apply HTF weak counter-trend lot penalty
    try:
        if _htf_lot_penalty < 1.0:
            order.lot_size = max(round(order.lot_size * _htf_lot_penalty, 2), 0.01)
            logger.info("[RiskGuard] HTF LOT PENALTY: lots -> %.2f (%.0fx)", order.lot_size, _htf_lot_penalty)
    except (NameError, AttributeError):
        pass

    # v8.0: Apply NY sell boost
    try:
        if _ny_sell_boost > 1.0:
            _max_lots = getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03)
            order.lot_size = min(round(order.lot_size * _ny_sell_boost, 2), _max_lots)
            logger.info("[RiskGuard] NY SELL BOOST: lots -> %.2f (1.2x)", order.lot_size)
    except (NameError, AttributeError):
        pass

    # == v8.0 EDGE 3: Lead Channel Priority Sizing ============================
    # Research: LEADING channels signal BEFORE the move. Give them max lots.
    try:
        _lead_channel_id = None
        if hasattr(signal, 'raw_source') and signal.raw_source.startswith("telegram:"):
            _lead_channel_id = int(signal.raw_source.split(":")[1])
        _lead_channels = getattr(config, "LEAD_CHANNELS", set())
        if _lead_channel_id and _lead_channel_id in _lead_channels:
            _lead_max = getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03)
            if order.lot_size < _lead_max:
                logger.info(
                    "[RiskGuard] LEAD CHANNEL: %s -- max lots (proven leader, signals BEFORE the move)",
                    signal.raw_source,
                )
                order.lot_size = _lead_max
                db_manager.log_audit("LEAD_CHANNEL_BOOST", {
                    "source": signal.raw_source, "lot_size": _lead_max,
                })
    except Exception:
        pass
    # == v7.0: Hard lot cap - larger lots amplify losses ======================
    _hard_max = getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03)
    if order is not None and order.lot_size > _hard_max:
        logger.info(
            "[RiskGuard] LOT CAP: %.2f -> %.2f (v7.0 hard max)",
            order.lot_size, _hard_max,
        )
        order.lot_size = _hard_max

    # == v7.0: Post-loss reduction - losses cluster (61.4% chance) ===========
    try:
        _last_closed = db_manager.get_last_closed_trade(signal.symbol)
        if _last_closed:
            _last_pnl = _last_closed.get("pnl", 0)
            _last_close_time = _last_closed.get("close_time")
            if _last_pnl and _last_pnl < 0 and _last_close_time:
                from datetime import datetime as _dt_loss
                if isinstance(_last_close_time, str):
                    for _lfmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            _last_close_time = _dt_loss.strptime(_last_close_time, _lfmt)
                            break
                        except ValueError:
                            continue
                if isinstance(_last_close_time, _dt_loss):
                    _mins_since_loss = (_dt_loss.now() - _last_close_time).total_seconds() / 60
                    if _mins_since_loss < 15:
                        _loss_reduced = 0.01
                        logger.warning(
                            "[RiskGuard] POST-LOSS REDUCTION: Last trade lost $%.2f "
                            "%.0fmin ago. Lots -> 0.01 for safety.",
                            _last_pnl, _mins_since_loss,
                        )
                        order.lot_size = _loss_reduced
    except Exception:
        pass

    # == v7.2: Cluster boost - crowd intelligence quick win =====================
    try:
        from datetime import datetime as _dt_cl, timedelta as _td_cluster
        _cluster_window = getattr(config, "CLUSTER_WINDOW_MINUTES", 10)
        _cluster_cutoff = (_dt_cl.now() - _td_cluster(minutes=_cluster_window)).strftime("%Y-%m-%d %H:%M:%S")

        with db_manager.get_connection() as _cl_conn:
            _cluster_rows = _cl_conn.execute(
                "SELECT DISTINCT source FROM signals "
                "WHERE received_at > ? "
                "AND status IN ('EXECUTED', 'REJECTED', 'PENDING') "
                "AND source != ? "
                "AND parsed_json LIKE ?",
                (_cluster_cutoff, signal.raw_source,
                 '%"' + signal.action + '"%'),
            ).fetchall()

        _cluster_count = len(_cluster_rows)

        if _cluster_count >= 3:
            _cluster_boost = 1.5
            logger.info(
                "[RiskGuard] CLUSTER BOOST: %d sources agree on %s "
                "in last %dmin - 1.5x lots",
                _cluster_count + 1, signal.action, _cluster_window,
            )
        elif _cluster_count >= 2:
            _cluster_boost = 1.3
            logger.info(
                "[RiskGuard] CLUSTER BOOST: %d sources agree on %s - 1.3x",
                _cluster_count + 1, signal.action,
            )
        elif _cluster_count >= 1:
            _cluster_boost = 1.15
        else:
            _cluster_boost = 1.0

        if _cluster_boost > 1.0:
            _max_lots = getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03)
            order.lot_size = min(
                round(order.lot_size * _cluster_boost, 2), _max_lots
            )
            db_manager.log_audit("CLUSTER_BOOST", {
                "sources_agreeing": _cluster_count + 1,
                "boost": _cluster_boost,
                "lot_size": order.lot_size,
                "action": signal.action,
            })
    except Exception:
        pass

    # == v7.2: Scanner consensus at execution time ============================
    try:
        _scanner_agreement = 0
        _scanner_total = 0
        _agreeing_names = []

        _scanners_to_check = []
        try:
            from quant.momentum_scanner import momentum_scanner as _ms
            _scanners_to_check.append(("Momentum", getattr(_ms, 'pressure', 0)))
        except Exception:
            pass
        try:
            from quant.smc_scanner import smc_scanner as _smc
            _scanners_to_check.append(("SMC", getattr(_smc, 'pressure', 0)))
        except Exception:
            pass
        try:
            from quant.liquidity_scanner import liquidity_scanner as _liq
            _scanners_to_check.append(("Liquidity", getattr(_liq, 'pressure', 0)))
        except Exception:
            pass
        try:
            from quant.tick_flow import tick_flow_engine as _tfi
            _scanners_to_check.append(("TFI", getattr(_tfi, 'pressure', 0)))
        except Exception:
            pass
        try:
            from quant.catcd_engine import catcd_engine as _cat
            _scanners_to_check.append(("CATCD", getattr(_cat, 'pressure', 0)))
        except Exception:
            pass
        try:
            from quant.mean_reversion_engine import mr_engine as _mr
            _scanners_to_check.append(("MR", getattr(_mr, 'pressure', 0)))
        except Exception:
            pass

        for _name, _pressure in _scanners_to_check:
            _scanner_total += 1
            if signal.action == "BUY" and _pressure > 0.15:
                _scanner_agreement += 1
                _agreeing_names.append(_name)
            elif signal.action == "SELL" and _pressure < -0.15:
                _scanner_agreement += 1
                _agreeing_names.append(_name)

        if _scanner_total >= 3:
            _agree_pct = _scanner_agreement / _scanner_total
            if _agree_pct < 0.33:
                order.lot_size = max(round(order.lot_size * 0.5, 2), 0.01)
                logger.warning(
                    "[RiskGuard] SCANNER DISAGREE: Only %d/%d scanners support "
                    "%s - lots halved to %.2f",
                    _scanner_agreement, _scanner_total,
                    signal.action, order.lot_size,
                )
            elif _agree_pct >= 0.66:
                logger.info(
                    "[RiskGuard] SCANNER CONFIRM: %d/%d scanners agree on %s (%s)",
                    _scanner_agreement, _scanner_total,
                    signal.action, ", ".join(_agreeing_names),
                )
    except Exception:
        pass

    # == v7.2: Celebration exhaustion detector =================================
    try:
        from datetime import datetime as _dt_cel, timedelta as _td_celeb
        _celeb_cutoff = (_dt_cel.now() - _td_celeb(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

        with db_manager.get_connection() as _cel_conn:
            _celeb_count = _cel_conn.execute(
                "SELECT COUNT(DISTINCT source) FROM signals "
                "WHERE received_at > ? "
                "AND source NOT LIKE 'AUTO_%' "
                "AND ("
                "  raw_text LIKE '%TP%hit%' OR raw_text LIKE '%TP%done%' "
                "  OR raw_text LIKE '%profit%secured%' OR raw_text LIKE '%pips%locked%' "
                "  OR raw_text LIKE '%target%achieved%'"
                ")",
                (_celeb_cutoff,),
            ).fetchone()[0]

        if _celeb_count >= 3:
            logger.warning(
                "[RiskGuard] CELEBRATION EXHAUSTION: %d channels celebrating "
                "in last 5min - move may be exhausted. Lots halved.",
                _celeb_count,
            )
            order.lot_size = max(round(order.lot_size * 0.5, 2), 0.01)
    except Exception:
        pass

    # v8.2: Breakout mode lot boost — override to max lots
    if _in_breakout_trade and order is not None:
        _bo_max = getattr(config, "MAX_SINGLE_TRADE_LOTS", 0.03)
        if order.lot_size < _bo_max:
            order.lot_size = _bo_max
            logger.info(
                "[RiskGuard] BREAKOUT MODE: lots -> %.2f (maximum)",
                _bo_max,
            )

    # v8.1: Enforce budget override / consecutive loss shrink (min lots)
    if getattr(signal, "_budget_override", False) and order is not None:
        order.lot_size = 0.01
        logger.info("[RiskGuard] BUDGET OVERRIDE applied: lots -> 0.01")
    if getattr(signal, "_consecutive_loss_shrink", False) and order is not None:
        if not _in_breakout_trade:
            order.lot_size = 0.01
            logger.info("[RiskGuard] CONSECUTIVE LOSS SHRINK applied: lots -> 0.01")

    if order is not None:
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

    # v4.3: Per-symbol position limit with pyramid bypass
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
        if _in_breakout_trade:
            max_per_sym = getattr(config, "BREAKOUT_MAX_CONCURRENT", 2)
        if len(same_symbol_same_dir) >= max_per_sym:
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
    # v4.4: Spread & Slippage Guard
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

    try:
        _blackout_alpha_tier = alpha_tier
    except NameError:
        _blackout_alpha_tier = None

    # v4.4: Dual Time Blackout
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

    # ── 5b. Global Bias Gate ──────────────────────────────────────────────────
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

    # v7.1: Weak HTF disagreement -> allow but halve lot size
    _htf_lot_penalty = 1.0

    # == v8.0: NY Sell Bias - NY SELL is +$413, best session+direction combo ===
    _ny_sell_boost = 1.0
    try:
        from datetime import datetime as _dt_ny, timezone as _tz_ny
        _ny_h = _dt_ny.now(_tz_ny.utc).hour
        if 13 <= _ny_h < 20:
            if signal.action == "BUY" and signal.confidence < 8 and not is_high_conviction:
                return _reject(
                    "NY_BUY_FILTER: BUY signals during NY need confidence >= 8 "
                    "(got %d). NY SELL has +$413, BUY is weaker." % signal.confidence,
                    "NY_BUY_FILTER",
                )
            elif signal.action == "SELL":
                _ny_sell_boost = 1.2
                logger.info("[RiskGuard] NY SELL BOOST: 1.2x lots (session bias +$413)")
    except Exception:
        pass
    if "HTF_WEAK_AGAINST" in htf_reason:
        _htf_lot_penalty = 0.5
        logger.info("[RiskGuard] v7.1 HTF_WEAK: lot penalty 0.5x for weak counter-trend")

    # ── 5b-2. v6.2 CHOP REGIME HARD GATE ────────────────────────────────────
    _chop_lot_dampener = 1.0
    if getattr(config, "CHOP_FILTER_ENABLED", True):
        try:
            from quant.chop_filter import check as chop_check, get_lot_dampener
            chop_tradeable, chop_score, chop_details = chop_check(signal.symbol)
            if not chop_tradeable:
                # v8.2: Breakout exempt from chop block
                # v8.1: Premium signals (Convergence, Canary, Tier S) exempt
                _chop_exempt = _in_breakout_trade
                if not _chop_exempt and signal.raw_source in (
                    "AUTO_CONVERGENCE", "AUTO_CANARY",
                ):
                    _chop_exempt = True
                if not _chop_exempt:
                    try:
                        if signal.raw_source.startswith("telegram:"):
                            _chop_ch = int(signal.raw_source.split(":")[1])
                            if _chop_ch in getattr(config, "TIER_S_CHANNELS", set()):
                                _chop_exempt = True
                    except Exception:
                        pass
                if _chop_exempt:
                    logger.warning(
                        "[RiskGuard] CHOP EXEMPT: %s bypasses chop block "
                        "(score=%.2f, breakout=%s)",
                        signal.raw_source, chop_score, _in_breakout_trade,
                    )
                else:
                    return _reject(
                        f"CHOP REGIME: tradeability={chop_score:.2f} "
                        f"(CI={chop_details['choppiness_index']:.1f} "
                        f"WDR={chop_details['wick_dominance']:.2f} "
                        f"DCS={chop_details['directional_consistency']:.2f}) "
                    f"\u2014 market is untradeable, blocking ALL entries",
                    "CHOP_REGIME",
                )
            _chop_lot_dampener = get_lot_dampener(signal.symbol)
        except Exception as _chop_err:
            logger.debug("[RiskGuard] Chop filter error (pass-through): %s", _chop_err)
            _chop_lot_dampener = 1.0

    # ── 5b-3. v6.2 SESSION LOSS DAMPENER ─────────────────────────────────
    _session_loss_dampener = 1.0
    if getattr(config, "SESSION_LOSS_DAMPENER_ENABLED", True):
        try:
            from quant.breakout_guard import get_session_loss_dampener
            _session_loss_dampener = get_session_loss_dampener(signal.symbol)
        except Exception:
            pass

    # ── 5c. Breakout Kill Switch ──────────────────────────────────────────────
    bo_blocked, bo_reason = is_counter_trend_blocked(signal.action, signal.raw_source)
    if bo_blocked:
        return _reject(bo_reason, "BREAKOUT_BLOCK")

    # ── 5c. Directional Circuit Breaker ───────────────────────────────────────
    dir_blocked, dir_reason = is_direction_blocked(signal.action, signal.symbol)
    if dir_blocked:
        if "Consecutive loss pause" in dir_reason:
            if _in_breakout_trade:
                logger.warning(
                    "[RiskGuard] BREAKOUT overrides consecutive loss pause"
                )
            else:
                signal._consecutive_loss_shrink = True
                logger.warning(
                    "[RiskGuard] CONSECUTIVE LOSS SHRINK: %s -- "
                    "continuing at 0.01 lots",
                    dir_reason[:80],
                )
        else:
            return _reject(dir_reason, "DIRECTION_BLOCK")

    # ── 5d. Order Flow Toxicity Gate + Signal Inversion Engine ────────────────
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
                    # v8.1+v8.2: Allow premium/breakout signals with min lots
                    _is_premium = _in_breakout_trade or signal.raw_source in (
                        "AUTO_CONVERGENCE", "AUTO_CANARY",
                    )
                    if not _is_premium:
                        try:
                            if signal.raw_source.startswith("telegram:"):
                                _bud_ch = int(signal.raw_source.split(":")[1])
                                if _bud_ch in getattr(config, "TIER_S_CHANNELS", set()):
                                    _is_premium = True
                                if _bud_ch in getattr(config, "LEAD_CHANNELS", set()):
                                    _is_premium = True
                        except Exception:
                            pass
                    if _is_premium:
                        signal._budget_override = True
                        logger.warning(
                            "[RiskGuard] SESSION BUDGET OVERRIDE: %s "
                            "allowed despite exhaustion (premium/breakout, "
                            "lots forced to 0.01)",
                            signal.raw_source,
                        )
                    else:
                        return _reject(
                            f"SESSION BUDGET EXHAUSTED: {_current_session} "
                            f"DD={_sess_dd:.1f}% >= session limit {_sess_dd_limit:.1f}% "
                            f"(budget={_sess_budget_pct:.0%} of daily {config.DAILY_DRAWDOWN_LIMIT_PCT}%)",
                            "SESSION_BUDGET",
                        )
    except Exception:
        pass

    # ── 5e. Execution Dedup ───────────────────────────────────────────────────
    dup_ok, dup_reason = check_execution_dedup(signal.symbol, signal.action)
    if not dup_ok:
        return _reject(dup_reason, "EXEC_DEDUP")

    # ── 6. News filter (Pillar 4) ─────────────────────────────────────────────
    await news_filter.refresh_if_needed()
    blocked, news_reason = news_filter.is_blocked(signal.symbol)
    if blocked:
        return _reject(news_reason, "NEWS")

    # ── 7. Confluence check (Pillars 2 & 3) ──────────────────────────────────
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

    # AI Confidence Override
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

    # ATR SL floor with R:R Preservation
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

    # Volatility-Adaptive TP Expansion
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

    # ── v6.4: Minimum R:R for scanner signals ──────────────────────────────
    if signal.raw_source.startswith('AUTO_') and signal.stop_loss and signal.tp1 and entry:
        _sl_dist = abs(entry - signal.stop_loss)
        _tp1_dist = abs(signal.tp1 - entry)
        _min_rr = getattr(config, 'MIN_RR_SCANNER', 1.3)
        if _sl_dist > 0:
            _actual_rr = _tp1_dist / _sl_dist
            if _actual_rr < _min_rr:
                return _reject(
                    'R:R too low for scanner signal: ' + format(_actual_rr, '.2f') + ':1 < ' + str(_min_rr) + ':1 minimum',
                    'LOW_RR',
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

    # v6.2: Apply chop and session-loss dampeners to lot size
    _v62_combined_dampener = min(_chop_lot_dampener, _session_loss_dampener)
    if _v62_combined_dampener < 1.0:
        sizing.lot_size = max(round(sizing.lot_size * _v62_combined_dampener, 2), 0.01)
        logger.info(
            "[RiskGuard] v6.2 DAMPENER: chop=%.2f session=%.2f combined=%.2f lots=%.2f",
            _chop_lot_dampener, _session_loss_dampener, _v62_combined_dampener, sizing.lot_size,
        )

    # Toxicity-adaptive sizing
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

    # v6.1: Momentum Decay + Tight SL Lot Dampener
    if getattr(config, 'SLOPE_DECAY_DAMPENER_ENABLED', False):
        try:
            _decay_applied = False
            _sl_dist_raw = abs(entry - signal.stop_loss) if signal.stop_loss else 0
            _atr_for_ratio = atr_value if atr_value > 0 else _sl_dist_raw

            _live_slope = 0.0
            _slope_delta_val = 0.0
            try:
                from quant.momentum_scanner import momentum_scanner as _mscan
                _live_slope = _mscan.last_slope
                _slope_delta_val = _mscan.slope_delta
            except Exception:
                pass

            # Penalty 1: Momentum decay
            _slope_threshold = getattr(config, 'SLOPE_DECAY_THRESHOLD', 0.40)
            if (signal.raw_source == "AUTO_PULLBACK"
                    and abs(_live_slope) < _slope_threshold
                    and abs(_live_slope) > 0.01):
                _decay_mult = getattr(config, 'SLOPE_DECAY_LOT_MULTIPLIER', 0.50)
                _old_lot = sizing.lot_size
                sizing.lot_size = round(sizing.lot_size * _decay_mult, 2)
                logger.warning(
                    "[RiskGuard] SLOPE_DECAY_DAMPENER: slope=%.2f (< %.2f threshold) "
                    "lots %.2f -> %.2f (%.0f%% penalty)",
                    _live_slope, _slope_threshold, _old_lot, sizing.lot_size,
                    (1 - _decay_mult) * 100,
                )
                _decay_applied = True

            # Penalty 2: Tight SL
            if _sl_dist_raw > 0 and _atr_for_ratio > 0:
                _sl_atr = _sl_dist_raw / _atr_for_ratio
                _tight_threshold = getattr(config, 'TIGHT_SL_ATR_RATIO', 0.65)
                if _sl_atr < _tight_threshold:
                    _tight_mult = getattr(config, 'TIGHT_SL_LOT_MULTIPLIER', 0.60)
                    _old_lot2 = sizing.lot_size
                    sizing.lot_size = round(sizing.lot_size * _tight_mult, 2)
                    logger.warning(
                        "[RiskGuard] TIGHT_SL_DAMPENER: SL/ATR=%.2f (< %.2f) "
                        "lots %.2f -> %.2f",
                        _sl_atr, _tight_threshold, _old_lot2, sizing.lot_size,
                    )

            # Penalty 3: Rapid same-direction repeat
            _cooldown = getattr(config, 'RAPID_REPEAT_COOLDOWN_SECS', 900)
            try:
                from database import db_manager as _rdb
                import time as _time
                _recent = _rdb.get_recent_signals(limit=5) if hasattr(_rdb, 'get_recent_signals') else []
                for _rsig in _recent:
                    _rsrc = _rsig.get('source', '')
                    _ract = _rsig.get('action', '')
                    _rtime = _rsig.get('created_at', '')
                    if (_rsrc == signal.raw_source
                            and _ract == signal.action
                            and _rtime):
                        from datetime import datetime as _dt
                        try:
                            if isinstance(_rtime, str):
                                _rtime = _dt.fromisoformat(_rtime)
                            _age = (_dt.now() - _rtime).total_seconds()
                            if _age < _cooldown:
                                _repeat_mult = getattr(config, 'RAPID_REPEAT_LOT_MULTIPLIER', 0.50)
                                _old_lot3 = sizing.lot_size
                                sizing.lot_size = round(sizing.lot_size * _repeat_mult, 2)
                                logger.warning(
                                    "[RiskGuard] RAPID_REPEAT_DAMPENER: %s %s repeated in %.0fs "
                                    "lots %.2f -> %.2f",
                                    signal.raw_source, signal.action, _age,
                                    _old_lot3, sizing.lot_size,
                                )
                                break
                        except Exception:
                            pass
            except Exception:
                pass

            # ── v6.3.1 FIX: DAMPENER LOT FLOOR ──────────────────────────────
            # Prevents stacking dampeners from crushing lots to 0.01
            # This was the March 23 problem: good signals, no conviction
            _dampener_floor = getattr(config, 'DAMPENER_LOT_FLOOR', 0.03)
            if sizing.lot_size < _dampener_floor:
                logger.info(
                    "[RiskGuard] DAMPENER FLOOR: %.2f -> %.2f "
                    "(stacking dampeners hit floor)",
                    sizing.lot_size, _dampener_floor,
                )
                sizing.lot_size = _dampener_floor

            if sizing.lot_size < 0.01:
                sizing.lot_size = 0.01
        except Exception as _damp_err:
            logger.error("[RiskGuard] Dampener block error: %s", _damp_err)

    # v3.7 VIP Lot Floor — v6.2 FIX: do NOT override safety dampeners
    vip_min = getattr(config, "VIP_MIN_LOT", 0.05)
    is_vip_signal = (
        (signal.confidence >= 10 and not signal.raw_source.startswith(chr(65)+chr(85)+chr(84)+chr(79)+chr(95)))
        or signal.raw_source == "AUTO_CONVERGENCE"
    )
    _any_dampener_active = False
    try:
        _any_dampener_active = _v62_combined_dampener < 1.0
    except NameError:
        pass
    if is_vip_signal and sizing.lot_size < vip_min and not _any_dampener_active:
        logger.info(
            "[RiskGuard] VIP LOT FLOOR: %.2f -> %.2f "
            "(VIP signal: conf=%d src=%s)",
            sizing.lot_size, vip_min, signal.confidence, signal.raw_source,
        )
        sizing.lot_size = vip_min
    elif is_vip_signal and sizing.lot_size < vip_min and _any_dampener_active:
        logger.info(
            "[RiskGuard] VIP LOT FLOOR SKIPPED: dampener active (chop=%.2f) "
            "— keeping dampened lots=%.2f instead of floor=%.2f",
            _v62_combined_dampener if '_v62_combined_dampener' in dir() else 0,
            sizing.lot_size, vip_min,
        )

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

    lot_size = round(sizing.lot_size, 2)

    # v6.0: Prop-Firm Finisher lot scaling
    if getattr(config, "PROP_FIRM_PHASE", "") == "CHALLENGE":
        try:
            from quant.prop_firm_finisher import prop_firm_finisher as _pfin
            _coast_check = _pfin.check_override(
                signal.raw_source, signal.confidence, signal.action
            )
            if _coast_check.active and 0 < _coast_check.lot_multiplier < 1.0:
                _coast_lot = max(round(lot_size * _coast_check.lot_multiplier, 2), 0.01)
                logger.info(
                    "[RiskGuard] COAST MODE SIZING: %.2f -> %.2f (%.0f%%)",
                    lot_size, _coast_lot, _coast_check.lot_multiplier * 100,
                )
                lot_size = _coast_lot
        except Exception:
            pass

    from quant.convexity_engine import HARD_LOT_CEILING

    # v5.0: ATO regime-adaptive lot sizing
    if _ato_available:
        _ato_lot_mult = get_lot_size_multiplier()
        if _ato_lot_mult != 1.0:
            lot_size = round(lot_size * _ato_lot_mult, 2)

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
    # -- Anti-Hedge Deferred Close: close conflicting position now that all gates passed
    if _ah_conflict is not None:
        _ah_ticket = _ah_conflict["ticket"]
        _ah_type = _ah_conflict["type"]
        _ah_profit = _ah_conflict.get("profit", 0)
        logger.warning(
            "[RiskGuard] ANTI-HEDGE OVERRIDE: Closing %s %s "
            "(ticket %d, PnL=$%.2f) to open %s -- "
            "all validation gates passed",
            signal.symbol, _ah_type, _ah_ticket,
            _ah_profit, signal.action,
        )
        try:
            from mt5_executor.mt5_executor import close_partial
            close_partial(_ah_ticket, _ah_conflict["volume"])
            db_manager.log_audit("ANTI_HEDGE_OVERRIDE", {
                "closed_ticket": _ah_ticket,
                "closed_type": _ah_type,
                "closed_pnl": round(_ah_profit, 2),
                "new_action": signal.action,
                "new_confidence": signal.confidence,
                "source": signal.raw_source,
            })
        except Exception as _close_err:
            logger.error("[RiskGuard] Failed to close conflicting position: %s", _close_err)
            return _reject(
                "ANTI-HEDGE: Failed to close conflicting " + _ah_type,
                "ANTI_HEDGE",
            )

    get_trade_logger().info("APPROVE | %s | %s | lots=%.2f tier=%s method=%s", signal.symbol, signal.action, sizing.lot_size, alpha_tier, sizing.method)
    db_manager.log_audit("RISK_APPROVED", {
        "symbol": signal.symbol, "action": signal.action, "lot": sizing.lot_size,
        "risk_pct": sizing.risk_pct, "method": sizing.method, "tier": alpha_tier,
        "source": signal.raw_source, "hc": is_high_conviction,
    })

    # v4.6: Graduated DD response — REDUCED mode
    try:
        _red_opening = db_manager.get_opening_equity()
        if _red_opening and _red_opening > 0 and account_equity > 0:
            _red_dd_pct = ((_red_opening - account_equity) / _red_opening) * 100.0
            _red_fraction = _red_dd_pct / config.DAILY_DRAWDOWN_LIMIT_PCT if config.DAILY_DRAWDOWN_LIMIT_PCT > 0 else 0
            _reduced_thr = getattr(config, "DD_REDUCED_MODE_THRESHOLD_PCT", 0.50)
            _reduced_mult = getattr(config, "DD_REDUCED_MODE_LOT_MULTIPLIER", 0.60)
            if _red_fraction >= _reduced_thr:
                _reduced_lot = max(round(order.lot_size * _reduced_mult, 2), 0.01)
                logger.warning(
                    "[RiskGuard] REDUCED MODE: DD=%.1f%% (%.0f%% of limit). Lots: %.2f -> %.2f",
                    _red_dd_pct, _red_fraction * 100, order.lot_size, _reduced_lot,
                )
                db_manager.log_audit("DD_REDUCED_SIZING", {
                    "dd_pct": round(_red_dd_pct, 2), "dd_fraction": round(_red_fraction, 2),
                    "original_lots": order.lot_size, "reduced_lots": _reduced_lot,
                })
                order.lot_size = _reduced_lot
                order.dd_mode = "REDUCED"
    except Exception as _red_err:
        logger.debug("[RiskGuard] Reduced DD check error: %s", _red_err)

    return True, "Approved", order


_velocity_halt_time: float = 0.0

def _check_equity_velocity(current_equity: float):
    global _velocity_halt_time
    now = time.time()
    _equity_history.append((now, current_equity))

    auto_resume_mins = getattr(config, 'VELOCITY_AUTO_RESUME_MINS', 15)
    if TRADING_HALTED and _velocity_halt_time > 0:
        elapsed_mins = (now - _velocity_halt_time) / 60.0
        if elapsed_mins >= auto_resume_mins:
            logger.info(
                "[RiskGuard] VELOCITY AUTO-RESUME: %.0f min since halt. Resuming.",
                elapsed_mins,
            )
            resume_trading()
            _velocity_halt_time = 0.0
            _equity_history.clear()
            db_manager.log_audit("VELOCITY_AUTO_RESUME", {
                "minutes_halted": round(elapsed_mins, 1),
                "equity_at_resume": round(current_equity, 2),
            })
        return

    if TRADING_HALTED:
        return

    cutoff = now - (config.EQUITY_VELOCITY_WINDOW_MINS * 60)
    window_start_equity = None
    for ts, eq in _equity_history:
        if ts >= cutoff:
            window_start_equity = eq
            break
    if window_start_equity is None or window_start_equity <= 0:
        return
    drop_pct = ((window_start_equity - current_equity) / window_start_equity) * 100
    if drop_pct >= config.EQUITY_VELOCITY_DROP_PCT:
        reason = (
            "Equity velocity breaker: "
            + str(round(window_start_equity, 0)) + " -> " + str(round(current_equity, 0))
            + " (" + str(round(drop_pct, 2)) + "% drop in "
            + str(config.EQUITY_VELOCITY_WINDOW_MINS) + " min)"
        )
        halt_trading(reason)
        _velocity_halt_time = now
        logger.critical("[RiskGuard] EQUITY_VELOCITY_HALT: %s", reason)
        db_manager.log_audit("EQUITY_VELOCITY_HALT", {
            "start_equity": round(window_start_equity, 2),
            "current_equity": round(current_equity, 2),
            "drop_pct": round(drop_pct, 2),
            "window_mins": config.EQUITY_VELOCITY_WINDOW_MINS,
        })


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
                    _check_equity_velocity(equity)
        except Exception as e:
            logger.error(f"[RiskGuard] Equity monitor error: {e}")
        await asyncio.sleep(getattr(config, "EQUITY_MONITOR_INTERVAL_SECS", 30))


async def daily_reset_watcher():
    """Clean daily reset — NO monkey-patching (v6.3.1 fix)."""
    from datetime import datetime, timezone
    last_day = datetime.now(timezone.utc).date()
    while True:
        try:
            now = datetime.now(timezone.utc).date()
            if now != last_day:
                last_day = now
                resume_trading()
                try:
                    from mt5_executor import mt5_executor as _mt5_rg
                    _new_eq = _mt5_rg.get_account_equity()
                    if _new_eq and _new_eq > 0:
                        db_manager.set_opening_equity(_new_eq)
                        logger.info('[RiskGuard] New day opening equity set: %.2f', _new_eq)
                except Exception as _eq_err:
                    logger.error(f'[RiskGuard] Failed to set opening equity on daily reset: {_eq_err}')
                logger.info("[RiskGuard] Daily reset — halt cleared for new day.")
        except Exception as e:
            logger.error(f"[RiskGuard] Daily reset error: {e}")
        await asyncio.sleep(60)







