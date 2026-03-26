"""
quant/self_correction.py — OmniSignal Alpha v2.0
Pillar 14: Automated Post-Trade Self-Correction

Feedback loop that runs every SELF_CORRECTION_REVIEW_HOURS:
  1. Queries closed trades + their original black-box decisions
  2. Uses Gemini to audit each decision for AI misinterpretations
  3. Identifies systematic patterns (e.g. "always misses 'cancel all' phrases")
  4. Generates correction rules and writes them to prompt_corrections.json
  5. Calls ai_engine.load_prompt_corrections() to inject rules into live prompts

Detection targets:
  - Signal parsed as BUY/SELL when message was a cancellation
  - Wrong symbol extracted (Gold → BTCUSDT)
  - Wrong direction (BUY when SELL was implied)
  - Confident parse on chatter/analysis (false positive)
  - Missed TP/SL updates that led to bad sizing
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from google import genai
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_AUDIT_PROMPT = """
You are a senior trading system quality auditor. Your job is to find AI parsing mistakes.

Below are recent closed trades with their original signal text and AI parse decisions.
Some trades may have been losers because the AI misinterpreted the signal.

Analyze each trade and identify SYSTEMATIC parsing mistakes the AI keeps making.
Focus on:
1. Signals that contained 'cancel', 'void', 'close early', 'target reached', 'be careful',
   'invalidated' — but were still parsed as new BUY/SELL entries
2. Symbol extraction errors (e.g. "Gold" parsed as BTCUSDT instead of XAUUSD)
3. Direction errors (BUY when message said 'looking to SELL', 'bearish', 'short')
4. False positives — commentary/analysis parsed as a trade signal
5. Missing management signals (TP modifications, SL updates ignored)

Closed Trades Data (JSON):
{trades_json}

Return ONLY a valid JSON object:
{{
  "mistakes_found": integer,
  "correction_rules": [
    "rule 1 text — specific, actionable instruction for the parser",
    "rule 2 text",
    ...
  ],
  "summary": "one paragraph describing the systematic issues found"
}}

Rules must be:
- Specific enough to change parsing behaviour (not vague like "be more careful")
- Written as direct instructions to the AI parser (e.g. "If message contains 'cancel' anywhere, set confidence=0")
- At most 10 rules total

Return valid JSON only. No markdown.
"""


class SelfCorrectionEngine:
    def __init__(self):
        self._last_run: Optional[datetime] = None
        self._current_rules: List[str] = []
        self._load_existing_rules()

    def _load_existing_rules(self):
        path = config.PROMPT_CORRECTIONS_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self._current_rules = data.get("corrections", [])
                logger.info(f"[SelfCorrection] Loaded {len(self._current_rules)} existing rules.")
            except Exception as e:
                logger.warning(f"[SelfCorrection] Could not load rules: {e}")

    async def run_review_loop(self):
        """Background task — runs every SELF_CORRECTION_REVIEW_HOURS."""
        logger.info("[SelfCorrection] Review loop started.")
        # Initial delay — don't run immediately at startup
        await asyncio.sleep(3600)
        while True:
            try:
                await self._run_review()
            except Exception as e:
                logger.error(f"[SelfCorrection] Review error: {e}", exc_info=True)
            await asyncio.sleep(config.SELF_CORRECTION_REVIEW_HOURS * 3600)

    async def _run_review(self):
        if not config.SELF_CORRECTION_ENABLED:
            return

        trades = self._fetch_recent_closed_trades()
        if len(trades) < config.SELF_CORRECTION_MIN_SAMPLES:
            logger.info(
                f"[SelfCorrection] Only {len(trades)} closed trades — need "
                f"{config.SELF_CORRECTION_MIN_SAMPLES} to run review."
            )
            return

        logger.info(f"[SelfCorrection] Running review on {len(trades)} trades...")

        try:
            trades_json = json.dumps(trades, indent=2, default=str)[:8000]  # Truncate for API
            prompt = _AUDIT_PROMPT.format(trades_json=trades_json)

            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: _client.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
                ),
                timeout=30.0
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            new_rules = result.get("correction_rules", [])
            summary   = result.get("summary", "")
            mistakes  = result.get("mistakes_found", 0)

            logger.info(
                f"[SelfCorrection] Review complete | mistakes={mistakes} "
                f"new_rules={len(new_rules)}"
            )
            logger.info(f"[SelfCorrection] Summary: {summary}")

            if new_rules:
                await self._merge_and_save_rules(new_rules, summary)
                # Reload into ai_engine immediately
                from ai_engine.ai_engine import load_prompt_corrections
                load_prompt_corrections()
                logger.info("[SelfCorrection] ✅ Prompt corrections updated and reloaded.")

                from utils.notifier import notify
                notify(
                    f"🧠 *Self-Correction Update*\n"
                    f"Found {mistakes} systematic mistakes.\n"
                    f"Added {len(new_rules)} new parsing rules."
                )

            self._last_run = datetime.now()

        except json.JSONDecodeError as e:
            logger.error(f"[SelfCorrection] JSON parse failed: {e}")
        except asyncio.TimeoutError:
            logger.error("[SelfCorrection] Review timed out.")
        except Exception as e:
            logger.error(f"[SelfCorrection] Unexpected error: {e}")

    async def _merge_and_save_rules(self, new_rules: List[str], summary: str):
        """
        Merge new rules with existing ones. Deduplicate and cap at 15 total.
        Older rules stay unless semantically superseded (basic dedup by similarity).
        """
        combined = list(self._current_rules)
        for rule in new_rules:
            # Simple dedup: skip if a very similar rule already exists
            rule_lower = rule.lower()
            is_dup = any(
                len(set(rule_lower.split()) & set(existing.lower().split())) /
                max(len(rule_lower.split()), 1) > 0.7
                for existing in combined
            )
            if not is_dup:
                combined.append(rule)

        # Keep most recent 15 rules
        combined = combined[-15:]
        self._current_rules = combined

        data = {
            "last_updated": datetime.now().isoformat(),
            "last_summary": summary,
            "corrections": combined,
        }
        path = config.PROMPT_CORRECTIONS_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[SelfCorrection] Saved {len(combined)} rules to {path}")

    def _fetch_recent_closed_trades(self) -> List[Dict]:
        """
        Fetches closed trades joined with their original black box decisions.
        Provides the raw signal text + AI decision for the auditor.
        """
        try:
            from database import db_manager
            from quant.black_box import query_decisions

            cutoff = (datetime.now() - timedelta(days=7)).isoformat()

            with db_manager.get_connection() as conn:
                rows = conn.execute("""
                    SELECT
                        t.ticket, t.symbol, t.action, t.lot_size,
                        t.entry_price, t.close_price, t.pnl,
                        t.open_time, t.close_time,
                        s.source, s.raw_text, s.ai_confidence, s.reject_reason
                    FROM trades t
                    LEFT JOIN signals s ON t.signal_id = s.id
                    WHERE t.status = 'CLOSED'
                      AND t.close_time >= ?
                    ORDER BY t.close_time DESC
                    LIMIT 100
                """, (cutoff,)).fetchall()

            trades = []
            for r in rows:
                trade = dict(r)
                # Attach black box decision if available
                bb_decisions = query_decisions(
                    limit=1, source=trade.get("source", ""),
                    symbol=trade.get("symbol", "")
                )
                if bb_decisions:
                    bb = bb_decisions[0]
                    trade["bb_ai_reasoning"]   = bb.get("ai_reasoning", "")
                    trade["bb_ai_category"]    = bb.get("ai_category", "")
                    trade["bb_confluence"]     = bb.get("confluence_details", "")
                    trade["bb_risk_rejected"]  = bb.get("risk_reject_reason", "")
                trades.append(trade)

            return trades
        except Exception as e:
            logger.error(f"[SelfCorrection] Fetch failed: {e}")
            return []



async def post_trade_forensic(ticket: int, pnl: float, tp1_hit: bool, entry_price: float, close_price: float, symbol: str, action: str):
    """
    v4.2 Instant Forensic Post-Mortem.
    Captures market microstructure at close, updates ML weights immediately.
    """
    try:
        from quant.win_model import win_model
        from database import db_manager

        pip_size = 0.1 if "XAU" in symbol else 0.0001
        entry_exit_pips = abs(close_price - entry_price) / pip_size

        result = "WIN" if pnl > 0 else "LOSS"
        efficiency = "TP_HIT" if tp1_hit else ("SL_HIT" if pnl < -1 else "EARLY_EXIT")

        # Capture market microstructure at close time
        microstructure = {}
        try:
            import MetaTrader5 as _mt5
            rates = _mt5.copy_rates_from_pos(symbol, _mt5.TIMEFRAME_M5, 0, 10)
            if rates is not None and len(rates) >= 5:
                import numpy as _np
                closes = rates["close"].astype(float)
                highs = rates["high"].astype(float)
                lows = rates["low"].astype(float)
                vols = rates["tick_volume"].astype(float)

                tr = _np.maximum(
                    highs[-5:] - lows[-5:],
                    _np.maximum(
                        _np.abs(highs[-5:] - closes[-6:-1]) if len(closes) > 5 else highs[-5:] - lows[-5:],
                        _np.abs(lows[-5:] - closes[-6:-1]) if len(closes) > 5 else highs[-5:] - lows[-5:],
                    ),
                )
                atr_at_close = float(_np.mean(tr))
                vol_at_close = float(_np.mean(vols[-5:]))
                momentum = float(closes[-1] - closes[-5]) / pip_size

                microstructure = {
                    "atr_at_close": round(atr_at_close / pip_size, 1),
                    "vol_at_close": round(vol_at_close, 0),
                    "momentum_5bar": round(momentum, 1),
                }
        except Exception:
            pass

        # Capture consensus state at close
        consensus_at_close = {}
        try:
            from quant.convergence_engine import convergence_engine
            consensus_at_close = convergence_engine.get_consensus_score()
        except Exception:
            pass

        # Capture AMD phase at close
        amd_at_close = {}
        try:
            from quant.amd_engine import amd_engine
            amd_at_close = amd_engine.get_state()
        except Exception:
            pass

        forensic = {
            "ticket": ticket,
            "result": result,
            "pnl": round(pnl, 2),
            "entry_exit_pips": round(entry_exit_pips, 1),
            "tp1_hit": tp1_hit,
            "efficiency": efficiency,
            "symbol": symbol,
            "action": action,
            "microstructure": microstructure,
            "consensus_at_close": consensus_at_close.get("score", 0),
            "amd_phase": amd_at_close.get("phase", "UNKNOWN"),
        }

        # SHAP Explainability: append top drivers to forensic record
        try:
            shap_result = win_model.explain_prediction({
                "symbol": symbol, "action": action,
                "ai_confidence": forensic.get("consensus_at_close", 7),
            })
            if shap_result:
                forensic["shap_top5"] = shap_result["top5_summary"]
                lesson = (
                    f"{result} ${pnl:+.2f} | {efficiency} | "
                    f"SHAP drivers: {shap_result['top5_summary']}"
                )
                forensic["lesson_learned"] = lesson
                logger.info("[Forensic] SHAP lesson: %s", lesson)
        except Exception as _shap_err:
            logger.debug("[Forensic] SHAP unavailable: %s", _shap_err)

        db_manager.log_audit("POST_TRADE_FORENSIC", forensic)

        # v5.0: Trade Autopsy Telegram Alert
        try:
            from utils.notifier import send_autopsy
            emoji = "🟢" if pnl > 0 else "🔴"
            res_label = "WIN" if pnl > 0 else "LOSS"
            duration_mins = 0
            try:
                trade_rec = db_manager.get_open_trades()
                t_rec = next((t for t in trade_rec if t.get("ticket") == ticket), None)
                if t_rec and t_rec.get("open_time"):
                    import datetime as _dt
                    ot = t_rec["open_time"]
                    if isinstance(ot, str):
                        ot = _dt.datetime.fromisoformat(ot)
                    duration_mins = int((_dt.datetime.utcnow() - ot).total_seconds() / 60)
            except Exception:
                pass

            shap_section = ""
            conclusion = ""
            shap_top5 = forensic.get("shap_top5", "")
            if shap_top5:
                drivers = [s.strip() for s in shap_top5.split(",")][:3]
                shap_lines = []
                for d in drivers:
                    parts = d.split("=")
                    if len(parts) == 2:
                        fname = parts[0].strip()
                        fval = parts[1].strip()
                        direction = "↑" if not fval.startswith("-") else "↓"
                        shap_lines.append(f"  {direction} {fname} ({fval})")
                if shap_lines:
                    shap_section = "*Top ML Drivers:*\n" + "\n".join(shap_lines)
                    top_feat = drivers[0].split("=")[0].strip() if "=" in drivers[0] else drivers[0]
                    top_val = drivers[0].split("=")[1].strip() if "=" in drivers[0] else ""
                    sign_word = "positive" if not top_val.startswith("-") else "negative"
                    conclusion = f"Trade {res_label.lower()}. Strongest {sign_word} factor was {top_feat} ({top_val})."
            else:
                conclusion = f"Trade {res_label.lower()}. SHAP analysis unavailable for this trade."

            dur_str = f" | Duration: {duration_mins}m" if duration_mins > 0 else ""
            autopsy_msg = (
                f"{emoji} *TRADE AUTOPSY: #{ticket}*\n"
                f"\n"
                f"Result: *{res_label}* ${pnl:+.2f}{dur_str}\n"
                f"Symbol: {symbol} | Action: {action}\n"
                f"Exit: {efficiency}\n"
                f"\n"
            )
            if shap_section:
                autopsy_msg += shap_section + "\n\n"
            autopsy_msg += f"_{conclusion}_"

            send_autopsy(autopsy_msg)
            logger.info("[Forensic] Trade autopsy sent for ticket %d", ticket)
        except Exception as _autopsy_err:
            logger.debug("[Forensic] Autopsy send failed: %s", _autopsy_err)

        # Immediate ML retraining trigger
        win_model.check_and_retrain()

        # Update convergence engine weights based on outcome
        try:
            from quant.convergence_engine import convergence_engine
            cs = consensus_at_close
            if cs and cs.get("components"):
                _update_engine_weights(cs["components"], action, pnl > 0)
        except Exception:
            pass

        # v4.3.1: Extract and amplify success fingerprint for big wins
        try:
            _extract_success_fingerprint(ticket, pnl, action, symbol)
        except Exception:
            pass

        if pnl < -20:
            logger.warning(
                "[Forensic] HEAVY LOSS $%.2f on %s %s | %s | consensus=%d | amd=%s",
                pnl, symbol, action, efficiency,
                consensus_at_close.get("score", 0),
                amd_at_close.get("phase", "?"),
            )
        elif pnl > 50:
            logger.info(
                "[Forensic] STRONG WIN $%.2f on %s %s | %s | consensus=%d | amd=%s",
                pnl, symbol, action, efficiency,
                consensus_at_close.get("score", 0),
                amd_at_close.get("phase", "?"),
            )

        logger.info(
            "[Forensic] %s | %s %s | P&L: $%.2f | %s | %.1f pips | consensus=%d",
            result, symbol, action, pnl, efficiency, entry_exit_pips,
            consensus_at_close.get("score", 0),
        )

    except Exception as e:
        logger.error("[Forensic] Post-trade analysis failed: %s", e)


def _extract_success_fingerprint(ticket: int, pnl: float, action: str, symbol: str):
    """
    v4.3.1: Extract and store the 'Success Fingerprint' of high-PnL trades.
    When a trade wins big ($50+), capture the exact market conditions for
    future high-conviction scaling decisions.
    """
    if pnl < 50:
        return

    try:
        import MetaTrader5 as _mt5
        from database import db_manager
        from quant.convergence_engine import convergence_engine, SCANNER_WEIGHTS
        from quant.amd_engine import amd_engine

        fingerprint = {
            "ticket": ticket,
            "pnl": round(pnl, 2),
            "action": action,
            "symbol": symbol,
        }

        # Capture consensus state
        try:
            cs = convergence_engine.get_consensus_score()
            fingerprint["consensus_score"] = cs.get("score", 0)
            fingerprint["consensus_direction"] = cs.get("direction", "")
            fingerprint["agreement_count"] = cs.get("agreement_count", 0)
            fingerprint["dominant_pressure"] = cs.get("dominant_pressure", 0)
        except Exception:
            pass

        # Capture AMD state
        try:
            amd = amd_engine.get_state()
            fingerprint["amd_phase"] = str(amd.get("phase", "UNKNOWN"))
            fingerprint["amd_confidence"] = amd.get("confidence", 0)
        except Exception:
            pass

        # Capture tick intensity at close
        try:
            from trade_manager.trade_manager import _measure_tick_intensity
            intensity, regime = _measure_tick_intensity(symbol)
            fingerprint["tick_intensity"] = round(intensity, 2)
            fingerprint["tick_regime"] = regime
        except Exception:
            pass

        # Capture current scanner weights (so we know which configuration produced this win)
        fingerprint["scanner_weights_at_win"] = dict(SCANNER_WEIGHTS)

        # Amplify weights: give a bigger boost to engines that produced big wins
        amplified_lr = 0.04  # 2x the normal learning rate for big wins
        components = {}
        try:
            cs = convergence_engine.get_consensus_score()
            components = cs.get("components", {})
        except Exception:
            pass

        for name, pressure in components.items():
            if name not in SCANNER_WEIGHTS:
                continue
            agreed = (
                (action == "BUY" and pressure > 0.15) or
                (action == "SELL" and pressure < -0.15)
            )
            if agreed:
                SCANNER_WEIGHTS[name] = min(0.50, SCANNER_WEIGHTS[name] + amplified_lr)

        total = sum(SCANNER_WEIGHTS.values())
        if total > 0:
            for k in SCANNER_WEIGHTS:
                SCANNER_WEIGHTS[k] = round(SCANNER_WEIGHTS[k] / total, 4)

        db_manager.log_audit("SUCCESS_FINGERPRINT", fingerprint)
        logger.info(
            "[Forensic] SUCCESS FINGERPRINT captured for ticket %d ($%.2f) | "
            "consensus=%s weights=%s",
            ticket, pnl, fingerprint.get("consensus_score", "?"),
            {k: round(v, 3) for k, v in SCANNER_WEIGHTS.items()},
        )

    except Exception as e:
        logger.debug("[Forensic] Fingerprint extraction failed: %s", e)


def _update_engine_weights(components: dict, action: str, was_win: bool):
    """
    v4.2: Micro-adjust convergence engine weights based on trade outcomes.
    Engines that agreed with winning trades get slightly boosted.
    """
    try:
        from quant.convergence_engine import SCANNER_WEIGHTS
        learning_rate = 0.02
        for name, pressure in components.items():
            if name not in SCANNER_WEIGHTS:
                continue
            agreed = (
                (action == "BUY" and pressure > 0.15) or
                (action == "SELL" and pressure < -0.15)
            )
            if agreed and was_win:
                SCANNER_WEIGHTS[name] = min(0.50, SCANNER_WEIGHTS[name] + learning_rate)
            elif agreed and not was_win:
                SCANNER_WEIGHTS[name] = max(0.05, SCANNER_WEIGHTS[name] - learning_rate)

        total = sum(SCANNER_WEIGHTS.values())
        if total > 0:
            for k in SCANNER_WEIGHTS:
                SCANNER_WEIGHTS[k] = round(SCANNER_WEIGHTS[k] / total, 4)

        logger.debug("[Forensic] Updated weights: %s", SCANNER_WEIGHTS)
    except Exception:
        pass


# Module-level singleton
self_correction = SelfCorrectionEngine()
