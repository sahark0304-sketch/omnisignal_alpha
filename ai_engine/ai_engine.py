"""
ai_engine/ai_engine.py — OmniSignal Alpha v2.0
Pillars 1 & 14: Two-stage parsing with noise gate, cancel/update detection,
                vision alignment, and runtime prompt injection from self-correction.
"""

import asyncio
import json
import base64
import os
from dataclasses import dataclass, field
from typing import Optional, List
from google import genai
from google.genai import types
import re
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)

# Runtime-injected corrections from Pillar 14 self_correction module
_ACTIVE_CORRECTIONS: List[str] = []


def load_prompt_corrections():
    """Called at startup and periodically by self_correction — injects learned rules."""
    global _ACTIVE_CORRECTIONS
    path = config.PROMPT_CORRECTIONS_FILE
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
                _ACTIVE_CORRECTIONS = data.get("corrections", [])
            logger.info(f"[AI] Loaded {len(_ACTIVE_CORRECTIONS)} prompt corrections.")
        except Exception as e:
            logger.warning(f"[AI] Could not load corrections: {e}")


# ── DATACLASS ────────────────────────────────────────────────────────────────

@dataclass
class ParsedSignal:
    symbol: str
    action: str                            # BUY / SELL / CANCEL / UPDATE / CLOSE
    entry_price: Optional[float]  = None
    stop_loss: Optional[float]    = None
    tp1: Optional[float]          = None
    tp2: Optional[float]          = None
    tp3: Optional[float]          = None
    confidence: int               = 0
    vision_confidence: int        = 0
    is_valid: bool                = False
    is_management_action: bool    = False  # True for CANCEL/UPDATE/CLOSE
    reject_reason: str            = ""
    raw_source: str               = ""
    ai_reasoning: str             = ""     # Pillar 12: saved to black box


# ── STAGE 1: NOISE GATE (fast single call, no full parse) ────────────────────

_NOISE_GATE_PROMPT = """
You are a trading signal gatekeeper. Classify the message below in ONE WORD.

Categories:
- SIGNAL  → new BUY or SELL trade setup with at least a direction and price level
- CANCEL  → explicitly cancels or invalidates a previous signal
- UPDATE  → modifies SL/TP of an existing position
- CLOSE   → instructs to close or exit a trade now
- NOISE   → commentary, analysis, news recap, greetings, emojis, polls, off-topic

Reply with EXACTLY ONE WORD from the list above. No other text.

Message:
---
{text}
---
"""


def _regex_fallback_parse(text: str, source: str) -> Optional[ParsedSignal]:
    """Degraded-mode parser: extract basic signal structure via regex when AI is down."""
    upper = text.upper()

    action_match = re.search(r'\b(BUY|SELL)\b', upper)
    if not action_match:
        return None
    action = action_match.group(1)

    sym_aliases = {
        "GOLD": "XAUUSD", "XAU": "XAUUSD", "CABLE": "GBPUSD",
        "FIBER": "EURUSD", "AUSSIE": "AUDUSD",
    }
    symbol = None
    sym_match = re.search(
        r'\b(XAUUSD|EURUSD|GBPUSD|USDJPY|AUDUSD|NZDUSD|USDCAD|USDCHF|'
        r'GBPJPY|EURJPY|XAGUSD|NAS100|US30|BTCUSDT|ETHUSDT|GOLD|XAU|CABLE|FIBER|AUSSIE)\b',
        upper,
    )
    if sym_match:
        raw_sym = sym_match.group(1)
        symbol = sym_aliases.get(raw_sym, raw_sym)

    if not symbol:
        return None

    def _extract_price(label: str) -> Optional[float]:
        pattern = rf'{label}\s*[:\-=]?\s*(\d+\.?\d*)'
        m = re.search(pattern, upper)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    entry = _extract_price("ENTRY") or _extract_price("@") or _extract_price("PRICE")
    sl = _extract_price("SL") or _extract_price("STOP")
    tp1 = _extract_price("TP1") or _extract_price("TP")
    tp2 = _extract_price("TP2")
    tp3 = _extract_price("TP3")

    if not sl:
        return None

    conf = 5
    if sl and tp1:
        conf = 6
    if sl and tp1 and entry:
        conf = 7

    signal = ParsedSignal(
        symbol=symbol, action=action, entry_price=entry,
        stop_loss=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        confidence=conf, is_valid=True,
        raw_source=source,
        ai_reasoning="REGEX_FALLBACK: Gemini unavailable, parsed via pattern matching",
    )
    logger.warning("[AI] REGEX FALLBACK: %s %s SL=%s TP1=%s (AI down)", symbol, action, sl, tp1)
    return signal


async def classify_message(text: str) -> str:
    """Fast gate — returns SIGNAL/CANCEL/UPDATE/CLOSE/NOISE before spending a full parse."""
    try:
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: _client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=_NOISE_GATE_PROMPT.format(text=text[:600]),
                )
            ),
            timeout=5.0
        )
        word = response.text.strip().upper().split()[0]
        return word if word in ("SIGNAL", "CANCEL", "UPDATE", "CLOSE", "NOISE") else "NOISE"
    except Exception:
        if _regex_fallback_parse(text, "gate_check"):
            return "SIGNAL"
        return "NOISE"


# ── STAGE 2: FULL PARSE PROMPT (with learned corrections injected) ──────────

def _build_full_parse_prompt(text: str) -> str:
    corrections_block = ""
    if _ACTIVE_CORRECTIONS:
        corrections_block = "\n\nLEARNED CORRECTIONS (apply these rules — derived from past mistakes):\n"
        for i, c in enumerate(_ACTIVE_CORRECTIONS, 1):
            corrections_block += f"{i}. {c}\n"

    return f"""
You are an institutional-grade Forex/Crypto signal parser with deep market expertise.
Your only job: extract clean, actionable trade data from the message below.{corrections_block}

Return ONLY a valid JSON object — no markdown, no backticks, no preamble:
{{
  "symbol":       "string — uppercase, no slash (EURUSD, XAUUSD, GBPJPY, BTCUSDT, US30)",
  "action":       "BUY or SELL",
  "entry_price":  number or null  (use midpoint for ranges like 1.0850-1.0860),
  "stop_loss":    number or null  (REQUIRED for confidence ≥7),
  "tp1":          number or null,
  "tp2":          number or null,
  "tp3":          number or null,
  "confidence":   integer 1-10,
  "reasoning":    "one sentence explaining your parse"
}}

Critical rules:
1. Common aliases: Gold/XAU=XAUUSD, Cable/GBP=GBPUSD, Fiber/EUR=EURUSD,
   Aussie=AUDUSD, Loonie=USDCAD, Kiwi=NZDUSD, Yen=USDJPY,
   Nas/NQ/Nasdaq=NAS100, Dow/US30=US30, Oil/Crude=XTIUSD
2. 'Market order' or 'at market' → entry_price = null
3. Multiple TPs listed → assign to tp1, tp2, tp3 in order
4. confidence ≥7 requires: symbol + action + stop_loss all present
5. If signal says 'cancel', 'void', 'disregard', 'closed early' → set confidence=0
6. R:R below 1:1 → reduce confidence by 2

Message:
---
{text}
---
"""

_MANAGEMENT_PROMPT = """
A trade management instruction was detected. Extract the details.

Return ONLY valid JSON:
{{
  "action":        "CANCEL or UPDATE or CLOSE",
  "symbol":        "string or null",
  "target_entry":  number or null  (the original entry price of the trade to act on),
  "new_sl":        number or null,
  "new_tp":        number or null,
  "reasoning":     "one sentence"
}}
Message:
---
{text}
---
"""

_VISION_PROMPT = """
You are a professional technical analyst reviewing a chart image.

Return ONLY valid JSON — no markdown:
{{
  "chart_bias":        "BULLISH or BEARISH or NEUTRAL",
  "trend_strength":    "STRONG or MODERATE or WEAK",
  "key_support":       number or null,
  "key_resistance":    number or null,
  "pattern":           "pattern name or null",
  "vision_confidence": integer 1-10,
  "reasoning":         "one sentence"
}}
"""


# ── CORE PARSE FUNCTION ──────────────────────────────────────────────────────

async def parse_text_signal(text: str, source: str) -> Optional[ParsedSignal]:
    """
    Two-stage pipeline:
      Stage 1: Noise gate (cheap, fast) — drop everything that isn't a signal
      Stage 2: Full structured parse with learned corrections injected
    """
    if not text or not text.strip():
        return None

    # Stage 1 — classify
    category = await classify_message(text)
    logger.debug(f"[AI] Gate={category} source={source}")

    if category == "NOISE":
        return None

    # Management actions routed separately
    if category in ("CANCEL", "UPDATE", "CLOSE"):
        return await _parse_management_action(text, source, category)

    # Stage 2 — full parse with retry
    data = None
    for attempt in range(config.AI_MAX_RETRIES + 1):
        try:
            prompt = _build_full_parse_prompt(text)
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: _client.models.generate_content(
                        model=config.GEMINI_MODEL, contents=prompt
                    )
                ),
                timeout=config.AI_PARSE_TIMEOUT_SECS
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            break
        except json.JSONDecodeError:
            if attempt < config.AI_MAX_RETRIES:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            logger.error("[AI] JSON parse failed after retries.")
            return None
        except asyncio.TimeoutError:
            logger.error("[AI] Parse timed out.")
            return None
        except Exception as e:
            logger.error(f"[AI] Unexpected: {e}")
            return None

    if data is None:
        fallback = _regex_fallback_parse(text, source)
        if fallback:
            return fallback
        return None

    sym = str(data.get("symbol", "")).upper().replace("/", "").replace("-", "").strip()
    act = str(data.get("action", "")).upper().strip()

    signal = ParsedSignal(
        symbol        = sym,
        action        = act,
        entry_price   = _safe_float(data.get("entry_price")),
        stop_loss     = _safe_float(data.get("stop_loss")),
        tp1           = _safe_float(data.get("tp1")),
        tp2           = _safe_float(data.get("tp2")),
        tp3           = _safe_float(data.get("tp3")),
        confidence    = int(data.get("confidence", 0)),
        ai_reasoning  = str(data.get("reasoning", "")),
        raw_source    = source,
    )

    # Validate
    if not signal.symbol or signal.action not in ("BUY", "SELL"):
        signal.reject_reason = f"Invalid symbol='{signal.symbol}' or action='{signal.action}'"
        return signal

    if signal.confidence < config.AI_CONFIDENCE_THRESHOLD:
        if not source.startswith("AUTO_"):
            signal.reject_reason = f"Low confidence {signal.confidence}/10 (threshold {config.AI_CONFIDENCE_THRESHOLD})"
            return signal

    # R:R check
    if signal.entry_price and signal.stop_loss and signal.tp1:
        sl_dist  = abs(signal.entry_price - signal.stop_loss)
        tp1_dist = abs(signal.tp1 - signal.entry_price)
        if sl_dist > 0 and (tp1_dist / sl_dist) < 0.7:
            signal.reject_reason = f"R:R={tp1_dist/sl_dist:.2f} < 0.7 — insufficient reward"
            return signal

    signal.is_valid = True
    logger.info(
        f"[AI] ✅ {signal.symbol} {signal.action} @ {signal.entry_price} "
        f"SL:{signal.stop_loss} TP1:{signal.tp1} Conf:{signal.confidence}/10"
    )
    return signal


async def _parse_management_action(text: str, source: str, category: str) -> Optional[ParsedSignal]:
    try:
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: _client.models.generate_content(
                    model=config.GEMINI_MODEL,
                    contents=_MANAGEMENT_PROMPT.format(text=text),
                )
            ),
            timeout=config.AI_PARSE_TIMEOUT_SECS
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        signal = ParsedSignal(
            symbol               = str(data.get("symbol") or "").upper().strip(),
            action               = category,
            entry_price          = _safe_float(data.get("target_entry")),
            stop_loss            = _safe_float(data.get("new_sl")),
            tp1                  = _safe_float(data.get("new_tp")),
            confidence           = 9,
            is_valid             = True,
            is_management_action = True,
            ai_reasoning         = str(data.get("reasoning", "")),
            raw_source           = source,
        )
        logger.info(f"[AI] Management: {category} | {signal.symbol} | {signal.ai_reasoning}")
        return signal
    except Exception as e:
        logger.error(f"[AI] Management parse failed: {e}")
        return None


async def analyze_chart_image(image_bytes: bytes, signal: ParsedSignal) -> ParsedSignal:
    """Vision alignment: penalise confidence if chart contradicts signal direction."""
    try:
        # Detect MIME
        import imghdr
        img_type = imghdr.what(None, h=image_bytes)
        mime = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(img_type, "image/png")
        image_part = types.Part(
            inline_data=types.Blob(mime_type=mime, data=image_bytes)
        )

        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: _client.models.generate_content(
                    model=config.GEMINI_VISION_MODEL,
                    contents=[_VISION_PROMPT, image_part],
                )
            ),
            timeout=config.AI_PARSE_TIMEOUT_SECS * 2
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        bias     = data.get("chart_bias", "NEUTRAL")
        strength = data.get("trend_strength", "WEAK")
        conf     = int(data.get("vision_confidence", 5))

        aligned = (
            (signal.action == "BUY"  and bias == "BULLISH") or
            (signal.action == "SELL" and bias == "BEARISH") or
            bias == "NEUTRAL"
        )
        if not aligned:
            penalty = 3 if strength == "STRONG" else 2
            signal.confidence = max(1, signal.confidence - penalty)
            if signal.confidence < config.AI_CONFIDENCE_THRESHOLD:
                signal.is_valid = False
            signal.reject_reason = f"Chart {bias}/{strength} vs signal {signal.action}"
            logger.warning(f"[AI-Vision] Misalignment penalty -{penalty} → conf={signal.confidence}")

        signal.vision_confidence = conf
        signal.ai_reasoning += f" | chart={bias}/{strength}"
        logger.info(f"[AI-Vision] bias={bias} strength={strength} conf={conf}/10 aligned={aligned}")
    except Exception as e:
        logger.warning(f"[AI-Vision] failed (non-fatal): {e}")
    return signal


def _safe_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
