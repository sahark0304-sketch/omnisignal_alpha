"""Inject trend win bias block before Module State marker in breakout_guard.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BREAKOUT_GUARD = ROOT / "quant" / "breakout_guard.py"

MARKER = (
    "# Module State\n"
    "_breakout_direction: Optional[str] = None\n"
    "_breakout_until: float = 0.0"
)

INJECTION = """
# v4.4: Post-Win Trend Bias Lock
TREND_WIN_BIAS_SECS = 300  # 5 minute counter-trend block after trend-aligned win
_trend_win_bias: Dict[str, Dict] = {}  # symbol -> {direction, until_ts}


def register_trend_win(symbol: str, action: str, regime: str):
    '''Called when a profitable trade closes in FAST_TREND regime.
    Blocks counter-trend signals for TREND_WIN_BIAS_SECS.'''
    if regime != "FAST_TREND":
        return
    until = time.time() + TREND_WIN_BIAS_SECS
    _trend_win_bias[symbol] = {"direction": action, "until_ts": until}
    logger.warning(
        "[BreakoutGuard] TREND BIAS LOCK: %s %s won in %s | "
        "counter-trend blocked for %ds",
        symbol, action, regime, TREND_WIN_BIAS_SECS,
    )


def is_trend_bias_blocked(signal_action: str, signal_symbol: str = "XAUUSD") -> Tuple[bool, str]:
    '''Block signals that oppose a recent trend-aligned win.'''
    bias = _trend_win_bias.get(signal_symbol)
    if not bias:
        return False, ""
    if time.time() > bias["until_ts"]:
        _trend_win_bias.pop(signal_symbol, None)
        return False, ""
    if signal_action == bias["direction"]:
        return False, ""
    remaining = int(bias["until_ts"] - time.time())
    return True, (
        f"Trend bias lock: {bias['direction']} won in FAST_TREND | "
        f"counter-trend {signal_action} blocked ({remaining}s remaining)"
    )


""".lstrip("\n")


def main() -> None:
    raw = BREAKOUT_GUARD.read_bytes()
    text = raw.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    idx = text.find(MARKER)
    if idx == -1:
        raise SystemExit(f"Marker not found in {BREAKOUT_GUARD}")

    new_text = text[:idx] + INJECTION + text[idx:]
    BREAKOUT_GUARD.write_text(new_text, encoding="utf-8", newline="\n")
    print("INJECT OK:", BREAKOUT_GUARD)


if __name__ == "__main__":
    main()
