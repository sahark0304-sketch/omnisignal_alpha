"""
config.py — OmniSignal Alpha v6.3.1
RESTORED March 17 Architecture + 3 Bug Fixes + Dampener Floor
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
#  OPERATING MODE
# ─────────────────────────────────────────────────────────────────────────────

class Mode:
    DEMO = "DEMO"
    LIVE = "LIVE"

OPERATING_MODE: str = os.getenv("OPERATING_MODE", Mode.DEMO)


# ─────────────────────────────────────────────────────────────────────────────
#  PROP FIRM PHASE
# ─────────────────────────────────────────────────────────────────────────────

class PropPhase:
    CHALLENGE = "CHALLENGE"
    FUNDED    = "FUNDED"
    PERSONAL  = "PERSONAL"

PROP_FIRM_PHASE: str = os.getenv("PROP_FIRM_PHASE", PropPhase.FUNDED)
CHALLENGE_PROFIT_TARGET_PCT: float = float(os.getenv("CHALLENGE_PROFIT_TARGET_PCT", "8.0"))
CHALLENGE_PROFIT_CURRENT_PCT: float = float(os.getenv("CHALLENGE_PROFIT_CURRENT_PCT", "0.0"))


# ─────────────────────────────────────────────────────────────────────────────
#  INGESTION
# ─────────────────────────────────────────────────────────────────────────────

TELEGRAM_API_ID: int         = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH: str       = os.getenv("TELEGRAM_API_HASH", "")
TELEGRAM_BOT_TOKEN: str      = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("NOTIFY_BOT_TOKEN") or ""
TELEGRAM_SESSION_NAME: str   = "omnisignal_session"

MONITORED_CHANNELS: List[int] = [
    -1001983734792,
    -1001347617494,
    -1001536621768,
    -1001309043988,
    -1001510927248,
    -1002489241398,
    -1002448604508,
    -1001872299004,
    -1002219243374,
    -1002293261831,
    -1003655941757,
    -1002494813464,
    -1001949192523,
    -1001623345960,
    -1001921425619,
    -1003372787430,
    -1003781656280,
    -1002273704999,
    -1001207301837,
    -1001785197109,
    -1001387511343,
    -1002176701424,
    -1001588519179,
    -1003738551742,
    -1002223574325,
    -1001814562728,
    -1001651583302,
    -1002399120063,
]

_telegram_channels_env = os.getenv("TELEGRAM_CHANNELS", "").strip()
TELEGRAM_CHANNELS: List[str] = (
    [c.strip() for c in _telegram_channels_env.split(",") if c.strip()]
    if _telegram_channels_env
    else [str(c) for c in MONITORED_CHANNELS]
)

DISCORD_BOT_TOKEN: str       = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_IDS: List[int] = [int(x) for x in os.getenv("DISCORD_CHANNEL_IDS", "").split(",") if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
#  AI ENGINE — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY: str          = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str            = "gemini-2.5-flash"
GEMINI_VISION_MODEL: str     = "gemini-2.5-flash"
AI_CONFIDENCE_THRESHOLD: int = 4
AI_PARSE_TIMEOUT_SECS: int   = 12
AI_MAX_RETRIES: int          = 2
SELF_CORRECTION_ENABLED: bool     = True
SELF_CORRECTION_MIN_SAMPLES: int  = 8
SELF_CORRECTION_REVIEW_HOURS: int = 6
PROMPT_CORRECTIONS_FILE: str      = "data/prompt_corrections.json"


# ─────────────────────────────────────────────────────────────────────────────
#  PROP FIRM DRAWDOWN GUARD
# ─────────────────────────────────────────────────────────────────────────────

MAX_DRAWDOWN_LIMIT_PCT: float  = float(os.getenv("MAX_DD_PCT", "8.0"))
DAILY_DRAWDOWN_LIMIT_PCT: float = float(os.getenv("DAILY_DD_PCT", "4.0"))
DD_OPENING_EQUITY_SERVER_UTC_OFFSET: int = int(os.getenv("BROKER_UTC_OFFSET", "2"))
EQUITY_MONITOR_INTERVAL_SECS: int = 5
DD_REDUCED_MODE_THRESHOLD_PCT: float = 0.50
DD_REDUCED_MODE_LOT_MULTIPLIER: float = 0.60
DD_BLOCK_THRESHOLD_PCT: float = 0.90
EQUITY_VELOCITY_DROP_PCT: float   = 3.50
EQUITY_VELOCITY_WINDOW_MINS: float = 5.0
VELOCITY_AUTO_RESUME_MINS: int     = 15
INITIAL_ACCOUNT_BALANCE: float = float(os.getenv("INITIAL_ACCOUNT_BALANCE", "10000"))
CHALLENGE_RISK_MULTIPLIER: float  = 1.20
FUNDED_CONSERVATIVE_MULTIPLIER: float = 0.70


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE FREQUENCY GOVERNOR — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

MAX_TRADES_PER_HOUR: int  = 6
MAX_TRADES_PER_DAY: int   = 20


# ─────────────────────────────────────────────────────────────────────────────
#  CONSECUTIVE LOSS CIRCUIT BREAKER — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONSECUTIVE_LOSSES: int           = 5
CONSECUTIVE_LOSS_MULTIPLIER: float    = 0.65
RECOVERY_STREAK_NEEDED: int           = 2
CONSECUTIVE_WIN_SCALE_UP_AFTER: int   = 3
CONSECUTIVE_WIN_SCALE_MULTIPLIER: float = 1.25


# ─────────────────────────────────────────────────────────────────────────────
#  CONFLUENCE ENGINE — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

CONFLUENCE_ENABLED: bool         = True
CONFLUENCE_TIMEFRAME: str        = "M15"
CONFLUENCE_HTF_TIMEFRAME: str    = "H4"
CONFLUENCE_MIN_SCORE: int        = 2
CONFLUENCE_RSI_PERIOD: int       = 14
CONFLUENCE_RSI_OB: float         = 65.0
CONFLUENCE_RSI_OS: float         = 35.0
CONFLUENCE_EMA_FAST: int         = 20
CONFLUENCE_EMA_SLOW: int         = 50
CONFLUENCE_EMA_TREND: int        = 200
CONFLUENCE_MACD_FAST: int        = 12
CONFLUENCE_MACD_SLOW: int        = 26
CONFLUENCE_MACD_SIGNAL: int      = 9
CONFLUENCE_OB_LOOKBACK: int      = 50
CONFLUENCE_OB_ZONE_PCT: float    = 0.002
HURST_ENABLED: bool              = True
HURST_LOOKBACK: int              = 100
HURST_MIN_THRESHOLD: float       = 0.52
HURST_MEAN_REVERSION_MAX: float  = 0.48
VWAP_ENABLED: bool               = True
VWAP_MAX_DISTANCE_ATR: float     = 2.5
SPREAD_PERCENTILE_ENABLED: bool  = True
SPREAD_PERCENTILE_LOOKBACK: int  = 60
SPREAD_PERCENTILE_MAX: float     = 0.75


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION-AWARE RISK MULTIPLIERS
# ─────────────────────────────────────────────────────────────────────────────

SESSION_RISK_BUDGET_PCT: Dict[str, float] = {
    "LONDON":   0.50,
    "NY":       0.35,
    "OVERLAP":  0.15,
    "ASIA":     0.10,
}
SESSION_HOURS_UTC: Dict[str, tuple] = {
    "ASIA":    (22, 7),
    "LONDON":  (7, 12),
    "OVERLAP": (12, 14),
    "NY":      (14, 21),
}


# ─────────────────────────────────────────────────────────────────────────────
#  WHALE / LIQUIDITY
# ─────────────────────────────────────────────────────────────────────────────

WHALE_ENABLED: bool              = True
WHALE_VOLUME_SPIKE_MULT: float   = 2.5
WHALE_SWEEP_LOOKBACK: int        = 10
WHALE_SWEEP_REJECTION_PCT: float = 0.003


# ─────────────────────────────────────────────────────────────────────────────
#  NEWS FILTER
# ─────────────────────────────────────────────────────────────────────────────

NEWS_API_URL: str                = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BLOCK_BEFORE_MINS: int      = 15
NEWS_BLOCK_AFTER_MINS: int       = 15
NEWS_HIGH_IMPACT_ONLY: bool      = True
NEWS_GOLD_MEDIUM_IMPACT: bool = True
NEWS_GOLD_SENSITIVE_EVENTS: List[str] = [
    "FOMC", "Fed", "CPI", "PPI", "NFP", "Non-Farm",
    "GDP", "PCE", "Retail Sales", "Unemployment",
    "ISM", "PMI", "ADP", "Durable Goods",
    "Consumer Confidence", "Jackson Hole",
    "Powell", "Yellen", "Treasury",
]
NEWS_GOLD_MEDIUM_BLOCK_BEFORE_MINS: int = 10
NEWS_GOLD_MEDIUM_BLOCK_AFTER_MINS: int = 10


# ─────────────────────────────────────────────────────────────────────────────
#  VOLATILITY SIZING — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

VOLATILITY_SIZING_ENABLED: bool  = True
ATR_PERIOD: int                  = 14
ATR_TIMEFRAME: str               = "M15"
ATR_MULTIPLIER: float            = 1.5
KELLY_ENABLED: bool              = True
KELLY_MIN_TRADES: int            = 15
KELLY_FRACTION: float            = 0.65
KELLY_MAX_RISK_PCT: float        = 2.0
RISK_PER_TRADE_PCT: float        = 0.75


# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA RANKER
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_RANKER_ENABLED: bool       = True
ALPHA_MIN_TRADES: int            = 5
ALPHA_ROLLING_DAYS: int          = 30
ALPHA_S_TIER_WR: float           = 0.65
ALPHA_A_TIER_WR: float           = 0.55
ALPHA_B_TIER_WR: float           = 0.45
ALPHA_C_TIER_WR: float           = 0.20
ALPHA_VETO_WR: float             = 0.35
ALPHA_VETO_MIN_TRADES: int       = 10
ALPHA_F_MUTE_BELOW: float        = 0.10
ALPHA_BAYESIAN_PRIOR_ALPHA: float = 3.0
ALPHA_BAYESIAN_PRIOR_BETA: float  = 3.0
ALPHA_TOXIC_MIN_TRADES: int       = 15
ALPHA_TOXIC_WR_THRESHOLD: float   = 0.10
AI_OVERRIDE_MIN_CONFIDENCE: int    = 9
AI_OVERRIDE_FLOOR_CONF_9: float    = 0.40
AI_OVERRIDE_FLOOR_CONF_10: float   = 0.50


# ─────────────────────────────────────────────────────────────────────────────
#  STATE RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

RECOVERY_ENABLED: bool           = True
RECOVERY_SNAPSHOT_FILE: str      = "data/recovery_snapshot.json"
RECOVERY_HEARTBEAT_SECS: int     = 60


# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTION — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

MT5_LOGIN: int              = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD: str           = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str             = os.getenv("MT5_SERVER", "")
MT5_PATH: str               = os.getenv("MT5_PATH", "C:/Program Files/MetaTrader 5/terminal64.exe")
MT5_MAGIC_NUMBER: int       = 20250101
MT5_SLIPPAGE: int           = 3
MT5_MAX_SPREAD_PIPS: float  = 12.0
EXEC_MAX_RETRIES: int       = 3
EXEC_RETRY_DELAY_MS: int    = 500
EXEC_PARTIAL_FILL_MIN_PCT: float = 0.80


# ─────────────────────────────────────────────────────────────────────────────
#  ML DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

ML_FEATURE_RECORDING_ENABLED: bool = True
ML_FEATURE_TIMEFRAMES: List[str]   = ["M15", "H1", "H4"]
ML_HURST_LOOKBACKS: List[int]      = [50, 100]
ML_REALIZED_VAR_WINDOW: int        = 12
ML_HAR_RV_ENABLED: bool            = True
ML_DOLLAR_BARS_ENABLED: bool       = False
ML_DOLLAR_BAR_SIZE_USD: float      = 50_000_000
ML_LABEL_LOOKFORWARD_BARS: int     = 30


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────

BACKTEST_DATA_DIR: str           = "data/historical"
BACKTEST_INITIAL_CAPITAL: float  = 10_000.0
BACKTEST_COMMISSION_PER_LOT: float = 7.0


# ─────────────────────────────────────────────────────────────────────────────
#  EXPOSURE GUARD — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

CORRELATION_GROUPS: List[List[str]] = [
    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],
    ["USDJPY", "USDCHF", "USDCAD"],
    ["XAUUSD", "XAGUSD"],
    ["BTCUSDT", "ETHUSDT"],
]
MAX_CURRENCY_EXPOSURE_PCT: float  = 5.0
MAX_CONCURRENT_TRADES: int        = 10
VIP_OVERFLOW_SLOTS: int           = 2
MAX_CONCURRENT_PER_SYMBOL: int    = 2
MAX_TOTAL_LOTS: float             = 0.50


# ─────────────────────────────────────────────────────────────────────────────
#  BLACK BOX
# ─────────────────────────────────────────────────────────────────────────────

BLACK_BOX_ENABLED: bool      = True
BLACK_BOX_DB_PATH: str       = "data/black_box.db"
DATA_DIR: str               = "data"


# ─────────────────────────────────────────────────────────────────────────────
#  LATENCY MONITOR
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_ENABLED: bool              = True
LATENCY_CHECK_INTERVAL_SECS: int   = 30
LATENCY_WARN_MS: float             = 200.0
LATENCY_SAFETY_MODE_MS: float      = 500.0
LATENCY_CRITICAL_MS: float         = 1000.0
LATENCY_SAMPLES: int               = 5
LATENCY_BROKER_HOST: str           = os.getenv("LATENCY_BROKER_HOST", "trade.icmarkets.com")


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE MANAGEMENT — SAME AS MARCH 17
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_EXPIRY_MINUTES: int           = 5
BE_TRIGGER_PCT: float                = 0.5
TP1_CLOSE_PCT: float                 = 0.34
TP2_CLOSE_PCT: float                 = 0.33
TRAILING_STOP_ACTIVATION_PIPS: float = 35.0
TRAILING_STOP_STEP_PIPS: float       = 12.0
MAX_ENTRY_DEVIATION_PIPS: float      = 80.0   # v7.1: was 150 (too loose), 50 (too tight)
CONSENSUS_WINDOW_MINUTES: int        = 15
CONSENSUS_MIN_SOURCES: int           = 2
PYRAMID_ENABLED: bool                = True
PYRAMID_ADD_PCT: float               = 0.25
STALE_EXIT_MINUTES: int              = 30
STALE_EXIT_MIN_PIPS: float           = 5.0


# v6.1: Momentum Decay Lot Dampener
SLOPE_DECAY_DAMPENER_ENABLED: bool = True
SLOPE_DECAY_THRESHOLD: float = 0.40
SLOPE_DECAY_LOT_MULTIPLIER: float = 0.50
TIGHT_SL_ATR_RATIO: float = 0.65
TIGHT_SL_LOT_MULTIPLIER: float = 0.60
RAPID_REPEAT_COOLDOWN_SECS: int = 900
RAPID_REPEAT_LOT_MULTIPLIER: float = 0.50

# v6.2: CHOP REGIME FILTER
CHOP_FILTER_ENABLED: bool          = True
CHOP_BLOCK_THRESHOLD: float        = 0.45    # v8.1: tightened from 0.35
CHOP_WARN_THRESHOLD: float         = 0.55    # v8.1: tightened from 0.50
CHOP_WARN_LOT_MULTIPLIER: float    = 0.50

# v6.2: Session-Level Loss Dampener
SESSION_LOSS_DAMPENER_ENABLED: bool = True
SESSION_SINGLE_LOSS_THRESHOLD: float = 50.0
SESSION_LOSS_LOT_REDUCTION: float   = 0.50
SESSION_LOSS_DAMPENER_DURATION_SECS: int = 3600

# v6.2: Misc
BE_NEUTRAL_PNL_THRESHOLD: float     = 3.0
VIP_MIN_LOT: float                   = 0.03
SESSION_BLACKOUT_START_UTC: int      = 99
SESSION_BLACKOUT_END_UTC: int        = 13
GLOBAL_BIAS_KILL_THRESHOLD: float    = 0.60
MAX_RR_MULTIPLIER: float             = 2.5


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE & NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH: str                = "data/omnisignal.db"
NOTIFY_CHAT_ID: str         = os.getenv("NOTIFY_CHAT_ID", "")
NOTIFY_ON_TRADE_OPEN: bool  = True
NOTIFY_ON_TRADE_CLOSE: bool = True
NOTIFY_ON_HALT: bool        = True


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTIVE TRADE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

ATO_ENABLED: bool = True

# v6.2.1: REMOVED — was blocking all scanners
# SCANNERS_DISABLED = True   # <-- DO NOT RE-ADD

# ─────────────────────────────────────────────────────────────────────────────
#  v6.3.1: RESTORED MARCH 17 SYSTEM + BUG FIXES
#  Scanners: FULLY ENABLED (same as March 17)
#  Anti-Hedge: prevents BUY+SELL same symbol (the only new safety feature)
#  Dampener Floor: prevents lot sizes from being crushed to 0.01
#  Bug fixes are in risk_guard.py and main.py, not here
# ─────────────────────────────────────────────────────────────────────────────

SCANNER_SIGNALS_ENABLED = True     # Scanners CAN trade (same as March 17)
TELEGRAM_BOOST_MODE = False        # Not needed — full system active
ANTI_HEDGE_ENABLED = True          # NEW: prevents hedge disaster

# v6.3.1 FIX: Dampener lot floor — prevents March 23 problem
# When multiple dampeners stack (chop × session × slope × tight_SL),
# they can multiply down to 0.01 lots. This floor prevents that.
DAMPENER_LOT_FLOOR: float = 0.03

# ─────────────────────────────────────────────────────────────────────────────
#  v6.4: Quality Over Quantity
# ─────────────────────────────────────────────────────────────────────────────

SOLO_PULLBACK_REQUIRE_CONSENSUS: bool = True
CATCD_MIN_SOLO_CONFIDENCE: int = 8
MIN_RR_SCANNER: float = 1.3


# ---------------------------------------------------------------------------
#  v7.0: Convergence Sniper
# ---------------------------------------------------------------------------

MIN_TRADE_GAP_MINUTES: int = 2        # Cooldown between trades (rapid-fire lost -$561)
MAX_HOLD_MINUTES: int = 45            # Hard time kill (>4hr trades lost -$699)
MAX_SINGLE_TRADE_LOTS: float = 0.03   # Hard lot cap (0.01 is only profitable tier)

# v7.2: Quick wins
CLUSTER_WINDOW_MINUTES: int = 10       # Cluster boost window for crowd consensus

# v8.0: Profit Machine
TIER_S_CHANNELS = {-1002223574325, -1001785197109}  # 100% WR elite channels

# v8.0: Lead channel classification from deep research
LEAD_CHANNELS = {
    -1002223574325,   # #1 LEADING, +1078 pips post-signal
    -1003715078909,   # LEADING, +634 pips post-signal
    -1002448604508,   # LEADING, +904 pips post-signal
    -1003655941757,   # LEADING, +742 pips post-signal
}



# ---------------------------------------------------------------------------
#  v8.2: Breakout Hunter
# ---------------------------------------------------------------------------

BREAKOUT_MIN_CONSECUTIVE_BARS: int = 3
BREAKOUT_VOLUME_SPIKE_MULT: float  = 1.5
BREAKOUT_MAX_CONCURRENT: int       = 2     # Allow stacking during breakout
BREAKOUT_COOLDOWN_MINUTES: int     = 1     # Fast entries during breakout
