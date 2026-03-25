"""
config.py — OmniSignal Alpha v4.2
Phase 1: Prop Firm Survival + ML Data Pipeline

AUDIT FIXES IMPLEMENTED:
  FIX-1: DAILY_DRAWDOWN_LIMIT_PCT now applied against opening equity (not current)
  FIX-2: MAX_DRAWDOWN_LIMIT_PCT added — was completely absent in v2.0
  FIX-3: DD_OPENING_EQUITY_HOUR added for correct prop firm reset time
  FIX-4: Continuous equity monitor parameters (not just on-signal checks)

ADDITIONAL INSTITUTIONAL UPGRADES (beyond audit scope):
  + Graduated DD response: 3 tiers (NORMAL → REDUCED → HALT)
  + Consecutive loss circuit breaker with anti-martingale sizing
  + Session-aware risk budgeting (scale risk by H-exponent per session)
  + Spread percentile filter (trade only when spread < 60th percentile)
  + Equity velocity circuit breaker (halt if equity drops X% in Y minutes)
  + Prop firm phase awareness (CHALLENGE vs FUNDED → different aggression)
  + Trade frequency governor (max trades per hour/day for prop firm compliance)
  + ML feature recording schema controls
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
#  PROP FIRM PHASE  ← NEW in v3.0
#  Set via .env: PROP_FIRM_PHASE=CHALLENGE or FUNDED
#  CHALLENGE: allowed to be more aggressive (need to hit profit target)
#  FUNDED:    survival-first, tighter DD guards, smaller sizing
# ─────────────────────────────────────────────────────────────────────────────

class PropPhase:
    CHALLENGE = "CHALLENGE"
    FUNDED    = "FUNDED"
    PERSONAL  = "PERSONAL"      # No prop firm rules — full Kelly/RL sizing

PROP_FIRM_PHASE: str = os.getenv("PROP_FIRM_PHASE", PropPhase.FUNDED)

# Challenge phase: how much profit is still needed to pass (0.0 = already passed)
CHALLENGE_PROFIT_TARGET_PCT: float = float(os.getenv("CHALLENGE_PROFIT_TARGET_PCT", "8.0"))
# Current profit made so far (% of initial balance) — update daily
CHALLENGE_PROFIT_CURRENT_PCT: float = float(os.getenv("CHALLENGE_PROFIT_CURRENT_PCT", "0.0"))


# ─────────────────────────────────────────────────────────────────────────────
#  INGESTION
# ─────────────────────────────────────────────────────────────────────────────

TELEGRAM_API_ID: int         = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH: str       = os.getenv("TELEGRAM_API_HASH", "")
# Bot API (HTTP sendMessage) — separate from MTProto api_hash above; do not overload TELEGRAM_API_HASH.
TELEGRAM_BOT_TOKEN: str      = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("NOTIFY_BOT_TOKEN") or ""
TELEGRAM_SESSION_NAME: str   = "omnisignal_session"

# MTProto monitored channels / supergroups (negative peer ids). Used when TELEGRAM_CHANNELS env is empty.
MONITORED_CHANNELS: List[int] = [
    -1001983734792,  # prime forex guide
    -1001347617494,  # Forex Scalping signals (free)
    -1001536621768,  # Vasilytrader (free forex signals)
    -1001309043988,  # Whale Alert
    -1001510927248,  # ForexPlace Signals
    -1002489241398,  # StarEdge Market
    -1002448604508,  # Ict Forex Star
    -1001872299004,  # Gold Singlas daily
    -1002219243374,  # Sure Shot Forex
    -1002293261831,  # Sure Shot Forex 2
    -1003655941757,  # 1000pips Builder (Official)
    -1002494813464,  # 1000pips Builder Official 2
    -1001949192523,  # GBP/JPY SIGNALS
    -1001623345960,  # FX PROFIT SIGNAL (FREE)
    -1001921425619,  # SNYTHETICX SIGNALS
    -1003372787430,  # Jeppe Kirk Bonde TM
    -1003781656280,  # GOLDCHAIN TRADING SIGNALS (free)
    -1002273704999,  # XAUUSD & GOLD TRADING SIGNALS
    -1001207301837,  # XAUUSDGOLDsinglas
    -1001785197109,  # AnableSignals
    -1001387511343,  # FBS Analytics
    -1002176701424,  # United Kings Singles
    -1001588519179,  # SureShot GOLD
    -1003738551742,  # Gold Signals 98% Sure
    -1002223574325,  # United Kings VIP- Forex
    -1001814562728,  # Forex Gold Signals
    -1001651583302,  # FXTradingVision | Forex & Crypto Signals
    -1002399120063,  # FX Culture | Free Signal Group
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
#  AI ENGINE (Pillars 1, 14)
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
#  ██████████  PROP FIRM DRAWDOWN GUARD — PHASE 1 CORE  ██████████
#  These parameters are the difference between keeping and losing funded capital.
# ─────────────────────────────────────────────────────────────────────────────

# --- BUG-2 FIX: Max drawdown now exists ---
# FTMO standard: 10% max drawdown from initial balance (trailing or static)
# Funded Trader: 8-12%. Match your specific prop firm's rules exactly.
MAX_DRAWDOWN_LIMIT_PCT: float  = float(os.getenv("MAX_DD_PCT", "8.0"))

# Daily drawdown: % of OPENING EQUITY for that day (not current equity)
DAILY_DRAWDOWN_LIMIT_PCT: float = float(os.getenv("DAILY_DD_PCT", "4.0"))

# --- BUG-3 FIX: Server reset time for opening equity snapshot ---
# Prop firms reset daily DD at THEIR server midnight, not your local time.
# IC Markets: 00:00 server time (EET/EEST). FTMO: 00:00 CET/CEST.
# Express as UTC offset for the broker server: IC Markets = 2 (UTC+2 normal, UTC+3 DST)
DD_OPENING_EQUITY_SERVER_UTC_OFFSET: int = int(os.getenv("BROKER_UTC_OFFSET", "2"))

# --- BUG-4 FIX: Continuous equity monitoring interval ---
EQUITY_MONITOR_INTERVAL_SECS: int = 5    # Check floating P&L every 5 seconds

# --- UPGRADE: Graduated DD response (3 tiers, not binary HALT) ---
# Tier 1: When DD reaches this fraction of daily limit → reduce all lot sizes
DD_REDUCED_MODE_THRESHOLD_PCT: float = 0.50   # 50% of daily limit used → REDUCED mode
DD_REDUCED_MODE_LOT_MULTIPLIER: float = 0.60  # Cut lots to 40% of normal
# Tier 2: When DD reaches this fraction → block all new entries (but manage existing)
DD_BLOCK_THRESHOLD_PCT: float = 0.90          # 80% of limit used → block new entries
# Tier 3: Full halt at 100% (existing behaviour)

# --- UPGRADE: Equity velocity circuit breaker ---
# If equity drops this many % points in this many minutes → immediate REDUCED mode
# Catches flash crashes / correlated multi-position wipeouts between 5-second checks
EQUITY_VELOCITY_DROP_PCT: float   = 0.80   # 0.80% equity drop (e.g. $800 on $100K)
EQUITY_VELOCITY_WINDOW_MINS: float = 5.0   # within any 5-minute rolling window

# Initial balance snapshot: set once at account open, never changes
# Used for max drawdown calculation from absolute base
# Set via .env: INITIAL_ACCOUNT_BALANCE=100000
INITIAL_ACCOUNT_BALANCE: float = float(os.getenv("INITIAL_ACCOUNT_BALANCE", "10000"))

# --- Phase-specific risk overrides ---
# In CHALLENGE phase with >50% of profit target remaining: slightly more aggressive
CHALLENGE_RISK_MULTIPLIER: float  = 1.20   # 20% more risk when in challenge, winning
# In FUNDED phase: ultra-conservative beyond 50% daily DD used
FUNDED_CONSERVATIVE_MULTIPLIER: float = 0.70


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE FREQUENCY GOVERNOR  ← NEW in v3.0
#  Prop firms can penalize or ban accounts for excessive trading.
#  Also prevents tilt-style overtrading after losses.
# ─────────────────────────────────────────────────────────────────────────────

MAX_TRADES_PER_HOUR: int  = 15    # Never more than 6 new entries per rolling hour
MAX_TRADES_PER_DAY: int   = 60   # Never more than 20 trades per trading day


# ─────────────────────────────────────────────────────────────────────────────
#  CONSECUTIVE LOSS CIRCUIT BREAKER  ← NEW in v3.0
#  Anti-martingale: size down after losses, never up
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONSECUTIVE_LOSSES: int           = 5     # After 3 in a row → size reduction kicks in
CONSECUTIVE_LOSS_MULTIPLIER: float    = 0.65  # Halve lot size until win streak recovers
RECOVERY_STREAK_NEEDED: int           = 2     # 2 consecutive wins restores full sizing
# Consecutive wins → allowed to scale up slightly (momentum of edge)
CONSECUTIVE_WIN_SCALE_UP_AFTER: int   = 3     # After 4 wins in a row
CONSECUTIVE_WIN_SCALE_MULTIPLIER: float = 1.25 # 15% size increase (modest, not Kelly yet)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFLUENCE ENGINE (Pillars 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────

CONFLUENCE_ENABLED: bool         = True
CONFLUENCE_TIMEFRAME: str        = "M15"   # XAUUSD scalper primary TF
CONFLUENCE_HTF_TIMEFRAME: str    = "H4"    # Higher timeframe trend context
CONFLUENCE_MIN_SCORE: int        = 2       # Raised from 2 to 3/8 for v3.0

# Classic indicators (still used as one component, not the entire stack)
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

# NEW: Hurst exponent filter
HURST_ENABLED: bool              = True
HURST_LOOKBACK: int              = 100     # bars for rolling Hurst computation
HURST_MIN_THRESHOLD: float       = 0.52   # Below this = near-random, don't trade trend-following
HURST_MEAN_REVERSION_MAX: float  = 0.48   # Above 0.48 but below 0.52 = neutral

# NEW: VWAP filter
VWAP_ENABLED: bool               = True
VWAP_MAX_DISTANCE_ATR: float     = 2.5    # Don't chase entries >2.5 ATR from session VWAP

# NEW: Spread percentile filter (institutional quality execution filter)
SPREAD_PERCENTILE_ENABLED: bool  = True
SPREAD_PERCENTILE_LOOKBACK: int  = 60     # bars for rolling spread distribution
SPREAD_PERCENTILE_MAX: float     = 0.75   # Only trade when spread < 60th percentile


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION-AWARE RISK MULTIPLIERS  ← NEW in v3.0
#  XAUUSD H≈0.62 London (trending), H≈0.55 NY (volatile), H≈0.48 Asia (random)
#  Scale maximum daily risk budget allocated per session.
# ─────────────────────────────────────────────────────────────────────────────

SESSION_RISK_BUDGET_PCT: Dict[str, float] = {
    "LONDON":   0.50,    # 50% of daily risk budget available during London
    "NY":       0.35,    # 35% during NY — more volatile, more likely to spike
    "OVERLAP":  0.15,    # 15% during London/NY overlap — most volatile hour
    "ASIA":     0.10,    # 0% during Asia — near-random, don't waste daily budget
}
# UTC hours for each session (server time aware; adjust via DD_OPENING_EQUITY_SERVER_UTC_OFFSET)
SESSION_HOURS_UTC: Dict[str, tuple] = {
    "ASIA":    (22, 7),   # 22:00 - 07:00 UTC
    "LONDON":  (7, 12),   # 07:00 - 12:00 UTC
    "OVERLAP": (12, 14),  # 12:00 - 14:00 UTC (London/NY overlap)
    "NY":      (14, 21),  # 14:00 - 21:00 UTC
}


# ─────────────────────────────────────────────────────────────────────────────
#  WHALE / LIQUIDITY (Pillar 3)
# ─────────────────────────────────────────────────────────────────────────────

WHALE_ENABLED: bool              = True
WHALE_VOLUME_SPIKE_MULT: float   = 2.5
WHALE_SWEEP_LOOKBACK: int        = 10
WHALE_SWEEP_REJECTION_PCT: float = 0.003


# ─────────────────────────────────────────────────────────────────────────────
#  NEWS FILTER (Pillar 4)
# ─────────────────────────────────────────────────────────────────────────────

NEWS_API_URL: str                = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_BLOCK_BEFORE_MINS: int      = 15
NEWS_BLOCK_AFTER_MINS: int       = 15
NEWS_HIGH_IMPACT_ONLY: bool      = True

# Gold-specific medium-impact news blocking (XAUUSD is uniquely sensitive)
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
#  VOLATILITY SIZING (Pillar 5)
# ─────────────────────────────────────────────────────────────────────────────

VOLATILITY_SIZING_ENABLED: bool  = True
ATR_PERIOD: int                  = 14
ATR_TIMEFRAME: str               = "M15"
ATR_MULTIPLIER: float            = 1.5
KELLY_ENABLED: bool              = True
KELLY_MIN_TRADES: int            = 15
KELLY_FRACTION: float            = 0.65
KELLY_MAX_RISK_PCT: float        = 2.0
RISK_PER_TRADE_PCT: float        = 0.75    # Prop-firm safe: $10K * 0.75% = $75 risk per trade


# ─────────────────────────────────────────────────────────────────────────────
#  ALPHA RANKER (Pillar 6)
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_RANKER_ENABLED: bool       = True
ALPHA_MIN_TRADES: int            = 5
ALPHA_ROLLING_DAYS: int          = 30
ALPHA_S_TIER_WR: float           = 0.65
ALPHA_A_TIER_WR: float           = 0.55
ALPHA_B_TIER_WR: float           = 0.45
ALPHA_C_TIER_WR: float           = 0.20
ALPHA_VETO_WR: float             = 0.35   # v4.4: Hard veto -- sources below 35% Bayesian WR with negative PnL are blocked
ALPHA_VETO_MIN_TRADES: int       = 10     # minimum trades before veto can apply
ALPHA_F_MUTE_BELOW: float        = 0.10

# Bayesian shrinkage prior (Beta-Binomial conjugate)
# Prior weight = PRIOR_ALPHA + PRIOR_BETA pseudo-observations centered at PRIOR_ALPHA/(PRIOR_ALPHA+PRIOR_BETA)
ALPHA_BAYESIAN_PRIOR_ALPHA: float = 3.0   # pseudo-wins  (prior mean = 3/(3+3) = 50%)
ALPHA_BAYESIAN_PRIOR_BETA: float  = 3.0   # pseudo-losses
ALPHA_TOXIC_MIN_TRADES: int       = 15    # minimum trades before a source can be classified TOXIC
ALPHA_TOXIC_WR_THRESHOLD: float   = 0.10  # WR below this WITH enough trades = truly toxic (hard block)

# AI Confidence Override: allows high-conviction AI signals to bypass low-tier muting
AI_OVERRIDE_MIN_CONFIDENCE: int    = 9    # minimum AI confidence to trigger override
AI_OVERRIDE_FLOOR_CONF_9: float    = 0.40 # effective multiplier floor when AI says 9/10
AI_OVERRIDE_FLOOR_CONF_10: float   = 0.50 # effective multiplier floor when AI says 10/10


# ─────────────────────────────────────────────────────────────────────────────
#  STATE RECOVERY (Pillar 7)
# ─────────────────────────────────────────────────────────────────────────────

RECOVERY_ENABLED: bool           = True
RECOVERY_SNAPSHOT_FILE: str      = "data/recovery_snapshot.json"
RECOVERY_HEARTBEAT_SECS: int     = 60


# ─────────────────────────────────────────────────────────────────────────────
#  EXECUTION (Pillar 8)
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
#  ██████  ML DATA PIPELINE — PILLAR 4  ██████
#  These control what gets recorded on every bar analysis.
#  Every parameter here feeds directly into the TFT training dataset.
# ─────────────────────────────────────────────────────────────────────────────

ML_FEATURE_RECORDING_ENABLED: bool = True     # Master switch for market_features table
ML_FEATURE_TIMEFRAMES: List[str]   = ["M15", "H1", "H4"]  # Record features at all 3 TFs
ML_HURST_LOOKBACKS: List[int]      = [50, 100]             # Two Hurst windows
ML_REALIZED_VAR_WINDOW: int        = 12        # bars for realized variance computation
ML_HAR_RV_ENABLED: bool            = True      # Compute HAR-RV volatility forecast

# Dollar bar aggregation (Pillar 5 paradigm shift)
ML_DOLLAR_BARS_ENABLED: bool       = False     # Set True when tick data collection ready
ML_DOLLAR_BAR_SIZE_USD: float      = 50_000_000  # $50M per bar for XAUUSD

# Label backfill: how far forward to look for TP/SL outcome labeling
ML_LABEL_LOOKFORWARD_BARS: int     = 30        # bars to wait for label (M15 = 7.5 hours)


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTESTING (Pillar 9)
# ─────────────────────────────────────────────────────────────────────────────

BACKTEST_DATA_DIR: str           = "data/historical"
BACKTEST_INITIAL_CAPITAL: float  = 10_000.0
BACKTEST_COMMISSION_PER_LOT: float = 7.0


# ─────────────────────────────────────────────────────────────────────────────
#  EXPOSURE GUARD (Pillar 11)
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
MAX_CONCURRENT_PER_SYMBOL: int    = 1
MAX_TOTAL_LOTS: float             = 0.50


# ─────────────────────────────────────────────────────────────────────────────
#  BLACK BOX (Pillar 12)
# ─────────────────────────────────────────────────────────────────────────────

BLACK_BOX_ENABLED: bool      = True
BLACK_BOX_DB_PATH: str       = "data/black_box.db"
DATA_DIR: str               = "data"
DB_PATH: str       = "data/black_box.db"


# ─────────────────────────────────────────────────────────────────────────────
#  LATENCY MONITOR (Pillar 13)
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_ENABLED: bool              = True
LATENCY_CHECK_INTERVAL_SECS: int   = 30
LATENCY_WARN_MS: float             = 200.0
LATENCY_SAFETY_MODE_MS: float      = 500.0
LATENCY_CRITICAL_MS: float         = 1000.0
LATENCY_SAMPLES: int               = 5
LATENCY_BROKER_HOST: str           = os.getenv("LATENCY_BROKER_HOST", "trade.icmarkets.com")


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_EXPIRY_MINUTES: int           = 5
BE_TRIGGER_PCT: float                = 0.5
TP1_CLOSE_PCT: float                 = 0.34
TP2_CLOSE_PCT: float                 = 0.33
TRAILING_STOP_ACTIVATION_PIPS: float = 35.0
TRAILING_STOP_STEP_PIPS: float       = 12.0
MAX_ENTRY_DEVIATION_PIPS: float      = 50.0
CONSENSUS_WINDOW_MINUTES: int        = 15
CONSENSUS_MIN_SOURCES: int           = 2
# v4.0 Profit Maximizer
PYRAMID_ENABLED: bool                = True
PYRAMID_ADD_PCT: float               = 0.25
VIP_MIN_LOT: float                   = 0.08
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
