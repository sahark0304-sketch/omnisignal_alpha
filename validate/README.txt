"""
validate/README.txt — Scanner Validation Suite
================================================

WHAT THIS DOES:
  Tests each of your 7 scanners INDEPENDENTLY against 6 months of
  real XAUUSD history. Produces a clear PASS/FAIL per scanner with
  win rate, expectancy, profit factor, and Sharpe ratio.

  Only scanners that PASS get re-enabled. Everything else stays off.
  This is how you stop bleeding money.


STEP-BY-STEP INSTRUCTIONS:
===========================

STEP 0: STOP LIVE TRADING (mandatory)
---------------------------------------
Before anything else, disable all AUTO scanners so the bot doesn't
trade while you're validating. Send this to Cursor:

--- COPY TO CURSOR (open config.py) ---

Add these lines at the very end of config.py, before any closing comments:

    # ═══════════════════════════════════════════════════════════════
    #  SCANNER VALIDATION MODE — ALL AUTO SCANNERS DISABLED
    #  Remove this block ONLY after validation passes specific scanners
    # ═══════════════════════════════════════════════════════════════
    SCANNERS_DISABLED = True

Then open main.py and find the section in startup() where scanner tasks
are created (the lines with _supervise("liquidity_scanner"...) etc).
Wrap ALL scanner task creation lines in:

    if not getattr(config, "SCANNERS_DISABLED", False):
        # ... all the scanner tasks go here ...

This keeps the core bot running (Telegram listener, trade manager,
equity monitor) but stops all autonomous signal generation.

--- END CURSOR PROMPT ---


STEP 1: DOWNLOAD DATA (run on VPS with MT5)
---------------------------------------------
    cd /path/to/omnisignal_alpha
    python validate/01_download_history.py

This downloads 6 months of XAUUSD M1/M5/M15/H1/H4 bars from MT5
into data/validation/*.csv files. Takes about 30 seconds.


STEP 2: RUN VALIDATION (can run anywhere)
-------------------------------------------
    python validate/02_run_validation.py

This replays each scanner's detection logic against historical bars
and simulates trades with realistic spread and commission.
Takes 1-3 minutes depending on your CPU.

Output:
    data/validation/scanner_report.txt     Terminal report
    data/validation/scanner_report.json    Machine-readable
    data/validation/trades_*.csv           Per-scanner trade logs


STEP 3: READ THE REPORT
--------------------------
The report tells you exactly which scanners to re-enable:

    ✅ PASS — ENABLE:     Positive expectancy confirmed. Turn it back on.
    ⚠️  MARGINAL:          Adjust parameters and re-test.
    ❌ FAIL — DISABLE:     Negative edge. Leave it off permanently.


STEP 4: RE-ENABLE ONLY PASSING SCANNERS
------------------------------------------
For each scanner that PASSED, remove it from the disabled block.
For example, if only LIQUIDITY and SMC pass:

--- COPY TO CURSOR (open main.py) ---

In startup(), only create tasks for LIQUIDITY and SMC scanners.
Comment out or delete the task creation lines for:
  - momentum_scanner
  - tick_flow_engine
  - catcd_engine
  - mr_engine
  - convergence_engine (disable until 2+ scanners prove profitable)

Keep these:
  - liquidity_scanner (if it passed)
  - smc_scanner (if it passed)

Also in config.py, remove the SCANNERS_DISABLED = True line.

--- END CURSOR PROMPT ---


STEP 5: PAPER TRADE FOR 1 WEEK
---------------------------------
After re-enabling only passing scanners, run the bot on your DEMO
account for at least 1 full trading week (5 sessions). Compare
live results to the backtest expectations.

If live results are within 30% of backtest expectations -> go live.
If live results are significantly worse -> the scanner has a data
snooping bias and should be disabled.


WHAT ABOUT TFI, CATCD, AND MR?
================================
These three scanners operate on tick-level data (individual bid/ask
updates, not OHLCV bars). They CANNOT be backtested from bar data.

To validate them, you need to:
1. Record live tick data for 2+ weeks (add a tick recorder task)
2. Replay ticks through the scanner logic
3. Run the same PASS/FAIL analysis

Until you've done this, keep them DISABLED. Running unvalidated
scanners is the same as gambling.


INTERPRETING THE METRICS:
==========================

Expectancy (pips/trade):
  The average pips you make per trade after spread and commission.
  Must be POSITIVE. If negative, the scanner loses money on average.
  Good: > 5 pips/trade.  Great: > 10 pips/trade.

Profit Factor:
  Gross wins / Gross losses. Must be > 1.0 to be profitable.
  Good: > 1.3.  Great: > 1.8.

Win Rate:
  Less important than expectancy. A 30% win rate with 3:1 R:R is
  much better than a 70% win rate with 0.3:1 R:R.

Sharpe Ratio:
  Risk-adjusted return. Annualized. 
  > 1.0 = good.  > 2.0 = excellent.  < 0 = losing money.

Max Drawdown:
  Largest peak-to-trough decline in pips. Tells you how much pain
  you'll experience before the strategy recovers.

Max Consecutive Losses:
  How many losses in a row to expect. If this is > 8, you'll likely
  panic and turn off the scanner manually — defeating the purpose.
"""
