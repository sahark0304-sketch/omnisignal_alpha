"""
validate/01_download_history.py — OmniSignal Alpha Scanner Validation Suite
============================================================================
STEP 1: Download 6 months of XAUUSD historical data from MT5.

Run this ONCE on your VPS where MT5 is connected:
    python validate/01_download_history.py

Creates CSV files in data/validation/
"""

import os
import sys
import csv
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join("data", "validation")
SYMBOL = "XAUUSD"
MONTHS_BACK = 6


def download():
    import MetaTrader5 as mt5

    try:
        import config
        kwargs = {}
        if getattr(config, "MT5_PATH", None):
            kwargs["path"] = config.MT5_PATH
        if getattr(config, "MT5_LOGIN", 0):
            kwargs["login"] = config.MT5_LOGIN
        if getattr(config, "MT5_PASSWORD", ""):
            kwargs["password"] = config.MT5_PASSWORD
        if getattr(config, "MT5_SERVER", ""):
            kwargs["server"] = config.MT5_SERVER
    except ImportError:
        kwargs = {}

    if not mt5.initialize(**kwargs):
        print(f"[ERROR] MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

    mt5.symbol_select(SYMBOL, True)
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        print(f"[ERROR] Symbol {SYMBOL} not found.")
        mt5.shutdown()
        sys.exit(1)

    print(f"[OK] MT5 connected. Symbol: {SYMBOL} Digits: {info.digits}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    utc_now = datetime.now(timezone.utc)
    date_from_6m = utc_now - timedelta(days=MONTHS_BACK * 30)

    timeframes = {
        "M1":  (mt5.TIMEFRAME_M1,  utc_now - timedelta(days=30)),   # M1: only ~1 month available
        "M5":  (mt5.TIMEFRAME_M5,  date_from_6m),
        "M15": (mt5.TIMEFRAME_M15, date_from_6m),
        "H1":  (mt5.TIMEFRAME_H1,  date_from_6m),
        "H4":  (mt5.TIMEFRAME_H4,  date_from_6m),
    }

    for tf_name, (tf_val, tf_from) in timeframes.items():
        print(f"\n[{tf_name}] Downloading...")

        try:
            rates = mt5.copy_rates_range(SYMBOL, tf_val, tf_from, utc_now)
        except Exception as e:
            print(f"  [WARN] Error fetching {tf_name}: {e}")
            continue

        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data returned for {tf_name}")
            # Try shorter range as fallback
            if tf_name == "M1":
                print(f"  [INFO] Trying last 7 days for M1...")
                try:
                    rates = mt5.copy_rates_range(SYMBOL, tf_val, utc_now - timedelta(days=7), utc_now)
                except Exception:
                    pass
                if rates is None or len(rates) == 0:
                    print(f"  [WARN] Still no M1 data. Skipping.")
                    continue

        # numpy structured array: access fields by name works, but .get() does NOT
        # Field names: 'time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'
        path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{tf_name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "open", "high", "low", "close",
                "tick_volume", "spread", "real_volume"
            ])
            for bar in rates:
                # bar is numpy.void — access fields by name like bar['open'], NOT bar.get()
                ts = int(bar['time'])
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                writer.writerow([
                    dt.strftime("%Y-%m-%d %H:%M:%S"),
                    round(float(bar['open']), 2),
                    round(float(bar['high']), 2),
                    round(float(bar['low']), 2),
                    round(float(bar['close']), 2),
                    int(bar['tick_volume']),
                    int(bar['spread']),
                    int(bar['real_volume']),
                ])

        first_dt = datetime.fromtimestamp(int(rates[0]['time']), tz=timezone.utc)
        last_dt = datetime.fromtimestamp(int(rates[-1]['time']), tz=timezone.utc)
        print(f"  [OK] {len(rates):,} bars saved to {path}")
        print(f"       Range: {first_dt.strftime('%Y-%m-%d')} -> {last_dt.strftime('%Y-%m-%d')}")

    mt5.shutdown()
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Files in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Next: python validate/02_run_validation.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    download()
