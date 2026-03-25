"""
quant/macro_collector.py -- Automatic COT data fetcher from CFTC.
"""
import requests
import pandas as pd
import io
import zipfile
from datetime import datetime
from quant.macro_filter import macro_filter, MacroState, COTState
from utils.logger import get_logger

logger = get_logger(__name__)

COT_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2026.zip"


def fetch_latest_cot():
    """Download and parse the latest CFTC COT report for Gold."""
    try:
        logger.info("[MacroCollector] Downloading latest COT report from CFTC...")
        response = requests.get(COT_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                df = pd.read_csv(f)

        gold_row = df[
            df["Market_and_Exchange_Names"].str.contains(
                "GOLD - COMMODITY EXCHANGE INC.", na=False
            )
        ].iloc[0]

        longs = int(gold_row["Prod_Merc_Positions_Long_All"])
        shorts = int(gold_row["Prod_Merc_Positions_Short_All"])
        net_commercial = longs - shorts

        report_date = str(gold_row.get("Report_Date_as_YYYY-MM-DD", ""))

        cot_data = COTState(
            commercial_net=net_commercial,
            spec_large_net=0,
            report_date=report_date or datetime.now().strftime("%Y-%m-%d"),
        )

        logger.info(f"[MacroCollector] COT updated: net_commercial={net_commercial:+,}")
        return cot_data

    except Exception as e:
        logger.error(f"[MacroCollector] COT download failed: {e}")
        return None


async def update_macro_automatically():
    """Main update function -- runs at bot startup."""
    state = macro_filter.state

    new_cot = fetch_latest_cot()
    if new_cot:
        state.cot = new_cot

    macro_filter.save(state)
