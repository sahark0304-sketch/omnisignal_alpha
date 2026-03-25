#!/usr/bin/env python3
import sys
import os

# Force Python to recognize the current directory (Fixes ModuleNotFoundError)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
from quant.macro_filter import macro_filter, MacroState, COTData, GammaData

def main():
    print("==================================================")
    print("  OmniSignal Alpha — Macro Framework Update")
    print("==================================================")
    print("Enter the latest weekly structural data.\n")

    cot_net = input("COT Commercial Net Position (e.g. -192000): ").replace(',', '')
    if not cot_net.strip('-').isnumeric():
        print("Invalid COT input. Exiting.")
        return

    gnp = input("Gamma Neutral Price (GNP) (e.g. 2488.0): ")
    max_pain = input("Max Pain Level (e.g. 2460.0): ")
    gex = input("Net GEX in $M (negative = short gamma) (e.g. -420): ")

    strikes_above_input = input("Strikes ABOVE (format: price:gex,price:gex) (Leave blank if none): ")
    strikes_below_input = input("Strikes BELOW (format: price:gex,price:gex) (Leave blank if none): ")

    cot = COTData(
        commercial_net=int(cot_net),
        large_speculator_net=0,
        update_date=macro_filter.state.cot.update_date
    )

    gamma = GammaData(
        gamma_neutral_price=float(gnp) if gnp else 0.0,
        max_pain=float(max_pain) if max_pain else 0.0,
        net_gex_millions=float(gex) if gex else 0.0,
        update_date=macro_filter.state.gamma.update_date,
        strikes_above=[],
        strikes_below=[],
        exclusion_zone_atr=1.0
    )

    if strikes_above_input:
        for s in strikes_above_input.split(","):
            parts = s.strip().split(":")
            gamma.strikes_above.append((float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0))

    if strikes_below_input:
        for s in strikes_below_input.split(","):
            parts = s.strip().split(":")
            gamma.strikes_below.append((float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0))

    state = MacroState(cot=cot, gamma=gamma)
    macro_filter.save(state)
    print(f"\n✅ Macro updated: COT={cot.bias} GNP={gamma.gamma_neutral_price:.1f} GEX=${gamma.net_gex_millions:.0f}M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniSignal Macro Framework Weekly Update")
    parser.add_argument("--cot-net",       type=int,   help="Net commercial position (integer)")
    parser.add_argument("--cot-spec",      type=int,   help="Net large speculator position")
    parser.add_argument("--gnp",           type=float, help="Gamma Neutral Price")
    parser.add_argument("--max-pain",      type=float, help="Options max pain level")
    parser.add_argument("--gex",           type=float, help="Net GEX in $M (negative=dealers short gamma)")
    parser.add_argument("--excl-atr",      type=float, help="Strike exclusion zone in ATR units", default=1.0)
    parser.add_argument("--strikes-above", type=str,   help="'price:gex,price:gex' e.g. '2500:180,2550:90'")
    parser.add_argument("--strikes-below", type=str,   help="'price:gex,price:gex'")

    args = parser.parse_args()

    if args.cot_net is not None:
        cot = COTData(commercial_net=args.cot_net, large_speculator_net=args.cot_spec or 0)
        gamma = GammaData(
            gamma_neutral_price=args.gnp or 0.0,
            max_pain=args.max_pain or 0.0,
            net_gex_millions=args.gex or 0.0,
            exclusion_zone_atr=args.excl_atr
        )
        if args.strikes_above:
            for s in args.strikes_above.split(","):
                parts = s.strip().split(":")
                gamma.strikes_above.append((float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0))
        if args.strikes_below:
            for s in args.strikes_below.split(","):
                parts = s.strip().split(":")
                gamma.strikes_below.append((float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0))
        state = MacroState(cot=cot, gamma=gamma)
        macro_filter.save(state)
        print(f"✅ Macro updated: COT={cot.bias} GNP={gamma.gamma_neutral_price:.1f} GEX=${gamma.net_gex_millions:.0f}M")
    else:
        main()