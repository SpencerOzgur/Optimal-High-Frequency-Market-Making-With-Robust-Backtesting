"""
run_with_wrds.py
================
End-to-end runner: WRDS TAQ data → replay simulator → results tables + plots.

Run this file directly:
    python run_with_wrds.py

You will be prompted for your WRDS credentials on first run.
Credentials are saved to ~/.pgpass so you won't be prompted again.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrds_loader import WRDSLoader, estimate_sigma
from market_maker import AvellanedaStoikov, InventoryModel, BaselineStrategy
from replay_simulator import ReplaySimulator
from helpers import summarise_results, summarise_order_stats, print_fill_analysis

# ---------------------------------------------------------------------------
# Configuration — matches the paper exactly
# ---------------------------------------------------------------------------

TICKERS = ['AAPL', 'AMZN', 'GE', 'IVV', 'M']

DATES = [
    '2017-06-12',
    '2017-06-13',
    '2017-06-14',
    '2017-06-15',
    '2017-06-16',
]

# Table 1 open/close spreads (used for A-S calibration)
SPREAD_PARAMS = {
    'AAPL': {'open_spread': 0.02, 'close_spread': 0.01, 'gamma': 0.001},
    'AMZN': {'open_spread': 0.49, 'close_spread': 0.56, 'gamma': 0.0001},
    'GE':   {'open_spread': 0.02, 'close_spread': 0.01, 'gamma': 0.001},
    'IVV':  {'open_spread': 0.02, 'close_spread': 0.01, 'gamma': 0.001},
    'M':    {'open_spread': 0.04, 'close_spread': 0.01, 'gamma': 0.001},
}

PHI_MAX = 100.0
ETA     = 0.005


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_wrds_experiment(tickers=TICKERS, dates=DATES):
    # 1. Load data from WRDS
    print("=" * 60)
    print("  Step 1: Loading TAQ data from WRDS")
    print("=" * 60)
    loader = WRDSLoader()
    raw_data = loader.load_week(tickers=tickers, dates=dates)
    loader.close()
    print()

    # 2. Build strategies and run simulation for each ticker
    print("=" * 60)
    print("  Step 2: Running simulations")
    print("=" * 60)

    sim = ReplaySimulator(dt=1.0, T_seconds=23400.0,
                          waiting_time=5.0, update_time=1.0)
    inv_model = InventoryModel(phi_max=PHI_MAX, eta=ETA)

    all_results = {}

    for ticker in tickers:
        sp = SPREAD_PARAMS[ticker]

        # Estimate sigma from actual trade data (average across days)
        valid_days = [raw_data[ticker][d] for d in dates
                      if raw_data[ticker][d] is not None]
        if not valid_days:
            print(f"  {ticker}: no valid data, skipping")
            continue

        sigma = estimate_sigma({d: raw_data[ticker][d]
                                for d in dates if raw_data[ticker][d]})
        print(f"  {ticker}: estimated sigma = {sigma:.6f} per second")

        # Build and calibrate A-S model
        as_model = AvellanedaStoikov(
            gamma=sp['gamma'],
            sigma=sigma,
            kappa=2.0,
            T=1.0
        )
        as_model.calibrate_from_market(sp['open_spread'], sp['close_spread'])

        baseline = BaselineStrategy()

        optimal_results  = []
        baseline_results = []

        for date in dates:
            day = raw_data[ticker][date]
            if day is None:
                print(f"    {ticker} {date}: skipped (no data)")
                continue

            print(f"    {ticker} {date}...", end=" ", flush=True)

            res_opt = sim.run(as_model, day, inv_model, strategy_type='optimal')
            res_base = sim.run(baseline, day, inv_model, strategy_type='baseline')

            optimal_results.append(res_opt)
            baseline_results.append(res_base)
            print(f"P&L opt={res_opt.pnl[-1]:,.0f}  base={res_base.pnl[-1]:,.0f}")

        all_results[ticker] = {
            'optimal':  optimal_results,
            'baseline': baseline_results,
            'as_model': as_model,
        }

    # 3. Print summary tables
    print("\n" + "=" * 60)
    print("  Step 3: Results")
    print("=" * 60)

    t2, t3 = summarise_results(all_results)
    print("\n--- Table 2: Average Terminal P&L and Position ---")
    print(t2.to_string(index=False))

    print("\n--- Table 3: Mean / Stdev of Daily P&L and Position ---")
    print(t3.to_string(index=False))

    t4, t5 = summarise_order_stats(all_results)
    print("\n--- Table 4: Avg Orders/Shares/Quotes (Optimal) ---")
    print(t4.to_string(index=False))
    print("\n--- Table 5: Avg Orders/Shares/Quotes (Baseline) ---")
    print(t5.to_string(index=False))

    print("\n--- Fill Analysis (replaces Markov chain — see helpers.py for rationale) ---")
    print_fill_analysis(all_results)

    return all_results


if __name__ == '__main__':
    results = run_wrds_experiment()

    # Optionally generate plots
    try:
        from analysis import plot_all
        plot_all(results, subfolder='wrds')
    except Exception as e:
        print(f"\nPlotting skipped: {e}")
