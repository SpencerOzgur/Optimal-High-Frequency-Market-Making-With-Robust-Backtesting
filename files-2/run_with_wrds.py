"""
run_with_wrds.py
================
End-to-end runner: WRDS TAQ data → replay simulator → results tables + plots.

Runs all three fill models for comparison:
  - first_in_queue    : original assumption, most optimistic
  - pro_rata          : proportional queue share, more conservative
  - pro_rata_imbalance: pro_rata + OBI adjustment with calibrated alpha

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
# Configuration
# ---------------------------------------------------------------------------

TICKERS = ['AAPL', 'AMZN', 'GE', 'IVV', 'M']

# Calibration period — used to estimate alpha for pro_rata_imbalance
# Kept separate from test dates so alpha is not fit on test data
CALIBRATION_DATES = [
    '2017-06-12',
    '2017-06-13',
    '2017-06-14',
    '2017-06-15',
    '2017-06-16',
]

# Test dates — the five regime days
TEST_DATES = [
    '2017-11-03',   # VIX all-time low
    '2018-02-05',   # Volmageddon
    '2015-08-24',   # Flash crash
    '2018-03-22',   # China tariff shock
    '2016-07-26',   # Quiet summer day
]

# Table 1 open/close spreads (used for A-S calibration)
SPREAD_PARAMS = {
    'AAPL': {'open_spread': 0.05,  'close_spread': 0.01, 'gamma': 0.1},
    'AMZN': {'open_spread': 0.49,  'close_spread': 0.56, 'gamma': 0.01},
    'GE':   {'open_spread': 0.04,  'close_spread': 0.01, 'gamma': 0.1},
    'IVV':  {'open_spread': 0.03,  'close_spread': 0.01, 'gamma': 0.1},
    'M':    {'open_spread': 0.09,  'close_spread': 0.01, 'gamma': 0.1},
}

PHI_MAX    = 100.0
ETA        = 0.005
FILL_MODELS = ['first_in_queue', 'pro_rata', 'pro_rata_imbalance']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_wrds_experiment(tickers=TICKERS,
                         calibration_dates=CALIBRATION_DATES,
                         test_dates=TEST_DATES):

    # 1. Load calibration data
    print("=" * 60)
    print("  Step 1: Loading calibration data from WRDS")
    print("=" * 60)
    loader = WRDSLoader()
    calib_data = loader.load_week(tickers=tickers, dates=calibration_dates)
    print()

    # 2. Load test data
    print("=" * 60)
    print("  Step 2: Loading test data from WRDS")
    print("=" * 60)
    test_data = loader.load_week(tickers=tickers, dates=test_dates)
    loader.close()
    print()

    # 3. Calibrate alpha using calibration period
    print("=" * 60)
    print("  Step 3: Calibrating OBI alpha")
    print("=" * 60)
    # Use all tickers' calibration days to get a robust estimate
    all_calib_days = [
        calib_data[ticker][date]
        for ticker in tickers
        for date in calibration_dates
        if calib_data[ticker].get(date) is not None
    ]
    # Calibrate using a temporary simulator instance
    calib_sim = ReplaySimulator(fill_model='pro_rata_imbalance')
    alpha = calib_sim.calibrate_alpha(all_calib_days)
    print()

    # 4. Run simulations for each ticker × fill model × strategy
    print("=" * 60)
    print("  Step 4: Running simulations")
    print("=" * 60)

    inv_model = InventoryModel(phi_max=PHI_MAX, eta=ETA)

    # Build one simulator per fill model, sharing the calibrated alpha
    simulators = {}
    for fm in FILL_MODELS:
        sim = ReplaySimulator(dt=1.0, T_seconds=23400.0,
                              waiting_time=5.0, update_time=1.0,
                              fill_model=fm)
        sim.alpha = alpha   # share calibrated alpha across all instances
        simulators[fm] = sim

    all_results = {}

    for ticker in tickers:
        sp = SPREAD_PARAMS[ticker]

        # Estimate sigma from calibration data
        valid_calib = {d: calib_data[ticker][d] for d in calibration_dates
                       if calib_data[ticker].get(d) is not None}
        if not valid_calib:
            print(f"  {ticker}: no calibration data, skipping")
            continue

        sigma = estimate_sigma(valid_calib)
        print(f"  {ticker}: sigma={sigma:.6f}")

        # Build and calibrate A-S model
        as_model = AvellanedaStoikov(
            gamma=sp['gamma'], sigma=sigma, kappa=2.0, T=1.0
        )
        as_model.calibrate_from_market(sp['open_spread'], sp['close_spread'])
        baseline = BaselineStrategy()

        all_results[ticker] = {'as_model': as_model}

        for fm in FILL_MODELS:
            sim = simulators[fm]
            optimal_results  = []
            baseline_results = []

            for date in test_dates:
                day = test_data[ticker].get(date)
                if day is None:
                    print(f"    {ticker} {date} [{fm}]: skipped")
                    continue

                res_opt  = sim.run(as_model, day, inv_model, 'optimal')
                res_base = sim.run(baseline, day, inv_model, 'baseline')
                optimal_results.append(res_opt)
                baseline_results.append(res_base)

            all_results[ticker][f'optimal_{fm}']  = optimal_results
            all_results[ticker][f'baseline_{fm}'] = baseline_results

            pnl_opt  = np.mean([r.pnl[-1] for r in optimal_results])  if optimal_results  else float('nan')
            pnl_base = np.mean([r.pnl[-1] for r in baseline_results]) if baseline_results else float('nan')
            print(f"    {ticker} [{fm}]  opt P&L={pnl_opt:,.0f}  base P&L={pnl_base:,.0f}")

    # 5. Print comparison tables
    print("\n" + "=" * 60)
    print("  Step 5: Results")
    print("=" * 60)

    # Reformat all_results into the shape summarise_results expects,
    # once per fill model so tables are comparable
    for fm in FILL_MODELS:
        print(f"\n{'='*60}")
        print(f"  Fill model: {fm}")
        print(f"{'='*60}")

        fm_results = {
            ticker: {
                'optimal':  all_results[ticker].get(f'optimal_{fm}',  []),
                'baseline': all_results[ticker].get(f'baseline_{fm}', []),
                'as_model': all_results[ticker]['as_model'],
            }
            for ticker in all_results
        }

        t2, t3 = summarise_results(fm_results)
        print("\n--- Avg Terminal P&L and Position ---")
        print(t2.to_string(index=False))

        print("\n--- Fill Analysis ---")
        print_fill_analysis(fm_results)

    return all_results


if __name__ == '__main__':
    results = run_wrds_experiment()

    try:
        from analysis import plot_all
        # Plot using first_in_queue results for figures (matches paper)
        paper_results = {
            ticker: {
                'optimal':  results[ticker].get('optimal_first_in_queue', []),
                'baseline': results[ticker].get('baseline_first_in_queue', []),
                'as_model': results[ticker]['as_model'],
            }
            for ticker in results
        }
        plot_all(results, test_dates=test_dates)
    except Exception as e:
        print(f"\nPlotting skipped: {e}")
