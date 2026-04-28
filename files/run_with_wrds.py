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
from helpers import summarise_results, summarise_order_stats, print_fill_analysis, export_quote_position_xlsx, export_fill_stats_xlsx

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

# Keep only gamma since that's a model parameter not a data parameter
GAMMA_PARAMS = {
    'AAPL': 0.001,
    'AMZN': 0.001,
    'GE':   0.001,
    'IVV':  0.001,
    'M':    0.001,
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
    best_bid_dict = {}
    best_ask_dict = {}

    for ticker in tickers:
        # Estimate spreads empirically from actual data
        open_spreads = []
        close_spreads = []
        for d in dates:
            if raw_data[ticker][d] is not None:
                spreads = raw_data[ticker][d]['best_ask'] - raw_data[ticker][d]['best_bid']
                open_spreads.append(float(np.nanmedian(spreads[:1800])))
                close_spreads.append(float(np.nanmedian(spreads[-1800:])))

        open_spread = float(np.mean(open_spreads))
        close_spread = float(np.mean(close_spreads))

        print(f"  {ticker}: empirical open_spread={open_spread:.4f} close_spread={close_spread:.4f}")

        sigma = estimate_sigma({d: raw_data[ticker][d]
                                for d in dates if raw_data[ticker][d]})

        as_model = AvellanedaStoikov(
            gamma=GAMMA_PARAMS[ticker],
            sigma=sigma,
            kappa=2.0,
            T=1.0
        )
        as_model.calibrate_from_market(open_spread, close_spread)
        as_model.gamma_sigma2 = as_model.gamma_sigma2 / 23400.0


        baseline = BaselineStrategy()

        optimal_front_results  = []
        optimal_back_results   = []
        baseline_front_results = []
        baseline_back_results  = []

        for date in dates:
            day = raw_data[ticker][date]
            if day is None:
                print(f"    {ticker} {date}: skipped (no data)")
                continue

            print(f"    {ticker} {date}...", end=" ", flush=True)

            res_opt_front  = sim.run(as_model, day, inv_model,
                                     strategy_type='optimal',  queue_model='front')
            res_opt_back   = sim.run(as_model, day, inv_model,
                                     strategy_type='optimal',  queue_model='back')
            res_base_front = sim.run(baseline, day, inv_model,
                                     strategy_type='baseline', queue_model='front')
            res_base_back  = sim.run(baseline, day, inv_model,
                                     strategy_type='baseline', queue_model='back')

            optimal_front_results.append(res_opt_front)
            optimal_back_results.append(res_opt_back)
            baseline_front_results.append(res_base_front)
            baseline_back_results.append(res_base_back)

            print(f"P&L opt(F)={res_opt_front.pnl[-1]:,.0f} "
                  f"opt(B)={res_opt_back.pnl[-1]:,.0f} "
                  f"base(F)={res_base_front.pnl[-1]:,.0f} "
                  f"base(B)={res_base_back.pnl[-1]:,.0f}")

        all_results[ticker] = {
            'optimal':       optimal_front_results,
            'optimal_back':  optimal_back_results,
            'baseline':      baseline_front_results,
            'baseline_back': baseline_back_results,
            'as_model':      as_model,
        }

        best_bid_dict[ticker] = [
            raw_data[ticker][d]['best_bid']
            for d in dates if raw_data[ticker][d] is not None
        ]
        best_ask_dict[ticker] = [
            raw_data[ticker][d]['best_ask']
            for d in dates if raw_data[ticker][d] is not None
        ]

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

    return all_results, best_bid_dict, best_ask_dict


if __name__ == '__main__':
    results, best_bid_dict, best_ask_dict = run_wrds_experiment()

    # Optionally generate plots
    try:
        from analysis import plot_all
        plot_all(results, subfolder='wrds')
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    try:
        export_fill_stats_xlsx(
            results,
            best_bid_dict=best_bid_dict,
            best_ask_dict=best_ask_dict,
            path='sheets/fill_stats.xlsx'
        )
    except Exception as e:
        print(f"\nFill stats export skipped: {e}")

    try:
        export_quote_position_xlsx(
            results,
            best_bid_dict,
            best_ask_dict,
            path='sheets/quote_position.xlsx'
        )


    except Exception as e:
        print(f"\nExport skipped: {e}")
