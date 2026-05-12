"""
run_with_wrds.py
================
End-to-end runner: WRDS TAQ data → replay simulator → results tables + plots.

"""

import numpy as np
import pandas as pd
import sys, os
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrds_loader import WRDSLoader, estimate_sigma, TICKER_VENUE
from market_maker import AvellanedaStoikov, InventoryModel, BaselineStrategy
from replay_simulator import ReplaySimulator
from helpers import summarise_results, summarise_order_stats, print_fill_analysis, export_quote_position_xlsx, export_fill_stats_xlsx

TICKERS = ['AAPL', 'AMZN', 'GE', 'IVV', 'M']

# Trading dates
DATES = [
    '2017-06-12',
    '2017-06-13',
    '2017-06-14',
    '2017-06-15',
    '2017-06-16',
]

# Risk aversion parameter tuned per asset
GAMMA_PARAMS = {
    'AAPL': 1.78e-7,
    'AMZN': 1.42e-7,
    'GE':   9.02e-5,
    'IVV':  1.23e-9,
    'M':    3.33e-4,
}

# Parameters turned via calibrate_params.py + manual adjustments
KAPPA_PARAMS = {
    'AAPL': 81.81,
    'AMZN': 7.57,
    'GE':   292.55,
    'IVV':  200,
    'M':    111.899,
}

"""
Laplace distribution for fills inside bbo:

λ(ξ) = A * exp(-ξ / b),       Δλ = λ(ξ_our) - λ(ξ_best)
A is fills/sec at the mid; b is the Laplace scale (decay width) in dollars.

Parameters turned via calibrate_params.py + manual adjustments

"""

A_PARAMS = {
    'AAPL': 0.23980,
    'AMZN': 0.02740,
    'GE':   0.00822,
    'IVV':  0.00215,
    'M':    0.07232,
}

B_PARAMS = {
    'AAPL': 0.00381,
    'AMZN': 0.12252,
    'GE':   0.00383,
    'IVV':  0.00506,
    'M':    0.00227,
}

PHI_MAX = 100.0
ETA     = 0.005

CACHE_PATH = 'sheets/raw_data.pkl'
CACHE_PATH_RESULTS = 'sheets/results.pkl'

# Calibration dates for sigma
CALIB_DATES = [
    '2017-06-05',
    '2017-06-06',
    '2017-06-07',
    '2017-06-08',
    '2017-06-09',
]
CACHE_PATH_CALIB = 'sheets/raw_data_calib.pkl'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_wrds_experiment(tickers=TICKERS, dates=DATES):
    # 1. Load data from WRDS
    print("=" * 60)
    print("  Step 1: Loading TAQ data from WRDS")
    print("=" * 60)
    try:
        with open(CACHE_PATH, 'rb') as f:
            raw_data = pickle.load(f)
        print("Loaded raw_data from cache.")
    except FileNotFoundError:
        loader = WRDSLoader()
        raw_data = loader.load_week(tickers=tickers, dates=dates)
        loader.close()
        os.makedirs('sheets', exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(raw_data, f)
        print("Fetched from WRDS and cached.")

    # Prior-week sigma calibration data (separate cache so eval and calib
    # weeks stay decoupled). Fetch on first use; reuse after.
    try:
        with open(CACHE_PATH_CALIB, 'rb') as f:
            calib_data = pickle.load(f)
        print(f"Loaded calibration-week data from {CACHE_PATH_CALIB}.")
    except FileNotFoundError:
        loader = WRDSLoader()
        calib_data = loader.load_week(tickers=tickers, dates=CALIB_DATES)
        loader.close()
        with open(CACHE_PATH_CALIB, 'wb') as f:
            pickle.dump(calib_data, f)
        print(f"Fetched calibration week from WRDS and cached at {CACHE_PATH_CALIB}.")
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
        # Estimate spreads empirically from data
        open_spreads = []
        close_spreads = []
        all_spreads = []
        for d in dates:
            if raw_data[ticker][d] is not None:
                spreads = raw_data[ticker][d]['best_ask'] - raw_data[ticker][d]['best_bid']
                open_spreads.append(float(np.nanmedian(spreads[:1800])))
                close_spreads.append(float(np.nanmedian(spreads[-1800:])))
                all_spreads.append(spreads)

        open_spread = float(np.mean(open_spreads))
        close_spread = float(np.mean(close_spreads))

        median_spread = float(np.nanmedian(np.concatenate(all_spreads)))

        # Per-ticker kappa is loaded from KAPPA_PARAMS
        venue_label = TICKER_VENUE.get(ticker, "NBBO")
        print(f"  {ticker} [{venue_label}]: empirical open_spread={open_spread:.4f} "
              f"close_spread={close_spread:.4f} "
              f"median_spread={median_spread:.4f} kappa={KAPPA_PARAMS[ticker]:.2f}")

        # Calibrate sigma off the PRIOR week
        sigma = estimate_sigma({d: calib_data[ticker][d]
                                for d in CALIB_DATES
                                if calib_data.get(ticker, {}).get(d)})



        as_model = AvellanedaStoikov(
            gamma=GAMMA_PARAMS[ticker],
            sigma=sigma,
            kappa=KAPPA_PARAMS[ticker],
            T=1.0,
        )

        """
        Toggle for Stanford paper-style calibration:
        """
        #
        #as_model.calibrate_from_market(open_spread, close_spread)
        #as_model.gamma_sigma2 = as_model.gamma_sigma2 / 23400.0

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

            # Per-ticker A/b for fills inside bbo
            a_t = A_PARAMS[ticker]
            b_t = B_PARAMS[ticker]

            res_opt_front  = sim.run(as_model, day, inv_model,
                                     strategy_type='optimal',  queue_model='front',
                                     a_intensity=a_t, b_scale=b_t)
            res_opt_back   = sim.run(as_model, day, inv_model,
                                     strategy_type='optimal',  queue_model='back',
                                     a_intensity=a_t, b_scale=b_t)
            res_base_front = sim.run(baseline, day, inv_model,
                                     strategy_type='baseline', queue_model='front',
                                     a_intensity=a_t, b_scale=b_t)
            res_base_back  = sim.run(baseline, day, inv_model,
                                     strategy_type='baseline', queue_model='back',
                                     a_intensity=a_t, b_scale=b_t)

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
    rerun = True
    if rerun:
        results, best_bid_dict, best_ask_dict = run_wrds_experiment()
        with open(CACHE_PATH_RESULTS, 'wb') as f:
            pickle.dump((results, best_bid_dict, best_ask_dict), f)
            print('cached.')
    else:
        with open(CACHE_PATH_RESULTS, 'rb') as f:
            results, best_bid_dict, best_ask_dict = pickle.load(f)

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
