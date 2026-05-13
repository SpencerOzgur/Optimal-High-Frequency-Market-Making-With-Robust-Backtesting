"""
calibrate_params.py
===================
Per-ticker calibration of (A, b) for the intra-spread Poisson-uplift fill
model used by replay_simulator._poisson_uplift_fill.

We default to the week prior to the run_with_wrds.py evaluation window:

    Calibration:   2017-06-05 ... 2017-06-09
    Evaluation:    2017-06-12 ... 2017-06-16   (run_with_wrds.py default)

Same TICKER_VENUE filter and addressable-flow rules apply (handled by
WRDSLoader._load_quotes / _load_trades), so the calibration sees the
exact trade universe the simulator will match against — no cross-venue
contamination.

The script PRINTS a table of fitted (A, b) per ticker to be inputted
into A_PARAMS / B_PARAMS in run_with_wrds.py (manually).

Usage
-----
    python3 source/calibrate_params.py
    python3 source/calibrate_params.py --tickers AAPL,GE
    python3 source/calibrate_params.py --dates 2017-06-05,2017-06-06,...
    python3 source/calibrate_params.py --no-cache       # force WRDS refetch
"""

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from wrds_loader import (  # noqa: E402
    WRDSLoader,
    calibrate_kappa,
    calibrate_poisson_uplift,
    TICKER_VENUE,
)

DEFAULT_CALIBRATION_DATES = [
    '2017-06-05',
    '2017-06-06',
    '2017-06-07',
    '2017-06-08',
    '2017-06-09',
]
DEFAULT_TICKERS = list(TICKER_VENUE.keys())
# Independent cache so the calibration week's pull doesn't clobber the
# evaluation week's sheets/raw_data.pkl.
DEFAULT_CACHE   = os.path.join(ROOT, 'sheets', 'raw_data_calib.pkl')


def fetch_or_load(cache: str, no_cache: bool, tickers, dates) -> dict:
    if not no_cache and os.path.exists(cache):
        print(f"Loading cached calibration data from {cache}")
        with open(cache, 'rb') as f:
            return pickle.load(f)
    loader = WRDSLoader()
    try:
        raw = loader.load_week(tickers=tickers, dates=dates)
    finally:
        loader.close()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(raw, f)
    print(f"Saved calibration cache to {cache}")
    return raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dates', default=','.join(DEFAULT_CALIBRATION_DATES),
                   help='Comma-separated YYYY-MM-DD dates (default: prior week of eval).')
    p.add_argument('--tickers', default=','.join(DEFAULT_TICKERS),
                   help='Comma-separated tickers (default: keys of TICKER_VENUE).')
    p.add_argument('--cache', default=DEFAULT_CACHE,
                   help=f'Calibration-week cache pickle (default: {DEFAULT_CACHE}).')
    p.add_argument('--no-cache', action='store_true',
                   help='Force a fresh WRDS pull, ignoring any existing cache.')
    args = p.parse_args()

    dates   = [d.strip() for d in args.dates.split(',')   if d.strip()]
    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]

    venue_map = {t: TICKER_VENUE.get(t, 'NBBO') for t in tickers}
    print(f"Calibration dates : {dates}")
    print(f"Tickers           : {tickers}")
    print(f"Per-ticker venue  : {venue_map}")
    print()

    raw = fetch_or_load(args.cache, args.no_cache, tickers, dates)

    print()
    print("=" * 78)
    print("  Per-ticker calibration (prior week, post TICKER_VENUE filter)")
    print("=" * 78)
    print("    kappa  : exponential fit on ALL trade depths from mid")
    print("             λ(δ) = A·exp(-kappa·δ); slope -> KAPPA_PARAMS")
    print("    A, b   : exponential fit on STRICTLY intra-spread trade depths")
    print("             λ(ξ) = A·exp(-ξ/b); intercept -> A_PARAMS, slope -> B_PARAMS")
    print()
    print(f"  {'ticker':<8} {'venue':<6} {'kappa':>10} "
          f"{'A (fills/sec)':>15} {'b ($)':>10} {'b (¢)':>9}")
    print("  " + "-" * 68)

    for ticker in tickers:
        days = raw.get(ticker, {})
        ticker_days = {d: days[d] for d in dates if days.get(d) is not None}
        kappa = calibrate_kappa(ticker_days)
        A, b  = calibrate_poisson_uplift(ticker_days)
        print(f"  {ticker:<8} {TICKER_VENUE.get(ticker, '-'):<6} "
              f"{kappa:>10.4f} {A:>15.5f} {b:>10.5f} {100*b:>9.3f}")

    print()
    print("Copy these values into KAPPA_PARAMS / A_PARAMS / B_PARAMS in run_with_wrds.py.")


if __name__ == '__main__':
    main()
