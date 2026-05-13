"""
exchange_diagnostics.py
=======================
Per-ticker breakdown of TAQ exchange (`ex`) codes, sorted descending by
percentage of rows.

  CTM (trades):  read from sheets/raw_data.pkl cache (already has `ex`).
  CQM (quotes):  queried fresh from WRDS — the loader strips `ex` during
                 processing so it is not in the cache.

Run from project root:
    python3 source/exchange_diagnostics.py
    python3 source/exchange_diagnostics.py --no-bbo   # trades only, skip WRDS
    python3 source/exchange_diagnostics.py --cache other.pkl
"""

import argparse
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_PATH = os.path.join(ROOT, "sheets", "raw_data.pkl")


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"Cache not found at {path}. Run run_with_wrds.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def print_ex_table(title: str, counts: dict, label: str) -> None:
    if not counts:
        print(f"\n{title}\n  (no rows)")
        return
    series = pd.Series(counts).sort_values(ascending=False)
    total = int(series.sum())
    pct = series / total * 100.0
    df = pd.DataFrame({
        "count": series.map(lambda v: f"{int(v):>12,d}"),
        "pct":   pct.map(lambda v: f"{v:6.2f}%"),
    })
    print(f"\n{title}  (n={total:,} {label})")
    print(df.to_string())


def trade_breakdowns(raw_data: dict) -> None:
    print("=" * 60)
    print("  CTM (trades) — exchange code distribution")
    print("=" * 60)
    for ticker, days in raw_data.items():
        counts: dict = {}
        for d in days.values():
            if d is None:
                continue
            vc = d["market_trades"]["ex"].value_counts(dropna=False)
            for ex, n in vc.items():
                counts[ex] = counts.get(ex, 0) + int(n)
        print_ex_table(ticker, counts, label="trades")


def quote_breakdowns(raw_data: dict) -> None:
    print("\n" + "=" * 60)
    print("  CQM (quotes) — exchange code distribution (querying WRDS)")
    print("=" * 60)
    sys.path.insert(0, HERE)
    from wrds_loader import WRDSLoader  # noqa: E402

    loader = WRDSLoader()
    try:
        for ticker, days in raw_data.items():
            counts: dict = {}
            for date, d in days.items():
                if d is None:
                    continue
                tbl = f"taqmsec.cqm_{date.replace('-', '')}"
                # Single SQL aggregation per (ticker, date) — only returns
                # one row per exchange code, so this is cheap regardless of
                # how many quote rows the underlying table holds.
                q = f"""
                    SELECT ex, COUNT(*) AS n
                    FROM {tbl}
                    WHERE sym_root = '{ticker}'
                      AND date     = '{date}'
                      AND time_m BETWEEN '09:30:00' AND '16:00:00'
                    GROUP BY ex
                """
                df = loader.db.raw_sql(q)
                for _, row in df.iterrows():
                    counts[row["ex"]] = counts.get(row["ex"], 0) + int(row["n"])
            print_ex_table(ticker, counts, label="quote rows")
    finally:
        loader.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=CACHE_PATH,
                   help=f"Path to raw_data pickle (default: {CACHE_PATH})")
    p.add_argument("--no-bbo", action="store_true",
                   help="Skip the BBO/CQM section (no WRDS connection needed).")
    args = p.parse_args()

    raw_data = load_cache(args.cache)
    trade_breakdowns(raw_data)
    if not args.no_bbo:
        quote_breakdowns(raw_data)


if __name__ == "__main__":
    main()
