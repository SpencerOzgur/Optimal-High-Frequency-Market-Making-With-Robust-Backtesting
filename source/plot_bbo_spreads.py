"""
plot_bbo_spreads.py
===================
Plot the distribution of BBO spreads (in pennies) per ticker, pulling
from the cached `sheets/raw_data.pkl`. One subplot per ticker with the
median marked, saved as a single combined PNG.

Run from project root:
    python3 source/plot_bbo_spreads.py
    python3 source/plot_bbo_spreads.py --cache other.pkl --out plots/x.png
    python3 source/plot_bbo_spreads.py --max-pennies 30
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE_PATH = os.path.join(ROOT, "sheets", "raw_data.pkl")
OUT_PATH   = os.path.join(ROOT, "plots", "bbo_spread_distribution.png")


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"Cache not found at {path}. Run run_with_wrds.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_spread_pennies(days: dict) -> np.ndarray:
    """Pool best_ask − best_bid across all loaded days, return integer pennies."""
    arrs = []
    for d in days.values():
        if d is None:
            continue
        s = d["best_ask"] - d["best_bid"]
        arrs.append(s)
    if not arrs:
        return np.array([], dtype=int)
    spreads = np.concatenate(arrs)
    spreads = spreads[np.isfinite(spreads) & (spreads >= 0)]
    # Cached best_bid/ask are already on the penny grid; rounding only
    # cleans up the float subtraction noise (e.g. 0.0099999...).
    return np.round(spreads * 100.0).astype(int)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=CACHE_PATH,
                   help=f"Path to raw_data pickle (default: {CACHE_PATH})")
    p.add_argument("--out", default=OUT_PATH,
                   help=f"Output PNG path (default: {OUT_PATH})")
    p.add_argument("--max-pennies", type=int, default=None,
                   help="Cap x-axis at this many pennies "
                        "(default: per-ticker 99th percentile + 2)")
    args = p.parse_args()

    raw = load_cache(args.cache)
    tickers = list(raw.keys())
    n = len(tickers)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, ticker in zip(axes, tickers):
        spreads = collect_spread_pennies(raw[ticker])
        if len(spreads) == 0:
            ax.set_title(f"{ticker} — no data")
            ax.axis("off")
            continue

        cap = args.max_pennies if args.max_pennies else int(np.percentile(spreads, 99)) + 2
        # Integer-penny bins, centred on each integer (so a 1¢ bar covers 0.5–1.5¢).
        bins = np.arange(-0.5, cap + 1.5)
        ax.hist(spreads, bins=bins, edgecolor="black", linewidth=0.4)

        median = float(np.median(spreads))
        mean   = float(np.mean(spreads))
        ax.axvline(median, color="red", linestyle="--", linewidth=1,
                   label=f"median = {median:.1f}¢")
        ax.set_title(f"{ticker}   n={len(spreads):,}   mean={mean:.2f}¢")
        ax.set_xlabel("BBO spread (¢)")
        ax.set_ylabel("count")
        ax.set_xlim(-0.5, cap + 0.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(tickers):]:
        ax.axis("off")

    fig.suptitle("BBO spread distribution per ticker", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
