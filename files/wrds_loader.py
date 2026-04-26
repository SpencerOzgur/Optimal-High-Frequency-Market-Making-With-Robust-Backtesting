"""
wrds_loader.py
==============
Loads and cleans TAQ data from WRDS for the Avellaneda-Stoikov simulator.

Requires:
    pip install wrds pandas numpy scipy

WRDS TAQ tables used:
    taqmsec.ctm_YYYYMMDD  — Consolidated Trades (millisecond)
    taqmsec.cqm_YYYYMMDD  — Consolidated Quotes (millisecond)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

pd.set_option("mode.copy_on_write", True)

try:
    import wrds
    WRDS_AVAILABLE = True
except ImportError:
    WRDS_AVAILABLE = False
    warnings.warn("wrds package not installed. Run: pip install wrds")


MARKET_OPEN  = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()
T_SECONDS    = 23400.0

BAD_TRADE_CONDITIONS = {
    "Z", "T", "U", "L", "G", "W", "J", "K"
}

BAD_QUOTE_CONDITIONS = {"C", "U", "D", "B", "W", "X", "Y"}


class WRDSLoader:
    """
    Connects to WRDS and downloads TAQ millisecond data.
    """

    def __init__(self, wrds_username: str = None):
        if not WRDS_AVAILABLE:
            raise ImportError("Install wrds: pip install wrds")

        print("Connecting to WRDS...")
        self.db = wrds.Connection(wrds_username=wrds_username)
        print("Connected.\n")

    def load_week(self, tickers: List[str], dates: List[str]) -> Dict[str, Dict[str, dict]]:
        result = {}

        for ticker in tickers:
            result[ticker] = {}

            for date in dates:
                print(f"  Loading {ticker} {date}...", end=" ", flush=True)

                try:
                    quotes = self._load_quotes(ticker, date)
                    trades = self._load_trades(ticker, date)

                    if quotes.empty:
                        raise ValueError("No valid quote data returned")
                    if trades.empty:
                        raise ValueError("No valid trade data returned")

                    processed = self._process_day(quotes, trades, date)
                    result[ticker][date] = processed

                    print(f"done  ({len(quotes):,} quotes, {len(trades):,} trades)")

                except Exception as e:
                    print(f"FAILED: {e}")
                    result[ticker][date] = None

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_timestamp(date: str, time_series: pd.Series) -> pd.Series:
        """
        Robust TAQ timestamp parser.

        TAQ time_m may appear as:
            09:30:00
            09:30:00.001
            13:48:22

        format='mixed' handles both second-level and millisecond-level strings.
        errors='coerce' prevents one bad timestamp from killing the full day.
        """
        return pd.to_datetime(
            date + " " + time_series.astype(str),
            format="mixed",
            errors="coerce"
        )

    def _get_quote_schema(self, table: str) -> Tuple[str, str]:
        """
        Detect whether WRDS TAQ quote table uses:
            ask / asksiz
        or legacy:
            ofr / ofrsiz
        """
        # table comes in like taqmsec.cqm_20170612
        library, table_name = table.split(".")

        desc = self.db.describe_table(library=library, table=table_name)
        cols = set(desc["name"].str.lower())

        if "ask" in cols:
            ask_col = "ask"
        elif "ofr" in cols:
            ask_col = "ofr"
        else:
            raise ValueError(f"No ask/ofr column found in {table}")

        if "asksiz" in cols:
            asksiz_col = "asksiz"
        elif "ofrsiz" in cols:
            asksiz_col = "ofrsiz"
        else:
            # Size is useful but not essential for your current processing.
            # If neither exists, return NULL as asksiz.
            asksiz_col = None

        return ask_col, asksiz_col

    # ------------------------------------------------------------------
    # Raw data fetchers
    # ------------------------------------------------------------------

    def _load_quotes(self, ticker: str, date: str) -> pd.DataFrame:
        table = f"taqmsec.cqm_{date.replace('-', '')}"

        ask_col, asksiz_col = self._get_quote_schema(table)
        asksiz_expr = f"{asksiz_col} AS asksiz" if asksiz_col else "NULL AS asksiz"

        query = f"""
            SELECT
                time_m,
                bid,
                {ask_col} AS ask,
                bidsiz,
                {asksiz_expr},
                qu_cond,
                natbbo_ind
            FROM {table}
            WHERE sym_root = '{ticker}'
              AND date     = '{date}'
              AND time_m BETWEEN '09:30:00' AND '16:00:00'
              AND bid > 0
              AND {ask_col} > 0
              AND bid < {ask_col}
        """

        df = self.db.raw_sql(query)

        if df.empty:
            return df

        if "qu_cond" in df.columns:
            df = df[~df["qu_cond"].isin(BAD_QUOTE_CONDITIONS)]

        if "natbbo_ind" in df.columns:
            df = df[df["natbbo_ind"].isin(["1", "4", 1, 4])]

        df = df.sort_values("time_m").reset_index(drop=True)
        return df

    def _load_trades(self, ticker: str, date: str) -> pd.DataFrame:
        table = f"taqmsec.ctm_{date.replace('-', '')}"

        query = f"""
            SELECT
                time_m,
                price,
                size,
                tr_corr,
                tr_scond
            FROM {table}
            WHERE sym_root = '{ticker}'
              AND date     = '{date}'
              AND time_m BETWEEN '09:30:00' AND '16:00:00'
              AND price > 0
              AND size  > 0
              AND tr_corr = '00'
        """

        df = self.db.raw_sql(query)

        if df.empty:
            return df

        if "tr_scond" in df.columns:
            mask = df["tr_scond"].apply(
                lambda c: not any(
                    ch in BAD_TRADE_CONDITIONS
                    for ch in str(c).split()
                ) if pd.notna(c) else True
            )
            df = df[mask]

        df = df.sort_values("time_m").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_day(self, quotes: pd.DataFrame, trades: pd.DataFrame, date: str) -> dict:
        open_ts  = pd.Timestamp(f"{date} 09:30:00")
        close_ts = pd.Timestamp(f"{date} 16:00:00")

        second_index = pd.date_range(open_ts, close_ts, freq="1s")

        # ---- Quotes -> 1-second grid ----
        quotes = quotes.copy()
        quotes["timestamp"] = self._safe_timestamp(date, quotes["time_m"])
        quotes = quotes.dropna(subset=["timestamp"])

        if quotes.empty:
            raise ValueError("All quote timestamps failed to parse")

        quotes = quotes.set_index("timestamp").sort_index()

        q_resampled = quotes[["bid", "ask"]].resample("1s").last()
        q_resampled = q_resampled.reindex(second_index).ffill().bfill()

        best_bid = q_resampled["bid"].to_numpy(dtype=float)
        best_ask = q_resampled["ask"].to_numpy(dtype=float)
        mid = (best_bid + best_ask) / 2.0

        # ---- Trades -> clean tick data + sigma ----
        trades_copy = trades.copy()
        trades_copy["timestamp"] = self._safe_timestamp(date, trades_copy["time_m"])
        trades_copy = trades_copy.dropna(subset=["timestamp"])

        if trades_copy.empty:
            raise ValueError("All trade timestamps failed to parse")

        trades_copy = trades_copy.set_index("timestamp").sort_index()

        trade_prices = trades_copy.resample("1s")["price"].last().ffill()
        trade_prices = trade_prices.replace([np.inf, -np.inf], np.nan).dropna()
        trade_prices = trade_prices[trade_prices > 0]

        if len(trade_prices) >= 2:
            log_returns = np.diff(np.log(trade_prices.to_numpy(dtype=float)))
            log_returns = log_returns[np.isfinite(log_returns)]
            sigma = float(np.std(log_returns)) if len(log_returns) > 0 else 0.0
        else:
            sigma = 0.0

        # ---- Market trades at tick level ----
        trades_copy = trades_copy.reset_index()
        trades_copy["t_sec"] = (
            trades_copy["timestamp"] - open_ts
        ).dt.total_seconds()

        trades_copy = trades_copy[
            (trades_copy["t_sec"] >= 0) &
            (trades_copy["t_sec"] <= T_SECONDS)
        ].reset_index(drop=True)

        return {
            "quotes": q_resampled,
            "trades": trades_copy,
            "mid": mid,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "sigma": sigma,
            "market_trades": trades_copy,
        }

    def close(self):
        self.db.close()


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def estimate_sigma(market_data: dict) -> float:
    sigmas = [day["sigma"] for day in market_data.values() if day is not None]
    return float(np.mean(sigmas)) if sigmas else 0.0


def estimate_bathtub_profile(
    market_data_all_days: List[dict],
    n_bins: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    all_t_sec = []
    all_sizes = []

    for day in market_data_all_days:
        if day is None:
            continue

        t = day["market_trades"]["t_sec"].values
        sz = day["market_trades"]["size"].values

        all_t_sec.append(t)
        all_sizes.append(sz)

    if not all_t_sec:
        raise ValueError("No valid days provided")

    t_all = np.concatenate(all_t_sec)
    sz_all = np.concatenate(all_sizes)

    bins = np.linspace(0, T_SECONDS, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2

    vol_profile, _ = np.histogram(t_all, bins=bins, weights=sz_all)

    if vol_profile.sum() > 0:
        vol_profile = vol_profile / vol_profile.sum()

    return centres, vol_profile


def estimate_depth_decay(
    market_data: dict,
    our_spread_samples: np.ndarray,
    n_bins: int = 20
) -> float:
    from scipy.optimize import curve_fit

    fill_rates = []
    depths = []

    for day in market_data.values():
        if day is None:
            continue

        trades = day["market_trades"]
        trade_seconds = set(trades["t_sec"].astype(int).values)

        for sec in range(int(T_SECONDS)):
            xi = float(np.random.choice(our_spread_samples))
            depths.append(xi)
            fill_rates.append(1.0 if sec in trade_seconds else 0.0)

    depths = np.array(depths)
    fill_rates = np.array(fill_rates)

    if len(depths) == 0:
        return 100.0

    bin_edges = np.percentile(depths, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(depths, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_depths = np.array([
        depths[bin_idx == i].mean()
        for i in range(n_bins)
        if (bin_idx == i).any()
    ])

    bin_rates = np.array([
        fill_rates[bin_idx == i].mean()
        for i in range(n_bins)
        if (bin_idx == i).any()
    ])

    try:
        popt, _ = curve_fit(
            lambda x, A, mu: A * np.exp(-mu * x),
            bin_depths,
            bin_rates,
            p0=[0.05, 100.0],
            maxfev=5000
        )
        return float(popt[1])
    except Exception:
        return 100.0
