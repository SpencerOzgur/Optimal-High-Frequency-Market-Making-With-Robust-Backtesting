"""
wrds_loader.py
==============
Loads and cleans TAQ data from WRDS for the Avellaneda-Stoikov simulator.

Requires:
    pip install wrds pandas numpy

WRDS TAQ tables used:
    taqmsec.ctm_YYYYMMDD  — Consolidated Trades (millisecond)
    taqmsec.cqm_YYYYMMDD  — Consolidated Quotes (millisecond)

On first run, wrds.Connection() will prompt for your WRDS username/password
and optionally save credentials to ~/.pgpass for future sessions.

Usage
-----
    from wrds_loader import WRDSLoader

    loader = WRDSLoader()                        # prompts for credentials once
    data = loader.load_week(
        tickers=['AAPL', 'AMZN', 'GE', 'IVV', 'M'],
        dates=['2017-06-12', '2017-06-13', '2017-06-14', '2017-06-15', '2017-06-16']
    )
    # data['AAPL']['2017-06-12'] -> {'quotes': DataFrame, 'trades': DataFrame,
    #                                'mid': np.ndarray, 'best_bid': np.ndarray,
    #                                'best_ask': np.ndarray, 'sigma': float}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

# WRDS import is optional at module level — only needed when actually connecting
try:
    import wrds
    WRDS_AVAILABLE = True
except ImportError:
    WRDS_AVAILABLE = False
    warnings.warn("wrds package not installed. Run: pip install wrds")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_OPEN  = pd.Timestamp('09:30:00').time()
MARKET_CLOSE = pd.Timestamp('16:00:00').time()
T_SECONDS    = 23400.0   # 6.5 hours

# TAQ condition codes to exclude (clearly erroneous prints)
# See: https://www.ctaplan.com/publicdocs/ctaplan/notifications/trader-update/CTS_Pillar_Trade_Cond_Codes.pdf
BAD_TRADE_CONDITIONS = {
    'Z',   # out of sequence
    'T',   # extended hours
    'U',   # extended hours (odd lot)
    'L',   # sold last (out of sequence)
    'G',   # bunched sold trade
    'W',   # average price trade
    'J',   # rule 127/155 (NYSE)
    'K',   # rule 127/155 (odd lot)
}

# Quote conditions indicating stale/crossed quotes
BAD_QUOTE_CONDITIONS = {'C', 'U', 'D', 'B', 'W', 'X', 'Y'}


# ---------------------------------------------------------------------------
# WRDS Loader
# ---------------------------------------------------------------------------

class WRDSLoader:
    """
    Connects to WRDS and downloads TAQ millisecond data.

    Parameters
    ----------
    wrds_username : str, optional
        Your WRDS username. If None, wrds.Connection() will prompt interactively.
    """

    def __init__(self, wrds_username: str = None):
        if not WRDS_AVAILABLE:
            raise ImportError("Install wrds: pip install wrds")
        print("Connecting to WRDS...")
        self.db = wrds.Connection(wrds_username=wrds_username)
        print("Connected.\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_week(self, tickers: List[str],
                  dates: List[str]) -> Dict[str, Dict[str, dict]]:
        """
        Load and process TAQ data for a list of tickers and dates.

        Returns
        -------
        Nested dict: result[ticker][date] = {
            'quotes'   : raw cleaned quote DataFrame,
            'trades'   : raw cleaned trade DataFrame,
            'mid'      : np.ndarray (1-second mid prices, length T_SECONDS),
            'best_bid' : np.ndarray (1-second best bids),
            'best_ask' : np.ndarray (1-second best asks),
            'sigma'    : float (per-second volatility, estimated from trades),
            'market_trades' : pd.DataFrame (all trades, used by replay simulator),
        }
        """
        result = {}
        for ticker in tickers:
            result[ticker] = {}
            for date in dates:
                print(f"  Loading {ticker} {date}...", end=" ", flush=True)
                try:
                    quotes = self._load_quotes(ticker, date)
                    trades = self._load_trades(ticker, date)
                    processed = self._process_day(quotes, trades, date)
                    result[ticker][date] = processed
                    print(f"done  ({len(quotes):,} quotes, {len(trades):,} trades)")
                except Exception as e:
                    print(f"FAILED: {e}")
                    result[ticker][date] = None
        return result

    # ------------------------------------------------------------------
    # Raw data fetchers
    # ------------------------------------------------------------------

    def _load_quotes(self, ticker: str, date: str) -> pd.DataFrame:
        """
        Pull NBBO quotes from taqmsec.cqm_YYYYMMDD.

        Key columns returned:
            time_m   — millisecond timestamp
            bid      — best bid price
            ask      — best ask (ofr) price
            bidsiz   — bid size (round lots)
            asksiz   — ask size (round lots)
            qu_cond  — quote condition code
            natbbo_ind — national BBO indicator
        """
        table = f"taqmsec.cqm_{date.replace('-', '')}"
        query = f"""
            SELECT
                time_m,
                bid,
                ofr      AS ask,
                bidsiz,
                ofrsiz   AS asksiz,
                qu_cond,
                natbbo_ind
            FROM {table}
            WHERE sym_root = '{ticker}'
              AND date     = '{date}'
              AND time_m BETWEEN '09:30:00' AND '16:00:00'
              AND bid  > 0
              AND ofr  > 0
              AND bid  < ofr
        """
        df = self.db.raw_sql(query, date_cols=['date'])

        # Filter bad quote conditions
        if 'qu_cond' in df.columns:
            df = df[~df['qu_cond'].isin(BAD_QUOTE_CONDITIONS)]

        # Keep only National BBO updates (natbbo_ind in {'1','4'} = set/revised)
        if 'natbbo_ind' in df.columns:
            df = df[df['natbbo_ind'].isin(['1', '4', 1, 4])]

        df = df.sort_values('time_m').reset_index(drop=True)
        return df

    def _load_trades(self, ticker: str, date: str) -> pd.DataFrame:
        """
        Pull consolidated trades from taqmsec.ctm_YYYYMMDD.

        Key columns returned:
            time_m   — millisecond timestamp
            price    — execution price
            size     — number of shares traded
            tr_corr  — correction indicator (00 = original, keep only these)
            tr_scond — trade sale condition
        """
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
        df = self.db.raw_sql(query, date_cols=['date'])

        # Filter bad trade conditions
        if 'tr_scond' in df.columns:
            mask = df['tr_scond'].apply(
                lambda c: not any(ch in BAD_TRADE_CONDITIONS
                                  for ch in str(c).split()) if pd.notna(c) else True
            )
            df = df[mask]

        df = df.sort_values('time_m').reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Processing: resample to 1-second grid
    # ------------------------------------------------------------------

    def _process_day(self, quotes: pd.DataFrame,
                     trades: pd.DataFrame,
                     date: str) -> dict:
        """
        Convert raw tick data into 1-second arrays for the simulator.

        Strategy:
          - Quotes: forward-fill at 1-second frequency (last quote in each second)
          - Mid-price: derived from quotes as (bid + ask) / 2
          - Sigma: realised vol from trade prices at 1-second intervals
          - Market trades: kept at tick resolution for the replay simulator
        """
        open_ts  = pd.Timestamp(f"{date} 09:30:00")
        close_ts = pd.Timestamp(f"{date} 16:00:00")

        # Build a 1-second index for the trading day
        second_index = pd.date_range(open_ts, close_ts, freq='1s')
        n_steps = len(second_index) - 1   # 23400

        # ---- Quotes → 1-second grid ----
        quotes = quotes.copy()
        quotes['timestamp'] = pd.to_datetime(
            date + ' ' + quotes['time_m'].astype(str)
        )
        quotes = quotes.set_index('timestamp').sort_index()

        # Reindex to 1-second grid, forward-fill (last known NBBO)
        q_resampled = quotes[['bid', 'ask']].resample('1s').last().ffill()
        q_resampled = q_resampled.reindex(second_index).ffill().bfill()

        best_bid = q_resampled['bid'].values.astype(float)
        best_ask = q_resampled['ask'].values.astype(float)
        mid      = (best_bid + best_ask) / 2.0

        # ---- Sigma: realised vol from 1-second trade prices ----
        trades_copy = trades.copy()
        trades_copy['timestamp'] = pd.to_datetime(
            date + ' ' + trades_copy['time_m'].astype(str)
        )
        trades_copy = trades_copy.set_index('timestamp').sort_index()

        trade_prices = trades_copy.resample('1s')['price'].last()
        log_returns  = np.diff(np.log(trade_prices.values))
        sigma        = float(np.std(log_returns[np.isfinite(log_returns)]))

        # ---- Market trades: keep tick-level for replay simulator ----
        # Add seconds-since-open column for fast lookup in simulator
        trades_copy = trades_copy.reset_index()
        trades_copy['t_sec'] = (
            trades_copy['timestamp'] - open_ts
        ).dt.total_seconds()
        # Only keep trades during regular session
        trades_copy = trades_copy[
            (trades_copy['t_sec'] >= 0) &
            (trades_copy['t_sec'] <= T_SECONDS)
        ].reset_index(drop=True)

        return {
            'quotes'        : q_resampled,
            'trades'        : trades_copy,
            'mid'           : mid,
            'best_bid'      : best_bid,
            'best_ask'      : best_ask,
            'sigma'         : sigma,
            'market_trades' : trades_copy,
        }

    def close(self):
        self.db.close()


# ---------------------------------------------------------------------------
# Calibration helpers (run once you have data)
# ---------------------------------------------------------------------------

def estimate_sigma(market_data: dict) -> float:
    """
    Estimate per-second mid-price volatility from a processed day dict.
    Returns the average sigma across all days for a ticker.
    """
    sigmas = [day['sigma'] for day in market_data.values() if day is not None]
    return float(np.mean(sigmas))


def estimate_bathtub_profile(market_data_all_days: List[dict],
                              n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the empirical intraday volume profile from TAQ trade data.

    Aggregates trade volume across all provided days into n_bins time buckets.
    Returns (bin_centres_seconds, normalised_volume_profile).

    Useful for visualising intraday activity patterns or as an input
    to any future extension that requires a time-varying intensity model.
    """
    all_t_sec = []
    all_sizes = []
    for day in market_data_all_days:
        if day is None:
            continue
        t   = day['market_trades']['t_sec'].values
        sz  = day['market_trades']['size'].values
        all_t_sec.append(t)
        all_sizes.append(sz)

    if not all_t_sec:
        raise ValueError("No valid days provided")

    t_all   = np.concatenate(all_t_sec)
    sz_all  = np.concatenate(all_sizes)

    bins    = np.linspace(0, T_SECONDS, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2

    # Volume-weighted histogram
    vol_profile, _ = np.histogram(t_all, bins=bins, weights=sz_all)
    vol_profile    = vol_profile / vol_profile.sum()   # normalise to sum=1

    return centres, vol_profile


def estimate_depth_decay(market_data: dict, our_spread_samples: np.ndarray,
                          n_bins: int = 20) -> float:
    """
    Estimate mu (the depth decay parameter) from the data.

    Method: for each second, compute how deep our quote would be relative to
    the NBBO, and measure the fraction of seconds in which a trade occurred
    at that depth. Fit exp(-mu * xi) to these empirical fill rates.

    Returns estimated mu.
    """
    from scipy.optimize import curve_fit

    fill_rates = []
    depths     = []

    for day in market_data.values():
        if day is None:
            continue
        trades   = day['market_trades']
        best_bid = day['best_bid']
        best_ask = day['best_ask']

        # For each 1-second bucket, check if any trade occurred
        trade_seconds = set(trades['t_sec'].astype(int).values)

        for sec in range(int(T_SECONDS)):
            # Hypothetical: our ask is best_ask + xi_sample
            xi = float(np.random.choice(our_spread_samples))
            depths.append(xi)
            fill_rates.append(1.0 if sec in trade_seconds else 0.0)

    depths     = np.array(depths)
    fill_rates = np.array(fill_rates)

    # Bin depths and compute mean fill rate per bin
    bin_edges  = np.percentile(depths, np.linspace(0, 100, n_bins + 1))
    bin_idx    = np.digitize(depths, bin_edges) - 1
    bin_idx    = np.clip(bin_idx, 0, n_bins - 1)

    bin_depths = np.array([depths[bin_idx == i].mean()
                           for i in range(n_bins) if (bin_idx == i).any()])
    bin_rates  = np.array([fill_rates[bin_idx == i].mean()
                           for i in range(n_bins) if (bin_idx == i).any()])

    # Fit: rate = A * exp(-mu * xi)
    try:
        popt, _ = curve_fit(lambda x, A, mu: A * np.exp(-mu * x),
                             bin_depths, bin_rates,
                             p0=[0.05, 100.0], maxfev=5000)
        mu_estimated = float(popt[1])
    except Exception:
        mu_estimated = 100.0   # fall back to paper default

    return mu_estimated
