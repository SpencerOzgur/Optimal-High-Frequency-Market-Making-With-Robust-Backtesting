"""
realized_vol_strategy.py
========================
Extension of the A-S model using rolling realized volatility
in the spread equation instead of a fixed sigma estimate.

Imports from existing files unchanged — no modifications to
market_maker.py or replay_simulator.py required.

Key difference from base model:
    Fixed:    spread = A*(T-t) + B
    Realized: spread = gamma_rv * sigma2_t * (T-t) + B

gamma_rv is calibrated so that at average realized vol the
RV spread equals the fixed spread — both models are equivalent
on average, but RV widens during volatile periods and tightens
during calm periods.

sigma2_t is a rolling 10-minute realized variance computed
from TAQ trade prices, updated every second.

Flat-spread tickers (open == close, A=0) are automatically
bypassed — the RV model degenerates to baseline for these
and the extension adds no value.

Run:
    python realized_vol_strategy.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Project root (one level above scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(ROOT, 'source'))

from market_maker import AvellanedaStoikov, InventoryModel, BaselineStrategy
from replay_simulator import ReplaySimulator, SimulationResult
from wrds_loader import WRDSLoader, estimate_sigma
from helpers import summarise_results, summarise_order_stats, print_fill_analysis

# Approximate prices for dollar vol conversion
S0_PRICES = {
    'AAPL': 145.0,
    'AMZN': 967.0,
    'GE':   29.0,
    'IVV':  244.0,
    'M':    23.0,
}


# ---------------------------------------------------------------------------
# Realized vol strategy — extends AvellanedaStoikov
# ---------------------------------------------------------------------------

class RealizedVolStrategy(AvellanedaStoikov):
    """
    A-S model with realized volatility in spread equation.
    Inherits everything from AvellanedaStoikov.
    Only overrides spread and quote methods.

    For flat-spread tickers (A < 1e-6), returns B directly
    so behavior is identical to fixed vol model.
    """

    def optimal_spread_realized(self, t: float, sigma2_t: float) -> float:
        if not hasattr(self, 'A') or self.A < 1e-6:
            return self.B
        # sigma2_t is dollar variance (rv2 * price²)
        time_component = self.gamma * sigma2_t * (self.T - t)
        return time_component + self.B

    def quotes_realized_vol(self, s: float, q: float,
                             t: float, sigma2_t: float):
        """Quotes using realized volatility spread."""
        r    = self.indifference_price(s, q, t)
        half = self.optimal_spread_realized(t, sigma2_t) / 2.0
        return r - half, r + half


# ---------------------------------------------------------------------------
# Realized vol simulator — extends ReplaySimulator
# ---------------------------------------------------------------------------

class RealizedVolSimulator(ReplaySimulator):
    """
    Extends ReplaySimulator to support realized vol quoting.
    Adds 'optimal_rv' strategy type.
    All other strategy types fall through to parent unchanged.
    """

    def run(self, strategy, day_data: dict,
            inventory_model, strategy_type: str = 'optimal') -> SimulationResult:

        # Non-rv strategies: delegate to parent unchanged
        if strategy_type != 'optimal_rv':
            return super().run(strategy, day_data,
                               inventory_model, strategy_type)

        # Flat-spread tickers: RV adds no value, treat as regular optimal
        if hasattr(strategy, 'A') and strategy.A < 1e-6:
            return super().run(strategy, day_data,
                               inventory_model, strategy_type='optimal')

        # Extract realized vol array
        realized_vol2 = day_data.get(
            'realized_vol2',
            np.full(len(day_data['mid']), day_data['sigma'] ** 2)
        )

        # Intercept quotes_calibrated via monkey-patch
        # so the parent run() loop calls RV quotes automatically
        original_quotes = strategy.quotes_calibrated

        def rv_quotes(s, q, t_norm):
            step     = int(t_norm * self.T_seconds)
            step     = min(step, len(realized_vol2) - 1)
            sigma2_t = realized_vol2[step]
            return strategy.quotes_realized_vol(s, q, t_norm, sigma2_t)

        strategy.quotes_calibrated = rv_quotes
        result = super().run(strategy, day_data,
                             inventory_model, strategy_type='optimal')
        strategy.quotes_calibrated = original_quotes  # always restore

        return result


# ---------------------------------------------------------------------------
# Realized variance computation
# ---------------------------------------------------------------------------

def compute_realized_vol2(trades: pd.DataFrame,
                           sigma_fallback: float,
                           window: int = 600,
                           price: float = 1.0) -> np.ndarray:
    """
    Compute rolling realized dollar variance from TAQ trade prices.
    Returns array of length 23401 (one value per second).

    Returns dollar variance: rv_logreturns² * price²
    so the spread equation gamma * sigma2_t * (T-t) operates
    in dollar units consistent with the spread calibration.

    Uses 10-minute (600 second) rolling window by default.
    Falls back to sigma_fallback² before min_periods=30 trades
    are available. Capped at 3x weekly sigma to prevent open
    spike blowout.

    Parameters
    ----------
    trades        : market_trades DataFrame with t_sec and price
    sigma_fallback: weekly per-second sigma — used before window fills
    window        : rolling window in seconds (default 600 = 10 minutes)
    price         : approximate stock price for dollar variance conversion
    """
    T_SECONDS    = 23400
    price_series = pd.Series(index=np.arange(T_SECONDS + 1), dtype=float)

    for _, row in trades.iterrows():
        sec = int(row['t_sec'])
        if 0 <= sec <= T_SECONDS:
            price_series[sec] = row['price']

    price_series  = price_series.ffill().bfill()
    log_ret       = np.log(price_series).diff()
    realized_vol  = log_ret.rolling(window, min_periods=30).std()
    realized_vol  = realized_vol.fillna(sigma_fallback)
    realized_vol2 = (realized_vol ** 2).to_numpy(dtype=float)

    # Convert to dollar variance
    realized_vol2_dollar = realized_vol2 * price ** 2

    # Cap at 3x weekly sigma to prevent open spike blowout
    sigma_dollar = sigma_fallback * price
    cap          = 2.25 * sigma_dollar ** 2   # (1.5 * sigma_dollar)²
    realized_vol2_dollar = np.minimum(realized_vol2_dollar, cap)

    return realized_vol2_dollar


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_realized_vol_experiment(tickers=None, dates=None):
    """
    Three-way comparison:
        1. Baseline   — always quotes at NBBO
        2. Optimal    — fixed vol A-S (from run_with_wrds.py)
        3. Optimal RV — realized vol A-S

    gamma_rv calibration:
        gamma_rv = A / avg_rv2_dollar
        where avg_rv2_dollar = mean(rv2) * price²

        This ensures RV spread == fixed spread at average realized vol.
        During high-vol periods: RV spread > fixed (wider, more cautious)
        During low-vol periods:  RV spread < fixed (tighter, more fills)

        Flat-spread tickers (A=0): gamma_rv = gamma_fixed, RV bypassed.
    """
    from run_with_wrds import TICKERS, DATES, GAMMA_PARAMS, PHI_MAX, ETA

    tickers = tickers or TICKERS
    dates   = dates   or DATES

    print("=" * 65)
    print("  Realized Vol Extension — A-S with rolling sigma²_t")
    print("  Window: 10 minutes (600 seconds)")
    print("=" * 65)

    loader   = WRDSLoader()
    raw_data = loader.load_week(tickers=tickers, dates=dates)
    loader.close()

    sim       = RealizedVolSimulator(dt=1.0, T_seconds=23400.0,
                                     waiting_time=5.0, update_time=1.0)
    inv_model = InventoryModel(phi_max=PHI_MAX, eta=ETA)

    all_results    = {}
    all_results_rv = {}

    for ticker in tickers:
        valid_days = [raw_data[ticker][d] for d in dates
                      if raw_data[ticker][d] is not None]
        if not valid_days:
            print(f"  {ticker}: no valid data, skipping")
            continue

        # --- Empirical spread calibration ---
        open_spreads, close_spreads = [], []
        for d in dates:
            if raw_data[ticker][d] is not None:
                spreads = (raw_data[ticker][d]['best_ask'] -
                           raw_data[ticker][d]['best_bid'])
                open_spreads.append(float(np.nanmedian(spreads[:1800])))
                close_spreads.append(float(np.nanmedian(spreads[-1800:])))

        open_spread  = float(np.mean(open_spreads))
        close_spread = float(np.mean(close_spreads))
        sigma        = estimate_sigma({d: raw_data[ticker][d]
                                       for d in dates
                                       if raw_data[ticker][d]})

        # --- Compute realized vol across all days ---
        price          = S0_PRICES.get(ticker, 100.0)
        all_rv2 = []
        for d in dates:
            if raw_data[ticker][d] is not None:
                rv2 = compute_realized_vol2(
                    raw_data[ticker][d]['market_trades'], sigma, price=price
                )
                all_rv2.extend(rv2.tolist())

        # --- Calibrate gamma_rv ---
        avg_rv2_dollar = np.nanmean(all_rv2)

        # A tells us if spread is flat — compute before building models
        temp_A = max((open_spread - close_spread) / 1.0, 0.0)

        if temp_A < 1e-6:
            # Flat spread: gamma_rv doesn't matter, RV will be bypassed
            gamma_rv = GAMMA_PARAMS[ticker]
        else:
            # gamma_rv so RV spread = fixed spread at avg realized vol
            gamma_rv = temp_A / (avg_rv2_dollar + 1e-15)
            gamma_rv = float(np.clip(gamma_rv, 0.001, 100.0))

        print(f"  avg_rv2            : {np.nanmean(all_rv2):.2e}")
        print(f"  price              : {price}")
        print(f"  avg_rv2_dollar     : {avg_rv2_dollar:.2e}")
        print(f"  temp_A             : {temp_A:.6f}")
        print(f"  gamma_rv (raw)     : {temp_A / (avg_rv2_dollar + 1e-15):.4f}")
        print(f"  gamma_rv (clipped) : {gamma_rv:.4f}")
        print(f"  gamma_rv*avg_rv2   : {gamma_rv * np.nanmean(all_rv2):.2e}")
        print(f"  should equal A     : {temp_A:.6f}")
        print(f"  spread at open     : {gamma_rv * np.nanmean(all_rv2) * 1.0 + close_spread:.6f}")
        print(f"  spread at midday   : {gamma_rv * np.nanmean(all_rv2) * 0.5 + close_spread:.6f}")

        print(f"\n  {ticker}: open={open_spread:.4f} close={close_spread:.4f} "
              f"A={temp_A:.6f} gamma_rv={gamma_rv:.4f} "
              f"{'[FLAT — RV bypassed]' if temp_A < 1e-6 else ''}")

        # --- Build models ---
        as_fixed = AvellanedaStoikov(
            gamma=GAMMA_PARAMS[ticker], sigma=sigma, kappa=2.0, T=1.0
        )
        as_fixed.calibrate_from_market(open_spread, close_spread)
        as_fixed.gamma_sigma2 = as_fixed.gamma_sigma2 / 23400.0

        as_rv = RealizedVolStrategy(
            gamma=gamma_rv, sigma=sigma, kappa=2.0, T=1.0
        )
        as_rv.calibrate_from_market(open_spread, close_spread)
        as_rv.gamma_sigma2 = as_rv.gamma_sigma2 / 23400.0

        baseline = BaselineStrategy()

        opt_results  = []
        rv_results   = []
        base_results = []

        for date in dates:
            day = raw_data[ticker][date]
            if day is None:
                continue

            # Attach realized vol to day_data for this day
            day['realized_vol2'] = compute_realized_vol2(
                day['market_trades'], sigma, price=price
            )

            print(f"    {ticker} {date}...", end=" ", flush=True)

            res_opt  = sim.run(as_fixed, day, inv_model,
                               strategy_type='optimal')
            res_rv   = sim.run(as_rv,    day, inv_model,
                               strategy_type='optimal_rv')
            res_base = sim.run(baseline, day, inv_model,
                               strategy_type='baseline')

            opt_results.append(res_opt)
            rv_results.append(res_rv)
            base_results.append(res_base)

            print(f"fixed={res_opt.pnl[-1]:>8,.0f}  "
                  f"rv={res_rv.pnl[-1]:>8,.0f}  "
                  f"base={res_base.pnl[-1]:>8,.0f}")

        all_results[ticker] = {
            'optimal':  opt_results,
            'baseline': base_results,
            'as_model': as_fixed,
        }
        all_results_rv[ticker] = {
            'optimal':  rv_results,
            'baseline': base_results,
            'as_model': as_rv,
        }

    # --- Summary table ---
    print("\n" + "=" * 65)
    print("  Fixed Vol vs Realized Vol vs Baseline")
    print("=" * 65)

    print(f"\n{'Ticker':<6} {'Fixed P&L':>12} {'RV P&L':>12} "
          f"{'Baseline P&L':>14} {'RV vs Fixed':>12}")
    print("-" * 58)

    for ticker in tickers:
        if ticker not in all_results:
            continue
        fixed_pnl = np.mean([d.pnl[-1] for d in all_results[ticker]['optimal']])
        rv_pnl    = np.mean([d.pnl[-1] for d in all_results_rv[ticker]['optimal']])
        base_pnl  = np.mean([d.pnl[-1] for d in all_results[ticker]['baseline']])
        diff      = rv_pnl - fixed_pnl
        print(f"{ticker:<6} {fixed_pnl:>12,.0f} {rv_pnl:>12,.0f} "
              f"{base_pnl:>14,.0f} {diff:>+12,.0f}")

    # --- Plots ---
    print("\nGenerating plots...")
    try:
        plot_realized_vol_comparison(all_results, all_results_rv,
                                     subfolder='wrds')
    except Exception as e:
        print(f"Plotting skipped: {e}")

    return all_results, all_results_rv

def plot_realized_vol_comparison(all_results: dict,
                                  all_results_rv: dict,
                                  subfolder: str = 'wrds'):
    """
    Plots for realized vol vs fixed vol vs baseline comparison.

    Figure 1: Cumulative P&L — three-way comparison per ticker
    Figure 2: Intraday realized vol vs fixed spread per ticker
    Figure 3: Spread comparison — fixed vs RV over trading day
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BASE_PLOTS_DIR = os.path.join(ROOT, 'plots', subfolder)
    os.makedirs(BASE_PLOTS_DIR, exist_ok=True)

    COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']
    DAY_LABELS = ['06/12', '06/13', '06/14', '06/15', '06/16']
    T_SECONDS  = 23400.0

    tickers = list(all_results.keys())

    # -----------------------------------------------------------------------
    # Figure 1: Three-way cumulative P&L per ticker
    # -----------------------------------------------------------------------
    for ticker in tickers:
        opt_days  = all_results[ticker]['optimal']
        rv_days   = all_results_rv[ticker]['optimal']
        base_days = all_results[ticker]['baseline']

        n_days = len(opt_days)
        fig, axes = plt.subplots(1, n_days, figsize=(4 * n_days, 4),
                                  sharey=False)
        if n_days == 1:
            axes = [axes]

        for i, (res_o, res_rv, res_b) in enumerate(
                zip(opt_days, rv_days, base_days)):
            ax      = axes[i]
            t_hours = res_o.times / 3600 + 9.5

            ax.plot(t_hours, res_b.pnl,  color='steelblue', lw=0.8,
                    alpha=0.7, label='Baseline')
            ax.plot(t_hours, res_o.pnl,  color='#E91E63',   lw=1.0,
                    label='Fixed vol')
            ax.plot(t_hours, res_rv.pnl, color='#4CAF50',   lw=1.0,
                    linestyle='--', label='Realized vol')

            ax.set_title(DAY_LABELS[i], fontsize=9)
            ax.set_xlim(9.5, 16.0)
            ax.set_xticks([9.5, 12.0, 14.0, 16.0])
            ax.set_xticklabels(['9:30', '12:00', '2:00', '4:00'],
                                fontsize=7, rotation=20)
            ax.set_ylabel('P&L ($)' if i == 0 else '')
            if i == 0:
                ax.legend(fontsize=7)

        fig.suptitle(f'{ticker}: Cumulative P&L — Fixed vs Realized Vol vs Baseline',
                     fontsize=10)
        plt.tight_layout()
        path = os.path.join(BASE_PLOTS_DIR, f'rv_pnl_{ticker}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # -----------------------------------------------------------------------
    # Figure 2: P&L summary bar chart — avg across days
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    x       = np.arange(len(tickers))
    width   = 0.25

    fixed_avgs = [np.mean([d.pnl[-1] for d in all_results[t]['optimal']])
                  for t in tickers]
    rv_avgs    = [np.mean([d.pnl[-1] for d in all_results_rv[t]['optimal']])
                  for t in tickers]
    base_avgs  = [np.mean([d.pnl[-1] for d in all_results[t]['baseline']])
                  for t in tickers]

    ax.bar(x - width, base_avgs,  width, label='Baseline',     color='steelblue', alpha=0.8)
    ax.bar(x,         fixed_avgs, width, label='Fixed vol',    color='#E91E63',   alpha=0.8)
    ax.bar(x + width, rv_avgs,    width, label='Realized vol', color='#4CAF50',   alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel('Average P&L ($)')
    ax.set_title('Average Daily P&L: Fixed Vol vs Realized Vol vs Baseline')
    ax.legend(fontsize=9)
    ax.axhline(0, color='black', lw=0.5, linestyle=':')

    plt.tight_layout()
    path = os.path.join(BASE_PLOTS_DIR, 'rv_pnl_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # -----------------------------------------------------------------------
    # Figure 3: Intraday spread comparison — fixed vs RV (Day 1 only)
    # -----------------------------------------------------------------------
    for ticker in tickers:
        opt_days = all_results[ticker]['optimal']
        rv_days  = all_results_rv[ticker]['optimal']

        if not opt_days:
            continue

        res_o  = opt_days[0]
        res_rv = rv_days[0]

        t_hours     = res_o.times / 3600 + 9.5
        fixed_spread = res_o.ask_prices  - res_o.bid_prices
        rv_spread    = res_rv.ask_prices - res_rv.bid_prices

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t_hours, fixed_spread, color='#E91E63', lw=0.8,
                alpha=0.7, label='Fixed vol spread')
        ax.plot(t_hours, rv_spread,   color='#4CAF50', lw=0.8,
                alpha=0.7, linestyle='--', label='Realized vol spread')

        ax.set_ylabel('Spread ($)')
        ax.set_xlim(9.5, 16.0)
        ax.set_xticks([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.0])
        ax.set_xticklabels(['9:30 AM', '10:30 AM', '11:30 AM', '12:30 PM',
                             '1:30 PM', '2:30 PM', '3:30 PM', '4:00 PM'],
                            rotation=30, ha='right', fontsize=8)
        ax.set_xlabel('time')
        ax.legend(fontsize=9)
        ax.set_title(f'{ticker}: Intraday Spread — Fixed vs Realized Vol (06/12)')

        plt.tight_layout()
        path = os.path.join(BASE_PLOTS_DIR, f'rv_spread_{ticker}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    # -----------------------------------------------------------------------
    # Figure 4: Inventory density — fixed vs RV
    # -----------------------------------------------------------------------
    for ticker in tickers:
        opt_days = all_results[ticker]['optimal']
        rv_days  = all_results_rv[ticker]['optimal']

        if not opt_days:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for i, (res_o, res_rv) in enumerate(zip(opt_days, rv_days)):
            c     = COLORS[i % len(COLORS)]
            label = DAY_LABELS[i] if i < len(DAY_LABELS) else f'Day {i+1}'
            bins  = np.linspace(-800, 800, 80)

            ax1.hist(res_o.inventory,  bins=bins, density=True,
                     histtype='step', color=c, lw=1.2, label=label)
            ax2.hist(res_rv.inventory, bins=bins, density=True,
                     histtype='step', color=c, lw=1.2, label=label)

        ax1.set_title('Fixed vol — Inventory Density', fontsize=9)
        ax2.set_title('Realized vol — Inventory Density', fontsize=9)
        for ax in [ax1, ax2]:
            ax.set_xlabel('position')
            ax.set_ylabel('density')
            ax.legend(fontsize=7)

        fig.suptitle(f'{ticker}: Inventory Density Comparison', fontsize=10)
        plt.tight_layout()
        path = os.path.join(BASE_PLOTS_DIR, f'rv_inventory_{ticker}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    print(f"\nAll realized vol plots saved to: {BASE_PLOTS_DIR}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_realized_vol_experiment()