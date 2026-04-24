"""
analysis.py
===========
Generates plots replicating and extending the paper's figures.

Figures produced:
  - Figure 1: Dynamic order size function
  - Figure 2: Intraday volume profile (bathtub shape)
  - Figure 3: Market vs Optimal spread over the trading day
  - Figure 4: Cumulative P&L and inventory density per stock
  - Figure 5: Fill model comparison — spread capture rate across regimes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_maker import AvellanedaStoikov, InventoryModel


T_SECONDS = 23400.0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']
FILL_MODEL_COLORS = {
    'first_in_queue':     '#2196F3',
    'pro_rata':           '#E91E63',
    'pro_rata_imbalance': '#4CAF50',
}


def bathtub_alpha(t: float, T: float = T_SECONDS) -> float:
    """Piece-wise linear intraday volume profile (bathtub shape)."""
    x = t / T
    if x <= 0.35:
        return 0.08 - (0.08 - 0.03) * (x / 0.35)
    elif x <= 0.65:
        return 0.03
    else:
        return 0.03 + (0.08 - 0.03) * ((x - 0.65) / 0.35)


# ---------------------------------------------------------------------------
# Shared axis helper — converts seconds-since-open to clock time tick labels
# ---------------------------------------------------------------------------

def _apply_time_axis(ax, x_hours: np.ndarray = None):
    """
    Apply proper time-of-day formatting to a plot whose x values are decimal
    hours since midnight (e.g. 9.5 = 9:30am, 13.0 = 1:00pm).

    Sets x-limits to the full trading session and formats ticks as HH:MM.
    """
    ticks      = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.0]
    tick_labels = ['9:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30', '16:00']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha='right')
    ax.set_xlim(9.5, 16.0)
    ax.set_xlabel('Time of day')


# ---------------------------------------------------------------------------
# Figure 1: Dynamic order size function
# ---------------------------------------------------------------------------

def plot_order_size_function(phi_max=100, eta=0.005, save=True):
    """Figure 1: phi_bid and phi_ask vs inventory position."""
    q_range = np.linspace(-600, 600, 500)
    inv     = InventoryModel(phi_max=phi_max, eta=eta)

    phi_bid = np.array([inv.bid_size(q) for q in q_range])
    phi_ask = np.array([inv.ask_size(q) for q in q_range])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q_range, phi_bid, color='red',  lw=2, label='Bid size (buy orders)')
    ax.plot(q_range, phi_ask, color='blue', lw=2, linestyle='--',
            label='Ask size (sell orders)')
    ax.axvline(0, color='gray', lw=0.8, linestyle=':')
    ax.axhline(phi_max, color='gray', lw=0.8, linestyle=':')
    ax.annotate('Long inventory:\nreduce bid size',
                xy=(300, inv.bid_size(300)), xytext=(350, 60),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='red')
    ax.annotate('Short inventory:\nreduce ask size',
                xy=(-300, inv.ask_size(-300)), xytext=(-580, 60),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='blue')
    ax.set_xlabel('Inventory position (shares)')
    ax.set_ylabel('Order size (shares)')
    ax.set_title('Figure 1: Dynamic Order Size Function')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-600, 600)
    ax.set_ylim(0, phi_max * 1.1)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'fig1_order_size.png')
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2: Intraday volume profile
# ---------------------------------------------------------------------------

def plot_intraday_profile(save=True):
    """Figure 2: bathtub-shaped intraday volume profile."""
    T     = 23400.0
    t_arr = np.linspace(0, T, 500)
    alphas = [bathtub_alpha(t, T) for t in t_arr]
    hours  = t_arr / 3600 + 9.5

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hours, alphas, color='red', lw=2)
    ax.axvline(9.5 + 2.0, color='gray', lw=0.8, linestyle='--')
    ax.axvline(9.5 + 4.5, color='gray', lw=0.8, linestyle='--')
    ax.text(10.2, 0.065, 'Open',   fontsize=8, color='gray')
    ax.text(12.2, 0.025, 'Midday', fontsize=8, color='gray')
    ax.text(14.5, 0.055, 'Close',  fontsize=8, color='gray')
    ax.set_ylabel('Fill probability at best price')
    ax.set_title('Figure 2: Intraday Volume Profile ($\\alpha_t$)')
    _apply_time_axis(ax)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'fig2_intraday_profile.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 3: Market vs Optimal spread
# ---------------------------------------------------------------------------

def plot_spreads(ticker: str, as_model: AvellanedaStoikov,
                 optimal_results: list, baseline_results: list,
                 day_idx: int = 0, save=True):
    """Figure 3: market spread vs optimal spread for one day."""
    if not optimal_results or not baseline_results:
        print(f"  No results to plot for {ticker}")
        return

    res_opt  = optimal_results[day_idx]
    res_base = baseline_results[day_idx]

    t_hours        = res_opt.times / 3600 + 9.5
    market_spread  = res_base.ask_prices - res_base.bid_prices
    optimal_spread = res_opt.ask_prices  - res_opt.bid_prices

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_hours, market_spread,  color='steelblue', lw=0.8,
            alpha=0.6, label='Market Spread')
    ax.plot(t_hours, optimal_spread, color='red', lw=1.5,
            label='Optimal Spread')
    ax.set_ylabel('Spread ($)')
    ax.set_title(f'{ticker}: Market vs Optimal Spread')
    ax.legend(fontsize=9)
    _apply_time_axis(ax)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f'fig3_spreads_{ticker}.png')
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4: Cumulative P&L and inventory density
# ---------------------------------------------------------------------------

def plot_pnl_and_inventory(ticker: str,
                            optimal_results: list,
                            baseline_results: list,
                            dates: list = None,
                            fill_model: str = 'first_in_queue',
                            save=True):
    """
    Figure 4: 2x2 grid of cumulative P&L and inventory density
    for optimal and baseline strategies across all simulated days.

    Parameters
    ----------
    dates      : list of date strings for legend labels (e.g. ['2017-06-12', ...])
                 If None, labels are Day 1, Day 2 etc.
    fill_model : which fill model these results used (shown in title)
    """
    if not optimal_results and not baseline_results:
        print(f"  No results to plot for {ticker}")
        return

    # Build day labels from dates if provided, else generic
    if dates is not None:
        day_labels = [d[5:] for d in dates]   # '2017-06-12' → '06-12'
    else:
        n = max(len(optimal_results), len(baseline_results))
        day_labels = [f'Day {i+1}' for i in range(n)]

    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax_pnl_opt  = fig.add_subplot(gs[0, 0])
    ax_pnl_base = fig.add_subplot(gs[0, 1])
    ax_inv_opt  = fig.add_subplot(gs[1, 0])
    ax_inv_base = fig.add_subplot(gs[1, 1])

    for i, (res_o, res_b) in enumerate(zip(optimal_results, baseline_results)):
        t_hours = res_o.times / 3600 + 9.5
        label   = day_labels[i] if i < len(day_labels) else f'Day {i+1}'
        c       = COLORS[i % len(COLORS)]

        ax_pnl_opt.plot(t_hours,  res_o.pnl, color=c, lw=0.8, label=label)
        ax_pnl_base.plot(t_hours, res_b.pnl, color=c, lw=0.8, label=label)

        bins = np.linspace(-800, 800, 80)
        ax_inv_opt.hist(res_o.inventory,  bins=bins, density=True,
                         histtype='step', color=c, lw=1.2, label=label)
        ax_inv_base.hist(res_b.inventory, bins=bins, density=True,
                          histtype='step', color=c, lw=1.2, label=label)

    for ax, title in [
        (ax_pnl_opt,  '(a) Cumulative P&L — Optimal'),
        (ax_pnl_base, '(b) Cumulative P&L — Baseline'),
        (ax_inv_opt,  '(c) Inventory Density — Optimal'),
        (ax_inv_base, '(d) Inventory Density — Baseline'),
    ]:
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc='best')

    for ax in [ax_pnl_opt, ax_pnl_base]:
        ax.set_ylabel('P&L ($)')
        _apply_time_axis(ax)

    for ax in [ax_inv_opt, ax_inv_base]:
        ax.set_xlabel('Inventory position (shares)')
        ax.set_ylabel('Density')

    fig.suptitle(f'{ticker} — P&L and Inventory  [{fill_model}]', fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR,
                            f'fig4_pnl_inventory_{ticker}_{fill_model}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 5: Fill model comparison across regimes
# ---------------------------------------------------------------------------

def plot_fill_model_comparison(all_results: dict,
                                test_dates: list,
                                save=True):
    """
    Figure 5: Bar chart comparing spread capture rate across fill models
    and test dates, for each ticker and strategy.

    This figure has no equivalent in the original paper — it is the
    main new contribution of the fill model extension.
    """
    from helpers import fill_analysis

    fill_models = ['first_in_queue', 'pro_rata', 'pro_rata_imbalance']
    tickers     = list(all_results.keys())
    n_tickers   = len(tickers)

    fig, axes = plt.subplots(1, n_tickers, figsize=(4 * n_tickers, 5),
                              sharey=True)
    if n_tickers == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        res = all_results[ticker]
        x      = np.arange(len(fill_models))
        width  = 0.35
        opt_rates  = []
        base_rates = []

        for fm in fill_models:
            opt_days  = res.get(f'optimal_{fm}',  [])
            base_days = res.get(f'baseline_{fm}', [])
            opt_rates.append(
                fill_analysis(opt_days)['spread_capture_rate'] * 100
                if opt_days else 0.0
            )
            base_rates.append(
                fill_analysis(base_days)['spread_capture_rate'] * 100
                if base_days else 0.0
            )

        ax.bar(x - width/2, opt_rates,  width, label='Optimal',
               color='#2196F3', alpha=0.85)
        ax.bar(x + width/2, base_rates, width, label='Baseline',
               color='#E91E63',  alpha=0.85)

        ax.set_title(ticker, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['First\nIn Queue', 'Pro\nRata', 'Pro Rata\n+OBI'],
                            fontsize=8)
        ax.legend(fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel('Spread capture rate (%)')

    fig.suptitle('Figure 5: Spread Capture Rate by Fill Model', fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'fig5_fill_model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Convenience: plot everything
# ---------------------------------------------------------------------------

def plot_all(all_results: dict, test_dates: list = None, save=True):
    """
    Generate all figures.

    all_results is keyed as:
        all_results[ticker]['optimal_first_in_queue']   → list of SimulationResult
        all_results[ticker]['baseline_first_in_queue']  → list of SimulationResult
        all_results[ticker]['optimal_pro_rata']         → ...
        etc.
        all_results[ticker]['as_model']                 → AvellanedaStoikov instance
    """
    plot_order_size_function(save=save)
    plot_intraday_profile(save=save)

    fill_models = ['first_in_queue', 'pro_rata', 'pro_rata_imbalance']

    for ticker, res in all_results.items():
        print(f"\nPlotting {ticker}...")

        # Figure 3: spread plot — use first_in_queue results
        opt_fiq  = res.get('optimal_first_in_queue',  [])
        base_fiq = res.get('baseline_first_in_queue', [])
        if opt_fiq and base_fiq:
            plot_spreads(ticker, res['as_model'],
                         opt_fiq, base_fiq, save=save)

        # Figure 4: one P&L/inventory plot per fill model
        for fm in fill_models:
            opt_days  = res.get(f'optimal_{fm}',  [])
            base_days = res.get(f'baseline_{fm}', [])
            if opt_days or base_days:
                plot_pnl_and_inventory(
                    ticker, opt_days, base_days,
                    dates=test_dates, fill_model=fm, save=save
                )

    # Figure 5: fill model comparison
    if test_dates:
        plot_fill_model_comparison(all_results, test_dates, save=save)


if __name__ == '__main__':
    plot_order_size_function()
    plot_intraday_profile()
