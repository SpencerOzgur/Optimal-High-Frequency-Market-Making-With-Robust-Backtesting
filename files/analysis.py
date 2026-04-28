"""
analysis.py
===========
Generates plots replicating the paper's figures:
  - Figure 3: Market vs Optimal spread over the trading day
  - Figure 4/5: Cumulative P&L and inventory density (optimal vs baseline)
  - Figure 1 equivalent: Dynamic order size function
  - Figure 2 equivalent: Intensity function components

Output routing:
  - Called from poisson_simulator.py  -> saved to plots/synthetic/
  - Called from run_with_wrds.py      -> saved to plots/wrds/

Usage:
  from analysis import plot_all
  plot_all(results, subfolder='synthetic')   # from poisson_simulator.py
  plot_all(results, subfolder='wrds')        # from run_with_wrds.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_maker import AvellanedaStoikov, InventoryModel


T_SECONDS = 23400.0

# Shared AM/PM tick positions and labels for intraday plots
TIME_TICKS  = [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.0]
TIME_LABELS = ['9:30 AM', '10:30 AM', '11:30 AM', '12:30 PM',
               '1:30 PM',  '2:30 PM',  '3:30 PM',  '4:00 PM']

# Sparse version for small subplots (P&L panels in 2x2 grid)
TIME_TICKS_SPARSE  = [9.5, 11.5, 13.5, 16.0]
TIME_LABELS_SPARSE = ['9:30 AM', '11:30 AM', '1:30 PM', '4:00 PM']

# Consistent color palette
COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']
DAY_LABELS = ['06/12', '06/13', '06/14', '06/15', '06/16']

# Base plots directory — subfolders created per run type
BASE_PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'plots')


def _get_results_dir(subfolder: str) -> str:
    """
    Return and create the output directory for a given subfolder.
    subfolder: 'synthetic' or 'wrds'
    """
    path = os.path.join(BASE_PLOTS_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    return path


def _apply_time_axis(ax, sparse=False, rotation=30):
    """Helper: apply AM/PM ticks to an intraday x-axis."""
    ticks  = TIME_TICKS_SPARSE  if sparse else TIME_TICKS
    labels = TIME_LABELS_SPARSE if sparse else TIME_LABELS
    ax.set_xlabel('time')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=rotation, ha='right', fontsize=8)


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
# Figure 1: Dynamic order size function
# ---------------------------------------------------------------------------

def plot_order_size_function(phi_max=100, eta=0.005,
                              save=True, subfolder='synthetic'):
    """Replicate Figure 1: phi_bid and phi_ask vs inventory position."""
    results_dir = _get_results_dir(subfolder)

    q_range = np.linspace(-600, 600, 500)
    inv     = InventoryModel(phi_max=phi_max, eta=eta)

    phi_bid = np.array([inv.bid_size(q) for q in q_range])
    phi_ask = np.array([inv.ask_size(q) for q in q_range])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q_range, phi_bid, color='red',  lw=2, label='Buy at function')
    ax.plot(q_range, phi_ask, color='blue', lw=2, linestyle='--',
            label='Sell at function')
    ax.axvline(0,       color='gray', lw=0.8, linestyle=':')
    ax.axhline(phi_max, color='gray', lw=0.8, linestyle=':')
    ax.annotate('Sell 100 shares\n Buy at function',
                xy=(300, inv.bid_size(300)), xytext=(350, 60),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='red')
    ax.annotate('Buy 100 shares\n Sell at function',
                xy=(-300, inv.ask_size(-300)), xytext=(-550, 60),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='blue')
    ax.set_xlabel('Position')
    ax.set_ylabel('Order Size')
    ax.set_title('Figure 1: Dynamic Order Size Function')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-600, 600)
    ax.set_ylim(0, phi_max * 1.1)
    plt.tight_layout()
    if save:
        path = os.path.join(results_dir, 'fig1_order_size.png')
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 2: Intensity function components
# ---------------------------------------------------------------------------

def plot_intensity_components(save=True, subfolder='synthetic'):
    """Replicate Figure 2: time component alpha_t and depth component exp(-mu*xi)."""
    results_dir = _get_results_dir(subfolder)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Left: time component
    T     = 23400.0
    t_arr = np.linspace(0, T, 500)
    alphas = [bathtub_alpha(t, T) for t in t_arr]
    hours  = t_arr / 3600 + 9.5

    ax1.plot(hours, alphas, color='red', lw=2)
    ax1.axvline(9.5 + 2.0, color='gray', lw=0.8, linestyle='--')
    ax1.axvline(9.5 + 4.5, color='gray', lw=0.8, linestyle='--')
    ax1.set_ylabel('Fill probability at best price')
    ax1.set_xticks([9.5, 11.5, 14.0, 16.0])
    ax1.set_xticklabels(['9:30 AM', '11:30 AM', '2:00 PM', '4:00 PM'],
                         rotation=20, ha='right', fontsize=8)
    ax1.set_xlabel('time')
    ax1.text(10.2, 0.065, 'Beginning', fontsize=8, color='gray')
    ax1.text(12.2, 0.025, 'Middle',    fontsize=8, color='gray')
    ax1.text(14.5, 0.055, 'End',       fontsize=8, color='gray')
    ax1.set_title('(a) Time component $\\alpha_t$')

    # Right: depth component
    xi_arr    = np.linspace(-0.10, 0.20, 500)
    mu        = 100.0
    intensity = np.exp(-mu * xi_arr)

    ax2.plot(xi_arr, np.clip(intensity, 0, 1.5), color='red', lw=2)
    ax2.axvline(0,     color='gray', lw=0.8, linestyle='--')
    ax2.axvline(-0.05, color='gray', lw=0.8, linestyle='--')
    ax2.axvline(0.05,  color='gray', lw=0.8, linestyle='--')
    ax2.set_xlabel('Price')
    ax2.set_ylabel('Intensity')
    ax2.set_xticks([-0.08, -0.05, 0, 0.05, 0.12])
    ax2.set_xticklabels(['Bid price', 'Best bid', 'Best ask', 'Ask price', ''],
                         fontsize=7.5)
    ax2.text(-0.09, 1.2, 'Buy side\n($\\xi < 0$)',  fontsize=7.5, color='gray')
    ax2.text(0.001, 1.3, 'Inside\nmarket',           fontsize=7.5, color='gray')
    ax2.text(0.07,  1.2, 'Sell side\n($\\xi > 0$)', fontsize=7.5, color='gray')
    ax2.set_title('(b) Depth component $e^{-\\mu\\xi}$')
    ax2.set_xlim(-0.12, 0.20)

    plt.suptitle('Figure 2: Intensity Function Components', y=1.01)
    plt.tight_layout()
    if save:
        path = os.path.join(results_dir, 'fig2_intensity.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 3: Market vs Optimal spread
# ---------------------------------------------------------------------------

def plot_spreads(ticker: str, as_model: AvellanedaStoikov,
                 optimal_results: list, baseline_results: list,
                 day_idx: int = 0, save=True, subfolder='synthetic'):
    """Replicate Figure 3: market spread vs optimal spread for one day."""
    results_dir = _get_results_dir(subfolder)

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
    ax.set_ylabel('dollar')
    ax.set_title(f'{ticker}: Market vs Optimal Spread')
    ax.legend(fontsize=9)
    ax.set_xlim(9.5, 16.0)
    _apply_time_axis(ax, sparse=False, rotation=30)
    plt.tight_layout()
    if save:
        path = os.path.join(results_dir, f'fig3_spreads_{ticker}.png')
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Figure 4/5: Cumulative P&L and inventory density
# ---------------------------------------------------------------------------

def plot_pnl_and_inventory(ticker: str,
                            optimal_results: list,
                            baseline_results: list,
                            save=True, subfolder='synthetic'):
    """
    Replicate Figures 4 & 5: 2x2 grid of cumulative P&L and inventory density
    for optimal and baseline strategies across all simulated days.
    """
    results_dir = _get_results_dir(subfolder)

    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax_pnl_opt  = fig.add_subplot(gs[0, 0])
    ax_pnl_base = fig.add_subplot(gs[0, 1])
    ax_inv_opt  = fig.add_subplot(gs[1, 0])
    ax_inv_base = fig.add_subplot(gs[1, 1])

    for i, (res_o, res_b) in enumerate(zip(optimal_results, baseline_results)):
        t_hours = res_o.times / 3600 + 9.5
        label   = DAY_LABELS[i] if i < len(DAY_LABELS) else f'Day {i+1}'
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
        ax.set_ylabel('dollar')
        ax.set_xlim(9.5, 16.0)
        _apply_time_axis(ax, sparse=True, rotation=20)

    for ax in [ax_inv_opt, ax_inv_base]:
        ax.set_xlabel('position')
        ax.set_ylabel('density')

    fig.suptitle(f'Figure 4: {ticker} — P&L and Inventory', fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(results_dir, f'fig4_pnl_inventory_{ticker}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Convenience: plot everything for all stocks
# ---------------------------------------------------------------------------

def plot_quote_dynamics(ticker: str,
                         as_model,
                         optimal_results: list,
                         day_idx: int = 0,
                         save: bool = True,
                         subfolder: str = 'wrds'):
    """
    Replicate Stanford paper Figure 3: mid-price, indifference price,
    bid quotes and ask quotes over the trading day.

    Main panel: full day with all four series
    Inset panel: zoomed first 30 minutes to show bid/ask spread separation

    Mid-price    — black line
    Indifference — green line
    Ask quotes   — red filled dots
    Bid quotes   — red open circles
    """
    results_dir = _get_results_dir(subfolder)
    res         = optimal_results[day_idx]

    T_SECONDS = 23400.0
    n         = len(res.times)
    t_norm    = res.times / T_SECONDS

    mid = res.mid_prices

    # Compute indifference price at each second
    r = np.array([
        as_model.indifference_price(mid[i], res.inventory[i], t_norm[i])
        for i in range(n)
    ])

    # Per-side active masks
    bid_active = ~np.isnan(res.bid_prices)
    ask_active = ~np.isnan(res.ask_prices)

    # Downsample for main panel — plot every 10th point to reduce density
    # while preserving the visual shape of the price path
    step       = 10
    t_ds       = t_norm[::step]
    mid_ds     = mid[::step]
    r_ds       = r[::step]

    fig, ax = plt.subplots(figsize=(13, 6))

    # --- Main panel ---

    # Mid-price — black line
    ax.plot(t_ds, mid_ds, color='black', lw=1.0,
            label='Mid-market price', zorder=4)

    # Indifference price — green line
    ax.plot(t_ds, r_ds, color='green', lw=1.0,
            label='Indifference Price', zorder=3, alpha=0.85)

    # Ask quotes — red filled dots (subsample for visibility)
    ask_idx = np.where(ask_active)[0][::5]
    ax.scatter(t_norm[ask_idx], res.ask_prices[ask_idx],
               color='red', s=6, marker='o',
               label='Price asked', zorder=2, alpha=0.5)

    # Bid quotes — red open circles (subsample for visibility)
    bid_idx = np.where(bid_active)[0][::5]
    ax.scatter(t_norm[bid_idx], res.bid_prices[bid_idx],
               color='red', s=6, marker='o',
               facecolors='none', edgecolors='red', linewidths=0.6,
               label='Price bid', zorder=2, alpha=0.5)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Stock Price', fontsize=11)
    ax.set_title(f'{ticker}: Mid-price, Indifference Price and Quotes',
                 fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc='upper right')

    # --- Inset: zoom first 30 minutes ---
    # Shows bid/ask spread separation and indifference price deviation
    zoom_end   = int(0.077 * n)   # first 30 min = 1800s / 23400s ≈ 0.077
    ax_inset   = fig.add_axes([0.13, 0.54, 0.22, 0.32])

    ax_inset.plot(t_norm[:zoom_end], mid[:zoom_end],
                  color='black', lw=1.0)
    ax_inset.plot(t_norm[:zoom_end], r[:zoom_end],
                  color='green', lw=1.0, alpha=0.85)

    ask_zoom = ask_active[:zoom_end]
    bid_zoom = bid_active[:zoom_end]

    ax_inset.scatter(t_norm[:zoom_end][ask_zoom],
                     res.ask_prices[:zoom_end][ask_zoom],
                     color='red', s=8, marker='o', alpha=0.7)
    ax_inset.scatter(t_norm[:zoom_end][bid_zoom],
                     res.bid_prices[:zoom_end][bid_zoom],
                     color='red', s=8, marker='o',
                     facecolors='none', edgecolors='red',
                     linewidths=0.7, alpha=0.7)

    ax_inset.set_title('Open (9:30–10:00)', fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.set_xlim(0, t_norm[zoom_end])

    # Draw rectangle on main panel showing inset region
    from matplotlib.patches import Rectangle
    y_min = min(np.nanmin(res.bid_prices[:zoom_end]),
                mid[:zoom_end].min()) - 0.05
    y_max = max(np.nanmax(res.ask_prices[:zoom_end]),
                mid[:zoom_end].max()) + 0.05
    rect = Rectangle((0, y_min), t_norm[zoom_end], y_max - y_min,
                      linewidth=0.8, edgecolor='gray',
                      facecolor='lightyellow', alpha=0.3, zorder=0)
    ax.add_patch(rect)

    plt.tight_layout()
    if save:
        path = os.path.join(results_dir, f'fig_quote_dynamics_{ticker}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

def plot_quoted_vs_actual_spread(ticker: str,
                                  optimal_results: list,
                                  baseline_results: list,
                                  day_idx: int = 0,
                                  save: bool = True,
                                  subfolder: str = 'wrds'):
    """
    Plot quoted spread (our bid/ask) vs actual market spread (NBBO)
    over the trading day for both optimal and baseline strategies.

    Shows:
    - Market spread (NBBO best_ask - best_bid) — blue
    - Optimal quoted spread — red
    - Baseline quoted spread — green dashed
    """
    results_dir = _get_results_dir(subfolder)
    res_opt     = optimal_results[day_idx]
    res_base    = baseline_results[day_idx]

    T_SECONDS = 23400.0
    t_hours   = res_opt.times / 3600 + 9.5

    # Quoted spreads — our bid/ask
    opt_spread  = res_opt.ask_prices  - res_opt.bid_prices
    base_spread = res_base.ask_prices - res_base.bid_prices

    # Market spread — from baseline quotes which track NBBO exactly
    # Baseline always quotes at best_bid/best_ask so its spread = market spread
    market_spread = base_spread.copy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              sharex=True, gridspec_kw={'hspace': 0.35})

    # --- Top panel: raw spreads ---
    ax = axes[0]

    ax.plot(t_hours, market_spread,  color='steelblue', lw=0.6,
            alpha=0.7, label='Market spread (NBBO)')
    ax.plot(t_hours, opt_spread,     color='red',       lw=0.8,
            alpha=0.8, label='Optimal quoted spread')
    ax.plot(t_hours, base_spread,    color='green',     lw=0.6,
            alpha=0.6, linestyle='--', label='Baseline quoted spread')

    ax.set_ylabel('Spread ($)', fontsize=10)
    ax.set_title(f'{ticker}: Quoted vs Market Spread', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(9.5, 16.0)
    _apply_time_axis(ax, sparse=False, rotation=30)

    # Add horizontal line at close_spread (B = minimum spread)
    if hasattr(optimal_results[day_idx], 'as_model'):
        pass  # as_model not in SimulationResult — skip

    # --- Bottom panel: difference (quoted - market) ---
    ax2    = axes[1]
    diff   = opt_spread - market_spread

    # Smooth with rolling mean for readability
    diff_series = pd.Series(diff).rolling(300, min_periods=1).mean()

    ax2.plot(t_hours, diff_series, color='purple', lw=1.0,
             label='Optimal spread - Market spread (300s rolling avg)')
    ax2.axhline(0, color='black', lw=0.6, linestyle=':')

    # Shade regions where optimal is wider vs tighter than market
    ax2.fill_between(t_hours, diff_series, 0,
                     where=diff_series > 0,
                     color='red', alpha=0.15,
                     label='Optimal wider than market')
    ax2.fill_between(t_hours, diff_series, 0,
                     where=diff_series < 0,
                     color='green', alpha=0.15,
                     label='Optimal tighter than market')

    ax2.set_ylabel('Spread difference ($)', fontsize=10)
    ax2.set_title('Optimal spread minus market spread (rolling 5-min avg)',
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_xlim(9.5, 16.0)
    _apply_time_axis(ax2, sparse=False, rotation=30)

    plt.tight_layout()
    if save:
        path = os.path.join(results_dir,
                            f'fig_spread_comparison_{ticker}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()

def plot_all(all_results: dict, save=True, subfolder='synthetic'):
    """
    Generate all figures for all stocks. When the results dict contains
    queue-model variants (`optimal_back` / `baseline_back`), figures for
    those are written to a parallel `<subfolder>_back/` directory.

    Parameters
    ----------
    all_results : dict from run_experiment() or run_wrds_experiment()
    save        : whether to save figures to disk
    subfolder   : 'synthetic' (from poisson_simulator.py)
                  'wrds'      (from run_with_wrds.py)
    """
    # The standalone reference figures (1, 2) are queue-model-independent —
    # they live in the front-of-queue folder only.
    print(f"\nSaving plots to: plots/{subfolder}/")
    plot_order_size_function(save=save, subfolder=subfolder)
    plot_intensity_components(save=save, subfolder=subfolder)

    variants = [
        ('optimal',      'baseline',      subfolder),
        ('optimal_back', 'baseline_back', f"{subfolder}_back"),
    ]

    for opt_key, base_key, folder in variants:
        if not all(opt_key in res and base_key in res
                   for res in all_results.values()):
            continue
        print(f"\nSaving plots to: plots/{folder}/")
        for ticker, res in all_results.items():
            print(f"  Plotting {ticker} ({folder})...")
            plot_spreads(ticker, res['as_model'],
                         res[opt_key], res[base_key],
                         save=save, subfolder=folder)
            plot_pnl_and_inventory(ticker,
                                    res[opt_key], res[base_key],
                                    save=save, subfolder=folder)
            plot_quote_dynamics(ticker, res['as_model'],
                                res[opt_key],
                                save=save, subfolder=folder)
            plot_quoted_vs_actual_spread(ticker,
                                         res[opt_key], res[base_key],
                                         save=save, subfolder=folder)

if __name__ == '__main__':
    plot_order_size_function(subfolder='synthetic')
    plot_intensity_components(subfolder='synthetic')