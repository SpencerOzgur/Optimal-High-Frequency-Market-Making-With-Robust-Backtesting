"""
poisson_simulator.py
====================
Closest possible reproduction of the Stanford A-S paper simulator.

Key design decisions matching the paper exactly:

Section 3.1 — Market Order Dynamics:
    Nt ~ Pois(integral_0^t lambda(s, xi) ds)
    lambda(t, xi) = alpha_t * exp(-mu * xi)
    xi = depth of quote measured from MID-PRICE (Figure 2b x-axis)
    alpha_t: piecewise linear bathtub (Cartea et al. 2015)
    mu = 100, set in Section 4

Section 3.2 — Order Execution:
    X ~ Ber(lambda(t, xi) * delta), delta = 1 second (Section 4)
    Y ~ Gamma(alpha=2, theta=1/1.65) (Section 4)
    executed_size = min(Y * order_size, order_size)

Section 4 — Experiments:
    Parameters: mu=100, alpha=2, theta=1/1.65, phi_max=100, eta=0.005
    Spread params calibrated from PREVIOUS week (we use Table 1 values directly)
    Both strategies run on IDENTICAL mid-price paths (same random seed)
    Baseline quotes at best bid/ask in order book
    In synthetic world, best bid/ask = mid +/- half_market_spread
    Market spread = time-varying, calibrated to match open/close spread params

Section 2.3 — Algorithm 1:
    waiting_time = 5 seconds
    update_time  = 1 second

AMZN fix:
    mu=100 is calibrated for penny-spread stocks like AAPL (open_spread=0.05).
    For AMZN (open_spread=0.49), exp(-100 * 0.245) = exp(-24.5) ≈ 0 — no fills.
    We scale mu inversely with open_spread so all tickers get reasonable fill
    probabilities at the baseline quote depth:
        effective_mu = MU * (REFERENCE_SPREAD / open_spread)
    AAPL: effective_mu = 100 * (0.05/0.05) = 100  (unchanged)
    AMZN: effective_mu = 100 * (0.05/0.49) ≈ 10.2  (reasonable fills)

Uses existing market_maker.py, helpers.py, analysis.py unchanged.
Returns SimulationResult so all downstream code works identically.

Run:
    python poisson_simulator.py
    python poisson_simulator.py --days 100   (for 100-simulation run)
"""

import numpy as np
import pandas as pd
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_maker import AvellanedaStoikov, InventoryModel, BaselineStrategy
from helpers import summarise_results, summarise_order_stats, print_fill_analysis, compute_quote_distance_stats
from replay_simulator import SimulationResult, Fill

# ---------------------------------------------------------------------------
# Paper parameters — Section 4, Table 1
# ---------------------------------------------------------------------------

T_SECONDS       = 23400.0       # 9:30am to 4:00pm = 6.5 hours
DT              = 1.0           # delta = 1 second (Section 4)
MU              = 100.0         # depth decay mu (Section 4)
REFERENCE_SPREAD = 0.05         # AAPL spread — mu=100 calibrated to this
ALPHA_GAMMA     = 2.0           # Gamma shape alpha (Section 4)
THETA           = 1.0 / 1.65    # Gamma scale theta (Section 4)
PHI_MAX         = 100.0         # phi_max (Section 4)
ETA             = 0.005         # eta (Section 2.2)
WAITING         = 5.0           # waiting time seconds (Section 2.3)
UPDATE          = 1.0           # update time seconds (Section 2.3)
N_DAYS          = 5             # default simulation days

# Table 1: calibrated from previous week's spreads
# sigma: annualized vol converted to per-second for arithmetic BM
# s0: approximate June 2017 prices
SPREAD_PARAMS = {
    'AAPL': {
        'open_spread':  0.05,
        'close_spread': 0.01,
        'gamma':        0.1,
        'sigma':        0.3 / np.sqrt(T_SECONDS),
        's0':           145.0,
    },
    'AMZN': {
        'open_spread':  0.49,
        'close_spread': 0.56,
        'gamma':        0.01,
        'sigma':        0.8 / np.sqrt(T_SECONDS),
        's0':           967.0,
    },
    'GE': {
        'open_spread':  0.04,
        'close_spread': 0.01,
        'gamma':        0.1,
        'sigma':        0.3 / np.sqrt(T_SECONDS),
        's0':           29.0,
    },
    'IVV': {
        'open_spread':  0.03,
        'close_spread': 0.01,
        'gamma':        0.1,
        'sigma':        0.2 / np.sqrt(T_SECONDS),
        's0':           244.0,
    },
    'M': {
        'open_spread':  0.09,
        'close_spread': 0.01,
        'gamma':        0.1,
        'sigma':        0.4 / np.sqrt(T_SECONDS),
        's0':           23.0,
    },
}

DATES      = ['2017-06-12', '2017-06-13', '2017-06-14', '2017-06-15', '2017-06-16']
COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']
DAY_LABELS = ['06/12', '06/13', '06/14', '06/15', '06/16']


# ---------------------------------------------------------------------------
# Bathtub alpha_t — Section 3.1, Cartea et al. (2015)
# ---------------------------------------------------------------------------

def bathtub_alpha(t: float) -> float:
    """Piecewise linear bathtub: 0.08 at open/close, 0.03 midday."""
    x = t / T_SECONDS
    if x <= 0.35:
        return 0.08 - (0.08 - 0.03) * (x / 0.35)
    elif x <= 0.65:
        return 0.03
    else:
        return 0.03 + (0.08 - 0.03) * ((x - 0.65) / 0.35)


# ---------------------------------------------------------------------------
# Poisson fill — Section 3.2
# xi measured from mid-price (Figure 2b)
# mu passed per-ticker to handle wide-spread stocks like AMZN
# ---------------------------------------------------------------------------

def poisson_fill(alpha_t: float, xi: float,
                 order_size: float, mu: float = MU) -> tuple:
    """
    Section 3.2 order execution.

    Step 1: X ~ Ber(lambda(t,xi) * delta)
            lambda = alpha_t * exp(-mu * xi)
            delta  = DT = 1 second

    Step 2: If X=1, Y ~ Gamma(alpha, theta)
            executed_size = min(Y * order_size, order_size)

    Parameters
    ----------
    alpha_t    : time component at current second
    xi         : depth from mid-price (positive = further from mid)
    order_size : our limit order size in shares
    mu         : depth decay parameter (per-ticker scaled version of MU)
    """
    # Guard against overflow before computing exp
    exponent = -mu * xi
    if exponent < -500:
        return False, 0.0

    lam  = alpha_t * np.exp(exponent)
    prob = min(lam * DT, 1.0)

    # Step 1: arrival indicator
    if np.random.rand() >= prob:
        return False, 0.0

    # Step 2: partial fill fraction
    Y         = np.random.gamma(ALPHA_GAMMA, THETA)
    fill_size = min(Y * order_size, order_size)
    fill_size = max(fill_size, 1.0)

    return True, fill_size


# ---------------------------------------------------------------------------
# Mid-price path — arithmetic BM (Section 2.1: dSt = sigma * dWt)
# ---------------------------------------------------------------------------

def generate_mid_price(s0: float, sigma: float, seed: int = None) -> np.ndarray:
    """Generate one day's mid-price path. Returns array of length n_steps+1."""
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T_SECONDS / DT)
    shocks  = sigma * np.random.randn(n_steps)
    mid     = np.zeros(n_steps + 1)
    mid[0]  = s0
    for i in range(1, n_steps + 1):
        mid[i] = mid[i-1] + shocks[i-1]
    return mid


# ---------------------------------------------------------------------------
# Simulator — Algorithm 1 + Section 3 Poisson fills
# ---------------------------------------------------------------------------

class PoissonSimulator:
    """
    Synthetic simulator implementing the paper's Poisson arrival model.

    Both optimal and baseline strategies are run on the SAME mid-price
    path per day (identical seed) for fair comparison.
    """

    def __init__(self,
                 dt: float = DT,
                 T_seconds: float = T_SECONDS,
                 waiting_time: float = WAITING,
                 update_time: float = UPDATE):
        self.dt           = dt
        self.T_seconds    = T_seconds
        self.waiting_time = waiting_time
        self.update_time  = update_time

    def run(self,
            strategy,
            mid: np.ndarray,
            open_spread: float,
            close_spread: float,
            inventory_model,
            strategy_type: str = 'optimal',
            mu: float = MU) -> SimulationResult:
        """
        Run one trading day.

        Parameters
        ----------
        strategy       : AvellanedaStoikov or BaselineStrategy
        mid            : pre-generated mid-price array (length n_steps+1)
        open_spread    : market spread at open
        close_spread   : market spread at close
        inventory_model: InventoryModel
        strategy_type  : 'optimal' or 'baseline'
        mu             : effective depth decay — scaled per ticker so wide-spread
                         stocks (AMZN) still receive realistic fill probabilities.
                         Default MU=100 matches paper for AAPL.
        """
        n_steps = len(mid) - 1
        T       = self.T_seconds

        cash      = 0.0
        inventory = 0.0
        pnl_arr   = np.zeros(n_steps + 1)
        inv_arr   = np.zeros(n_steps + 1)
        bid_arr   = np.full(n_steps + 1, np.nan)
        ask_arr   = np.full(n_steps + 1, np.nan)

        fills         = []
        n_buy_orders  = 0
        n_sell_orders = 0
        shares_bought = 0.0
        shares_sold   = 0.0
        n_quotes      = 0

        active_bid      = None
        active_ask      = None
        last_quote_time = -np.inf
        last_exec_time  = -np.inf

        for step in range(n_steps):
            t      = step * self.dt
            t_norm = t / T
            s      = mid[step]

            # Time-varying market spread: decays linearly open -> close
            mkt_spread = open_spread + (close_spread - open_spread) * t_norm
            half_mkt   = mkt_spread / 2.0
            best_bid   = s - half_mkt
            best_ask   = s + half_mkt

            alpha_t  = bathtub_alpha(t)
            n_active = (active_bid is not None) + (active_ask is not None)

            # --- Algorithm 1 (Section 2.3) ---
            if n_active == 0:
                phi_bid, phi_ask = inventory_model.order_sizes(inventory)

                if strategy_type == 'optimal':
                    bid_q, ask_q = strategy.quotes_calibrated(s, inventory, t_norm)
                else:
                    bid_q, ask_q = strategy.quotes(best_bid, best_ask)

                if bid_q < ask_q:
                    active_bid      = {'price': bid_q, 'size': phi_bid}
                    active_ask      = {'price': ask_q, 'size': phi_ask}
                    last_quote_time = t
                    n_quotes       += 1

            elif n_active == 1:
                if t - last_exec_time > self.waiting_time:
                    active_bid = None
                    active_ask = None

            else:  # n_active == 2
                if t - last_quote_time > self.update_time:
                    phi_bid, phi_ask = inventory_model.order_sizes(inventory)

                    if strategy_type == 'optimal':
                        bid_q, ask_q = strategy.quotes_calibrated(s, inventory, t_norm)
                    else:
                        bid_q, ask_q = strategy.quotes(best_bid, best_ask)

                    if bid_q < ask_q:
                        active_bid      = {'price': bid_q, 'size': phi_bid}
                        active_ask      = {'price': ask_q, 'size': phi_ask}
                        last_quote_time = t
                        n_quotes       += 1

            # --- Section 3.2: Poisson fills ---
            # xi measured from mid-price (Figure 2b)
            # bid: xi = s - bid_price  (positive = below mid)
            # ask: xi = ask_price - s  (positive = above mid)

            if active_bid is not None:
                xi_bid = s - active_bid['price']
                filled, fill_size = poisson_fill(alpha_t, xi_bid,
                                                  active_bid['size'], mu=mu)
                if filled:
                    cash          -= active_bid['price'] * fill_size
                    inventory     += fill_size
                    shares_bought += fill_size
                    n_buy_orders  += 1
                    fills.append(Fill('bid', active_bid['price'], fill_size, t))
                    active_bid     = None
                    last_exec_time = t

            if active_ask is not None:
                xi_ask = active_ask['price'] - s
                filled, fill_size = poisson_fill(alpha_t, xi_ask,
                                                  active_ask['size'], mu=mu)
                if filled:
                    cash          += active_ask['price'] * fill_size
                    inventory     -= fill_size
                    shares_sold   += fill_size
                    n_sell_orders += 1
                    fills.append(Fill('ask', active_ask['price'], fill_size, t))
                    active_ask     = None
                    last_exec_time = t

            pnl_arr[step] = cash + inventory * s
            inv_arr[step] = inventory
            bid_arr[step] = active_bid['price'] if active_bid else np.nan
            ask_arr[step] = active_ask['price'] if active_ask else np.nan

        # Final mark-to-market
        pnl_arr[n_steps] = cash + inventory * mid[n_steps]
        inv_arr[n_steps] = inventory
        bid_arr[n_steps] = bid_arr[n_steps - 1]
        ask_arr[n_steps] = ask_arr[n_steps - 1]

        times = np.arange(n_steps + 1) * self.dt

        return SimulationResult(
            times=times,
            mid_prices=mid,
            pnl=pnl_arr,
            inventory=inv_arr,
            bid_prices=bid_arr,
            ask_prices=ask_arr,
            fills=fills,
            n_buy_orders=n_buy_orders,
            n_sell_orders=n_sell_orders,
            shares_bought=shares_bought,
            shares_sold=shares_sold,
            n_quotes=n_quotes,
        )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(n_days: int = N_DAYS):
    """
    Run the full experiment matching paper Section 4.

    Both strategies run on identical mid-price paths per day.

    gamma_sigma2 rescaled by T_SECONDS: fixes inventory shading magnitude.
    effective_mu scaled by open_spread: fixes AMZN fill probability.
    """
    sim       = PoissonSimulator()
    inv_model = InventoryModel(phi_max=PHI_MAX, eta=ETA)

    all_results = {}

    print("=" * 65)
    print("  Poisson Simulator — Stanford A-S paper (Section 3)")
    print(f"  mu={MU}, alpha={ALPHA_GAMMA}, theta={THETA:.4f}")
    print(f"  phi_max={PHI_MAX}, eta={ETA}, delta={DT}s, days={n_days}")
    print("=" * 65)

    for ticker, sp in SPREAD_PARAMS.items():

        as_model = AvellanedaStoikov(
            gamma=sp['gamma'],
            sigma=sp['sigma'],
            kappa=2.0,
            T=1.0
        )
        as_model.calibrate_from_market(sp['open_spread'], sp['close_spread'])

        # Rescale gamma_sigma2 only — keeps spread decay correct but
        # fixes inventory shading magnitude to per-second scale.
        as_model.gamma_sigma2 = as_model.gamma_sigma2 / T_SECONDS

        # Scale mu inversely with open_spread.
        # mu=100 is calibrated for AAPL's 0.05 spread — baseline xi = 0.025
        # gives exp(-100*0.025) = exp(-2.5) ≈ 0.08, a reasonable fill rate.
        # For AMZN baseline xi = 0.245, exp(-100*0.245) ≈ 0 — no fills ever.
        # Scaling: effective_mu = 100 * (0.05 / open_spread) ensures baseline
        # always quotes at an equivalent relative depth across all tickers.
        effective_mu = MU * (REFERENCE_SPREAD / sp['open_spread'])

        print(f"\n  {ticker}: A={as_model.A:.6f} B={as_model.B:.4f} "
              f"gs2={as_model.gamma_sigma2:.8f} "
              f"effective_mu={effective_mu:.2f} "
              f"spread(0)={as_model.optimal_spread_calibrated(0):.4f} "
              f"spread(1)={as_model.optimal_spread_calibrated(1):.4f}")

        baseline = BaselineStrategy()

        optimal_results  = []
        baseline_results = []

        for day_i in range(n_days):
            seed = hash(ticker + str(day_i)) % (2**31)
            mid  = generate_mid_price(sp['s0'], sp['sigma'], seed=seed)

            res_opt  = sim.run(as_model, mid,
                               sp['open_spread'], sp['close_spread'],
                               inv_model, strategy_type='optimal',
                               mu=effective_mu)

            res_base = sim.run(baseline, mid,
                               sp['open_spread'], sp['close_spread'],
                               inv_model, strategy_type='baseline',
                               mu=effective_mu)

            optimal_results.append(res_opt)
            baseline_results.append(res_base)

            if day_i < 5:
                date = DATES[day_i % len(DATES)]
                print(f"    {ticker} {date}: "
                      f"opt={res_opt.pnl[-1]:>10,.0f}  "
                      f"base={res_base.pnl[-1]:>10,.0f}  "
                      f"inv_opt={res_opt.inventory[-1]:>6.0f}  "
                      f"inv_base={res_base.inventory[-1]:>6.0f}")

        all_results[ticker] = {
            'optimal':  optimal_results,
            'baseline': baseline_results,
            'as_model': as_model,
        }

    # Tables
    print("\n" + "=" * 65)
    print("  Results")
    print("=" * 65)

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

    print("\n--- Fill Analysis ---")
    print_fill_analysis(all_results)

    # Figures
    try:
        from analysis import plot_all
        plot_all(all_results, subfolder='synthetic')
    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A-S Poisson Simulator')
    parser.add_argument('--days', type=int, default=N_DAYS,
                        help='Number of days to simulate (default 5, use 100 for full run)')
    args = parser.parse_args()

    all_results = run_experiment(n_days=args.days)

    print("\n--- Quote Distance from NBBO ---")
    dist_df = compute_quote_distance_stats(all_results, SPREAD_PARAMS)
    print(dist_df.to_string(index=False))