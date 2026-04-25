"""
debug_wrds.py
=============
Diagnostic script that runs the WRDS pipeline for ONE ticker, ONE day,
and prints everything relevant along the way to pinpoint exactly where
the model breaks down on real data.

Run:
    python debug_wrds.py
    python debug_wrds.py --ticker AMZN --date 2017-06-12
"""

import numpy as np
import pandas as pd
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wrds_loader import WRDSLoader, estimate_sigma
from market_maker import AvellanedaStoikov, InventoryModel, BaselineStrategy
from replay_simulator import ReplaySimulator, SimulationResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPREAD_PARAMS = {
    'AAPL': {'open_spread': 0.02, 'close_spread': 0.01, 'gamma': 0.001},
    'AMZN': {'open_spread': 0.49, 'close_spread': 0.56, 'gamma': 0.0001},
    'GE':   {'open_spread': 0.02, 'close_spread': 0.01, 'gamma': 0.001},
    'IVV':  {'open_spread': 0.02, 'close_spread': 0.01, 'gamma': 0.001},
    'M':    {'open_spread': 0.04, 'close_spread': 0.01, 'gamma': 0.001},
}

SEP  = "=" * 70
SEP2 = "-" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subsection(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_debug(ticker: str = 'AAPL', date: str = '2017-06-12'):

    sp = SPREAD_PARAMS[ticker]

    # -----------------------------------------------------------------------
    # STEP 1: Load data
    # -----------------------------------------------------------------------
    section(f"STEP 1: Loading TAQ data — {ticker} {date}")

    loader   = WRDSLoader()
    raw_data = loader.load_week(tickers=[ticker], dates=[date])
    loader.close()

    day = raw_data[ticker][date]
    if day is None:
        print("ERROR: No data loaded. Check ticker/date.")
        return

    mid         = day['mid']
    best_bid    = day['best_bid']
    best_ask    = day['best_ask']
    mkt_trades  = day['market_trades']
    sigma       = day['sigma']

    subsection("Mid-price statistics")
    print(f"  Array length      : {len(mid)} seconds")
    print(f"  Open price        : {mid[0]:.4f}")
    print(f"  Close price       : {mid[-1]:.4f}")
    print(f"  Min               : {np.nanmin(mid):.4f}")
    print(f"  Max               : {np.nanmax(mid):.4f}")
    print(f"  Daily range       : {np.nanmax(mid) - np.nanmin(mid):.4f}")
    print(f"  Sigma (per sec)   : {sigma:.8f}")
    print(f"  Sigma (annualized): {sigma * np.sqrt(252 * 23400):.4f}")

    subsection("Market spread statistics (best_ask - best_bid)")
    spreads = best_ask - best_bid
    print(f"  Mean spread       : ${np.nanmean(spreads):.4f}")
    print(f"  Median spread     : ${np.nanmedian(spreads):.4f}")
    print(f"  Min spread        : ${np.nanmin(spreads):.4f}")
    print(f"  Max spread        : ${np.nanmax(spreads):.4f}")
    print(f"  Stdev spread      : ${np.nanstd(spreads):.4f}")
    print(f"  % time at 1 cent  : {(spreads <= 0.011).mean()*100:.1f}%")
    print(f"  % time at 2 cents : {(spreads <= 0.021).mean()*100:.1f}%")

    subsection("Trade data")
    print(f"  Total trades      : {len(mkt_trades):,}")
    print(f"  Trades per second : {len(mkt_trades) / 23400:.2f}")
    print(f"  Mean trade size   : {mkt_trades['size'].mean():.1f} shares")
    print(f"  Median trade size : {mkt_trades['size'].median():.1f} shares")
    print(f"  Total volume      : {mkt_trades['size'].sum():,.0f} shares")
    print(f"  Price range       : {mkt_trades['price'].min():.4f} - {mkt_trades['price'].max():.4f}")

    # Seconds with at least one trade
    trade_seconds = set(mkt_trades['t_sec'].astype(int).values)
    print(f"  Seconds with trades: {len(trade_seconds):,} / 23400 ({len(trade_seconds)/234:.1f}%)")

    # -----------------------------------------------------------------------
    # STEP 2: Model calibration
    # -----------------------------------------------------------------------
    section("STEP 2: Model calibration")

    as_model = AvellanedaStoikov(
        gamma=sp['gamma'],
        sigma=sigma,
        kappa=2.0,
        T=1.0
    )
    as_model.calibrate_from_market(sp['open_spread'], sp['close_spread'])
    as_model.gamma_sigma2 = as_model.gamma_sigma2 / 23400.0  # add this line

    print(f"\n  Input parameters:")
    print(f"    gamma            : {sp['gamma']}")
    print(f"    sigma (per sec)  : {sigma:.8f}")
    print(f"    open_spread      : {sp['open_spread']}")
    print(f"    close_spread     : {sp['close_spread']}")

    print(f"\n  Calibrated parameters:")
    print(f"    A (spread slope) : {as_model.A:.6f}")
    print(f"    B (close spread) : {as_model.B:.6f}")
    print(f"    gamma_sigma2     : {as_model.gamma_sigma2:.8f}")
    print(f"    kappa            : {as_model.kappa:.4f}")

    print(f"\n  Spread verification:")
    print(f"    spread(t=0.0)    : {as_model.optimal_spread_calibrated(0.0):.6f}  (should = open_spread={sp['open_spread']})")
    print(f"    spread(t=0.5)    : {as_model.optimal_spread_calibrated(0.5):.6f}")
    print(f"    spread(t=1.0)    : {as_model.optimal_spread_calibrated(1.0):.6f}  (should = close_spread={sp['close_spread']})")

    subsection("Quote samples at q=0 (no inventory)")
    s_sample = mid[0]
    print(f"\n  Mid price used: {s_sample:.4f}")
    print(f"  {'t_norm':<10} {'spread':<12} {'bid':<12} {'ask':<12} {'vs market bid':<15} {'vs market ask'}")
    for t_norm in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        step     = int(t_norm * 23399)
        s        = mid[step]
        bid, ask = as_model.quotes_calibrated(s, 0, t_norm)
        mkt_b    = best_bid[step]
        mkt_a    = best_ask[step]
        print(f"  {t_norm:<10.2f} {ask-bid:<12.4f} {bid:<12.4f} {ask:<12.4f} "
              f"{bid - mkt_b:<+15.4f} {ask - mkt_a:<+.4f}")

    subsection("Indifference price shift at various inventory levels (t=0.5)")
    t_norm = 0.5
    step   = int(t_norm * 23399)
    s      = mid[step]
    print(f"\n  Mid price: {s:.4f}")
    print(f"  {'q':<10} {'r':<12} {'shift from mid':<18} {'bid':<12} {'ask':<12} {'spread'}")
    for q in [-200, -100, -50, 0, 50, 100, 200]:
        r        = as_model.indifference_price(s, q, t_norm)
        bid, ask = as_model.quotes_calibrated(s, q, t_norm)
        print(f"  {q:<10} {r:<12.4f} {r-s:<+18.6f} {bid:<12.4f} {ask:<12.4f} {ask-bid:.4f}")

    subsection("Key diagnostic: Is inventory shading in correct range?")
    half_spread = as_model.optimal_spread_calibrated(0.5) / 2
    max_inv_shift = 200 * as_model.gamma_sigma2 * 0.5
    print(f"\n  Half spread at t=0.5   : ${half_spread:.6f}")
    print(f"  Inv shift at q=200,t=0.5: ${max_inv_shift:.6f}")
    if max_inv_shift > half_spread:
        print(f"  WARNING: Inventory shift ({max_inv_shift:.6f}) > half spread ({half_spread:.6f})")
        print(f"           This means large inventory pushes quotes PAST mid-price")
        print(f"           Ratio: {max_inv_shift / half_spread:.1f}x — should be < 1.0")
    else:
        print(f"  OK: Inventory shift is within half-spread bounds")
        print(f"  Ratio: {max_inv_shift / half_spread:.2f}x")

    # -----------------------------------------------------------------------
    # STEP 3: Fill analysis — what would actually get filled
    # -----------------------------------------------------------------------
    section("STEP 3: Fill environment analysis")

    subsection("Optimal quotes vs market at t=0 (open)")
    bid_opt, ask_opt = as_model.quotes_calibrated(mid[0], 0, 0.0)
    print(f"\n  Optimal bid        : {bid_opt:.4f}")
    print(f"  Market best bid    : {best_bid[0]:.4f}")
    print(f"  Difference         : {bid_opt - best_bid[0]:+.4f}  ({'inside' if bid_opt > best_bid[0] else 'outside'} market)")
    print(f"\n  Optimal ask        : {ask_opt:.4f}")
    print(f"  Market best ask    : {best_ask[0]:.4f}")
    print(f"  Difference         : {ask_opt - best_ask[0]:+.4f}  ({'inside' if ask_opt < best_ask[0] else 'outside'} market)")

    subsection("Fill rate simulation — how often would quotes get crossed?")

    # Simulate fill rates at different depth levels
    from replay_simulator import ReplaySimulator as RS
    sim_test = RS()
    trade_index = sim_test._build_trade_index(mkt_trades)

    # Check fill rates for quotes at various depths from mid
    depth_results = []
    n_seconds = min(1000, len(mid) - 1)  # check first 1000 seconds

    for depth_offset in [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]:
        bid_fills = 0
        ask_fills = 0
        for step in range(n_seconds):
            s   = mid[step]
            bid = s - depth_offset
            ask = s + depth_offset
            trades_now = trade_index.get(step, None)
            b_filled, _ = sim_test._check_bid_fill(bid, 100.0, trades_now)
            a_filled, _ = sim_test._check_ask_fill(ask, 100.0, trades_now)
            if b_filled:
                bid_fills += 1
            if a_filled:
                ask_fills += 1

        depth_results.append({
            'depth': depth_offset,
            'bid_fill_rate': bid_fills / n_seconds,
            'ask_fill_rate': ask_fills / n_seconds,
        })

    print(f"\n  Fill rates over first {n_seconds} seconds (quote = mid ± depth):")
    print(f"  {'Depth from mid':<18} {'Bid fill rate':<18} {'Ask fill rate'}")
    for r in depth_results:
        flag = " ← baseline" if abs(r['depth'] - spreads[:n_seconds].mean()/2) < 0.005 else ""
        print(f"  {r['depth']:+.4f}            {r['bid_fill_rate']*100:>8.1f}%          "
              f"{r['ask_fill_rate']*100:>8.1f}%{flag}")

    # What depth does the optimal strategy quote at?
    opt_depths = []
    for step in range(0, n_seconds, 100):
        t_norm   = step / 23399
        s        = mid[step]
        bid, ask = as_model.quotes_calibrated(s, 0, t_norm)
        opt_depths.append(s - bid)  # distance from mid to our bid

    print(f"\n  Optimal strategy avg depth from mid: {np.mean(opt_depths):.4f}")
    print(f"  Optimal strategy at open (t=0):      {opt_depths[0]:.4f}")
    print(f"  Optimal strategy at midday (t=0.5):  {opt_depths[len(opt_depths)//2]:.4f}")

    # -----------------------------------------------------------------------
    # STEP 4: Run single simulation with verbose per-fill output
    # -----------------------------------------------------------------------
    section("STEP 4: Single simulation run — first 300 seconds verbose")

    sim = ReplaySimulator(dt=1.0, T_seconds=23400.0,
                          waiting_time=5.0, update_time=1.0)
    inv_model = InventoryModel(phi_max=100.0, eta=0.005)
    baseline  = BaselineStrategy()

    # Run both strategies
    res_opt  = sim.run(as_model, day, inv_model, strategy_type='optimal')
    res_base = sim.run(baseline,  day, inv_model, strategy_type='baseline')

    subsection("Terminal results")
    print(f"\n  {'Metric':<25} {'Optimal':>15} {'Baseline':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Terminal P&L':<25} {res_opt.pnl[-1]:>15,.2f} {res_base.pnl[-1]:>15,.2f}")
    print(f"  {'Terminal inventory':<25} {res_opt.inventory[-1]:>15.1f} {res_base.inventory[-1]:>15.1f}")
    print(f"  {'Buy orders':<25} {res_opt.n_buy_orders:>15,} {res_base.n_buy_orders:>15,}")
    print(f"  {'Sell orders':<25} {res_opt.n_sell_orders:>15,} {res_base.n_sell_orders:>15,}")
    print(f"  {'Shares bought':<25} {res_opt.shares_bought:>15,.0f} {res_base.shares_bought:>15,.0f}")
    print(f"  {'Shares sold':<25} {res_opt.shares_sold:>15,.0f} {res_base.shares_sold:>15,.0f}")
    print(f"  {'Quotes posted':<25} {res_opt.n_quotes:>15,} {res_base.n_quotes:>15,}")
    print(f"  {'Fill imbalance':<25} {abs(res_opt.n_buy_orders - res_opt.n_sell_orders):>15,} "
          f"{abs(res_base.n_buy_orders - res_base.n_sell_orders):>15,}")

    subsection("Fill analysis — optimal")
    opt_fills  = res_opt.fills
    bid_fills  = [f for f in opt_fills if f.side == 'bid']
    ask_fills  = [f for f in opt_fills if f.side == 'ask']
    if bid_fills and ask_fills:
        avg_bid_price = np.mean([f.price for f in bid_fills])
        avg_ask_price = np.mean([f.price for f in ask_fills])
        avg_spread_captured = avg_ask_price - avg_bid_price
        print(f"\n  Total fills        : {len(opt_fills)}")
        print(f"  Bid fills          : {len(bid_fills)}")
        print(f"  Ask fills          : {len(ask_fills)}")
        print(f"  Avg bid fill price : {avg_bid_price:.4f}")
        print(f"  Avg ask fill price : {avg_ask_price:.4f}")
        print(f"  Avg spread captured: {avg_spread_captured:+.4f}  ({'POSITIVE — earning spread' if avg_spread_captured > 0 else 'NEGATIVE — paying spread'})")
        print(f"  Avg fill size      : {np.mean([f.filled_size for f in opt_fills]):.1f} shares")
    else:
        print(f"\n  Total fills: {len(opt_fills)}")
        print(f"  One or both sides have zero fills — strategy is not functioning")

    subsection("Fill analysis — baseline")
    base_fills     = res_base.fills
    bid_fills_base = [f for f in base_fills if f.side == 'bid']
    ask_fills_base = [f for f in base_fills if f.side == 'ask']
    if bid_fills_base and ask_fills_base:
        avg_bid_base = np.mean([f.price for f in bid_fills_base])
        avg_ask_base = np.mean([f.price for f in ask_fills_base])
        print(f"\n  Total fills        : {len(base_fills)}")
        print(f"  Bid fills          : {len(bid_fills_base)}")
        print(f"  Ask fills          : {len(ask_fills_base)}")
        print(f"  Avg bid fill price : {avg_bid_base:.4f}")
        print(f"  Avg ask fill price : {avg_ask_base:.4f}")
        print(f"  Avg spread captured: {avg_ask_base - avg_bid_base:+.4f}")
    else:
        print(f"\n  Total fills: {len(base_fills)}")

    subsection("PnL decomposition — optimal")
    # Decompose PnL into spread capture vs inventory mark-to-market
    gross_spread = sum(
        f.price * f.filled_size if f.side == 'ask' else -f.price * f.filled_size
        for f in opt_fills
    )
    inventory_mtm = res_opt.inventory[-1] * mid[-1]
    print(f"\n  Gross trading cash : {gross_spread:>12,.2f}")
    print(f"  Inventory MTM      : {inventory_mtm:>12,.2f}  ({res_opt.inventory[-1]:.0f} shares @ {mid[-1]:.2f})")
    print(f"  Total P&L          : {gross_spread + inventory_mtm:>12,.2f}")
    print(f"\n  Interpretation:")
    if abs(inventory_mtm) > abs(gross_spread):
        print(f"  Mark-to-market loss ({inventory_mtm:,.0f}) dominates spread capture ({gross_spread:,.0f})")
        print(f"  The strategy is accumulating one-sided inventory and getting hurt by price moves")
    else:
        print(f"  Spread capture ({gross_spread:,.0f}) dominates MTM ({inventory_mtm:,.0f})")

    subsection("First 10 fills — optimal (showing what's happening)")
    print(f"\n  {'Time':<8} {'Side':<6} {'Price':<10} {'Size':<8} {'Inventory after':<18} {'PnL after'}")
    running_cash = 0.0
    running_inv  = 0.0
    for i, f in enumerate(opt_fills[:10]):
        step = int(f.time)
        if f.side == 'bid':
            running_cash -= f.price * f.filled_size
            running_inv  += f.filled_size
        else:
            running_cash += f.price * f.filled_size
            running_inv  -= f.filled_size
        pnl_now = running_cash + running_inv * mid[min(step, len(mid)-1)]
        print(f"  {f.time:<8.0f} {f.side:<6} {f.price:<10.4f} {f.filled_size:<8.0f} "
              f"{running_inv:<18.0f} {pnl_now:.2f}")

    # -----------------------------------------------------------------------
    # STEP 5: The core diagnosis
    # -----------------------------------------------------------------------
    section("STEP 5: Core diagnosis")

    print(f"\n  Market spread (mean)   : ${np.nanmean(spreads):.4f}")
    print(f"  Optimal spread (open)  : ${as_model.optimal_spread_calibrated(0.0):.4f}")
    print(f"  Optimal spread (mid)   : ${as_model.optimal_spread_calibrated(0.5):.4f}")
    print(f"  Optimal spread (close) : ${as_model.optimal_spread_calibrated(1.0):.4f}")

    mkt_mean  = np.nanmean(spreads)
    opt_open  = as_model.optimal_spread_calibrated(0.0)
    opt_close = as_model.optimal_spread_calibrated(1.0)

    print(f"\n  Is optimal spread wider than market spread?")
    wider_pct = (np.nanmean(spreads) < opt_open) * 100
    print(f"  At open: optimal ({opt_open:.4f}) {'>' if opt_open > mkt_mean else '<'} market ({mkt_mean:.4f})")

    crossover = None
    for t in np.linspace(0, 1, 1000):
        if as_model.optimal_spread_calibrated(t) <= mkt_mean:
            crossover = t
            break

    if crossover:
        crossover_hour = 9.5 + crossover * 6.5
        print(f"  Crossover point: t={crossover:.3f} (~{crossover_hour:.1f}h, "
              f"{int(crossover_hour)}:{int((crossover_hour % 1)*60):02d})")
        print(f"  Optimal is wider than market for {crossover*100:.0f}% of the trading day")
        print(f"  This means fills are scarce for {crossover*100:.0f}% of the day")
    else:
        print(f"  Optimal spread never crosses below market spread")
        print(f"  Optimal is ALWAYS wider than market — fills are always outside NBBO")

    print(f"\n  Fill rate comparison:")
    print(f"  Optimal:  {res_opt.n_buy_orders + res_opt.n_sell_orders:,} fills in 23400 seconds "
          f"= {(res_opt.n_buy_orders + res_opt.n_sell_orders)/234:.2f}% fill rate per second")
    print(f"  Baseline: {res_base.n_buy_orders + res_base.n_sell_orders:,} fills in 23400 seconds "
          f"= {(res_base.n_buy_orders + res_base.n_sell_orders)/234:.2f}% fill rate per second")

    print(f"\n  SUMMARY:")
    print(f"  The A-S model quotes at a spread calibrated to ({sp['open_spread']:.2f}, {sp['close_spread']:.2f})")
    print(f"  The real market spread is averaging ${mkt_mean:.4f}")
    if opt_open > mkt_mean:
        print(f"  The optimal spread is {opt_open/mkt_mean:.1f}x wider than the real market at open")
        print(f"  On real TAQ, quoting outside the NBBO means almost no fills")
        print(f"  The few fills that do occur are asymmetric — causing inventory drift")
        print(f"  Inventory drift causes mark-to-market losses that overwhelm spread capture")
    else:
        print(f"  Optimal spread is within market spread — fill environment is reasonable")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WRDS debug diagnostic')
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Ticker to diagnose (default AAPL)')
    parser.add_argument('--date', type=str, default='2017-06-12',
                        help='Date to diagnose (default 2017-06-12)')
    args = parser.parse_args()

    run_debug(ticker=args.ticker, date=args.date)