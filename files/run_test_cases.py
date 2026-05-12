"""
test_units.py
=============
Unit tests for market_maker.py, replay_simulator.py, and helpers.py.
Tests the math and logic of each component independently.

Run from the project root:
    python test_units.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_maker import (AvellanedaStoikov, InventoryModel, BaselineStrategy,
                           INVENTORY_MODEL_ENABLED, BASELINE_INVENTORY_MODEL_ENABLED)
from replay_simulator import ReplaySimulator
from helpers import fill_analysis

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

results = []

def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"{status}  {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, condition))


# ===========================================================================
# AvellanedaStoikov
# ===========================================================================

print("\n--- AvellanedaStoikov ---")

sp = {'open': 0.05, 'close': 0.01, 'gamma': 0.1}
m  = AvellanedaStoikov(gamma=sp['gamma'], sigma=0.01, T=1.0, kappa=2.0)
m.calibrate_from_market(sp['open'], sp['close'])

# Calibration
check("A = (open - close) / T",
      abs(m.A - 0.04) < 1e-10,
      f"A={m.A}")

check("B = close_spread",
      abs(m.B - 0.01) < 1e-10,
      f"B={m.B}")

check("gamma_sigma2 = A after calibration",
      abs(m.gamma_sigma2 - m.A) < 1e-10,
      f"gamma_sigma2={m.gamma_sigma2}")

# Spread endpoints
check("spread at t=0 equals open_spread",
      abs(m.optimal_spread(0.0) - sp['open']) < 1e-10,
      f"spread(0)={m.optimal_spread(0.0)}")

check("spread at t=1 equals close_spread",
      abs(m.optimal_spread(1.0) - sp['close']) < 1e-10,
      f"spread(1)={m.optimal_spread(1.0)}")

# Spread is monotonically decreasing
spreads = [m.optimal_spread(t) for t in np.linspace(0, 1, 200)]
check("spread decreases monotonically over time",
      all(spreads[i] >= spreads[i+1] for i in range(len(spreads)-1)))

# Indifference price
check("indifference price = s when q=0",
      abs(m.indifference_price(145.0, 0, 0.5) - 145.0) < 1e-10,
      f"r={m.indifference_price(145.0, 0, 0.5)}")

check("indifference price < s when long (q>0)",
      m.indifference_price(145.0, 100, 0.5) < 145.0,
      f"r={m.indifference_price(145.0, 100, 0.5):.4f}")

check("indifference price > s when short (q<0)",
      m.indifference_price(145.0, -100, 0.5) > 145.0,
      f"r={m.indifference_price(145.0, -100, 0.5):.4f}")

check("indifference price moves more at open than close (larger T-t)",
      abs(m.indifference_price(145.0, 100, 0.0) - 145.0) >
      abs(m.indifference_price(145.0, 100, 0.9) - 145.0))

# Quotes
bid, ask = m.quotes_calibrated(145.0, 0, 0.0)
check("bid < ask",
      bid < ask,
      f"bid={bid:.4f} ask={ask:.4f}")

check("ask - bid = open_spread at t=0, q=0",
      abs((ask - bid) - sp['open']) < 1e-10,
      f"spread={ask-bid:.4f}")

check("quotes symmetric around indifference price",
      abs((ask + bid) / 2 - m.indifference_price(145.0, 0, 0.0)) < 1e-10)

# Inventory shading moves quotes in right direction
bid_long, ask_long   = m.quotes_calibrated(145.0,  100, 0.5)
bid_short, ask_short = m.quotes_calibrated(145.0, -100, 0.5)
bid_flat, ask_flat   = m.quotes_calibrated(145.0,    0, 0.5)

check("long inventory shifts quotes down (sell pressure)",
      bid_long < bid_flat and ask_long < ask_flat)

check("short inventory shifts quotes up (buy pressure)",
      bid_short > bid_flat and ask_short > ask_flat)

# AMZN: open < close so A=0, spread is flat
m_amzn = AvellanedaStoikov(gamma=0.01, sigma=0.01, T=1.0, kappa=2.0)
m_amzn.calibrate_from_market(0.49, 0.56)
check("AMZN: A=0 when open_spread < close_spread (clamped)",
      m_amzn.A == 0.0,
      f"A={m_amzn.A}")

check("AMZN: spread is flat at B=close_spread all day",
      abs(m_amzn.optimal_spread(0.0) - 0.56) < 1e-10 and
      abs(m_amzn.optimal_spread(0.5) - 0.56) < 1e-10)


# ===========================================================================
# InventoryModel
# ===========================================================================

print("\n--- InventoryModel (enabled=True) ---")

inv = InventoryModel(phi_max=100.0, eta=0.005, enabled=True)

check("bid_size = phi_max when q=0",
      inv.bid_size(0) == 100.0)

check("ask_size = phi_max when q=0",
      inv.ask_size(0) == 100.0)

check("bid_size = phi_max when q<0 (short, want to buy)",
      inv.bid_size(-200) == 100.0)

check("ask_size = phi_max when q>0 (long, want to sell)",
      inv.ask_size(200) == 100.0)

check("bid_size shrinks when q>0 (long, reduce buying)",
      inv.bid_size(200) < 100.0,
      f"bid_size(200)={inv.bid_size(200):.2f}")

check("ask_size shrinks when q<0 (short, reduce selling)",
      inv.ask_size(-200) < 100.0,
      f"ask_size(-200)={inv.ask_size(-200):.2f}")

check("bid_size never goes negative",
      inv.bid_size(100000) > 0,
      f"bid_size(100000)={inv.bid_size(100000):.6f}")

check("ask_size never goes negative",
      inv.ask_size(-100000) > 0,
      f"ask_size(-100000)={inv.ask_size(-100000):.6f}")

check("bid and ask sizes are symmetric around q=0",
      abs(inv.bid_size(200) - inv.ask_size(-200)) < 1e-10)

# Verify eta=0.005 at q=200: exp(-0.005*200) = exp(-1) ≈ 0.368
expected = 100.0 * np.exp(-0.005 * 200)
check("bid_size(200) = phi_max * exp(-eta*200)",
      abs(inv.bid_size(200) - expected) < 1e-10,
      f"expected={expected:.4f} got={inv.bid_size(200):.4f}")

# order_sizes convenience method
b, a = inv.order_sizes(100)
check("order_sizes returns (bid_size, ask_size)",
      b == inv.bid_size(100) and a == inv.ask_size(100))


# --- Section 2.2 toggle (enabled=False) ---

print("\n--- InventoryModel (enabled=False) ---")

inv_off = InventoryModel(phi_max=100.0, eta=0.005, enabled=False)

check("disabled: bid_size = phi_max when q=0",
      inv_off.bid_size(0) == 100.0)

check("disabled: bid_size stays phi_max when q>0 (no decay)",
      inv_off.bid_size(200) == 100.0,
      f"bid_size(200)={inv_off.bid_size(200)}")

check("disabled: ask_size stays phi_max when q<0 (no decay)",
      inv_off.ask_size(-200) == 100.0,
      f"ask_size(-200)={inv_off.ask_size(-200)}")

check("disabled: bid_size constant across extreme inventory",
      inv_off.bid_size(0) == inv_off.bid_size(10_000) == 100.0)

check("disabled: ask_size constant across extreme inventory",
      inv_off.ask_size(0) == inv_off.ask_size(-10_000) == 100.0)

check("disabled: order_sizes returns (phi_max, phi_max) at any q",
      inv_off.order_sizes(500) == (100.0, 100.0))


# --- Module-level toggle constants ---

print("\n--- Inventory toggle constants ---")

check("INVENTORY_MODEL_ENABLED is a bool",
      isinstance(INVENTORY_MODEL_ENABLED, bool),
      f"type={type(INVENTORY_MODEL_ENABLED).__name__}")

check("BASELINE_INVENTORY_MODEL_ENABLED is a bool",
      isinstance(BASELINE_INVENTORY_MODEL_ENABLED, bool),
      f"type={type(BASELINE_INVENTORY_MODEL_ENABLED).__name__}")

# Default of the `enabled` kwarg should track the module constant
inv_default = InventoryModel(phi_max=100.0, eta=0.005)
check("InventoryModel default enabled tracks INVENTORY_MODEL_ENABLED",
      inv_default.enabled == INVENTORY_MODEL_ENABLED,
      f"inv.enabled={inv_default.enabled}  const={INVENTORY_MODEL_ENABLED}")


# ===========================================================================
# BaselineStrategy
# ===========================================================================

print("\n--- BaselineStrategy ---")

base = BaselineStrategy()
bid, ask = base.quotes(144.95, 145.05)

check("baseline returns best_bid unchanged",
      bid == 144.95)

check("baseline returns best_ask unchanged",
      ask == 145.05)


# ===========================================================================
# ReplaySimulator fill logic
# ===========================================================================

print("\n--- ReplaySimulator fill logic ---")

sim = ReplaySimulator()

# Signature: _check_bid_fill(bid_price, bid_size, trades, queue_ahead, queue_model)
# Returns: (filled, fill_size, queue_ahead)
# 'front' queue model = best case (queue_ahead ignored), matches paper baseline.
def bid_fill(bid_price, bid_size, trades, queue_ahead=0.0, queue_model='front'):
    return sim._check_bid_fill(bid_price, bid_size, trades, queue_ahead, queue_model)

def ask_fill(ask_price, ask_size, trades, queue_ahead=0.0, queue_model='front'):
    return sim._check_ask_fill(ask_price, ask_size, trades, queue_ahead, queue_model)

# --- _check_bid_fill ---

# Trade below our bid: should fill
t = pd.DataFrame({'price': [144.90], 'size': [200.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid filled when trade price < bid price",
      filled and sz > 0,
      f"filled={filled} size={sz}")

# Trade at exactly our bid: should fill
t = pd.DataFrame({'price': [145.00], 'size': [200.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid filled when trade price = bid price",
      filled and sz > 0)

# Trade above our bid: should not fill
t = pd.DataFrame({'price': [145.10], 'size': [200.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid NOT filled when trade price > bid price",
      not filled)

# Fill size capped at our order size
t = pd.DataFrame({'price': [144.90], 'size': [500.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid fill size capped at order size",
      sz == 100.0,
      f"fill_size={sz}")

# Fill size capped at available volume
t = pd.DataFrame({'price': [144.90], 'size': [30.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid fill size capped at available volume when volume < order size",
      sz == 30.0,
      f"fill_size={sz}")

# No trades: no fill
filled, sz, _ = bid_fill(145.00, 100.0, None)
check("bid no fill when no trades",
      not filled)

filled, sz, _ = bid_fill(145.00, 100.0, pd.DataFrame(columns=['price', 'size']))
check("bid no fill when empty trades DataFrame",
      not filled)

# --- _check_ask_fill ---

# Trade above our ask: should fill
t = pd.DataFrame({'price': [145.10], 'size': [200.0]})
filled, sz, _ = ask_fill(145.00, 100.0, t)
check("ask filled when trade price > ask price",
      filled and sz > 0,
      f"filled={filled} size={sz}")

# Trade at exactly our ask: should fill
t = pd.DataFrame({'price': [145.00], 'size': [200.0]})
filled, sz, _ = ask_fill(145.00, 100.0, t)
check("ask filled when trade price = ask price",
      filled and sz > 0)

# Trade below our ask: should not fill
t = pd.DataFrame({'price': [144.90], 'size': [200.0]})
filled, sz, _ = ask_fill(145.00, 100.0, t)
check("ask NOT filled when trade price < ask price",
      not filled)

# Fill size capped at order size
t = pd.DataFrame({'price': [145.10], 'size': [500.0]})
filled, sz, _ = ask_fill(145.00, 100.0, t)
check("ask fill size capped at order size",
      sz == 100.0,
      f"fill_size={sz}")

# Multiple trades: aggregate volume
t = pd.DataFrame({'price': [144.85, 144.90, 144.95], 'size': [20.0, 30.0, 40.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid fill aggregates volume across multiple qualifying trades",
      filled and sz == 90.0,
      f"fill_size={sz}")

# Only qualifying trades count
t = pd.DataFrame({'price': [144.90, 145.10], 'size': [30.0, 50.0]})
filled, sz, _ = bid_fill(145.00, 100.0, t)
check("bid fill only counts trades at or below bid price",
      filled and sz == 30.0,
      f"fill_size={sz}")

# --- queue_model='back': queue_ahead consumed before fills ---

# All addressable volume consumed by queue ahead -> no fill
t = pd.DataFrame({'price': [144.90], 'size': [50.0]})
filled, sz, qa_after = bid_fill(145.00, 100.0, t, queue_ahead=200.0, queue_model='back')
check("back queue: no fill when queue_ahead > addressable volume",
      not filled and qa_after == 150.0,
      f"filled={filled} sz={sz} qa_after={qa_after}")

# Queue ahead partially consumed; remainder fills our order
t = pd.DataFrame({'price': [144.90], 'size': [200.0]})
filled, sz, qa_after = bid_fill(145.00, 100.0, t, queue_ahead=50.0, queue_model='back')
check("back queue: partial queue consumption leaves room to fill",
      filled and sz == 100.0 and qa_after == 0.0,
      f"sz={sz} qa_after={qa_after}")


# ===========================================================================
# helpers.py — fill_analysis
# ===========================================================================

print("\n--- helpers.fill_analysis ---")

from replay_simulator import Fill, SimulationResult

def make_mock_result(bid_fills, ask_fills, n_quotes):
    """Build a minimal SimulationResult for testing fill_analysis."""
    fills = (
        [Fill('bid', 145.00, 100.0, float(i)) for i in range(bid_fills)] +
        [Fill('ask', 145.05, 100.0, float(i)) for i in range(ask_fills)]
    )
    return SimulationResult(
        times=np.array([0.0]),
        mid_prices=np.array([145.0]),
        pnl=np.array([0.0]),
        inventory=np.array([0.0]),
        bid_prices=np.array([144.95]),
        ask_prices=np.array([145.05]),
        fills=fills,
        n_buy_orders=bid_fills,
        n_sell_orders=ask_fills,
        shares_bought=float(bid_fills * 100),
        shares_sold=float(ask_fills * 100),
        n_quotes=n_quotes,
    )

# Balanced fills
res = make_mock_result(bid_fills=10, ask_fills=10, n_quotes=100)
stats = fill_analysis([res])
check("fill_analysis: low imbalance when fills balanced",
      stats['imbalance_ratio'] < 0.05,
      f"imbalance={stats['imbalance_ratio']}")

check("fill_analysis: spread_capture_rate > 0 when both sides fill",
      stats['spread_capture_rate'] > 0)

check("fill_analysis: avg_spread_captured > 0 when ask > bid",
      stats['avg_spread_captured'] > 0,
      f"avg_spread={stats['avg_spread_captured']:.4f}")

# One-sided fills
res_onesided = make_mock_result(bid_fills=20, ask_fills=0, n_quotes=100)
stats_os = fill_analysis([res_onesided])
check("fill_analysis: high imbalance when all fills one-sided",
      stats_os['imbalance_ratio'] == 1.0,
      f"imbalance={stats_os['imbalance_ratio']}")

# No fills
res_none = make_mock_result(bid_fills=0, ask_fills=0, n_quotes=100)
stats_none = fill_analysis([res_none])
check("fill_analysis: handles zero fills gracefully",
      stats_none['fills_per_quote'] == 0.0)


# ===========================================================================
# Summary
# ===========================================================================

total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print(f"\n{'='*50}")
print(f"  {passed}/{total} tests passed", end="")
if failed:
    print(f"  —  {failed} FAILED:")
    for name, ok in results:
        if not ok:
            print(f"    ✗  {name}")
else:
    print("  — all good")
print(f"{'='*50}\n")