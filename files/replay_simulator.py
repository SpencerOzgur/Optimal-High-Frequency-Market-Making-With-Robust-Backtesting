"""
replay_simulator.py
===================
Event-driven simulator that replays actual TAQ market orders
instead of drawing from the Poisson arrival model.

Key difference from simulator.py:
  - BEFORE (synthetic): at each second, draw Bernoulli(lambda*dt) to decide
    if a market order arrives, then draw Gamma for partial fill size.
  - NOW (data-driven): at each second, look up ALL real trades that occurred
    in that second from ctm. If any trade price crosses our limit order price,
    we're filled — no probability draws needed.

Fill logic (realistic but still assumption-free on arrival):
  - A bid order at price P_bid is filled if any trade occurs at price <= P_bid
    (a seller hit the bid at or below our level)
  - An ask order at price P_ask is filled if any trade occurs at price >= P_ask
    (a buyer lifted the offer at or above our level)
  - Fill size = min(our_order_size, total_traded_size_at_that_level)

This is the standard "price priority" fill assumption used in academic
backtests when full order book depth is unavailable.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class Fill:
    side: str
    price: float
    filled_size: float
    time: float

@dataclass
class SimulationResult:
    times: np.ndarray
    mid_prices: np.ndarray
    pnl: np.ndarray
    inventory: np.ndarray
    bid_prices: np.ndarray
    ask_prices: np.ndarray
    fills: List[Fill]
    n_buy_orders: int
    n_sell_orders: int
    shares_bought: float
    shares_sold: float
    n_quotes: int


# ---------------------------------------------------------------------------
# Replay simulator
# ---------------------------------------------------------------------------

class ReplaySimulator:
    """
    Simulates market making by replaying real TAQ trade data.

    Parameters
    ----------
    dt            : time step in seconds (should match TAQ resolution; use 1)
    T_seconds     : trading day length in seconds
    waiting_time  : seconds to wait for outstanding order after one-side fill
    update_time   : seconds between quote refreshes with 2 orders in book
    """

    def __init__(self,
                 dt: float = 1.0,
                 T_seconds: float = 23400.0,
                 waiting_time: float = 5.0,
                 update_time: float = 1.0):
        self.dt           = dt
        self.T_seconds    = T_seconds
        self.waiting_time = waiting_time
        self.update_time  = update_time

    def _build_trade_index(self, market_trades: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Pre-index trades by integer second for O(1) lookup during simulation.
        Returns dict: {second: DataFrame of trades in that second}
        """
        market_trades = market_trades.copy()
        market_trades['sec_int'] = market_trades['t_sec'].astype(int)
        return {
            sec: grp.reset_index(drop=True)
            for sec, grp in market_trades.groupby('sec_int')
        }

    def _check_bid_fill(self, bid_price: float, bid_size: float,
                         trades_this_second: Optional[pd.DataFrame]) -> tuple:
        """
        A bid (buy limit order) at bid_price is filled if a real trade
        occurred at price <= bid_price (seller crossed the spread to hit us).

        Returns (filled: bool, fill_size: float)
        """
        if trades_this_second is None or len(trades_this_second) == 0:
            return False, 0.0

        # Trades at or below our bid price
        matching = trades_this_second[trades_this_second['price'] <= bid_price]
        if len(matching) == 0:
            return False, 0.0

        # Fill size = min(our order size, total volume that crossed our price)
        available_volume = matching['size'].sum()
        fill_size = min(bid_size, available_volume)
        return True, fill_size

    def _check_ask_fill(self, ask_price: float, ask_size: float,
                         trades_this_second: Optional[pd.DataFrame]) -> tuple:
        """
        An ask (sell limit order) at ask_price is filled if a real trade
        occurred at price >= ask_price (buyer lifted our offer).

        Returns (filled: bool, fill_size: float)
        """
        if trades_this_second is None or len(trades_this_second) == 0:
            return False, 0.0

        matching = trades_this_second[trades_this_second['price'] >= ask_price]
        if len(matching) == 0:
            return False, 0.0

        available_volume = matching['size'].sum()
        fill_size = min(ask_size, available_volume)
        return True, fill_size

    def run(self,
            strategy,
            day_data: dict,
            inventory_model,
            strategy_type: str = 'optimal') -> SimulationResult:
        """
        Run one trading day using real TAQ data.

        Parameters
        ----------
        strategy       : AvellanedaStoikov or BaselineStrategy instance
        day_data       : dict from WRDSLoader._process_day(), must contain:
                         'mid', 'best_bid', 'best_ask', 'market_trades'
        inventory_model: InventoryModel instance
        strategy_type  : 'optimal' (uses quotes_calibrated) or 'baseline'

        Returns
        -------
        SimulationResult
        """
        mid_prices = day_data['mid']
        best_bids  = day_data['best_bid']
        best_asks  = day_data['best_ask']
        market_trades = day_data['market_trades']

        n_steps = len(mid_prices) - 1
        T = self.T_seconds

        # Pre-index trades by second for fast lookup
        trade_index = self._build_trade_index(market_trades)

        # State
        cash      = 0.0
        inventory = 0.0
        pnl_arr   = np.zeros(n_steps + 1)
        inv_arr   = np.zeros(n_steps + 1)
        bid_arr   = np.full(n_steps + 1, np.nan)
        ask_arr   = np.full(n_steps + 1, np.nan)

        fills: List[Fill] = []
        n_buy_orders  = 0
        n_sell_orders = 0
        shares_bought = 0.0
        shares_sold   = 0.0
        n_quotes      = 0

        # Order book state machine (same logic as Algorithm 1 in paper)
        active_bid: Optional[dict] = None
        active_ask: Optional[dict] = None
        last_quote_time = -np.inf
        last_exec_time  = -np.inf
        one_side_filled = False

        for step in range(n_steps):
            t      = step * self.dt
            t_norm = t / T

            s        = mid_prices[step]
            best_bid = best_bids[step]
            best_ask = best_asks[step]

            # Trades that actually happened in this second
            sec_int = int(t)
            trades_now = trade_index.get(sec_int, None)

            # ----------------------------------------------------------------
            # Algorithm 1: quote management state machine
            # ----------------------------------------------------------------
            n_active = (active_bid is not None) + (active_ask is not None)

            if n_active == 0:
                # Place new bid and ask
                phi_bid, phi_ask = inventory_model.order_sizes(inventory)

                if strategy_type == 'optimal':
                    bid_q, ask_q = strategy.quotes_calibrated(s, inventory, t_norm)
                else:
                    bid_q, ask_q = strategy.quotes(best_bid, best_ask)

                # Sanity check: don't post crossed quotes
                if bid_q < ask_q:
                    active_bid = {'price': bid_q, 'size': phi_bid, 'placed_at': t}
                    active_ask = {'price': ask_q, 'size': phi_ask, 'placed_at': t}
                    last_quote_time = t
                    n_quotes += 1
                    one_side_filled = False

            elif n_active == 1:
                # Waiting for the outstanding order
                if t - last_exec_time > self.waiting_time:
                    # Cancel and re-quote
                    active_bid = None
                    active_ask = None
                    one_side_filled = False

            elif n_active == 2:
                # Refresh quotes every update_time seconds
                if t - last_quote_time > self.update_time:
                    phi_bid, phi_ask = inventory_model.order_sizes(inventory)

                    if strategy_type == 'optimal':
                        bid_q, ask_q = strategy.quotes_calibrated(s, inventory, t_norm)
                    else:
                        bid_q, ask_q = strategy.quotes(best_bid, best_ask)

                    if bid_q < ask_q:
                        active_bid = {'price': bid_q, 'size': phi_bid, 'placed_at': t}
                        active_ask = {'price': ask_q, 'size': phi_ask, 'placed_at': t}
                        last_quote_time = t
                        n_quotes += 1

            # ----------------------------------------------------------------
            # Check fills against real market trades
            # ----------------------------------------------------------------
            if active_bid is not None:
                filled, fill_size = self._check_bid_fill(
                    active_bid['price'], active_bid['size'], trades_now
                )
                if filled and fill_size > 0:
                    cash      -= active_bid['price'] * fill_size
                    inventory += fill_size
                    shares_bought += fill_size
                    n_buy_orders  += 1
                    fills.append(Fill('bid', active_bid['price'], fill_size, t))
                    active_bid     = None
                    last_exec_time = t
                    one_side_filled = True

            if active_ask is not None:
                filled, fill_size = self._check_ask_fill(
                    active_ask['price'], active_ask['size'], trades_now
                )
                if filled and fill_size > 0:
                    cash      += active_ask['price'] * fill_size
                    inventory -= fill_size
                    shares_sold   += fill_size
                    n_sell_orders += 1
                    fills.append(Fill('ask', active_ask['price'], fill_size, t))
                    active_ask     = None
                    last_exec_time = t
                    one_side_filled = True

            # Mark-to-market PnL
            pnl_arr[step] = cash + inventory * s
            inv_arr[step] = inventory
            bid_arr[step] = active_bid['price'] if active_bid else np.nan
            ask_arr[step] = active_ask['price'] if active_ask else np.nan

        # Final step
        pnl_arr[n_steps] = cash + inventory * mid_prices[n_steps]
        inv_arr[n_steps] = inventory
        bid_arr[n_steps] = bid_arr[n_steps - 1]
        ask_arr[n_steps] = ask_arr[n_steps - 1]

        times = np.arange(n_steps + 1) * self.dt

        return SimulationResult(
            times=times,
            mid_prices=mid_prices,
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
