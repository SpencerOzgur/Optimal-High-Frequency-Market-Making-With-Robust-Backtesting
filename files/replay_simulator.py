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

Two queue-position models are supported:

  queue_model='front'  (best-case, optimistic)
    Equivalent to first-in-line. Any addressable print at price <= our bid
    (>= our ask) fills us up to min(our_size, sum_of_print_sizes). Ignores
    pre-existing displayed depth at our level.

  queue_model='back'   (worst-case, conservative)
    Last-in-line. At quote-post time we record `queue_ahead` = NBBO size
    at our quote price (0 if we improve the NBBO; +inf if our quote is
    outside the NBBO). Each addressable print first decrements queue_ahead
    until it hits 0; only the residual reaches us, capped by our_size.

Running both gives a best/worst execution band for the same strategy.
"""

import math
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

    @staticmethod
    def _initial_queue_ahead(our_price: float, best_price: float,
                              best_size: float, side: str) -> float:
        """
        Queue depth ahead of us when posting `our_price`, given the prevailing
        NBBO (`best_price`, `best_size`). Used only under queue_model='back'.

          - improved the NBBO  -> 0     (we are alone at the new top)
          - joined the NBBO    -> best_size  (last in line)
          - outside the NBBO   -> +inf  (off the protected quote; cannot fill)

        side is 'bid' or 'ask' so the comparison direction is correct.
        """
        if side == 'bid':
            improved = our_price > best_price
            joined   = our_price == best_price
        else:  # 'ask'
            improved = our_price < best_price
            joined   = our_price == best_price

        if improved:
            return 0.0
        if joined:
            return float(best_size) if best_size and not math.isnan(best_size) else 0.0
        return math.inf

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
                         trades_this_second: Optional[pd.DataFrame],
                         queue_ahead: float,
                         queue_model: str) -> tuple:
        """
        A bid (buy limit order) at bid_price is filled if a real trade
        occurred at price <= bid_price.

        Returns (filled: bool, fill_size: float, queue_ahead: float).
        queue_ahead is updated under 'back' mode (decremented by addressable
        size that didn't reach us); under 'front' mode it is passed through.
        """
        if trades_this_second is None or len(trades_this_second) == 0:
            return False, 0.0, queue_ahead

        matching = trades_this_second[trades_this_second['price'] <= bid_price]
        if len(matching) == 0:
            return False, 0.0, queue_ahead

        addressable = float(matching['size'].sum())

        if queue_model == 'back':
            consumed = min(queue_ahead, addressable)
            queue_ahead -= consumed
            remaining = addressable - consumed
            fill_size = min(bid_size, remaining)
        else:
            fill_size = min(bid_size, addressable)

        return fill_size > 0, fill_size, queue_ahead

    def _check_ask_fill(self, ask_price: float, ask_size: float,
                         trades_this_second: Optional[pd.DataFrame],
                         queue_ahead: float,
                         queue_model: str) -> tuple:
        """
        An ask (sell limit order) at ask_price is filled if a real trade
        occurred at price >= ask_price.

        Returns (filled: bool, fill_size: float, queue_ahead: float).
        """
        if trades_this_second is None or len(trades_this_second) == 0:
            return False, 0.0, queue_ahead

        matching = trades_this_second[trades_this_second['price'] >= ask_price]
        if len(matching) == 0:
            return False, 0.0, queue_ahead

        addressable = float(matching['size'].sum())

        if queue_model == 'back':
            consumed = min(queue_ahead, addressable)
            queue_ahead -= consumed
            remaining = addressable - consumed
            fill_size = min(ask_size, remaining)
        else:
            fill_size = min(ask_size, addressable)

        return fill_size > 0, fill_size, queue_ahead

    def run(self,
            strategy,
            day_data: dict,
            inventory_model,
            strategy_type: str = 'optimal',
            queue_model: str = 'front') -> SimulationResult:
        """
        Run one trading day using real TAQ data.

        Parameters
        ----------
        strategy       : AvellanedaStoikov or BaselineStrategy instance
        day_data       : dict from WRDSLoader._process_day(), must contain:
                         'mid', 'best_bid', 'best_ask',
                         'best_bid_size', 'best_ask_size', 'market_trades'
        inventory_model: InventoryModel instance
        strategy_type  : 'optimal' (uses quotes_calibrated) or 'baseline'
        queue_model    : 'front' (best-case, current default) or 'back'
                         (last-in-line at NBBO size)

        Returns
        -------
        SimulationResult
        """
        if queue_model not in ('front', 'back'):
            raise ValueError(f"queue_model must be 'front' or 'back', got {queue_model!r}")

        mid_prices = day_data['mid']
        best_bids  = day_data['best_bid']
        best_asks  = day_data['best_ask']
        best_bid_sizes = day_data.get('best_bid_size')
        best_ask_sizes = day_data.get('best_ask_size')
        if best_bid_sizes is None or best_ask_sizes is None:
            best_bid_sizes = np.zeros_like(best_bids)
            best_ask_sizes = np.zeros_like(best_asks)
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
                    qa_bid = self._initial_queue_ahead(
                        bid_q, best_bid, best_bid_sizes[step], 'bid')
                    qa_ask = self._initial_queue_ahead(
                        ask_q, best_ask, best_ask_sizes[step], 'ask')
                    active_bid = {'price': bid_q, 'size': phi_bid,
                                  'placed_at': t, 'queue_ahead': qa_bid}
                    active_ask = {'price': ask_q, 'size': phi_ask,
                                  'placed_at': t, 'queue_ahead': qa_ask}
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
                        qa_bid = self._initial_queue_ahead(
                            bid_q, best_bid, best_bid_sizes[step], 'bid')
                        qa_ask = self._initial_queue_ahead(
                            ask_q, best_ask, best_ask_sizes[step], 'ask')
                        active_bid = {'price': bid_q, 'size': phi_bid,
                                      'placed_at': t, 'queue_ahead': qa_bid}
                        active_ask = {'price': ask_q, 'size': phi_ask,
                                      'placed_at': t, 'queue_ahead': qa_ask}
                        last_quote_time = t
                        n_quotes += 1

            # ----------------------------------------------------------------
            # Check fills against real market trades
            # ----------------------------------------------------------------
            if active_bid is not None:
                filled, fill_size, qa_new = self._check_bid_fill(
                    active_bid['price'], active_bid['size'], trades_now,
                    active_bid['queue_ahead'], queue_model
                )
                active_bid['queue_ahead'] = qa_new
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
                filled, fill_size, qa_new = self._check_ask_fill(
                    active_ask['price'], active_ask['size'], trades_now,
                    active_ask['queue_ahead'], queue_model
                )
                active_ask['queue_ahead'] = qa_new
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
