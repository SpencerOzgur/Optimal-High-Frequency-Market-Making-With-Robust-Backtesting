"""
replay_simulator.py
===================
Event-driven simulator that replays actual TAQ market orders.

THREE FILL MODELS
-----------------
All three share the same price-crossing condition: a fill can only occur
if a real TAQ trade happened at or through our limit price. They differ
in how much of that volume we claim.

1. first_in_queue
   Assumes we are always at the front of the queue. We capture all volume
   traded at our price level up to our order size. This is the most
   optimistic model and the standard academic simplification.

   fill_size = min(our_size, volume_traded_at_our_price)

2. pro_rata
   Assumes we receive a proportional share of volume based on our order
   size relative to the total quoted size at that price level (bidsiz or
   asksiz from TAQ cqm). More conservative — directly addresses the queue
   priority problem without requiring full order book data.

   our_share = our_size / (our_size + quoted_size_at_level)
   fill_size = volume_traded_at_our_price * our_share

3. pro_rata_imbalance
   Extends pro_rata with an order book imbalance (OBI) adjustment. The
   OBI signal captures directional pressure:

       OBI = (bidsiz - asksiz) / (bidsiz + asksiz)  ∈ [-1, +1]

   High OBI (more bid volume) → buying pressure → ask fills more likely
   Low OBI  (more ask volume) → selling pressure → bid fills more likely

   The sensitivity of fills to OBI is controlled by alpha, estimated from
   a logistic regression on a calibration dataset:

       P(fill | ask) = sigmoid(beta_0 + alpha * OBI)
       P(fill | bid) = sigmoid(beta_0 - alpha * OBI)

   The regression is fit on the calibration data passed to
   calibrate_alpha(), then alpha scales a multiplier applied on top of
   the pro_rata fill size:

       ask_multiplier = 1 + alpha_scaled * OBI
       bid_multiplier = 1 - alpha_scaled * OBI

   alpha_scaled is normalised so the multiplier stays in [0.5, 1.5],
   preventing extreme values from dominating.

CALIBRATION
-----------
Call sim.calibrate_alpha(calibration_days) before running with
fill_model='pro_rata_imbalance'. calibration_days should be a list of
day dicts from a held-out period (e.g. the paper's original June 2017
week). If not calibrated, alpha defaults to 0.5.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict

try:
    from scipy.special import expit          # sigmoid
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
    fill_model: str          # which fill model was used — for comparison tables


# ---------------------------------------------------------------------------
# Replay simulator
# ---------------------------------------------------------------------------

class ReplaySimulator:
    """
    Simulates market making by replaying real TAQ trade data.

    Parameters
    ----------
    dt            : time step in seconds (use 1 to match TAQ resolution)
    T_seconds     : trading day length in seconds
    waiting_time  : seconds to wait for outstanding order after one-side fill
    update_time   : seconds between quote refreshes with 2 orders in book
    fill_model    : 'first_in_queue', 'pro_rata', or 'pro_rata_imbalance'
    """

    FILL_MODELS = ('first_in_queue', 'pro_rata', 'pro_rata_imbalance')

    def __init__(self,
                 dt: float = 1.0,
                 T_seconds: float = 23400.0,
                 waiting_time: float = 5.0,
                 update_time: float = 1.0,
                 fill_model: str = 'first_in_queue'):

        if fill_model not in self.FILL_MODELS:
            raise ValueError(f"fill_model must be one of {self.FILL_MODELS}")

        self.dt           = dt
        self.T_seconds    = T_seconds
        self.waiting_time = waiting_time
        self.update_time  = update_time
        self.fill_model   = fill_model
        self.alpha        = 0.5    # default; overwritten by calibrate_alpha()

    # ------------------------------------------------------------------
    # Alpha calibration (logistic regression on OBI vs fill outcomes)
    # ------------------------------------------------------------------

    def calibrate_alpha(self, calibration_days: List[dict]) -> float:
        """
        Estimate alpha from a held-out calibration dataset using logistic
        regression of fill outcomes on OBI.

        For each second in calibration_days:
          - Compute OBI = (bidsiz - asksiz) / (bidsiz + asksiz)
          - Label ask seconds: y=1 if any trade occurred at ask or above
          - Label bid seconds: y=1 if any trade occurred at bid or below
          - Fit: P(y=1) = sigmoid(beta_0 + alpha * OBI * direction)
            where direction = +1 for ask side, -1 for bid side

        Sets self.alpha and returns it.
        """
        if not SCIPY_AVAILABLE:
            print("  scipy not available — using default alpha=0.5")
            return self.alpha

        X = []   # OBI * direction
        y = []   # fill occurred (1) or not (0)

        for day in calibration_days:
            if day is None:
                continue

            bid_sizes    = day.get('bid_sizes', None)
            ask_sizes    = day.get('ask_sizes', None)
            best_bids    = day['best_bid']
            best_asks    = day['best_ask']
            market_trades = day['market_trades']

            if bid_sizes is None or ask_sizes is None:
                print("  Warning: bid_sizes/ask_sizes missing — skipping day")
                continue

            # Index trades by second
            trade_index = self._build_trade_index(market_trades)
            n_steps     = len(best_bids) - 1

            for step in range(n_steps):
                bs = float(bid_sizes[step])
                as_ = float(ask_sizes[step])
                total = bs + as_
                if total == 0:
                    continue

                obi = (bs - as_) / total
                trades_now = trade_index.get(step, None)

                # Ask side: trade at or above best ask
                ask_fill = 0
                bid_fill = 0
                if trades_now is not None and len(trades_now) > 0:
                    if (trades_now['price'] >= best_asks[step]).any():
                        ask_fill = 1
                    if (trades_now['price'] <= best_bids[step]).any():
                        bid_fill = 1

                # direction = +1 for ask, -1 for bid
                X.append(obi * 1.0)
                y.append(ask_fill)
                X.append(obi * -1.0)
                y.append(bid_fill)

        if len(X) < 100:
            print("  Warning: insufficient calibration data — using default alpha=0.5")
            return self.alpha

        X = np.array(X).reshape(-1, 1)
        y = np.array(y)

        # Logistic regression: minimise negative log-likelihood
        # params = [beta_0, alpha]
        def neg_log_likelihood(params):
            beta_0, alpha = params
            logits = beta_0 + alpha * X.ravel()
            # clip for numerical stability
            logits = np.clip(logits, -20, 20)
            probs  = 1 / (1 + np.exp(-logits))
            probs  = np.clip(probs, 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        result = minimize(neg_log_likelihood, x0=[0.0, 0.5],
                          method='L-BFGS-B',
                          bounds=[(-10, 10), (0.0, 10.0)])

        if result.success:
            self.alpha = float(result.x[1])
            print(f"  Calibrated alpha = {self.alpha:.4f} "
                  f"(beta_0 = {result.x[0]:.4f}, n={len(y):,} observations)")
        else:
            print(f"  Calibration did not converge — using default alpha=0.5")

        return self.alpha

    # ------------------------------------------------------------------
    # Trade index helper
    # ------------------------------------------------------------------

    def _build_trade_index(self, market_trades: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """Pre-index trades by integer second for O(1) lookup."""
        market_trades = market_trades.copy()
        market_trades['sec_int'] = market_trades['t_sec'].astype(int)
        return {
            sec: grp.reset_index(drop=True)
            for sec, grp in market_trades.groupby('sec_int')
        }

    # ------------------------------------------------------------------
    # Fill logic — three models
    # ------------------------------------------------------------------

    def _check_bid_fill(self,
                         bid_price: float,
                         bid_size: float,
                         trades_now: Optional[pd.DataFrame],
                         quoted_ask_size: float,
                         obi: float) -> tuple:
        """
        Check whether our bid order is filled this second.

        All three models require a real trade to have occurred at or
        below our bid price. They differ in how much volume we capture.

        Parameters
        ----------
        bid_price       : our limit bid price
        bid_size        : our order size
        trades_now      : TAQ trades in this second (or None)
        quoted_ask_size : shares quoted at best ask (for pro_rata denominator)
        obi             : order book imbalance = (bidsiz-asksiz)/(bidsiz+asksiz)

        Returns (filled: bool, fill_size: float)
        """
        if trades_now is None or len(trades_now) == 0:
            return False, 0.0

        matching = trades_now[trades_now['price'] <= bid_price]
        if len(matching) == 0:
            return False, 0.0

        available_volume = matching['size'].sum()

        if self.fill_model == 'first_in_queue':
            fill_size = min(bid_size, available_volume)

        elif self.fill_model == 'pro_rata':
            # Our share = our_size / (our_size + total quoted at this level)
            # Use ask_size as proxy for depth at our bid level
            total_depth = bid_size + max(quoted_ask_size, 1.0)
            our_share   = bid_size / total_depth
            fill_size   = available_volume * our_share
            fill_size   = min(fill_size, bid_size)

        else:  # pro_rata_imbalance
            # Pro-rata base
            total_depth = bid_size + max(quoted_ask_size, 1.0)
            our_share   = bid_size / total_depth
            fill_size   = available_volume * our_share

            # OBI adjustment: negative OBI (more ask pressure) → easier bid fills
            # Normalise alpha so multiplier stays in [0.5, 1.5]
            alpha_scaled = min(self.alpha, 1.0) * 0.5
            bid_multiplier = 1.0 - alpha_scaled * obi   # obi negative → multiplier > 1
            bid_multiplier = np.clip(bid_multiplier, 0.5, 1.5)
            fill_size = fill_size * bid_multiplier
            fill_size = min(fill_size, bid_size)

        return True, max(0.0, fill_size)

    def _check_ask_fill(self,
                         ask_price: float,
                         ask_size: float,
                         trades_now: Optional[pd.DataFrame],
                         quoted_bid_size: float,
                         obi: float) -> tuple:
        """
        Check whether our ask order is filled this second.

        Parameters
        ----------
        ask_price       : our limit ask price
        ask_size        : our order size
        trades_now      : TAQ trades in this second (or None)
        quoted_bid_size : shares quoted at best bid (for pro_rata denominator)
        obi             : order book imbalance = (bidsiz-asksiz)/(bidsiz+asksiz)

        Returns (filled: bool, fill_size: float)
        """
        if trades_now is None or len(trades_now) == 0:
            return False, 0.0

        matching = trades_now[trades_now['price'] >= ask_price]
        if len(matching) == 0:
            return False, 0.0

        available_volume = matching['size'].sum()

        if self.fill_model == 'first_in_queue':
            fill_size = min(ask_size, available_volume)

        elif self.fill_model == 'pro_rata':
            total_depth = ask_size + max(quoted_bid_size, 1.0)
            our_share   = ask_size / total_depth
            fill_size   = available_volume * our_share
            fill_size   = min(fill_size, ask_size)

        else:  # pro_rata_imbalance
            total_depth = ask_size + max(quoted_bid_size, 1.0)
            our_share   = ask_size / total_depth
            fill_size   = available_volume * our_share

            # OBI adjustment: positive OBI (more bid pressure) → easier ask fills
            alpha_scaled   = min(self.alpha, 1.0) * 0.5
            ask_multiplier = 1.0 + alpha_scaled * obi   # obi positive → multiplier > 1
            ask_multiplier = np.clip(ask_multiplier, 0.5, 1.5)
            fill_size = fill_size * ask_multiplier
            fill_size = min(fill_size, ask_size)

        return True, max(0.0, fill_size)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

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
                         'mid', 'best_bid', 'best_ask', 'market_trades',
                         'bid_sizes', 'ask_sizes'
        inventory_model: InventoryModel instance
        strategy_type  : 'optimal' (uses quotes_calibrated) or 'baseline'

        Returns
        -------
        SimulationResult
        """
        mid_prices    = day_data['mid']
        best_bids     = day_data['best_bid']
        best_asks     = day_data['best_ask']
        market_trades = day_data['market_trades']

        # bid_sizes/ask_sizes needed for pro_rata models; fall back to ones
        # if missing (e.g. old data without these columns)
        bid_sizes = day_data.get('bid_sizes', np.ones(len(mid_prices)))
        ask_sizes = day_data.get('ask_sizes', np.ones(len(mid_prices)))

        n_steps = len(mid_prices) - 1
        T       = self.T_seconds

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
            bs       = float(bid_sizes[step])
            as_      = float(ask_sizes[step])

            # Order book imbalance for this second
            total = bs + as_
            obi   = (bs - as_) / total if total > 0 else 0.0

            trades_now = trade_index.get(int(t), None)

            # ----------------------------------------------------------------
            # Algorithm 1: quote management state machine
            # ----------------------------------------------------------------
            n_active = (active_bid is not None) + (active_ask is not None)

            if n_active == 0:
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
                    one_side_filled = False

            elif n_active == 1:
                if t - last_exec_time > self.waiting_time:
                    active_bid = None
                    active_ask = None
                    one_side_filled = False

            elif n_active == 2:
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
            # Check fills
            # ----------------------------------------------------------------
            if active_bid is not None:
                filled, fill_size = self._check_bid_fill(
                    active_bid['price'], active_bid['size'],
                    trades_now, as_, obi
                )
                if filled and fill_size > 0:
                    cash          -= active_bid['price'] * fill_size
                    inventory     += fill_size
                    shares_bought += fill_size
                    n_buy_orders  += 1
                    fills.append(Fill('bid', active_bid['price'], fill_size, t))
                    active_bid      = None
                    last_exec_time  = t
                    one_side_filled = True

            if active_ask is not None:
                filled, fill_size = self._check_ask_fill(
                    active_ask['price'], active_ask['size'],
                    trades_now, bs, obi
                )
                if filled and fill_size > 0:
                    cash          += active_ask['price'] * fill_size
                    inventory     -= fill_size
                    shares_sold   += fill_size
                    n_sell_orders += 1
                    fills.append(Fill('ask', active_ask['price'], fill_size, t))
                    active_ask      = None
                    last_exec_time  = t
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

        return SimulationResult(
            times=np.arange(n_steps + 1) * self.dt,
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
            fill_model=self.fill_model,
        )
