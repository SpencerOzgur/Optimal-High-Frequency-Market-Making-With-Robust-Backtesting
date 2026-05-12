"""
market_maker.py
===============
Implements the Avellaneda-Stoikov (2008) high-frequency market making model
combined with a proprietary dynamic order-size inventory control model.

Key equations from the paper:
  Indifference price:  r(s,t) = s - q * gamma * sigma^2 * (T - t)
  Optimal spread:      delta_a + delta_b = gamma*sigma^2*(T-t) + ln(1 + gamma/kappa)
  Order size (bid):    phi_bid = phi_max * exp(-eta * q)  if q > 0, else phi_max
  Order size (ask):    phi_ask = phi_max * exp( eta * q)  if q < 0, else phi_max
"""

import numpy as np


# Section 2.2 inventory order-size framework toggles.
# Optimal and baseline are independent so you can A/B-test the framework
# while holding the other side fixed.
INVENTORY_MODEL_ENABLED          = True   # applies to the optimal strategy
BASELINE_INVENTORY_MODEL_ENABLED = True    # applies to the baseline strategy


class AvellanedaStoikov:
    """
    Optimal bid/ask quote-setting strategy from Avellaneda & Stoikov (2008).

    Parameters
    ----------
    gamma : float
        Risk-aversion parameter. Higher gamma -> wider spread, more conservative.
    sigma : float
        Volatility of the mid-price (annualised or per-second, must match T units).
    kappa : float
        Order-book depth parameter used in spread calibration via B = ln(1 + gamma/kappa).
        Not used as a Poisson intensity parameter in this implementation — we use
        real TAQ trades for fills instead.
    T : float
        Terminal time horizon (in the same units as t). Typically 1 trading day = 1.0.
    """

    def __init__(self, gamma: float, sigma: float, kappa: float, T: float = 1.0):
        self.gamma = gamma
        self.sigma = sigma
        self.kappa = kappa
        self.T = T

    def indifference_price(self, s: float, q: float, t: float) -> float:
        gs2 = getattr(self, 'gamma_sigma2', self.gamma * self.sigma ** 2)
        return s - q * gs2 * (self.T - t)

    def optimal_spread(self, t: float) -> float:
        gs2 = getattr(self, 'gamma_sigma2', self.gamma * self.sigma ** 2)
        time_component = gs2 * (self.T - t)
        # print(f'sigma: {self.sigma}, gamma: {self.gamma}, gs2: {gs2}, self.T: {self.T}, t {t}, time_component {time_component}')
        closing_spread = (2.0 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return time_component + closing_spread

    def quotes(self, s: float, q: float, t: float):
        """
        Compute optimal bid and ask prices.

        Returns
        -------
        bid_price, ask_price : float, float
        """
        r = self.indifference_price(s, q, t)
        half_spread = self.optimal_spread(t) / 2.0
        bid_price = r - half_spread
        ask_price = r + half_spread
        return bid_price, ask_price

    def calibrate_from_market(self, open_spread: float, close_spread: float):
        """
        Calibrate (gamma * sigma^2) and kappa from observed opening/closing spreads.

        The spread equation is linear in (T - t):
            spread(t) = A*(T - t) + B
        where:
            A = gamma * sigma^2
            B = ln(1 + gamma/kappa)

        At t=0 (open):  spread = A*T + B = open_spread
        At t=T (close): spread = B       = close_spread

        So:  B = close_spread
             A = (open_spread - close_spread) / T
        """
        B = close_spread
        A = (open_spread - close_spread) / self.T

        # Store calibrated slope and intercept
        self.A = max(A, 0.0)   # ensure non-negative
        self.B = B

        # Back out effective gamma*sigma^2 = A
        # We keep gamma and sigma separately but rescale sigma so A holds
        self.gamma_sigma2 = self.A

        # Back out kappa from B = ln(1 + gamma/kappa)
        # => gamma/kappa = exp(B) - 1  => kappa = gamma / (exp(B) - 1)
        exp_term = np.exp(B * self.gamma / 2.0)
        if exp_term > 1:
            self.kappa = self.gamma / (exp_term - 1)
        # else kappa stays as initialised

        return self.A, self.B

    def optimal_spread_calibrated(self, t: float) -> float:
        """
        Spread using calibrated A and B (call calibrate_from_market first).
        """
        if hasattr(self, 'A'):
            return self.A * (self.T - t) + self.B
        return self.optimal_spread(t)

    def quotes_calibrated(self, s: float, q: float, t: float):
        """
        Quotes using calibrated parameters.
        """
        r = self.indifference_price(s, q, t)
        half_spread = self.optimal_spread(t) / 2.0
        bid_price = r - half_spread
        ask_price = r + half_spread
        return bid_price, ask_price


class InventoryModel:
    """
    Proprietary dynamic order-size model to control inventory risk.

    Unlike Guéant et al. (2013) who stop quoting at inventory limits, this model
    *keeps trading* but reduces order size in the direction of excess accumulation.

    Order size equations:
        phi_bid = phi_max * exp(-eta * q)   if q > 0  (long: reduce buy size)
               = phi_max                   if q <= 0

        phi_ask = phi_max * exp(eta * q)    if q < 0  (short: q<0 so eta*q<0, exp<1, reduces size)
               = phi_max                   if q >= 0

    With eta = 0.005 (positive), exp(-eta*q) < 1 when q > 0, so bid size
    shrinks as long inventory grows, and vice versa for asks.

    Parameters
    ----------
    phi_max : float
        Maximum order size (shares). Paper uses 100.
    eta : float
        Shape parameter controlling how quickly size decays with inventory.
        Paper uses eta = 0.005.
    enabled : bool
        If False, bid_size and ask_size always return phi_max regardless of q
        (i.e. the Section 2.2 framework is disabled and order size is constant).
        Useful for A/B testing the inventory framework.
    """

    def __init__(self, phi_max: float = 100.0, eta: float = 0.005,
                 enabled: bool = INVENTORY_MODEL_ENABLED):
        self.phi_max = phi_max
        self.eta = eta
        self.enabled = enabled

    def bid_size(self, q: float) -> float:
        """
        Bid (buy) order size. Reduced when already long (q > 0).
        """
        if not self.enabled or q <= 0:
            return self.phi_max
        return self.phi_max * np.exp(-self.eta * q)

    def ask_size(self, q: float) -> float:
        """
        Ask (sell) order size. Reduced when already short (q < 0).
        """
        if not self.enabled or q >= 0:
            return self.phi_max
        return self.phi_max * np.exp(self.eta * q)   # q<0, so eta*q<0, exp<1

    def order_sizes(self, q: float):
        """
        Return (bid_size, ask_size) for current inventory q.
        """
        return self.bid_size(q), self.ask_size(q)


class BaselineStrategy:
    """
    Baseline: always quote at the current best bid and ask in the order book.
    No inventory adjustment to spread; identical inventory model.
    """

    def quotes(self, best_bid: float, best_ask: float):
        """Simply return the best bid/ask as our quotes."""
        return best_bid, best_ask
