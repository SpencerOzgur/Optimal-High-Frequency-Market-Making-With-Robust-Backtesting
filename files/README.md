[README.md](https://github.com/user-attachments/files/27060463/README.md)
# Avellaneda-Stoikov Market Making

Python implementation of **"Optimal High-Frequency Market Making"**
(Fushimi, González Rojas & Herman, 2018), which implements and extends
the Avellaneda & Stoikov (2008) pricing model.

Uses real TAQ millisecond data from WRDS rather than synthetic order arrivals.

---

## Project Structure

```
avellaneda_stoikov/
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── market_maker.py       # A-S pricing model + inventory control
    ├── helpers.py            # Summary tables + Markov chain analysis
    ├── wrds_loader.py        # TAQ data loader (quotes + trades from WRDS)
    ├── replay_simulator.py   # Event-driven simulator using real TAQ trades
    ├── run_with_wrds.py      # ← entry point
    └── analysis.py           # Plots replicating paper figures
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/avellaneda_stoikov.git
cd avellaneda_stoikov

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### WRDS credentials
On first run, `wrds.Connection()` will prompt for your WRDS username and
password and offer to save them to `~/.pgpass` — accept this so you aren't
prompted on every run. Your credentials are never stored in the repo
(`.pgpass` is in `.gitignore`).

If you're running on a new machine and want to pre-configure credentials,
create `~/.pgpass` manually:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
```

Then set permissions (Mac/Linux only):
```bash
chmod 600 ~/.pgpass
```

---

## Quick Start

```bash
python src/run_with_wrds.py
```

This will:
1. Connect to WRDS and pull TAQ quote and trade data for all 5 stocks over the week of June 12-16 2017
2. Run both the optimal (A-S) and baseline strategies through the replay simulator
3. Print Tables 2-8 from the paper to the console
4. Save plots to `results/`

### Use individual components
```python
from src.market_maker import AvellanedaStoikov, InventoryModel

# Build and calibrate the A-S model from observed open/close spreads
model = AvellanedaStoikov(gamma=0.1, sigma=0.03, kappa=2.0, T=1.0)
model.calibrate_from_market(open_spread=0.05, close_spread=0.01)

# Get optimal quotes at t=0.5 (midday), inventory q=10 shares, mid=$155
bid, ask = model.quotes_calibrated(s=155.0, q=10, t=0.5)
print(f"Bid: {bid:.4f}  Ask: {ask:.4f}  Spread: {ask-bid:.4f}")

# Dynamic order sizing
inv_model = InventoryModel(phi_max=100, eta=0.005)
phi_bid, phi_ask = inv_model.order_sizes(q=10)   # long: buy size reduced
print(f"Bid size: {phi_bid:.1f}  Ask size: {phi_ask:.1f}")
```

---

## Key Model Components

### 1. Pricing (Section 2.1)
The **indifference (reservation) price** shifts the mid-price based on
current inventory to reduce directional risk:

```
r(s, t) = s - q * gamma * sigma^2 * (T - t)
```

The **optimal spread** decreases linearly toward close so the market maker
becomes more aggressive to unwind inventory before the session ends:

```
delta_a + delta_b = gamma*sigma^2*(T - t) + ln(1 + gamma/kappa)
```

Calibration fits slope `A = gamma*sigma^2` and intercept `B = ln(1 + gamma/kappa)`
directly from the observed open/close spreads in Table 1 of the paper.

### 2. Inventory Control (Section 2.2)
Instead of halting when inventory limits are breached (Guéant et al. 2013),
order sizes are dynamically shrunk in the direction of excess accumulation:

```
phi_bid = phi_max * exp(-eta * q)   if q > 0  (already long:  reduce buys)
phi_bid = phi_max                   if q <= 0

phi_ask = phi_max * exp(eta * q)    if q < 0  (already short: reduce sells)
phi_ask = phi_max                   if q >= 0
```

### 3. Simulator — Event-Driven TAQ Replay
Rather than generating synthetic order arrivals (as in the original paper,
which lacked tick data), this implementation replays actual TAQ trades from WRDS.

**Data sources (both from `taqmsec` on WRDS):**
- `cqm` (Consolidated Quotes): provides `best_bid` and `best_ask` at every
  quote update, resampled to a 1-second grid via forward-fill
- `ctm` (Consolidated Trades): provides every executed trade with price and
  size, used directly to determine whether our limit orders get filled

**Fill logic:** at each second, a limit order is filled if a real market
order crossed its price level:
- Bid at price P is filled if any trade occurred at price <= P
- Ask at price P is filled if any trade occurred at price >= P
- Fill size = min(our order size, volume traded at that level)

This replaces the paper's Poisson arrival model and Gamma partial-fill draws
entirely — no distributional assumptions, no parameters to estimate.

**Mid-price** is derived as `(best_bid + best_ask) / 2` from the quote data.
**Sigma** is estimated from realised 1-second log-returns of trade prices.

### 4. Markov Chain Analysis (Section 4.2)
States: `{Quoting, Waiting, Spread}`

Probability of capturing the spread per quote cycle:
```
p* = p(0,2) + sum_{n=0}^{5} p(0,1) * p(1,1)^n * p(1,2)
```

Probability of a one-sided fill (increases inventory risk):
```
q* = p(0,1) + p(0,1) * p(1,1)^5 * p(1,0)
```

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `phi_max` | 100 shares | Maximum order size |
| `eta` | 0.005 | Inventory size decay shape |
| `dt` | 1 second | Simulation time step |
| `waiting_time` | 5 seconds | Cancel outstanding order after one-sided fill |
| `update_time` | 1 second | Quote refresh interval with 2 orders in book |

---

## Stocks Traded (Table 1 of paper)

| Ticker | Volume | Performance | Open Spread | Close Spread |
|--------|--------|-------------|-------------|--------------|
| AAPL | High | High | $0.05 | $0.01 |
| AMZN | Low  | High | $0.49 | $0.56 |
| GE   | High | Low  | $0.04 | $0.01 |
| IVV  | Low  | High | $0.03 | $0.01 |
| M    | Low  | Low  | $0.09 | $0.01 |

---

## Extending the Project

1. **Different date range**: change `DATES` in `run_with_wrds.py`
2. **Different stocks**: add tickers and spread params to `SPREAD_PARAMS` in `run_with_wrds.py`
3. **Guéant et al. (2013)**: implement the closed-form solution with hard inventory limits and compare against A-S
4. **Transaction costs**: add maker rebates and taker fees to the cash process in `replay_simulator.py`
5. **Mid-price signal**: add a short-term drift predictor to shift the indifference price in `market_maker.py`

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217-224.
- Guéant, O., Lehalle, C-A., & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk.* Mathematics and Financial Economics, 7(4), 477-507.
- Ho, T. & Stoll, H.R. (1981). *Optimal dealer pricing under transactions and return uncertainty.* Journal of Financial Economics, 9, 47-73.
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
