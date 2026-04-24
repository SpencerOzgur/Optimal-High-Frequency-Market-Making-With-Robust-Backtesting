# Avellaneda-Stoikov Market Making

Python implementation of **"Optimal High-Frequency Market Making"**
(Fushimi, González Rojas & Herman, 2018), which implements and extends
the Avellaneda & Stoikov (2008) pricing model.

---

## Project Structure

```
avellaneda_stoikov/
├── src/
│   ├── market_maker.py     # A-S pricing model + inventory control model
│   ├── simulator.py        # Trading simulator (Poisson arrivals, partial fills)
│   ├── run_experiment.py   # Replicates paper's Tables 1-8 + Markov chain analysis
│   └── analysis.py         # Generates plots replicating Figures 1-5
└── results/                # Output plots saved here
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

### Run the full experiment (all 5 stocks, 5 days each)
```python
from src.run_experiment import run_all_stocks
results = run_all_stocks()  # prints Tables 2-8 to console
```

### Generate all plots
```python
from src.run_experiment import run_all_stocks
from src.analysis import plot_all

results = run_all_stocks()
plot_all(results)           # saves PNGs to results/
```

### Use individual components
```python
from src.market_maker import AvellanedaStoikov, InventoryModel

# Build and calibrate the A-S model
model = AvellanedaStoikov(gamma=0.1, sigma=0.03, kappa=2.0, T=1.0)
model.calibrate_from_market(open_spread=0.05, close_spread=0.01)

# Get optimal quotes at t=0.5 (midday), inventory q=10 shares, mid=155.0
bid, ask = model.quotes_calibrated(s=155.0, q=10, t=0.5)
print(f"Bid: {bid:.4f}, Ask: {ask:.4f}, Spread: {ask-bid:.4f}")

# Dynamic order sizing
inv_model = InventoryModel(phi_max=100, eta=0.005)
phi_bid, phi_ask = inv_model.order_sizes(q=10)   # long: reduce bid size
print(f"Bid size: {phi_bid:.1f}, Ask size: {phi_ask:.1f}")
```

---

## Key Model Components

### 1. Pricing (Section 2.1)
The **indifference (reservation) price** shifts the mid away from the current
position to reduce inventory risk:

```
r(s, t) = s - q · γ · σ² · (T - t)
```

The **optimal spread** decreases linearly toward close (γ > 0), so the MM
becomes more aggressive to unwind inventory before the session ends:

```
δᵃ + δᵇ = γσ²(T-t) + ln(1 + γ/κ)
```

Calibration fits the slope `A = γσ²` and intercept `B = ln(1+γ/κ)` from the
observed open/close spreads in Table 1.

### 2. Inventory Control (Section 2.2)
Instead of halting when limits are breached (Guéant et al. 2013), order
sizes are dynamically shrunk in the direction of excess inventory:

```
φ_bid = φ_max · exp(-η · q)   if q > 0  (already long: reduce buys)
φ_bid = φ_max                 if q ≤ 0

φ_ask = φ_max · exp(η · q)    if q < 0  (already short: reduce sells)
φ_ask = φ_max                 if q ≥ 0
```

### 3. Trading Simulator (Section 3)
- **Mid-price**: arithmetic Brownian motion `dS = σ dW`
- **Order arrivals**: time-inhomogeneous Poisson with bathtub time profile
- **Execution intensity**: `λ(t, ξ) = α_t · exp(-μ · ξ)`
- **Partial fills**: `Y ~ Gamma(κ, θ)`; fill size = `min(Y, 1) × order_size`

### 4. Markov Chain Analysis (Section 4.2)
States: `{Quoting, Waiting, Spread}`

The spread-capture probability is:
```
p* = p(0,2) + Σₙ₌₀⁵ p(0,1) · p(1,1)ⁿ · p(1,2)
```

---

## Parameters Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| `phi_max` | 100   | Max order size (shares) |
| `eta`     | 0.005 | Inventory shape parameter |
| `mu`      | 100   | Depth decay in intensity |
| `kappa_fill` | 2  | Gamma shape for partial fills |
| `theta_fill` | 1/1.65 | Gamma scale |
| `dt`      | 1 sec | Time step |
| `waiting_time` | 5 sec | Cancel wait after one-sided fill |
| `update_time`  | 1 sec | Quote refresh interval |

---

## Extending the Project

Some ideas for further development:

1. **Real data**: plug in actual TAQ or Thesys data by replacing `simulate_market_data()` in `run_experiment.py`
2. **Guéant et al. (2013)**: implement the closed-form solution with hard inventory limits and compare
3. **Mid-price prediction**: add a drift term `μ dt` to the Brownian motion, calibrated from a short-term signal
4. **Transaction costs**: add maker rebates and taker fees to the cash process
5. **Multi-asset**: extend the model to correlated assets with a joint inventory penalty

---

## References

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
- Guéant, O., Lehalle, C-A., & Fernandez-Tapia, J. (2013). *Dealing with the inventory risk.* Mathematics and Financial Economics, 7(4), 477–507.
- Ho, T. & Stoll, H.R. (1981). *Optimal dealer pricing under transactions and return uncertainty.* Journal of Financial Economics, 9, 47–73.
- Cartea, A., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.
