# Optimal High-Frequency Market Making With Robust Backtesting

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Status-Research%20Framework-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/Focus-Market%20Microstructure-black?style=for-the-badge">
  <img src="https://img.shields.io/badge/Data-WRDS%20TAQ-darkred?style=for-the-badge">
  <img src="https://img.shields.io/badge/Model-Avellaneda--Stoikov-darkgreen?style=for-the-badge">
  <img src="https://img.shields.io/badge/Backtesting-Replay%20Simulation-purple?style=for-the-badge">
</p>

<p align="left">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=flat-square">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white">
  <img src="https://img.shields.io/badge/Quantitative%20Finance-black?style=flat-square">
  <img src="https://img.shields.io/badge/Algorithmic%20Trading-black?style=flat-square">
</p>

Implementation and empirical evaluation of the Avellaneda-Stoikov (2008) market making model, extended with a dynamic inventory control framework and tested against real TAQ data from WRDS.

Based on: *Optimal High-Frequency Trading in a Pro-Rata Microstructure with Predictive Information* (Stanford, 2018).

---

## Project Structure

```
.
├── source/                         # Core library
│   ├── market_maker.py             # A-S model, inventory model, baseline strategy
│   ├── replay_simulator.py         # Event-driven simulator using real TAQ trades
│   ├── wrds_loader.py              # WRDS TAQ data loader and calibration helpers
│   ├── helpers.py                  # Summary tables, fill analysis, xlsx export
│   ├── analysis.py                 # All plots (figures 1–5 + diagnostics)
│   ├── realized_vol_strategy.py    # Rolling realized vol extension of A-S
│   ├── calibrate_params.py         # Calibrate kappa, A, b from prior week
│   └── data/                       # Cached pkl files (gitignored)
│       ├── raw_data.pkl
│       ├── raw_data_calib.pkl
│       └── results.pkl
│
├── scripts/                        # Entry-point runners
│   ├── run_with_wrds.py            # Main empirical run (TAQ replay)
│   ├── run_synthetic.py            # Synthetic run (paper-style Poisson fills)
│   ├── run_realized_vol.py         # Realized vol experiment runner
│
├── plots/                          # Generated figures (gitignored)
│   ├── wrds/                       # Figures from TAQ replay run
│   ├── wrds_back/                  # Same figures, back-of-queue model
│   └── synthetic/                  # Figures from Poisson simulator
│
└── sheets/                         # Excel outputs (gitignored)
    ├── fill_stats.xlsx
    └── quote_position.xlsx
```

---

## Setup

### Requirements

```bash
pip install numpy pandas scipy matplotlib wrds openpyxl
```

WRDS access is required for the empirical runs. You will be prompted for your WRDS username and password on first connection.

### Data

On first run, `run_with_wrds.py` fetches TAQ data from WRDS and caches it to `source/data/`. Subsequent runs load from cache. The calibration week (June 5–9, 2017) and evaluation week (June 12–16, 2017) are cached separately.

---

## Usage

### Empirical run (TAQ replay)

```bash
python scripts/run_with_wrds.py
```

Loads one week of TAQ data for AAPL, AMZN, GE, IVV, and M. Runs the optimal A-S strategy and the NBBO baseline under both front-of-queue and back-of-queue fill assumptions. Prints summary tables and saves plots to `plots/wrds/`.

### Synthetic run (paper-style Poisson simulator)

```bash
python scripts/run_synthetic.py
python scripts/run_synthetic.py --days 100 --seed 42
```

Replicates the Stanford paper's simulation environment using arithmetic Brownian motion for mid-price paths and Poisson-Gamma fills. Both strategies run on identical price paths per day for a fair comparison.

### Realized volatility extension

```bash
python scripts/realized_vol_strategy.py
```

Three-way comparison: baseline vs. fixed-vol A-S vs. realized-vol A-S. The RV model replaces the fixed `gamma * sigma^2` term in the spread equation with a rolling 10-minute realized variance computed from TAQ trade prices, updated every second.

### Parameter calibration

```bash
python scripts/calibrate_params.py
python scripts/calibrate_params.py --tickers AAPL,GE
python scripts/calibrate_params.py --no-cache   # force WRDS refetch
```

Fits `kappa`, `A`, and `b` from the prior week's TAQ data. Copy the printed values into `KAPPA_PARAMS`, `A_PARAMS`, and `B_PARAMS` in `run_with_wrds.py`.

---

## Model Overview

### Avellaneda-Stoikov (2008)

The optimal market maker posts a bid and ask symmetric around the **indifference price**:

```
r(s, t) = s - (q / lot_size) * gamma * sigma^2 * (T - t)
```

The optimal spread decays linearly from open to close:

```
spread(t) = A * (T - t) + B
```

where `A = gamma * sigma^2` and `B = (2/gamma) * ln(1 + gamma/kappa)`. Parameters are calibrated directly from empirical open and close spreads rather than estimated structurally.

### Inventory Model

Order sizes are adjusted exponentially with inventory:

```
phi_bid = phi_max * exp(-eta * q)   when q > 0  (long: reduce buying)
phi_ask = phi_max * exp( eta * q)   when q < 0  (short: reduce selling)
```

This keeps the market maker quoting at all times (unlike hard position limits) while naturally reducing directional exposure as inventory accumulates. Default parameters: `phi_max = 100`, `eta = 0.005`.

### Fill Model (empirical)

Fills combine two components:

**Observed fills** — at each second, look up all TAQ prints at or through the quoted price. Addressable volume is compared against queue position (`front` or `back` model).

**Poisson uplift** — for quotes posted strictly inside the BBO, an incremental fill probability is added using a Laplace intensity model:

```
lambda(xi) = A * exp(-xi / b)
delta_lambda = lambda(xi_our) - lambda(xi_best)
```

`A` and `b` are calibrated from intra-spread TAQ prints in the prior week via `calibrate_params.py`.

### Queue Models

`queue_model='front'` — assumes the market maker is first in queue at the quoted price level. All addressable volume is available.

`queue_model='back'` — assumes the market maker joins the back of the queue. The prevailing NBBO size is placed ahead; fill only occurs if total addressable volume exceeds that queue. Both models are run simultaneously in `run_with_wrds.py` and reported separately.

### Realized Vol Extension

`realized_vol_strategy.py` replaces the fixed `gamma * sigma^2` term with a rolling realized dollar variance:

```
spread(t) = gamma_rv * sigma2_t * (T - t) + B
```

`gamma_rv` is calibrated so that at average realized vol the RV spread equals the fixed spread — the two models are identical on average, but RV widens during volatile periods and tightens during calm periods. Flat-spread tickers (AMZN, where `open_spread <= close_spread`) bypass the RV logic automatically.

---

## Tickers and Venues

| Ticker | Exchange | Notes |
|--------|----------|-------|
| AAPL   | NASDAQ (Q) | Reference penny-spread stock |
| AMZN   | NASDAQ (Q) | Wide spread; A=0, flat spread model |
| GE     | Cboe BYX (Y) | |
| IVV    | NYSE Arca (P) | ETF |
| M      | NASDAQ (T) | NYSE-listed, NASDAQ-traded |

Each ticker is filtered to its primary venue in both the quote and trade streams to avoid cross-venue contamination. The `effective_mu` in the Poisson simulator is scaled inversely with `open_spread` so wide-spread tickers (AMZN) receive realistic fill probabilities.

---

## Output

### Tables (printed to console)

- **Table 2** — Average terminal P&L and inventory position
- **Table 3** — Mean and standard deviation of daily P&L and position
- **Table 4/5** — Orders, shares, and quote counts (optimal and baseline)
- **Fill Analysis** — Spread capture rate, imbalance ratio, fill-per-quote rate

### Plots (`plots/`)

| File | Description |
|------|-------------|
| `fig1_order_size.png` | Dynamic order size function vs inventory |
| `fig2_intensity.png` | Poisson intensity: time component and depth component |
| `fig3_spreads_<TICKER>.png` | Market spread vs optimal quoted spread |
| `fig4_pnl_inventory_<TICKER>.png` | Cumulative P&L and inventory density (2×2 grid) |
| `fig_quote_dynamics_<TICKER>.png` | Mid-price, indifference price, bid/ask quotes intraday |
| `fig_spread_comparison_<TICKER>.png` | Quoted spread vs NBBO, with rolling difference |
| `bbo_spread_distribution.png` | BBO spread distribution in pennies per ticker |
| `rv_pnl_<TICKER>.png` | Fixed vol vs realized vol vs baseline P&L by day |
| `rv_pnl_summary.png` | Average daily P&L bar chart across all tickers |
| `rv_spread_<TICKER>.png` | Intraday spread: fixed vol vs realized vol |
| `rv_inventory_<TICKER>.png` | Inventory density: fixed vol vs realized vol |

### Excel (`sheets/`)

- `fill_stats.xlsx` — Fill rate, spread capture, and volume statistics per ticker
- `quote_position.xlsx` — Intraday quote position relative to NBBO

---

## References

- Avellaneda, M. and Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217–224.
- Cartea, Á., Jaimungal, S., and Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Ho, T. and Stoll, H. (1981). *Optimal dealer pricing under transactions and return uncertainty*. Journal of Financial Economics, 9(1), 47–73.
