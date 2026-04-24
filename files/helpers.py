"""
helpers.py
==========
Summary statistics and Markov chain analysis helpers.
Used by run_with_wrds.py and analysis.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def summarise_results(all_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Table 2 (avg terminal P&L and position) and
    Table 3 (mean/stdev of daily profits and position).
    """
    table2_rows = []
    table3_rows = []

    for ticker, res in all_results.items():
        for strategy_key in ['optimal', 'baseline']:
            days = res[strategy_key]
            terminal_pnl = [d.pnl[-1] for d in days]
            terminal_pos = [d.inventory[-1] for d in days]

            table2_rows.append({
                'Ticker':       ticker,
                'Strategy':     strategy_key.capitalize(),
                'Avg P&L':      round(np.mean(terminal_pnl), 2),
                'Avg Position': round(np.mean(terminal_pos), 2),
            })
            table3_rows.append({
                'Ticker':    ticker,
                'Strategy':  strategy_key.capitalize(),
                'P&L Mean':  round(np.mean(terminal_pnl), 2),
                'P&L Stdev': round(np.std(terminal_pnl),  2),
                'Pos Mean':  round(np.mean(terminal_pos),  2),
                'Pos Stdev': round(np.std(terminal_pos),   2),
            })

    return pd.DataFrame(table2_rows), pd.DataFrame(table3_rows)


def summarise_order_stats(all_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Table 4 and Table 5: avg orders, shares, quotes per day.
    """
    rows_opt  = []
    rows_base = []

    for ticker, res in all_results.items():
        for strategy_key, rows in [('optimal', rows_opt), ('baseline', rows_base)]:
            days = res[strategy_key]
            rows.append({
                'Ticker':        ticker,
                'Buy Orders':    round(np.mean([d.n_buy_orders  for d in days])),
                'Sell Orders':   round(np.mean([d.n_sell_orders for d in days])),
                'Shares Bought': round(np.mean([d.shares_bought for d in days])),
                'Shares Sold':   round(np.mean([d.shares_sold   for d in days])),
                'Quotes':        round(np.mean([d.n_quotes      for d in days])),
            })

    return pd.DataFrame(rows_opt), pd.DataFrame(rows_base)


# ---------------------------------------------------------------------------
# Markov Chain analysis (Section 4.2)
# ---------------------------------------------------------------------------

def markov_chain_analysis(result_days: List) -> Dict:
    """
    Estimate Markov chain transition probabilities from simulation results.

    States: 0=Quoting, 1=Waiting, 2=Spread (both sides filled)

    p* = probability of capturing the spread per quote cycle
    q* = probability of a one-sided fill (increases inventory risk)
    """
    daily_bid_fills = [sum(1 for f in d.fills if f.side == 'bid') for d in result_days]
    daily_ask_fills = [sum(1 for f in d.fills if f.side == 'ask') for d in result_days]
    daily_quotes    = [d.n_quotes for d in result_days]

    p_spread = np.mean([
        min(b, a) / max(q, 1)
        for b, a, q in zip(daily_bid_fills, daily_ask_fills, daily_quotes)
    ])
    p_one_side = np.mean([
        abs(b - a) / max(q, 1)
        for b, a, q in zip(daily_bid_fills, daily_ask_fills, daily_quotes)
    ])

    p_no_fill = max(0.0, 1.0 - p_spread - p_one_side)
    p_1_2 = p_spread   * 0.3
    p_1_0 = p_one_side * 0.5
    p_1_1 = max(0.0, 1.0 - p_1_2 - p_1_0)

    # p* = p(0,2) + sum_{n=0}^{5} p(0,1)*p(1,1)^n*p(1,2)
    p_star = p_spread + sum(
        p_one_side * (p_1_1 ** n) * p_1_2 for n in range(6)
    )
    # q* = p(0,1) + p(0,1)*p(1,1)^5*p(1,0)
    q_star = p_one_side + p_one_side * (p_1_1 ** 5) * p_1_0

    return {
        'p(0,0)':                  round(p_no_fill,    4),
        'p(0,1)':                  round(p_one_side,   4),
        'p(0,2)':                  round(p_spread,     4),
        'p(1,0)':                  round(p_1_0,        4),
        'p(1,1)':                  round(p_1_1,        4),
        'p(1,2)':                  round(p_1_2,        4),
        'p_star (spread capture)': round(p_star * 100, 2),
        'q_star (one-side fill)':  round(q_star * 100, 2),
    }
