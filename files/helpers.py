"""
helpers.py
==========
Summary statistics and fill analysis helpers.
Used by run_with_wrds.py and analysis.py.
"""

import numpy as np
import pandas as pd
import os
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
# Fill analysis (replaces Markov chain from Section 4.2 of the paper)
# ---------------------------------------------------------------------------
#
# WHY WE REPLACED THE MARKOV CHAIN:
#
# The paper's Markov chain analysis was designed for a *generative* simulator
# where order arrivals were drawn from a Poisson process. In that setting,
# transition probabilities like p(0,2) — both sides fill in one second — were
# genuine forward-looking probabilities arising from the Poisson intensity,
# and the formula p* = p(0,2) + sum p(0,1)*p(1,1)^n*p(1,2) made sense as an
# analytic expression for the spread-capture probability under that model.
#
# With TAQ replay, fills are deterministic given the historical data: for a
# specific day, our bid either got crossed by a real trade or it didn't. There
# is no probability to estimate — only frequencies to count. Fitting a Markov
# chain to these frequencies and then applying the geometric waiting-time
# formula would give numbers that look like probabilities but aren't: the
# underlying assumption of memoryless transitions is violated because real
# market order flow has intraday patterns and serial correlation.
#
# We therefore replace the Markov chain with direct empirical statistics
# computed from the fills list. These are honest about what the replay
# simulation actually measures: observed rates over the specific week simulated,
# not forward-looking probabilities for future trading days.
#
# The quantities reported are directly comparable to Tables 7 and 8 of the
# paper as performance metrics — they just have a cleaner interpretation.
# ---------------------------------------------------------------------------

def fill_analysis(result_days: List) -> Dict:
    """
    Compute empirical fill statistics from replay simulation results.

    All rates are computed per quote cycle, averaged across days.

    Returns
    -------
    dict with:
        spread_capture_rate : fraction of quote cycles where both sides filled
        one_side_fill_rate  : fraction of quote cycles where only one side filled
        no_fill_rate        : fraction of quote cycles with no fill at all
        fills_per_quote     : average total fills per quote posted
        avg_spread_captured : average spread earned on two-sided fills (dollars)
        imbalance_ratio     : abs(buy_fills - sell_fills) / total_fills
                              — measures how lopsided fills are; lower is better
                              for inventory control
    """
    daily_stats = []

    for day in result_days:
        n_quotes = max(day.n_quotes, 1)
        bid_fills = [f for f in day.fills if f.side == 'bid']
        ask_fills = [f for f in day.fills if f.side == 'ask']
        n_bid = len(bid_fills)
        n_ask = len(ask_fills)
        total_fills = n_bid + n_ask

        # Spread captured = both sides filled in the same quote cycle.
        # Best approximation from aggregate data: min(bid_fills, ask_fills)
        # counts the number of fully paired round-trips.
        n_spread = min(n_bid, n_ask)
        n_one_side = abs(n_bid - n_ask)

        # Average spread earned per two-sided fill
        if n_spread > 0 and len(bid_fills) > 0 and len(ask_fills) > 0:
            avg_ask_price = np.mean([f.price for f in ask_fills])
            avg_bid_price = np.mean([f.price for f in bid_fills])
            avg_spread_earned = avg_ask_price - avg_bid_price
        else:
            avg_spread_earned = 0.0

        # Imbalance: 0 means perfectly balanced, 1 means entirely one-sided
        imbalance = abs(n_bid - n_ask) / max(total_fills, 1)

        daily_stats.append({
            'spread_capture_rate': n_spread    / n_quotes,
            'one_side_fill_rate':  n_one_side  / n_quotes,
            'no_fill_rate':        max(0.0, 1.0 - (n_spread + n_one_side) / n_quotes),
            'fills_per_quote':     total_fills / n_quotes,
            'avg_spread_captured': avg_spread_earned,
            'imbalance_ratio':     imbalance,
        })

    return {
        k: round(float(np.mean([d[k] for d in daily_stats])), 4)
        for k in daily_stats[0]
    }


def print_fill_analysis(all_results: Dict) -> None:
    """
    Print fill analysis table for all tickers and both strategies.
    Analogous to Tables 7 and 8 in the paper.
    """
    print(f"\n{'Ticker':<8} {'Strategy':<12} "
          f"{'Spread Cap%':>11} {'One-Side%':>10} {'No Fill%':>9} "
          f"{'Fills/Quote':>12} {'Avg Spread$':>12} {'Imbalance':>10}")
    print("-" * 80)

    for ticker, res in all_results.items():
        for strat_key in ['optimal', 'baseline']:
            stats = fill_analysis(res[strat_key])
            print(
                f"{ticker:<8} {strat_key.capitalize():<12} "
                f"{stats['spread_capture_rate']*100:>10.1f}% "
                f"{stats['one_side_fill_rate']*100:>9.1f}% "
                f"{stats['no_fill_rate']*100:>8.1f}% "
                f"{stats['fills_per_quote']:>12.3f} "
                f"{stats['avg_spread_captured']:>11.4f}  "
                f"{stats['imbalance_ratio']:>9.4f}"
            )


def export_quote_position_xlsx(all_results: dict,
                               best_bid_dict: dict,
                               best_ask_dict: dict,
                               path: str = 'sheets/quote_position.xlsx'):
    """
    Export quote position analysis to Excel.

    Layout per ticker: 2x3 table
    Rows:    Bid, Ask
    Columns: Inside NBBO, At NBBO, Outside NBBO

    One sheet per ticker, one table per day.
    """
    TICK = 0.001

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        for ticker, res in all_results.items():

            # Collect all day tables then write to one sheet
            all_day_frames = []

            for day_i, day_res in enumerate(res['optimal']):
                bid_arr = day_res.bid_prices
                ask_arr = day_res.ask_prices
                bb = best_bid_dict[ticker][day_i]
                ba = best_ask_dict[ticker][day_i]

                # Only seconds where we had active quotes
                active = ~np.isnan(bid_arr)
                n_active = int(active.sum())

                # Bid counts
                bid_inside = int(((bid_arr > bb + TICK) & active).sum())
                bid_at = int(((np.abs(bid_arr - bb) <= TICK) & active).sum())
                bid_outside = int(((bid_arr < bb - TICK) & active).sum())

                # Ask counts
                ask_inside = int(((ask_arr < ba - TICK) & active).sum())
                ask_at = int(((np.abs(ask_arr - ba) <= TICK) & active).sum())
                ask_outside = int(((ask_arr > ba + TICK) & active).sum())

                # Build 2x3 count table
                count_df = pd.DataFrame(
                    {
                        'Inside NBBO': [bid_inside, ask_inside],
                        'At NBBO': [bid_at, ask_at],
                        'Outside NBBO': [bid_outside, ask_outside],
                    },
                    index=['Bid', 'Ask']
                )

                # Build 2x3 percentage table
                pct_df = pd.DataFrame(
                    {
                        'Inside NBBO': [round(bid_inside / max(n_active, 1) * 100, 1),
                                        round(ask_inside / max(n_active, 1) * 100, 1)],
                        'At NBBO': [round(bid_at / max(n_active, 1) * 100, 1),
                                    round(ask_at / max(n_active, 1) * 100, 1)],
                        'Outside NBBO': [round(bid_outside / max(n_active, 1) * 100, 1),
                                         round(ask_outside / max(n_active, 1) * 100, 1)],
                    },
                    index=['Bid', 'Ask']
                )

                all_day_frames.append({
                    'day': day_i + 1,
                    'count': count_df,
                    'pct': pct_df,
                    'active': n_active,
                })

            # Write all days to one sheet per ticker
            sheet = ticker
            startrow = 0

            # Write to Excel manually so we can control layout
            workbook = writer.book
            worksheet = workbook.create_sheet(title=sheet)

            for d in all_day_frames:
                # Day header
                worksheet.cell(row=startrow + 1, column=1,
                               value=f"Day {d['day']} — Active seconds: {d['active']}")

                # Count table
                worksheet.cell(row=startrow + 2, column=1, value='Counts')
                worksheet.cell(row=startrow + 2, column=2, value='Inside NBBO')
                worksheet.cell(row=startrow + 2, column=3, value='At NBBO')
                worksheet.cell(row=startrow + 2, column=4, value='Outside NBBO')

                for row_i, idx in enumerate(['Bid', 'Ask']):
                    worksheet.cell(row=startrow + 3 + row_i, column=1, value=idx)
                    worksheet.cell(row=startrow + 3 + row_i, column=2,
                                   value=d['count'].loc[idx, 'Inside NBBO'])
                    worksheet.cell(row=startrow + 3 + row_i, column=3,
                                   value=d['count'].loc[idx, 'At NBBO'])
                    worksheet.cell(row=startrow + 3 + row_i, column=4,
                                   value=d['count'].loc[idx, 'Outside NBBO'])

                # Percentage table
                worksheet.cell(row=startrow + 6, column=1, value='Percentages (%)')
                worksheet.cell(row=startrow + 6, column=2, value='Inside NBBO')
                worksheet.cell(row=startrow + 6, column=3, value='At NBBO')
                worksheet.cell(row=startrow + 6, column=4, value='Outside NBBO')

                for row_i, idx in enumerate(['Bid', 'Ask']):
                    worksheet.cell(row=startrow + 7 + row_i, column=1, value=idx)
                    worksheet.cell(row=startrow + 7 + row_i, column=2,
                                   value=d['pct'].loc[idx, 'Inside NBBO'])
                    worksheet.cell(row=startrow + 7 + row_i, column=3,
                                   value=d['pct'].loc[idx, 'At NBBO'])
                    worksheet.cell(row=startrow + 7 + row_i, column=4,
                                   value=d['pct'].loc[idx, 'Outside NBBO'])

                startrow += 10
    print(f"\nExported quote position to: {full_path}")