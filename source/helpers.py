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


# Strategy key -> display label, in the order rows should appear in tables.
# Keys absent from a result dict are skipped, so legacy callers (e.g. the
# synthetic runner that only stores 'optimal' / 'baseline') still work.
STRATEGY_VARIANTS: List[Tuple[str, str]] = [
    ('optimal',       'Optimal Front'),
    ('optimal_back',  'Optimal Back'),
    ('baseline',      'Baseline Front'),
    ('baseline_back', 'Baseline Back'),
]


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
        for strategy_key, label in STRATEGY_VARIANTS:
            if strategy_key not in res:
                continue
            days = res[strategy_key]
            terminal_pnl = [d.pnl[-1] for d in days]
            terminal_pos = [d.inventory[-1] for d in days]

            table2_rows.append({
                'Ticker':       ticker,
                'Strategy':     label,
                'Avg P&L':      round(np.mean(terminal_pnl), 2),
                'Avg Position': round(np.mean(terminal_pos), 2),
            })
            table3_rows.append({
                'Ticker':    ticker,
                'Strategy':  label,
                'P&L Mean':  round(np.mean(terminal_pnl), 2),
                'P&L Stdev': round(np.std(terminal_pnl),  2),
                'Pos Mean':  round(np.mean(terminal_pos),  2),
                'Pos Stdev': round(np.std(terminal_pos),   2),
            })

    return pd.DataFrame(table2_rows), pd.DataFrame(table3_rows)


def summarise_order_stats(all_results: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Table 4 and Table 5: avg orders, shares, quotes per day.

    Table 4 covers the optimal strategy under both queue models;
    Table 5 covers the baseline strategy under both queue models.
    """
    rows_opt  = []
    rows_base = []

    optimal_keys  = [k for k in ('optimal', 'optimal_back')
                     if any(k in res for res in all_results.values())]
    baseline_keys = [k for k in ('baseline', 'baseline_back')
                     if any(k in res for res in all_results.values())]

    label_for = dict(STRATEGY_VARIANTS)

    def _emit(rows, ticker, label, days):
        rows.append({
            'Ticker':        ticker,
            'Strategy':      label,
            'Buy Orders':    round(np.mean([d.n_buy_orders  for d in days])),
            'Sell Orders':   round(np.mean([d.n_sell_orders for d in days])),
            'Shares Bought': round(np.mean([d.shares_bought for d in days])),
            'Shares Sold':   round(np.mean([d.shares_sold   for d in days])),
            'Quotes':        round(np.mean([d.n_quotes      for d in days])),
        })

    for ticker, res in all_results.items():
        for k in optimal_keys:
            if k in res:
                _emit(rows_opt, ticker, label_for[k], res[k])
        for k in baseline_keys:
            if k in res:
                _emit(rows_base, ticker, label_for[k], res[k])

    return pd.DataFrame(rows_opt), pd.DataFrame(rows_base)



# Fill analysis (replaces Markov chain from Section 4.2 of the Stanford paper

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
    Print fill analysis table for all tickers under each (strategy, queue)
    variant present in the results dict. Analogous to Tables 7 and 8.
    """
    print(f"\n{'Ticker':<8} {'Strategy':<16} "
          f"{'Spread Cap%':>11} {'One-Side%':>10} {'No Fill%':>9} "
          f"{'Fills/Quote':>12} {'Avg Spread$':>12} {'Imbalance':>10}")
    print("-" * 84)

    for ticker, res in all_results.items():
        for strat_key, label in STRATEGY_VARIANTS:
            if strat_key not in res:
                continue
            stats = fill_analysis(res[strat_key])
            print(
                f"{ticker:<8} {label:<16} "
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

    Layout per ticker/queue variant: 2x3 table
    Rows:    Bid, Ask
    Columns: Inside NBBO, At NBBO, Outside NBBO

    Sheets: AAPL_Opt_F, AAPL_Opt_B, AAPL_Base_F, AAPL_Base_B, etc.
    One sheet per (ticker, strategy, queue model) combination.
    One day block per sheet — counts on top, percentages below.
    """
    TICK = 0.001

    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    sheet_variants = [
        ('optimal',       'Opt_F'),
        ('optimal_back',  'Opt_B'),
        ('baseline',      'Base_F'),
        ('baseline_back', 'Base_B'),
    ]

    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:

        # Remove default empty sheet created by openpyxl
        if 'Sheet' in writer.book.sheetnames:
            del writer.book['Sheet']

        for ticker, res in all_results.items():
            for strategy_key, sheet_suffix in sheet_variants:

                # Skip if this variant doesn't exist in results
                if strategy_key not in res:
                    continue

                sheet_name = f"{ticker}_{sheet_suffix}"
                workbook   = writer.book
                worksheet  = workbook.create_sheet(title=sheet_name)
                startrow   = 0

                for day_i, day_res in enumerate(res[strategy_key]):
                    bid_arr = day_res.bid_prices
                    ask_arr = day_res.ask_prices
                    bb      = best_bid_dict[ticker][day_i]
                    ba      = best_ask_dict[ticker][day_i]

                    # Per-side active masks — bid and ask can differ
                    # during one-sided fill waiting periods
                    active_bid   = ~np.isnan(bid_arr)
                    active_ask   = ~np.isnan(ask_arr)
                    n_active_bid = int(active_bid.sum())
                    n_active_ask = int(active_ask.sum())

                    # Bid counts
                    bid_inside  = int(((bid_arr > bb + TICK) & active_bid).sum())
                    bid_at      = int(((np.abs(bid_arr - bb) <= TICK) & active_bid).sum())
                    bid_outside = int(((bid_arr < bb - TICK) & active_bid).sum())

                    # Ask counts
                    ask_inside  = int(((ask_arr < ba - TICK) & active_ask).sum())
                    ask_at      = int(((np.abs(ask_arr - ba) <= TICK) & active_ask).sum())
                    ask_outside = int(((ask_arr > ba + TICK) & active_ask).sum())

                    # --- Day header ---
                    worksheet.cell(row=startrow + 1, column=1,
                                   value=f"Day {day_i + 1} — "
                                         f"Active bid: {n_active_bid} "
                                         f"ask: {n_active_ask}")

                    # --- Count table ---
                    worksheet.cell(row=startrow + 2, column=1, value='Counts')
                    worksheet.cell(row=startrow + 2, column=2, value='Inside NBBO')
                    worksheet.cell(row=startrow + 2, column=3, value='At NBBO')
                    worksheet.cell(row=startrow + 2, column=4, value='Outside NBBO')

                    for row_i, (side, inside, at, outside) in enumerate([
                        ('Bid', bid_inside, bid_at, bid_outside),
                        ('Ask', ask_inside, ask_at, ask_outside),
                    ]):
                        worksheet.cell(row=startrow + 3 + row_i, column=1, value=side)
                        worksheet.cell(row=startrow + 3 + row_i, column=2, value=inside)
                        worksheet.cell(row=startrow + 3 + row_i, column=3, value=at)
                        worksheet.cell(row=startrow + 3 + row_i, column=4, value=outside)

                    # --- Percentage table ---
                    worksheet.cell(row=startrow + 6, column=1, value='Percentages (%)')
                    worksheet.cell(row=startrow + 6, column=2, value='Inside NBBO')
                    worksheet.cell(row=startrow + 6, column=3, value='At NBBO')
                    worksheet.cell(row=startrow + 6, column=4, value='Outside NBBO')

                    for row_i, (side, inside, at, outside, n_active) in enumerate([
                        ('Bid', bid_inside, bid_at, bid_outside, n_active_bid),
                        ('Ask', ask_inside, ask_at, ask_outside, n_active_ask),
                    ]):
                        denom = max(n_active, 1)
                        worksheet.cell(row=startrow + 7 + row_i, column=1, value=side)
                        worksheet.cell(row=startrow + 7 + row_i, column=2,
                                       value=round(inside  / denom * 100, 1))
                        worksheet.cell(row=startrow + 7 + row_i, column=3,
                                       value=round(at      / denom * 100, 1))
                        worksheet.cell(row=startrow + 7 + row_i, column=4,
                                       value=round(outside / denom * 100, 1))

                    startrow += 10

    print(f"\nExported quote position to: {full_path}")


def compute_quote_distance_stats(all_results: dict,
                                 spread_params: dict) -> pd.DataFrame:
    """
    For each ticker and day, compute squared distance between
    optimal quotes and the synthetic NBBO (best_bid/best_ask).

    In the Poisson simulator, best_bid = mid - half_mkt_spread
    and best_ask = mid + half_mkt_spread where mkt_spread decays
    linearly from open_spread to close_spread.
    """
    T_SECONDS = 23400.0
    rows = []

    for ticker, res in all_results.items():
        sp = spread_params[ticker]
        open_spread = sp['open_spread']
        close_spread = sp['close_spread']

        for day_i, day_res in enumerate(res['optimal']):
            bid_arr = day_res.bid_prices
            ask_arr = day_res.ask_prices
            mid_arr = day_res.mid_prices
            n = len(mid_arr)

            # Reconstruct synthetic NBBO at each second
            t_norm_arr = np.arange(n) / T_SECONDS
            mkt_spread = open_spread + (close_spread - open_spread) * t_norm_arr
            best_bid = mid_arr - mkt_spread / 2.0
            best_ask = mid_arr + mkt_spread / 2.0

            # Only active seconds
            active = ~np.isnan(bid_arr)

            bid_dist_sq = np.where(active, (bid_arr - best_bid) ** 2, np.nan)
            ask_dist_sq = np.where(active, (ask_arr - best_ask) ** 2, np.nan)

            rows.append({
                'Ticker': ticker,
                'Day': day_i + 1,
                'Active Seconds': int(active.sum()),
                'Bid Mean Sq Dist': round(float(np.nanmean(bid_dist_sq)), 8),
                'Ask Mean Sq Dist': round(float(np.nanmean(ask_dist_sq)), 8),
                'Bid RMSE': round(float(np.sqrt(np.nanmean(bid_dist_sq))), 6),
                'Ask RMSE': round(float(np.sqrt(np.nanmean(ask_dist_sq))), 6),
                'Bid Max Dist': round(float(np.sqrt(np.nanmax(bid_dist_sq))), 6),
                'Ask Max Dist': round(float(np.sqrt(np.nanmax(ask_dist_sq))), 6),
                'Bid Pct Within Tick': round(float(
                    np.nanmean(np.abs(bid_arr - best_bid)[active] <= 0.01)
                ) * 100, 1),
                'Ask Pct Within Tick': round(float(
                    np.nanmean(np.abs(ask_arr - best_ask)[active] <= 0.01)
                ) * 100, 1),
            })

    return pd.DataFrame(rows)

def export_fill_stats_xlsx(all_results: dict,
                            best_bid_dict: dict,
                            best_ask_dict: dict,
                            path: str = 'sheets/fill_stats.xlsx'):
    """
    Export fill analysis for all four strategy/queue combinations to Excel.
    One sheet per ticker, one table per day plus average summary.

    Columns:
        Strategy, Spread Cap%, One-Side%, No Fill%, Fills/Quote,
        Avg Spread$, Imbalance,
        Fill Inside%, Fill At%, Fill Outside%

    Fill Inside/At/Outside: percentage of fills that occurred when
    the quote was inside, at, or outside the NBBO respectively.
    """
    TICK = 0.001

    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    headers = [
        'Strategy', 'Spread Cap%', 'One-Side%', 'No Fill%',
        'Fills/Quote', 'Avg Spread$', 'Imbalance',
        'Fill Inside%', 'Fill At%', 'Fill Outside%'
    ]

    def _fill_position_stats(days: list, ticker: str,
                              bb_list: list, ba_list: list,
                              day_indices: list) -> dict:
        """
        For each fill, determine whether the quote was inside, at,
        or outside the NBBO at the time of fill.

        Returns dict with fill_inside_pct, fill_at_pct, fill_outside_pct.
        """
        inside  = 0
        at_nbbo = 0
        outside = 0

        for day_i, day_res in zip(day_indices, days):
            if day_i >= len(bb_list):
                continue
            bb = bb_list[day_i]
            ba = ba_list[day_i]

            for fill in day_res.fills:
                sec = int(fill.time)
                sec = min(sec, len(bb) - 1)

                if fill.side == 'bid':
                    ref   = bb[sec]
                    price = fill.price
                    # bid fill: inside if our bid > best_bid
                    if price > ref + TICK:
                        inside += 1
                    elif abs(price - ref) <= TICK:
                        at_nbbo += 1
                    else:
                        outside += 1
                else:  # ask
                    ref   = ba[sec]
                    price = fill.price
                    # ask fill: inside if our ask < best_ask
                    if price < ref - TICK:
                        inside += 1
                    elif abs(price - ref) <= TICK:
                        at_nbbo += 1
                    else:
                        outside += 1

        total = inside + at_nbbo + outside
        if total == 0:
            return {'fill_inside_pct': 0.0,
                    'fill_at_pct':     0.0,
                    'fill_outside_pct':0.0}

        return {
            'fill_inside_pct':  round(inside  / total * 100, 1),
            'fill_at_pct':      round(at_nbbo / total * 100, 1),
            'fill_outside_pct': round(outside / total * 100, 1),
        }

    def _write_row(worksheet, row, label, stats, pos_stats):
        worksheet.cell(row=row, column=1,  value=label)
        worksheet.cell(row=row, column=2,  value=round(stats['spread_capture_rate'] * 100, 1))
        worksheet.cell(row=row, column=3,  value=round(stats['one_side_fill_rate']  * 100, 1))
        worksheet.cell(row=row, column=4,  value=round(stats['no_fill_rate']        * 100, 1))
        worksheet.cell(row=row, column=5,  value=round(stats['fills_per_quote'],     3))
        worksheet.cell(row=row, column=6,  value=round(stats['avg_spread_captured'], 4))
        worksheet.cell(row=row, column=7,  value=round(stats['imbalance_ratio'],     4))
        worksheet.cell(row=row, column=8,  value=pos_stats['fill_inside_pct'])
        worksheet.cell(row=row, column=9,  value=pos_stats['fill_at_pct'])
        worksheet.cell(row=row, column=10, value=pos_stats['fill_outside_pct'])

    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:

        if 'Sheet' in writer.book.sheetnames:
            del writer.book['Sheet']

        for ticker, res in all_results.items():
            workbook  = writer.book
            worksheet = workbook.create_sheet(title=ticker)

            bb_list = best_bid_dict.get(ticker, [])
            ba_list = best_ask_dict.get(ticker, [])

            variants = [
                ('Optimal(F)',  res['optimal']),
                ('Optimal(B)',  res.get('optimal_back',  res['optimal'])),
                ('Baseline(F)', res['baseline']),
                ('Baseline(B)', res.get('baseline_back', res['baseline'])),
            ]

            n_days   = max(len(days) for _, days in variants)
            startrow = 0

            # --- Per-day tables ---
            for day_i in range(n_days):
                worksheet.cell(row=startrow + 1, column=1,
                               value=f"Day {day_i + 1}")

                for col_i, h in enumerate(headers):
                    worksheet.cell(row=startrow + 2,
                                   column=col_i + 1, value=h)

                for row_i, (label, days) in enumerate(variants):
                    if day_i >= len(days):
                        continue

                    day_list   = [days[day_i]]
                    stats      = fill_analysis(day_list)
                    pos_stats  = _fill_position_stats(
                        day_list, ticker, bb_list, ba_list, [day_i]
                    )
                    _write_row(worksheet, startrow + 3 + row_i,
                               label, stats, pos_stats)

                startrow += 4 + len(variants) + 1

            # --- Average across all days ---
            worksheet.cell(row=startrow + 1, column=1,
                           value="Average across all days")

            for col_i, h in enumerate(headers):
                worksheet.cell(row=startrow + 2,
                               column=col_i + 1, value=h)

            for row_i, (label, days) in enumerate(variants):
                if not days:
                    continue

                stats     = fill_analysis(days)
                pos_stats = _fill_position_stats(
                    days, ticker, bb_list, ba_list,
                    list(range(len(days)))
                )
                _write_row(worksheet, startrow + 3 + row_i,
                           label, stats, pos_stats)

    print(f"\nExported fill stats to: {full_path}")