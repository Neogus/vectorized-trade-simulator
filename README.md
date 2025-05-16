# Fractal Heuristic Backtesting Engine for SL/TP Strategies

## Overview

This repository contains a **heuristic, vectorized, iterative backtesting framework** designed to simulate **Stop Loss (SL) and Take Profit (TP)** mechanics on OHLC time series data, supporting both long and short trading positions.

The core algorithm applies **forward-agnostic vectorized logic repeatedly** to refine trade entry and exit signals until convergence. It merges overlapping signals heuristically and calculates realistic trade returns considering fees and intrabar price action.

---

## Features

- Supports **long and short entries and exits** with separate SL and TP logic.
- Employs an **iterative fractal/heuristic algorithm** for refining trades on price bars.
- Calculates **final trade returns as a pandas Series**, enabling performance analysis.
- Vectorized approach ensures efficient processing of large datasets.
- Designed to work with pandas DataFrames containing OHLC data.
- Includes fee adjustment and handles intrabar high/low price triggers.

---

## Usage

1. Prepare a pandas DataFrame containing your price data with `Open`, `High`, `Low`, `Close` columns.
2. Add the **entry signal columns** named `Long_Trade` and `Short_Trade`:
   - Set entries in `Long_Trade` column to **-1** to signal long entries.
   - Set entries in `Short_Trade` column to **-2** to signal short entries.
3. Call the `get_ret` function with appropriate parameters including:
   - The DataFrame
   - The list of signal column names: `['Long_Trade', 'Short_Trade']`
   - SL and TP percentages
   - Fee per trade
   - Minimum trade count threshold (`t_min`)
4. `get_ret` returns a pandas Series of trade returns (`Ret`), which you can use for further performance evaluation.

---

## Example

```python
import pandas as pd
# Assuming 'df' is your OHLC DataFrame with 'Long_Trade' and 'Short_Trade' columns defined

take_profit_pct = 0.03
stop_loss_pct = 0.02
fee_per_trade = 0.0005
minimum_trades = 5
signal_columns = ['Long_Trade', 'Short_Trade']

returns_series = get_ret(df, signal_columns, [0, 0, 0, 0], stop_loss_pct, take_profit_pct, fee_per_trade, minimum_trades)

print(returns_series.describe())
