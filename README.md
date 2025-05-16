# Fractal Heuristic Backtesting Engine for SL/TP Strategies

## Overview

This repository contains a **heuristic, vectorized, iterative backtesting framework** designed to simulate **Stop Loss (SL) and Take Profit (TP)** mechanics on time series data for both long and short trading positions.

Unlike traditional sequential backtesting methods that process trades one by one, this approach applies **forward-agnostic vectorized logic repeatedly** to refine trade entries and exits. This iterative refinement enables efficient and realistic trade signal generation without relying on explicit stateful tracking.

---

## Features

- Supports **long and short trade signals** with independent SL and TP logic.
- Uses **vectorized operations** for high performance over large datasets.
- Employs an **iterative heuristic algorithm** that refines trade signals until convergence.
- Handles intrabar high/low price triggers for accurate SL/TP exits.
- Designed for use with pandas DataFrames containing OHLC (Open, High, Low, Close) price data.

---

## Usage

1. Prepare your price data in a pandas DataFrame with columns such as `Open`, `High`, `Low`, `Close`.
2. Generate your entry signals as separate columns for long and short entries.
3. Use the provided functions to calculate exit signals applying your SL and TP percentages.
4. Extract trade returns or performance metrics from the output DataFrame.

---

## Example

```python
import pandas as pd

# Load or generate your OHLC price data in a DataFrame 'df'
# Add your entry signals columns 'Long_Trade' and 'Short_Trade'

# Define your SL and TP thresholds (e.g., 2% stop loss, 3% take profit)
stop_loss_pct = 0.02
take_profit_pct = 0.03

# Run the backtesting algorithm (example function names)
df_result = exit_loop(df, 'Long_Trade', take_profit_pct, stop_loss_pct)
df_result = exit_loop(df_result, 'Short_Trade', take_profit_pct, stop_loss_pct, second=True)

# Analyze resulting trades in df_result['Trade']
