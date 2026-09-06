"""Page 5: Guide & Glossary — how to use the simulator."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Guide", page_icon="📖", layout="wide")

# ── Custom sidebar navigation ──────────────────────────────────────────

st.markdown(
    """<style>[data-testid="stSidebarNav"] { display: none; }</style>""",
    unsafe_allow_html=True,
)
st.sidebar.title("📊 Trade Simulator")
st.sidebar.markdown("**Sections**")
st.sidebar.page_link("app.py", label="🏠 App")
st.sidebar.page_link("pages/1_Data_Source.py", label="📂 Data Source")
st.sidebar.page_link("pages/2_Configure.py", label="⚙️ Configure")
st.sidebar.page_link("pages/3_Run_Backtest.py", label="▶️ Run Backtest")
st.sidebar.page_link("pages/4_Results.py", label="📈 Results")
st.sidebar.page_link("pages/5_Guide.py", label="📖 Guide")
st.sidebar.markdown("---")
st.sidebar.caption(
    "☕ [Support this project](https://github.com/sponsors/Neogus) · "
    "[Ko-fi](https://ko-fi.com/neogus43222)"
)

st.title("📖 Guide & Glossary")

# ── Quick Start ────────────────────────────────────────────────────────

st.markdown("""
## 🚀 Quick Start

1. Go to **📂 Data Source** → upload a CSV file or download data from Yahoo Finance / crypto exchange
2. Go to **⚙️ Configure** → set your stop-loss, take-profit, and select signals
3. Go to **▶️ Run Backtest** → click "Run" and wait for results
4. Go to **📈 Results** → explore charts and download the trade log

**Tip:** Start with the defaults (RSI signal, SL=1.5, TP=2.0) and adjust from there.
""")

# ── How It Works ───────────────────────────────────────────────────────

st.markdown("""
## ⚙️ How the Engine Works

The simulator runs a **vectorized, bar-by-bar state machine**:

1. **ATR Calculation** — Computes the Average True Range (volatility measure) using Wilder's
   smoothing over the ATR window (default: 14 bars)

2. **Barrier Construction** — For each bar, pre-computes 4 price levels:
   - Long Take-Profit = `close + tp_mult × ATR`
   - Long Stop-Loss = `close - sl_mult × ATR`
   - Short Take-Profit = `close - tp_mult × ATR`
   - Short Stop-Loss = `close + sl_mult × ATR`

3. **Signal Processing** — Entry signals from Tecana (or built-in RSI/SMA) are shifted by 1 bar
   to prevent look-ahead bias. The signal fires on bar N, the trade enters on bar N+1.

4. **Trade Simulation** — A position-aware state machine scans through bars:
   - On an entry signal: opens a position at the close price, locks in barrier levels
   - Each bar while in position: checks if High/Low crosses any barrier
   - On barrier hit: closes the position, records the return
   - If both SL and TP are hit on the same bar: SL wins (conservative)

5. **Return Calculation** — Per-trade returns include round-trip fees and slippage:
   - Long: `exit_price / entry_price - 1 - (2×fee + 2×slippage)`
   - Short: `entry_price / exit_price - 1 - (2×fee + 2×slippage)`
""")

# ── Parameters ─────────────────────────────────────────────────────────

st.markdown("## 📋 Parameter Glossary")

st.markdown("""
### Trading Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| **SL Mult** | Stop-loss distance as ATR multiple. Higher = wider stop, fewer triggers. | 1.5 | 0.1 – 50.0 |
| **TP Mult** | Take-profit distance as ATR multiple. Higher = larger targets, fewer hits. | 2.0 | 0.1 – 50.0 |
| **ATR Window** | Wilder smoothing period for Average True Range. | 14 | 1 – 500 |
| **Fee** | Per-side trading fee as a fraction (e.g., 0.0004 = 0.04%). Applied on both entry and exit. | 0.0004 | 0 – 0.1 |
| **Slippage** | Per-side slippage as a fraction. Models price impact. | 0.0002 | 0 – 0.1 |
| **Max Hold** | Maximum bars to hold a position. Forces exit if no barrier is hit. 0 = no limit. | 200 | 0 – 10000 |

### SL/TP Mode

- **Fixed** — Uses a single SL and TP value for all trades. Fast, good for quick tests.
- **Search Range** — Sweeps a grid of SL/TP combinations and scores each one. Use this to find the optimal parameters for your data.
""")

# ── Signals ────────────────────────────────────────────────────────────

st.markdown("""
### Signals

Signals tell the engine when to enter trades. They come from the **Tecana** technical analysis library.

| Family | Suffix | Description | Example |
|--------|--------|-------------|---------|
| **Momentum** | `_m` | Crossover / zero-cross reversals | `rsi_m` — RSI crosses out of oversold/overbought |
| **Zone** | `_z` | Overbought / oversold conditions | `rsi_z` — RSI enters extreme zone |
| **Trend** | `_t` | Price vs indicator direction | `dema_t` — Price above/below DEMA |
| **Volatility** | `_v` | High/low volatility flags | `natr_v` — NATR exceeds 1.5× median |

### Aggregation Modes

When you select **multiple signals**, the aggregation mode determines when to enter:

| Mode | Logic | Best for |
|------|-------|----------|
| **Any signal** | Enter when **at least 1** signal fires | Most trades, exploratory analysis |
| **Majority agree** | Enter when **more than half** agree | Balanced — fewer but higher-confidence entries |
| **All agree (unanimous)** | Enter when **every** signal agrees | Fewest trades, highest conviction only |

**⚠️ Tip:** Start with "Any signal" and 1–2 signals to get plenty of trades. "Unanimous" with many signals will often produce zero entries.
""")

# ── Signal Convention ──────────────────────────────────────────────────

st.markdown("""
### Signal Convention

This simulator and Tecana use **opposite conventions** — but the app handles this automatically:

| | Tecana (raw) | Simulator (internal) |
|---|---|---|
| **Bullish / Buy / Long** | `-1` | `+1` |
| **Bearish / Sell / Short** | `+1` | `-1` |
| **Neutral** | `0` | `0` |

The app **negates Tecana signals** when importing them, so you never need to think about this.
If you provide your own signals, use the **simulator convention**: `+1 = LONG`, `-1 = SHORT`.
""")

# ── Scoring ────────────────────────────────────────────────────────────

st.markdown("""
## 🏆 Scoring Formulas

Scoring formulas rank strategies by combining multiple performance metrics into a single number.

| Formula | Name | What it measures | Best for |
|---------|------|-----------------|----------|
| **A** | Composite | Sharpe × drawdown control × alpha | General-purpose balanced ranking |
| **B** | Sharpe Focus | `sharpe_avg - sharpe_dev` | Risk-adjusted return consistency |
| **C** | Sharpe Focus | Same as B (legacy) | — |
| **D** | Accuracy + DD | Win rate × drawdown factor | Strategies where hit rate matters |
| **E** | Alpha Focus | `alpha_avg - return_dev` | Market-beating strategies |
| **F** | Alpha Focus | Same as E (legacy) | — |
| **G** | Alpha Focus | Same as E (legacy) | — |
| **H** | Full Composite | Sharpe + drawdown + alpha + accuracy bonus | Most thorough, recommended for final ranking |

**Recommendation:** Start with **Formula A** for a quick balanced view. Use **Formula H** for comprehensive evaluation.
""")

# ── Metrics ────────────────────────────────────────────────────────────

st.markdown("""
## 📊 Performance Metrics Glossary

| Metric | Description | Good value |
|--------|-------------|------------|
| **Total Return** | Compounded product of all trade returns | > 0% |
| **Max Drawdown** | Worst peak-to-trough equity decline | > -20% |
| **Sharpe Ratio** | Risk-adjusted return (mean / std dev, annualized) | > 1.0 |
| **Win Rate** | Fraction of trades that were profitable | > 50% |
| **Profit Factor** | Gross profit / gross loss | > 1.5 |
| **Alpha** | Excess return over a benchmark (market) | > 0 |
| **Beta** | Correlation / sensitivity to market returns | ~0 (market-neutral) |
| **Avg Win** | Average return on winning trades | Depends on strategy |
| **Avg Loss** | Average return on losing trades | Smaller is better |
""")

# ── Data Tips ──────────────────────────────────────────────────────────

st.markdown("""
## 📂 Data Tips

### File Upload
The parser auto-detects columns from 30+ naming conventions:
- `Open, High, Low, Close, Volume` (standard)
- `o, h, l, c, vol` (abbreviated)
- `open_price, high_price, ...` (verbose)
- `Date, DateTime, timestamp, dt` (datetime)

Supported formats: **CSV**, **TSV**, **Parquet**, **Excel** (.xlsx / .xls)

### API Downloads
- **Yahoo Finance** — Works for stocks, ETFs, indices, and crypto (e.g., `BTC-USD`). Intraday data limited to 60 days.
- **Crypto (ccxt)** — Supports 100+ exchanges. Use `BTC/USDT` format for symbols.

### Data Quality
After loading, the app checks for:
- Duplicate timestamps
- Out-of-order rows
- Missing bars (gaps)
- NaN values and negative prices
- High < Low rows (corrupted)

Click **Auto-fix** to apply forward-fill cleaning.
""")

st.markdown("---")

col1, col2 = st.columns([3, 1])
with col1:
    st.caption(
        "⚠️ *This tool is for educational and informational purposes only. "
        "There is no guarantee that calculations are error-free. "
        "Always conduct your own research before making investment decisions.*"
    )
with col2:
    st.markdown(
        '<p style="text-align:right; font-size:0.85em; opacity:0.7;">'
        '☕ <a href="https://github.com/sponsors/Neogus">Sponsor</a> · '
        '<a href="https://ko-fi.com/neogus43222">Ko-fi</a></p>',
        unsafe_allow_html=True,
    )
