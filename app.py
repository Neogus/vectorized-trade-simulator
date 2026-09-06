"""Vectorized Trade Simulator — Streamlit App (main entry point)."""

import streamlit as st

st.set_page_config(
    page_title="Vectorized Trade Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────

st.sidebar.title("📊 Trade Simulator")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Workflow:**\n"
    "1. 📂 Load Data\n"
    "2. ⚙️ Configure\n"
    "3. ▶️ Run Backtest\n"
    "4. 📈 Results"
)

# Show data status in sidebar
if "ohlcv_data" in st.session_state and st.session_state.ohlcv_data is not None:
    df = st.session_state.ohlcv_data
    st.sidebar.success(f"✅ Data loaded: {len(df):,} bars")
    st.sidebar.caption(
        f"{df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}"
    )
else:
    st.sidebar.warning("No data loaded yet")

if "backtest_results" in st.session_state and st.session_state.backtest_results is not None:
    n = len(st.session_state.backtest_results)
    st.sidebar.success(f"✅ {n} trades simulated")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with [Tecana](https://github.com/Neogus/tecana) · "
    "[GitHub](https://github.com/Neogus/vectorized-trade-simulator)"
)

# ── Main page ──────────────────────────────────────────────────────────

st.title("📊 Vectorized Trade Simulator")
st.markdown(
    """
    A **vectorized backtesting engine** for testing stop-loss / take-profit
    strategies on historical OHLCV data. Powered by
    [Tecana](https://pypi.org/project/tecana/) for 90+ technical indicators.

    ### Getting Started

    Use the **sidebar pages** to navigate through the workflow:

    | Step | Page | Description |
    |------|------|-------------|
    | 1 | **📂 Data Source** | Upload a CSV/Parquet file or download from yfinance/ccxt |
    | 2 | **⚙️ Configure** | Set trading parameters, select signals, choose scoring |
    | 3 | **▶️ Run Backtest** | Execute the simulation and see the trade log |
    | 4 | **📈 Results** | Interactive charts, performance metrics, CSV export |

    ### Signal Convention

    | Value | Meaning |
    |-------|---------|
    | **+1** | LONG entry (bullish) |
    | **-1** | SHORT entry (bearish) |
    | **0** | No signal (neutral) |

    ---

    *⚠️ Disclaimer: This tool is for educational and informational purposes only.
    There is no guarantee that calculations are free of errors.
    The author is not responsible for any financial losses.
    Always conduct your own research before making investment decisions.*
    """
)
