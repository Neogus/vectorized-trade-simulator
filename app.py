"""Vectorized Trade Simulator — Streamlit App (main entry point)."""

import streamlit as st

st.set_page_config(
    page_title="Vectorized Trade Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide default auto-navigation and build custom sidebar ──────────────

st.markdown(
    """
    <style>
    /* Hide the auto-generated page navigation */
    [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Custom sidebar ─────────────────────────────────────────────────────

st.sidebar.title("📊 Trade Simulator")
st.sidebar.markdown("**Sections**")

st.sidebar.page_link("app.py", label="🏠 App", icon=None)
st.sidebar.page_link("pages/1_Data_Source.py", label="📂 Data Source")
st.sidebar.page_link("pages/2_Configure.py", label="⚙️ Configure")
st.sidebar.page_link("pages/3_Run_Backtest.py", label="▶️ Run Backtest")
st.sidebar.page_link("pages/4_Results.py", label="📈 Results")

st.sidebar.markdown("---")

# Workflow guide
st.sidebar.markdown(
    "**Workflow:**\n"
    "1. 📂 Load data\n"
    "2. ⚙️ Configure params\n"
    "3. ▶️ Run backtest\n"
    "4. 📈 View results"
)

st.sidebar.markdown("---")

# Show data status
if "ohlcv_data" in st.session_state and st.session_state.ohlcv_data is not None:
    df = st.session_state.ohlcv_data
    st.sidebar.success(f"✅ Data: {len(df):,} bars")
    st.sidebar.caption(
        f"{df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}"
    )
else:
    st.sidebar.warning("No data loaded")

if "backtest_results" in st.session_state and st.session_state.backtest_results is not None:
    n = len(st.session_state.backtest_results)
    st.sidebar.success(f"✅ {n} trades simulated")

st.sidebar.markdown("---")
st.sidebar.caption(
    "[Tecana](https://github.com/Neogus/tecana) · "
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

    Use the **sidebar** to navigate through the workflow:

    | Step | Page | Description |
    |------|------|-------------|
    | 1 | **📂 Data Source** | Upload a CSV/Parquet file or download from yfinance/ccxt |
    | 2 | **⚙️ Configure** | Set trading parameters, select signals, choose scoring |
    | 3 | **▶️ Run Backtest** | Execute the simulation and see the trade log |
    | 4 | **📈 Results** | Interactive charts, performance metrics, CSV export |

    ### Signal Convention

    This simulator uses `+1 = LONG` / `-1 = SHORT` internally.
    If you use [Tecana](https://pypi.org/project/tecana/) signals (which use
    the opposite convention: `-1 = buy`, `+1 = sell`), the app **automatically
    negates** them so the two systems stay aligned. You don't need to do
    anything — the bridge is built in.

    | Simulator | Tecana (raw) | After auto-negation |
    |-----------|-------------|---------------------|
    | **+1** LONG (bullish) | -1 buy (bullish) | → +1 ✅ |
    | **-1** SHORT (bearish) | +1 sell (bearish) | → -1 ✅ |
    | **0** No signal | 0 no signal | → 0 ✅ |

    ---

    *⚠️ Disclaimer: This tool is for educational and informational purposes only.
    There is no guarantee that calculations are free of errors.
    The author is not responsible for any financial losses.
    Always conduct your own research before making investment decisions.*
    """
)
