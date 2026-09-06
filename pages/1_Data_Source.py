"""Page 1: Data Source — upload file or connect to API."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.parser import load_ohlcv, detect_columns
from data.validator import validate_ohlcv, fix_issues

st.set_page_config(page_title="Data Source", page_icon="📂", layout="wide")

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

st.title("📂 Data Source")

tab_upload, tab_api = st.tabs(["📁 Upload File", "🌐 Download from API"])

# ── Tab 1: File Upload ─────────────────────────────────────────────────

with tab_upload:
    st.markdown("Upload a CSV, Parquet, or Excel file with OHLCV data. Columns are auto-detected.")

    uploaded = st.file_uploader(
        "Choose file",
        type=["csv", "txt", "tsv", "parquet", "xlsx", "xls"],
        help="Supported formats: CSV, TSV, Parquet, Excel. Columns like Open/High/Low/Close are auto-detected.",
    )

    if uploaded is not None:
        # Save to temp file for parser
        import tempfile, os
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            df = load_ohlcv(tmp_path)
            st.session_state.ohlcv_data = df
            st.session_state.data_source = f"File: {uploaded.name}"

            st.success(f"✅ Loaded {len(df):,} bars from **{uploaded.name}**")

            # Show column mapping
            raw = pd.read_csv(tmp_path, nrows=3) if suffix in (".csv", ".txt") else None
            if raw is not None:
                mapping = detect_columns(raw)
                st.caption(f"Column mapping: {mapping}")

            # Preview
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**First 5 rows:**")
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                st.markdown("**Last 5 rows:**")
                st.dataframe(df.tail(), use_container_width=True)

            # Validate
            report = validate_ohlcv(df)
            st.markdown("### Data Quality")
            if report.is_clean:
                st.success(report.summary())
            else:
                st.warning(report.summary())
                if st.button("🔧 Auto-fix issues"):
                    df = fix_issues(df)
                    st.session_state.ohlcv_data = df
                    report2 = validate_ohlcv(df)
                    st.success(f"Fixed! {report2.summary()}")
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
        finally:
            os.unlink(tmp_path)

# ── Tab 2: API Download ───────────────────────────────────────────────

with tab_api:
    source = st.radio("Data source:", ["Yahoo Finance (Equities)", "Crypto Exchange (ccxt)"], horizontal=True)

    if source == "Yahoo Finance (Equities)":
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Ticker symbol", value="AAPL", help="e.g. AAPL, MSFT, BTC-USD, ETH-USD")
        with col2:
            interval = st.selectbox("Interval", ["1d", "1h", "5m", "15m", "30m", "1wk"], index=0)

        col3, col4 = st.columns(2)
        with col3:
            start = st.date_input("Start date", value=pd.Timestamp("2023-01-01"))
        with col4:
            end = st.date_input("End date", value=pd.Timestamp.now())

        if st.button("📥 Download", type="primary"):
            with st.spinner(f"Downloading {ticker}..."):
                try:
                    from data.sources import download_yfinance
                    df = download_yfinance(ticker, start=str(start), end=str(end), interval=interval)
                    st.session_state.ohlcv_data = df
                    st.session_state.data_source = f"yfinance: {ticker} ({interval})"
                    st.success(f"✅ Downloaded {len(df):,} bars for **{ticker}**")
                    st.dataframe(df.tail(10), use_container_width=True)
                except ImportError:
                    st.error("❌ yfinance not installed. Run: `pip install yfinance`")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")

    else:  # Crypto
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", value="BTC/USDT", help="e.g. BTC/USDT, ETH/USDT")
        with col2:
            exchange = st.selectbox("Exchange", ["binance", "bybit", "coinbase", "kraken", "kucoin", "okx"])
        with col3:
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)

        limit = st.slider("Number of bars", min_value=100, max_value=5000, value=1000, step=100)

        if st.button("📥 Download", type="primary", key="ccxt_download"):
            with st.spinner(f"Downloading {symbol} from {exchange}..."):
                try:
                    from data.sources import download_ccxt
                    df = download_ccxt(symbol, exchange, timeframe=timeframe, limit=limit)
                    st.session_state.ohlcv_data = df
                    st.session_state.data_source = f"ccxt: {symbol} @ {exchange} ({timeframe})"
                    st.success(f"✅ Downloaded {len(df):,} bars for **{symbol}**")
                    st.dataframe(df.tail(10), use_container_width=True)
                except ImportError:
                    st.error("❌ ccxt not installed. Run: `pip install ccxt`")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")

# ── Data summary (always visible if loaded) ────────────────────────────

st.markdown("---")
if "ohlcv_data" in st.session_state and st.session_state.ohlcv_data is not None:
    df = st.session_state.ohlcv_data
    st.markdown(f"### 📊 Current Dataset: {st.session_state.get('data_source', 'Unknown')}")

    # Price chart
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    ))
    fig.update_layout(
        title="Price Chart", xaxis_title="Date", yaxis_title="Price",
        height=450, xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("✅ Data is ready. Go to **⚙️ Configure** to set up your backtest.")
else:
    st.info("👆 Upload a file or download data to get started.")
