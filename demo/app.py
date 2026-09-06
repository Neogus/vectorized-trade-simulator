"""Streamlit Cloud Demo — resource-capped version with sample data."""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Trade Simulator Demo", page_icon="📊", layout="wide")

# ── Resource caps ──────────────────────────────────────────────────────

MAX_BARS = 500
MAX_TIMEOUT_SEC = 30

# ── Sidebar ────────────────────────────────────────────────────────────

st.sidebar.title("📊 Demo Mode")
st.sidebar.info(f"⚡ Cloud limits: {MAX_BARS} bars max, {MAX_TIMEOUT_SEC}s timeout")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Full version:**\n"
    "```bash\n"
    "git clone https://github.com/Neogus/vectorized-trade-simulator\n"
    "pip install -r requirements.txt\n"
    "streamlit run app.py\n"
    "```"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "[GitHub](https://github.com/Neogus/vectorized-trade-simulator) · "
    "[Tecana](https://github.com/Neogus/tecana)"
)
st.sidebar.caption(
    "☕ [Support this project](https://github.com/sponsors/Neogus) · "
    "[Ko-fi](https://ko-fi.com/neogus43222)"
)

# ── Main ───────────────────────────────────────────────────────────────

st.title("📊 Vectorized Trade Simulator — Demo")
st.markdown(
    "Try the backtesting engine with sample or uploaded data. "
    f"**Cloud demo** is limited to {MAX_BARS} bars — install locally for full power."
)

# ── Step 1: Data ───────────────────────────────────────────────────────

st.markdown("### 1️⃣ Load Data")

data_source = st.radio("Choose data source:", ["📦 Sample Data (BTC-like)", "📁 Upload CSV"], horizontal=True)

if data_source == "📦 Sample Data (BTC-like)":
    np.random.seed(42)
    n = MAX_BARS
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 40000 + np.random.randn(n).cumsum() * 100
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 30,
        "high": close + abs(np.random.randn(n)) * 80,
        "low": close - abs(np.random.randn(n)) * 80,
        "close": close,
        "volume": np.random.randint(100, 5000, n).astype(float),
    }, index=dates)
    df.index.name = "datetime"
    st.success(f"✅ Sample data: {len(df)} bars of BTC-like hourly data")
else:
    uploaded = st.file_uploader("Upload OHLCV file", type=["csv", "parquet", "xlsx"])
    if uploaded is None:
        st.info("👆 Upload a file to continue")
        st.stop()

    import tempfile, os
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    try:
        from data.parser import load_ohlcv
        df = load_ohlcv(tmp_path)
        if len(df) > MAX_BARS:
            df = df.tail(MAX_BARS)
            st.warning(f"⚡ Trimmed to last {MAX_BARS} bars for cloud demo.")
        st.success(f"✅ Loaded {len(df)} bars from {uploaded.name}")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)

# Price preview
import plotly.graph_objects as go

fig = go.Figure(go.Candlestick(
    x=df.index, open=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
))
fig.update_layout(title="Price Data", height=350, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ── Step 2: Configure ─────────────────────────────────────────────────

st.markdown("### 2️⃣ Configure & Run")

col1, col2, col3, col4 = st.columns(4)
with col1:
    sl_mult = st.number_input("Stop Loss (ATR×)", value=1.5, step=0.5, format="%.1f")
with col2:
    tp_mult = st.number_input("Take Profit (ATR×)", value=2.0, step=0.5, format="%.1f")
with col3:
    atr_window = st.number_input("ATR Window", value=14, min_value=5, max_value=50)
with col4:
    fee = st.number_input("Fee", value=0.0004, step=0.0001, format="%.4f")

# Simple signal generation (no tecana dependency for cloud)
st.markdown("**Signal method:**")
signal_method = st.selectbox("", [
    "SMA Crossover (fast/slow)",
    "RSI Crossover (overbought/oversold)",
    "Random Entries (demo only)",
])

# ── Step 3: Run ───────────────────────────────────────────────────────

if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
    from simulator.returns import simulate_trades_detail
    from simulator.metrics import total_return, max_drawdown, sharpe, accuracy

    progress = st.progress(0, text="Computing signals...")

    # Generate entries based on method
    entries = np.zeros(len(df), dtype=np.int8)

    if signal_method.startswith("RSI"):
        # Simple RSI crossover
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        prev_rsi = rsi.shift(1)
        entries[(prev_rsi < 35) & (rsi >= 35)] = 1   # Long on oversold exit
        entries[(prev_rsi > 65) & (rsi <= 65)] = -1   # Short on overbought exit

    elif signal_method.startswith("SMA"):
        fast = df["close"].rolling(10).mean()
        slow = df["close"].rolling(30).mean()
        prev_fast, prev_slow = fast.shift(1), slow.shift(1)
        entries[(prev_fast <= prev_slow) & (fast > slow)] = 1   # Golden cross
        entries[(prev_fast >= prev_slow) & (fast < slow)] = -1  # Death cross

    else:
        np.random.seed(42)
        locs = np.random.choice(range(50, len(df) - 20), size=min(15, len(df) // 30), replace=False)
        for i, loc in enumerate(sorted(locs)):
            entries[loc] = 1 if i % 2 == 0 else -1

    n_entries = np.count_nonzero(entries)
    progress.progress(30, text=f"Generated {n_entries} entry signals...")

    if n_entries == 0:
        st.warning("No entry signals generated. Try a different method or data.")
        st.stop()

    progress.progress(60, text="Simulating trades...")

    try:
        detail = simulate_trades_detail(
            df, entries, sl_mult=sl_mult, tp_mult=tp_mult,
            atr_window=int(atr_window), fee=fee,
        )
        returns = detail["ret"] if len(detail) > 0 else pd.Series(dtype=float)
        progress.progress(100, text="Done!")
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    # ── Results ────────────────────────────────────────────────────────

    if len(returns) == 0:
        st.warning("No trades completed. Adjust parameters.")
        st.stop()

    st.markdown("### 3️⃣ Results")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trades", len(returns))
    col2.metric("Return", f"{total_return(returns):.2%}")
    col3.metric("Max DD", f"{max_drawdown(returns):.2%}")
    col4.metric("Sharpe", f"{sharpe(returns):.2f}" if len(returns) > 1 else "N/A")
    col5.metric("Win Rate", f"{accuracy(returns):.1%}")

    # Equity curve + drawdown
    from plotly.subplots import make_subplots

    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         row_heights=[0.65, 0.35], subplot_titles=["Equity Curve", "Drawdown"])
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity",
                              line=dict(color="royalblue", width=2),
                              fill="tozeroy", fillcolor="rgba(65,105,225,0.1)"), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                              line=dict(color="red", width=1.5),
                              fill="tozeroy", fillcolor="rgba(255,0,0,0.1)"), row=2, col=1)
    fig.update_layout(height=450, hovermode="x unified")
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Trade log
    st.markdown("**Trade Log:**")
    st.dataframe(detail, use_container_width=True, height=250)

    # Export
    st.download_button("📥 Download CSV", detail.to_csv(index=False),
                        file_name="demo_trades.csv", mime="text/csv")

st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(
        "⚠️ *Demo mode with limited data. Install locally for full features. "
        "This is for educational purposes only — not financial advice.*"
    )
with col2:
    st.markdown(
        '<p style="text-align:right; font-size:0.85em; opacity:0.7;">'
        '☕ <a href="https://github.com/sponsors/Neogus">Sponsor</a> · '
        '<a href="https://ko-fi.com/neogus43222">Ko-fi</a></p>',
        unsafe_allow_html=True,
    )
