"""Page 4: Results Dashboard — interactive charts, metrics, export."""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.metrics import total_return, max_drawdown, sharpe, accuracy, alpha, capm_beta

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")

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

st.title("📈 Results Dashboard")

if "backtest_results" not in st.session_state or st.session_state.backtest_results is None:
    st.warning("⚠️ No results yet. Go to **▶️ Run Backtest** first.")
    st.stop()

detail = st.session_state.backtest_results
returns = st.session_state.backtest_returns
df = st.session_state.ohlcv_data

if len(returns) == 0:
    st.warning("No trades were generated.")
    st.stop()

# ── Performance Metrics ────────────────────────────────────────────────

st.markdown("### 📊 Performance Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", f"{len(returns)}")
col2.metric("Total Return", f"{total_return(returns):.2%}")
col3.metric("Max Drawdown", f"{max_drawdown(returns):.2%}")

if len(returns) > 1:
    col4.metric("Sharpe Ratio", f"{sharpe(returns):.3f}")
else:
    col4.metric("Sharpe Ratio", "N/A")

col5.metric("Win Rate", f"{accuracy(returns):.1%}")

wins = returns[returns > 0]
losses = returns[returns < 0]
avg_win = wins.mean() if len(wins) > 0 else 0
avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")
col6.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞")

# ── Equity Curve ───────────────────────────────────────────────────────

st.markdown("### 💰 Equity Curve")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

equity = (1 + returns).cumprod()
equity_with_start = pd.concat([pd.Series([1.0], index=[equity.index[0]]), equity])

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     vertical_spacing=0.08, row_heights=[0.65, 0.35],
                     subplot_titles=["Equity Curve", "Drawdown"])

# Equity
fig.add_trace(go.Scatter(
    x=equity_with_start.index, y=equity_with_start.values,
    mode="lines", name="Equity",
    line=dict(color="royalblue", width=2),
    fill="tozeroy", fillcolor="rgba(65,105,225,0.1)",
), row=1, col=1)

fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)

# Drawdown
peak = equity.cummax()
dd = (equity - peak) / peak
fig.add_trace(go.Scatter(
    x=dd.index, y=dd.values,
    mode="lines", name="Drawdown",
    line=dict(color="red", width=1.5),
    fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
), row=2, col=1)

fig.update_layout(height=550, showlegend=True, hovermode="x unified")
fig.update_yaxes(title_text="Equity", row=1, col=1)
fig.update_yaxes(title_text="Drawdown %", tickformat=".1%", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── Trade Distribution ─────────────────────────────────────────────────

st.markdown("### 📊 Trade Analysis")

col1, col2 = st.columns(2)

with col1:
    # Returns histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns.values, nbinsx=30, name="Returns",
        marker_color=np.where(returns.values > 0, "green", "red"),
    ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
    fig_hist.update_layout(title="Return Distribution", xaxis_title="Return",
                           yaxis_title="Count", height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Win/Loss breakdown
    n_wins = (returns > 0).sum()
    n_losses = (returns <= 0).sum()

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Wins", "Losses"],
        values=[n_wins, n_losses],
        marker_colors=["green", "red"],
        hole=0.4,
    )])
    fig_pie.update_layout(title=f"Win/Loss: {n_wins}/{n_losses}", height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Price Chart with Trades ────────────────────────────────────────────

if "entry_time" in detail.columns and "exit_time" in detail.columns:
    st.markdown("### 📍 Trades on Price Chart")

    fig_price = go.Figure()

    # Candlestick
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color="green", decreasing_line_color="red",
        opacity=0.5,
    ))

    # Trade entries
    longs = detail[detail["direction"] == 1]
    shorts = detail[detail["direction"] == -1]

    if len(longs) > 0:
        fig_price.add_trace(go.Scatter(
            x=longs["entry_time"], y=longs["entry_price"],
            mode="markers", name="Long Entry",
            marker=dict(symbol="triangle-up", size=12, color="green"),
        ))
    if len(shorts) > 0:
        fig_price.add_trace(go.Scatter(
            x=shorts["entry_time"], y=shorts["entry_price"],
            mode="markers", name="Short Entry",
            marker=dict(symbol="triangle-down", size=12, color="red"),
        ))

    # Trade exits
    if "exit_time" in detail.columns:
        winning = detail[detail["ret"] > 0]
        losing = detail[detail["ret"] <= 0]
        if len(winning) > 0:
            fig_price.add_trace(go.Scatter(
                x=winning["exit_time"], y=winning["exit_price"],
                mode="markers", name="Win Exit",
                marker=dict(symbol="circle", size=8, color="lime"),
            ))
        if len(losing) > 0:
            fig_price.add_trace(go.Scatter(
                x=losing["exit_time"], y=losing["exit_price"],
                mode="markers", name="Loss Exit",
                marker=dict(symbol="circle", size=8, color="orangered"),
            ))

    fig_price.update_layout(
        title="Trade Entries & Exits", height=500,
        xaxis_rangeslider_visible=False, hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ── Detailed Stats ─────────────────────────────────────────────────────

st.markdown("### 📋 Detailed Statistics")

stats = {
    "Total Trades": len(returns),
    "Winning Trades": int((returns > 0).sum()),
    "Losing Trades": int((returns <= 0).sum()),
    "Win Rate": f"{accuracy(returns):.2%}",
    "Total Return": f"{total_return(returns):.4%}",
    "Max Drawdown": f"{max_drawdown(returns):.4%}",
    "Avg Win": f"{avg_win:.4%}",
    "Avg Loss": f"{-abs(avg_loss):.4%}",
    "Profit Factor": f"{profit_factor:.2f}",
    "Best Trade": f"{returns.max():.4%}",
    "Worst Trade": f"{returns.min():.4%}",
}

if len(returns) > 1:
    stats["Sharpe Ratio"] = f"{sharpe(returns):.4f}"
    stats["Return Std Dev"] = f"{returns.std():.4%}"

col1, col2 = st.columns(2)
items = list(stats.items())
mid = len(items) // 2
with col1:
    for k, v in items[:mid]:
        st.markdown(f"**{k}:** {v}")
with col2:
    for k, v in items[mid:]:
        st.markdown(f"**{k}:** {v}")

# ── Export ─────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 💾 Export")

col1, col2 = st.columns(2)

with col1:
    csv = detail.to_csv(index=False)
    st.download_button(
        "📥 Download Trade Log (CSV)", csv,
        file_name="trade_log.csv", mime="text/csv",
    )

with col2:
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    st.download_button(
        "📥 Download Metrics (CSV)", stats_df.to_csv(index=False),
        file_name="metrics.csv", mime="text/csv",
    )

st.markdown("---")
st.caption(
    "⚠️ *Disclaimer: Results are based on historical data simulation. "
    "There is no guarantee that calculations are error-free. "
    "Past performance does not predict future results. "
    "The author is not responsible for any financial losses.*"
)
