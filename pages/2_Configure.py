"""Page 2: Configure Backtest — trading params, signals, scoring."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Configure Backtest", page_icon="⚙️", layout="wide")
st.title("⚙️ Configure Backtest")

if "ohlcv_data" not in st.session_state or st.session_state.ohlcv_data is None:
    st.warning("⚠️ No data loaded. Go to **📂 Data Source** first.")
    st.stop()

st.success(f"Dataset: {st.session_state.get('data_source', 'Unknown')} — {len(st.session_state.ohlcv_data):,} bars")

# ── Trading Parameters ─────────────────────────────────────────────────

st.markdown("### 🎯 Trading Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    sl_mode = st.radio("SL/TP Mode", ["Fixed", "Search Range"], horizontal=True)

if sl_mode == "Fixed":
    with col1:
        sl_mult = st.number_input("Stop Loss (ATR multiple)", min_value=0.1, max_value=50.0,
                                   value=1.5, step=0.1, format="%.1f")
    with col2:
        tp_mult = st.number_input("Take Profit (ATR multiple)", min_value=0.1, max_value=50.0,
                                   value=2.0, step=0.1, format="%.1f")
    with col3:
        atr_window = st.number_input("ATR Window", min_value=1, max_value=500, value=14, step=1)

    st.session_state.sl_config = {"mode": "fixed", "sl_mult": sl_mult, "tp_mult": tp_mult, "atr_window": atr_window}
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Stop Loss Range:**")
        sl_min = st.number_input("SL Min", value=0.5, step=0.5, format="%.1f")
        sl_max = st.number_input("SL Max", value=3.0, step=0.5, format="%.1f")
        sl_step = st.number_input("SL Step", value=0.5, step=0.1, format="%.1f")
    with col_b:
        st.markdown("**Take Profit Range:**")
        tp_min = st.number_input("TP Min", value=1.0, step=0.5, format="%.1f")
        tp_max = st.number_input("TP Max", value=5.0, step=0.5, format="%.1f")
        tp_step = st.number_input("TP Step", value=0.5, step=0.1, format="%.1f")

    atr_window = st.number_input("ATR Window", min_value=1, max_value=500, value=14, step=1, key="atr_search")

    st.session_state.sl_config = {
        "mode": "search",
        "sl_range": (sl_min, sl_max, sl_step),
        "tp_range": (tp_min, tp_max, tp_step),
        "atr_window": atr_window,
    }

# Fees & slippage
st.markdown("### 💰 Costs")
col1, col2, col3 = st.columns(3)
with col1:
    fee = st.number_input("Fee (per side)", min_value=0.0, max_value=0.1,
                           value=0.0004, step=0.0001, format="%.4f",
                           help="Trading fee as fraction (e.g. 0.0004 = 0.04%)")
with col2:
    slippage = st.number_input("Slippage (per side)", min_value=0.0, max_value=0.1,
                                value=0.0002, step=0.0001, format="%.4f")
with col3:
    max_hold = st.number_input("Max Hold (bars)", min_value=0, max_value=10000,
                                value=200, step=10,
                                help="Force-close after N bars. 0 = no limit.")
    max_hold = max_hold if max_hold > 0 else None

st.session_state.trading_params = {
    "fee": fee, "slippage": slippage, "max_hold": max_hold,
    "intrabar": "sl_first",
}

# ── Signal Selection ───────────────────────────────────────────────────

st.markdown("### 📶 Signal Selection")

# Check if tecana is available
try:
    import tecana
    TECANA_AVAILABLE = True
    all_signals = [m for m in sorted(tecana.__all__) if "_" in m and not m.startswith("_")]
except ImportError:
    TECANA_AVAILABLE = False
    all_signals = []

if TECANA_AVAILABLE:
    st.caption(f"Tecana v{tecana.__version__} — {len(all_signals)} signals available")

    # Group signals by type
    momentum = [s for s in all_signals if s.endswith("_m")]
    zone = [s for s in all_signals if s.endswith("_z")]
    trend = [s for s in all_signals if s.endswith("_t")]
    volatility = [s for s in all_signals if s.endswith("_v")]

    tab_m, tab_z, tab_t, tab_v = st.tabs([
        f"Momentum ({len(momentum)})",
        f"Zone ({len(zone)})",
        f"Trend ({len(trend)})",
        f"Volatility ({len(volatility)})",
    ])

    with tab_m:
        sel_m = st.multiselect("Momentum signals", momentum,
                                default=["rsi_m", "macd_m"] if "rsi_m" in momentum else momentum[:2])
    with tab_z:
        sel_z = st.multiselect("Zone signals", zone,
                                default=["rsi_z"] if "rsi_z" in zone else [])
    with tab_t:
        sel_t = st.multiselect("Trend signals", trend, default=[])
    with tab_v:
        sel_v = st.multiselect("Volatility flags", volatility, default=[])

    selected_signals = sel_m + sel_z + sel_t + sel_v
else:
    st.warning("⚠️ Tecana not installed. Using random signals for demo. Install with: `pip install tecana`")
    selected_signals = ["random_signal"]

st.session_state.selected_signals = selected_signals
if selected_signals:
    st.info(f"**{len(selected_signals)} signals selected:** {', '.join(selected_signals)}")

# ── Scoring ────────────────────────────────────────────────────────────

st.markdown("### 🏆 Scoring")

from simulator.scoring import SCORE_FORMULAS

col1, col2 = st.columns(2)
with col1:
    formula = st.selectbox("Scoring Formula", list(SCORE_FORMULAS.keys()),
                            help="How to rank strategies. A=simple Sharpe, H=most comprehensive.")
with col2:
    st.markdown(f"**Formula {formula}** function: `{SCORE_FORMULAS[formula].__name__}`")

st.session_state.scoring = {"formula": formula}

# ── Summary ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### ✅ Configuration Summary")

config = {
    "SL/TP": st.session_state.sl_config,
    "Costs": st.session_state.trading_params,
    "Signals": len(selected_signals),
    "Scoring": formula,
}
st.json(config)

st.info("Configuration saved. Go to **▶️ Run Backtest** to execute.")
