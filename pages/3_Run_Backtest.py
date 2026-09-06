"""Page 3: Run Backtest — execute simulation with progress."""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.returns import simulate_trades, simulate_trades_detail
from simulator.metrics import total_return, max_drawdown, sharpe, accuracy
from simulator.signals import Signal

st.set_page_config(page_title="Run Backtest", page_icon="▶️", layout="wide")

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
    "☕ [Support](https://github.com/sponsors/Neogus) · "
    "[Ko-fi](https://ko-fi.com/neogus43222) · "
    "[BMC](https://buymeacoffee.com/neogus)"
)

st.title("▶️ Run Backtest")

# ── Validate prerequisites ─────────────────────────────────────────────

if "ohlcv_data" not in st.session_state or st.session_state.ohlcv_data is None:
    st.warning("⚠️ No data loaded. Go to **📂 Data Source** first.")
    st.stop()

if "sl_config" not in st.session_state:
    st.warning("⚠️ Not configured. Go to **⚙️ Configure** first.")
    st.stop()

df = st.session_state.ohlcv_data
config = st.session_state.sl_config
params = st.session_state.trading_params
signals = st.session_state.get("selected_signals", [])

st.success(f"**{len(df):,} bars** | SL/TP: {config.get('sl_mult', 'search')} / {config.get('tp_mult', 'search')} | Signals: {len(signals)}")

# ── Generate signals ───────────────────────────────────────────────────

def compute_entries(df, signal_names, agg_mode="Any signal (union)"):
    """Compute aggregated entry signals from selected signal names."""

    # Built-in RSI fallback (no tecana or explicit builtin)
    if signal_names == ["builtin_rsi"] or not signal_names:
        return _builtin_rsi_sma(df)

    try:
        import tecana
        ta = tecana.Tecana()
    except ImportError:
        return _builtin_rsi_sma(df)

    # Compute each signal via Tecana
    signal_arrays = []
    work_df = df.copy()

    for sig_name in signal_names:
        try:
            method = getattr(ta, sig_name)
            result = method(work_df)
            raw = result[sig_name].to_numpy(dtype=np.int8)
            # Negate to canonical: Tecana +1=sell → canonical -1=SHORT
            canonical = -raw
            signal_arrays.append(canonical)
        except Exception:
            continue

    if not signal_arrays:
        return _builtin_rsi_sma(df)

    # Aggregate based on mode
    mat = np.column_stack(signal_arrays)  # (rows, k)
    k = mat.shape[1]
    entries = np.zeros(len(df), dtype=np.int8)

    if "Any" in agg_mode or k == 1:
        for i in range(k):
            long_mask = (entries == 0) & (mat[:, i] == 1)
            short_mask = (entries == 0) & (mat[:, i] == -1)
            entries[long_mask] = 1
            entries[short_mask] = -1
    elif "Majority" in agg_mode:
        row_sum = mat.sum(axis=1)
        threshold = k / 2
        entries[row_sum > threshold] = 1
        entries[row_sum < -threshold] = -1
    else:  # Unanimous
        all_long = (mat == 1).all(axis=1)
        all_short = (mat == -1).all(axis=1)
        entries[all_long] = 1
        entries[all_short] = -1

    # Shift by 1 to prevent look-ahead
    entries = np.roll(entries, 1)
    entries[0] = 0

    # If Tecana signals produced nothing, fall back to built-in
    if (entries != 0).sum() == 0:
        return _builtin_rsi_sma(df)

    return entries


def _builtin_rsi_sma(df):
    """Built-in signal generator that always produces entries on any OHLCV data.

    Combines RSI crossovers (with relaxed 35/65 thresholds) and SMA crossovers
    to ensure at least some entries are generated regardless of market conditions.
    """
    entries = np.zeros(len(df), dtype=np.int8)

    # SMA crossover (fast=10, slow=30) — reliable on any dataset
    fast = df["close"].rolling(10).mean()
    slow = df["close"].rolling(30).mean()
    prev_fast, prev_slow = fast.shift(1), slow.shift(1)
    entries[(prev_fast <= prev_slow) & (fast > slow)] = 1   # Golden cross → Long
    entries[(prev_fast >= prev_slow) & (fast < slow)] = -1  # Death cross → Short

    # Also add RSI with relaxed thresholds (35/65 instead of 30/70)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    prev_rsi = rsi.shift(1)
    # Only add RSI entries where SMA didn't already fire
    rsi_long = (prev_rsi < 35) & (rsi >= 35) & (entries == 0)
    rsi_short = (prev_rsi > 65) & (rsi <= 65) & (entries == 0)
    entries[rsi_long] = 1
    entries[rsi_short] = -1

    # Shift to prevent look-ahead
    entries = np.roll(entries, 1)
    entries[0] = 0

    return entries


# ── Run backtest ───────────────────────────────────────────────────────

if st.button("🚀 Run Backtest", type="primary", use_container_width=True):

    progress = st.progress(0, text="Computing signals...")

    if config["mode"] == "fixed":
        # Single run
        sl_mult = config["sl_mult"]
        tp_mult = config["tp_mult"]
        atr_window = config["atr_window"]

        progress.progress(20, text="Computing entry signals...")
        agg_mode = st.session_state.get("aggregation_mode", "Any signal (union)")
        entries = compute_entries(df, signals, agg_mode)
        n_entries = np.count_nonzero(entries)
        st.caption(f"Entry signals: {n_entries} ({(entries == 1).sum()} long, {(entries == -1).sum()} short)")

        progress.progress(50, text=f"Simulating trades (SL={sl_mult}, TP={tp_mult})...")
        try:
            detail = simulate_trades_detail(
                df, entries,
                sl_mult=sl_mult, tp_mult=tp_mult,
                atr_window=atr_window,
                fee=params["fee"], slippage=params["slippage"],
                max_hold=params["max_hold"],
            )
            returns = detail["ret"] if len(detail) > 0 else pd.Series(dtype=float)

            progress.progress(100, text="Done!")

            # Store results
            st.session_state.backtest_results = detail
            st.session_state.backtest_returns = returns
            st.session_state.backtest_entries = entries

        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
            st.stop()

    else:
        # Grid search
        import itertools

        sl_min, sl_max, sl_step = config["sl_range"]
        tp_min, tp_max, tp_step = config["tp_range"]
        atr_window = config["atr_window"]

        sl_values = np.arange(sl_min, sl_max + sl_step / 2, sl_step)
        tp_values = np.arange(tp_min, tp_max + tp_step / 2, tp_step)
        combos = list(itertools.product(sl_values, tp_values))

        progress.progress(10, text="Computing entry signals...")
        agg_mode = st.session_state.get("aggregation_mode", "Any signal (union)")
        entries = compute_entries(df, signals, agg_mode)

        results_list = []
        for i, (sl, tp) in enumerate(combos):
            pct = 10 + int(85 * (i + 1) / len(combos))
            progress.progress(pct, text=f"Testing SL={sl:.1f} TP={tp:.1f} ({i+1}/{len(combos)})...")

            try:
                detail = simulate_trades_detail(
                    df, entries,
                    sl_mult=float(sl), tp_mult=float(tp),
                    atr_window=atr_window,
                    fee=params["fee"], slippage=params["slippage"],
                    max_hold=params["max_hold"],
                )
                ret = detail["ret"] if len(detail) > 0 else pd.Series(dtype=float)

                results_list.append({
                    "sl_mult": float(sl), "tp_mult": float(tp),
                    "trades": len(ret),
                    "total_return": float(total_return(ret)) if len(ret) > 0 else 0,
                    "max_drawdown": float(max_drawdown(ret)) if len(ret) > 0 else 0,
                    "sharpe": float(sharpe(ret)) if len(ret) > 1 else 0,
                    "accuracy": float(accuracy(ret)) if len(ret) > 0 else 0,
                })
            except Exception:
                continue

        progress.progress(100, text="Done!")

        if results_list:
            grid_df = pd.DataFrame(results_list).sort_values("sharpe", ascending=False)
            st.session_state.grid_results = grid_df

            # Use the best combo
            best = grid_df.iloc[0]
            best_detail = simulate_trades_detail(
                df, entries,
                sl_mult=best["sl_mult"], tp_mult=best["tp_mult"],
                atr_window=atr_window,
                fee=params["fee"], slippage=params["slippage"],
                max_hold=params["max_hold"],
            )
            st.session_state.backtest_results = best_detail
            st.session_state.backtest_returns = best_detail["ret"] if len(best_detail) > 0 else pd.Series(dtype=float)
            st.session_state.backtest_entries = entries

# ── Display results ────────────────────────────────────────────────────

if "backtest_results" in st.session_state and st.session_state.backtest_results is not None:
    detail = st.session_state.backtest_results
    returns = st.session_state.backtest_returns

    st.markdown("---")
    st.markdown("### 📊 Results Summary")

    if len(returns) > 0:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Trades", f"{len(returns)}")
        col2.metric("Total Return", f"{total_return(returns):.2%}")
        col3.metric("Max Drawdown", f"{max_drawdown(returns):.2%}")
        col4.metric("Sharpe Ratio", f"{sharpe(returns):.2f}" if len(returns) > 1 else "N/A")
        col5.metric("Win Rate", f"{accuracy(returns):.1%}")

        st.markdown("### 📋 Trade Log")
        st.dataframe(detail, use_container_width=True, height=400)

        # Grid search results
        if "grid_results" in st.session_state:
            st.markdown("### 🔍 Grid Search Results")
            st.dataframe(
                st.session_state.grid_results.style.format({
                    "sl_mult": "{:.1f}", "tp_mult": "{:.1f}",
                    "total_return": "{:.2%}", "max_drawdown": "{:.2%}",
                    "sharpe": "{:.2f}", "accuracy": "{:.1%}",
                }),
                use_container_width=True,
            )
    else:
        st.warning("No trades were generated. Try adjusting the signals or SL/TP parameters.")

    st.info("Go to **📈 Results** for interactive charts and export.")
