"""Vectorized Trade Simulator — a standalone backtesting engine.

Provides vectorized trade simulation with ATR-based stop-loss / take-profit
barriers, performance metrics, scoring formulas, and signal management.

Requires: numpy, pandas. Optional: numba (for JIT-compiled simulation),
tecana (for indicator-derived signals).
"""

__version__ = "2.0.0"

# Core simulation
from simulator.returns import (
    simulate_trades,
    simulate_trades_detail,
    compute_atr_barriers,
)

# Performance metrics
from simulator.metrics import (
    total_return,
    max_drawdown,
    alpha,
    capm_beta,
    sharpe,
    accuracy,
)

# Signals
from simulator.signals import (
    Signal,
    encode_entries,
    aggregate_signals,
)

# Scoring
from simulator.scoring import (
    SCORE_FORMULAS,
    register_score,
    get_score,
)

# Utilities
from simulator.resample import resample_ohlcv
from simulator.decay import adjust_data_decay
from simulator._helpers import wilder_rma, prepare_df

__all__ = [
    # Simulation
    "simulate_trades",
    "simulate_trades_detail",
    "compute_atr_barriers",
    # Metrics
    "total_return",
    "max_drawdown",
    "alpha",
    "capm_beta",
    "sharpe",
    "accuracy",
    # Signals
    "Signal",
    "encode_entries",
    "aggregate_signals",
    # Scoring
    "SCORE_FORMULAS",
    "register_score",
    "get_score",
    # Utilities
    "resample_ohlcv",
    "adjust_data_decay",
    "wilder_rma",
    "prepare_df",
]
