"""OHLCV resampling for the Backtest Core (Req 3.6, 13.6).

This module provides :func:`resample_ohlcv`, the single canonical replacement
for the Old Project's ``Indicators2.resample``. The old implementation
capitalized columns and called ``fillna(ffill)`` after resampling, which
silently masked genuine data gaps. The rewrite:

* Enforces the lowercase OHLCV column contract (reusing
  :func:`simulator._helpers.prepare_df`).
* Aggregates each resampled interval as open=first, high=max, low=min,
  close=last, volume=sum (Req 3.6).
* Does **not** force forward-fill; gap handling is left to the caller so that
  genuine gaps remain visible.
* Keeps the DatetimeIndex and exposes explicit ``label``/``closed`` controls.
"""

from __future__ import annotations

import pandas as pd

from simulator._helpers import prepare_df

__all__ = ["resample_ohlcv"]

# Aggregation contract: open=first, high=max, low=min, close=last, volume=sum.
_OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(
    df: pd.DataFrame,
    tf: str,
    *,
    label: str = "right",
    closed: str = "right",
) -> pd.DataFrame:
    """Resample a lowercase OHLCV frame to the caller-specified frequency.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV market data with a :class:`~pandas.DatetimeIndex`. Column names
        are lowercased and validated; the required columns are ``open``,
        ``high``, ``low``, ``close`` and ``volume``. The caller's frame is
        never mutated.
    tf : str
        Target resampling frequency understood by :meth:`pandas.DataFrame.resample`
        (for example ``"5min"``, ``"1h"``, ``"1D"``).
    label : {"right", "left"}, optional
        Which bin edge labels each resampled bucket. Defaults to ``"right"``.
    closed : {"right", "left"}, optional
        Which bin edge is closed (inclusive). Defaults to ``"right"``.

    Returns
    -------
    pandas.DataFrame
        A frame indexed by a :class:`~pandas.DatetimeIndex` with columns
        ``open``, ``high``, ``low``, ``close``, ``volume`` aggregated within
        each resampled interval. No forward-fill is applied, so intervals with
        no source rows surface as missing values rather than being masked.

    Raises
    ------
    ValueError
        If any required OHLCV column is missing after lowercasing, or if the
        frame is not indexed by a :class:`~pandas.DatetimeIndex`.
    """
    prepared = prepare_df(df, list(_OHLCV_AGG))

    if not isinstance(prepared.index, pd.DatetimeIndex):
        raise ValueError(
            "resample_ohlcv requires a DatetimeIndex; got index of type "
            f"{type(prepared.index).__name__}"
        )

    resampled = prepared.resample(_normalize_freq(tf), label=label, closed=closed).agg(_OHLCV_AGG)
    # Preserve the canonical OHLCV column order regardless of source ordering.
    return resampled[list(_OHLCV_AGG)]


def _normalize_freq(tf: str) -> str:
    """Convert user-facing timeframe strings to pandas-compatible frequency aliases.

    Pandas 2.2+ deprecated 'm' (month-end); minute-based frequencies must use
    'min' or 'T'. This maps common crypto timeframe notation to valid pandas offsets.
    """
    # If it already ends with 'min', 'h', 'D', 'W' etc — leave it alone
    if tf.endswith("min") or tf.endswith("T"):
        return tf
    # Map shorthand minute notation: "1m" -> "1min", "5m" -> "5min", etc.
    if tf.endswith("m") and tf[:-1].isdigit():
        return tf[:-1] + "min"
    # Map day notation: "1d" -> "1D", "2d" -> "2D"
    if tf.endswith("d") and tf[:-1].isdigit():
        return tf[:-1] + "D"
    # Map week notation: "1w" -> "1W"
    if tf.endswith("w") and tf[:-1].isdigit():
        return tf[:-1] + "W"
    return tf
