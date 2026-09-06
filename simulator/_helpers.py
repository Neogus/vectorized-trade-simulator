"""Shared utility functions (standalone replacements for tkit/tecana internals)."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["wilder_rma", "prepare_df"]


def wilder_rma(series: pd.Series, window: int) -> pd.Series:
    """Wilder's Running Moving Average (RMA / SMMA).

    Equivalent to ``ewm(alpha=1/window, adjust=False)`` seeded by the
    first *window* observations.

    Parameters
    ----------
    series : pd.Series
    window : int

    Returns
    -------
    pd.Series
    """
    return series.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def prepare_df(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Lowercase columns, validate required columns, return a copy.

    Parameters
    ----------
    df : pd.DataFrame
        Input market data.
    required_cols : list[str] | None
        Column names (lowercase) that must be present.

    Returns
    -------
    pd.DataFrame
        A copy with lowercased column names.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    return df
