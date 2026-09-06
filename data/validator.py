"""OHLCV data quality validation — gaps, duplicates, sort order, corruption."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["validate_ohlcv", "ValidationReport", "fix_issues"]


@dataclass
class ValidationReport:
    """Results of OHLCV data quality checks."""

    rows: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    missing_bars: int = 0
    corrupted: int = 0
    nan_values: int = 0
    negative_prices: int = 0
    high_lt_low: int = 0
    has_datetime_index: bool = True
    detected_timeframe: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicates == 0
            and self.out_of_order == 0
            and self.corrupted == 0
            and self.nan_values == 0
        )

    def summary(self) -> str:
        lines = [f"Rows: {self.rows}"]
        if self.detected_timeframe:
            lines.append(f"Detected timeframe: {self.detected_timeframe}")
        if self.is_clean:
            lines.append("✅ Data is clean — no issues found.")
        else:
            if self.duplicates:
                lines.append(f"⚠️  Duplicate timestamps: {self.duplicates}")
            if self.out_of_order:
                lines.append(f"⚠️  Out-of-order rows: {self.out_of_order}")
            if self.missing_bars:
                lines.append(f"⚠️  Missing bars (estimated): {self.missing_bars}")
            if self.nan_values:
                lines.append(f"⚠️  NaN values: {self.nan_values}")
            if self.negative_prices:
                lines.append(f"⚠️  Negative prices: {self.negative_prices}")
            if self.high_lt_low:
                lines.append(f"⚠️  High < Low rows: {self.high_lt_low}")
        return "\n".join(lines)


def _detect_timeframe(index: pd.DatetimeIndex) -> str:
    """Detect the most common bar interval from timestamps."""
    if len(index) < 2:
        return ""
    deltas = index.to_series().diff().dropna()
    median_sec = int(deltas.median().total_seconds())

    tf_map = {
        60: "1m", 180: "3m", 300: "5m", 600: "10m", 900: "15m",
        1200: "20m", 1800: "30m", 3600: "1h", 7200: "2h",
        10800: "3h", 14400: "4h", 21600: "6h", 28800: "8h",
        43200: "12h", 86400: "1d", 604800: "1w",
    }

    for expected_sec, label in tf_map.items():
        if abs(median_sec - expected_sec) / max(expected_sec, 1) < 0.15:
            return label
    return f"{median_sec}s"


def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Run all data quality checks on an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have lowercase columns (open, high, low, close) and
        ideally a DatetimeIndex.

    Returns
    -------
    ValidationReport
        Detailed results of each check.
    """
    report = ValidationReport(rows=len(df))

    # Check datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        report.has_datetime_index = False
        report.issues.append("Index is not a DatetimeIndex")
    else:
        # Detect timeframe
        report.detected_timeframe = _detect_timeframe(df.index)

        # Duplicates
        dup_mask = df.index.duplicated(keep="first")
        report.duplicates = int(dup_mask.sum())
        if report.duplicates:
            report.issues.append(f"{report.duplicates} duplicate timestamps")

        # Sort order
        if not df.index.is_monotonic_increasing:
            diffs = np.diff(df.index.asi8)
            report.out_of_order = int((diffs < 0).sum())
            report.issues.append(f"{report.out_of_order} out-of-order rows")

        # Missing bars (estimated)
        if report.detected_timeframe and len(df) >= 2:
            expected = pd.date_range(
                df.index.min(), df.index.max(),
                freq=report.detected_timeframe
            )
            report.missing_bars = max(0, len(expected) - len(df))
            if report.missing_bars > 0:
                report.issues.append(f"~{report.missing_bars} missing bars")

    # NaN values
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    nan_mask = df[ohlcv_cols].isna()
    report.nan_values = int(nan_mask.values.sum())
    if report.nan_values:
        report.issues.append(f"{report.nan_values} NaN values")

    # Negative prices
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    neg_mask = df[price_cols] < 0
    report.negative_prices = int(neg_mask.values.sum())
    if report.negative_prices:
        report.issues.append(f"{report.negative_prices} negative prices")

    # High < Low
    if "high" in df.columns and "low" in df.columns:
        hl_mask = df["high"] < df["low"]
        report.high_lt_low = int(hl_mask.sum())
        if report.high_lt_low:
            report.issues.append(f"{report.high_lt_low} rows where high < low")

    report.corrupted = report.negative_prices + report.high_lt_low

    return report


def fix_issues(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    sort: bool = True,
    fill_strategy: str = "ffill",
) -> pd.DataFrame:
    """Apply common auto-fixes to an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    drop_duplicates : bool, default True
        Remove duplicate timestamp rows (keep first).
    sort : bool, default True
        Sort by index.
    fill_strategy : str, default 'ffill'
        How to fill NaN values: 'ffill', 'interpolate', or 'drop'.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the data.
    """
    df = df.copy()

    if sort and isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    if drop_duplicates and isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="first")]

    # Fix NaN values
    if fill_strategy == "ffill":
        df = df.ffill()
    elif fill_strategy == "interpolate":
        df = df.interpolate(method="linear")
    elif fill_strategy == "drop":
        df = df.dropna()

    return df
