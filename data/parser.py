"""Smart OHLCV file parser with auto-detection of format, columns, and dates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_ohlcv", "detect_columns", "COLUMN_ALIASES"]

# ── Column alias sets (lowercase, stripped, underscored) ───────────────

COLUMN_ALIASES: dict[str, set[str]] = {
    "datetime": {"datetime", "date", "timestamp", "time", "date_time", "dt",
                 "date/time", "unix", "epoch", "utc", "ts"},
    "open":     {"open", "o", "open_price", "first", "price_open"},
    "high":     {"high", "h", "high_price", "max", "price_high"},
    "low":      {"low", "l", "low_price", "min", "price_low"},
    "close":    {"close", "c", "close_price", "last", "adj_close",
                 "adj close", "adjusted_close", "adjusted close", "price_close"},
    "volume":   {"volume", "vol", "v", "qty", "quantity", "amount",
                 "base_volume", "quote_volume"},
}


def _normalize_col(name: str) -> str:
    """Normalize a column name for alias matching."""
    return name.strip().lower().replace(" ", "_")


def _guess_role(col_name: str) -> str | None:
    """Guess OHLCV role from column name, or None if unrecognized."""
    norm = _normalize_col(col_name)
    for role, aliases in COLUMN_ALIASES.items():
        if norm in aliases:
            return role
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Auto-detect OHLCV column mapping.

    Returns
    -------
    dict
        Maps role ('datetime', 'open', 'high', 'low', 'close', 'volume')
        to the original column name in ``df``. Missing roles are omitted.
    """
    mapping = {}
    for col in df.columns:
        role = _guess_role(col)
        if role and role not in mapping:
            mapping[role] = col
    return mapping


def _detect_delimiter(path: Path) -> str:
    """Probe CSV delimiter by trying common separators."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        head = "".join(f.readline() for _ in range(5))

    for sep in [",", ";", "\t", "|"]:
        cols = head.split("\n")[0].split(sep)
        if len(cols) >= 4:
            return sep
    return ","


def _load_raw(path: Path) -> pd.DataFrame:
    """Load a raw file based on extension, with delimiter probing for CSV."""
    suffix = path.suffix.lower()

    if suffix in (".csv", ".txt", ".tsv"):
        sep = _detect_delimiter(path)
        return pd.read_csv(path, sep=sep, engine="python")

    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    elif suffix == ".parquet":
        return pd.read_parquet(path)

    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: .csv, .txt, .tsv, .xlsx, .xls, .parquet"
        )


def load_ohlcv(
    path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Load an OHLCV file with smart format and column detection.

    Parameters
    ----------
    path : str or Path
        Path to CSV, Parquet, or Excel file.
    column_map : dict, optional
        Explicit mapping of role → original column name.
        If None, columns are auto-detected from aliases.
    tz : str, default 'UTC'
        Timezone to localize naive datetime indices to.

    Returns
    -------
    pd.DataFrame
        DataFrame with lowercase columns ('open', 'high', 'low', 'close',
        'volume') and a DatetimeIndex (UTC-aware).

    Raises
    ------
    ValueError
        If required columns (open, high, low, close) cannot be detected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw = _load_raw(path)

    # Determine column mapping
    if column_map is None:
        column_map = detect_columns(raw)

    # Validate minimum required columns
    required = {"open", "high", "low", "close"}
    detected = set(column_map.keys()) & required
    if detected != required:
        missing = required - detected
        raise ValueError(
            f"Cannot detect required columns: {missing}. "
            f"Detected: {column_map}. "
            f"Available columns: {list(raw.columns)}"
        )

    # Build the normalized DataFrame
    rename = {column_map[role]: role for role in column_map}
    df = raw.rename(columns=rename)

    # Keep only OHLCV columns
    keep = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    # Set datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif df.index.dtype == "object":
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = "datetime"
        except Exception:
            pass
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "datetime"

    # Ensure UTC-aware
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize(tz)

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add volume=0 if missing
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Sort by index
    df = df.sort_index()

    return df
