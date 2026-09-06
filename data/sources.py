"""Public API data connectors for equities (yfinance) and crypto (ccxt)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

__all__ = ["download_yfinance", "download_ccxt", "search_tickers"]


def download_yfinance(
    ticker: str,
    *,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. 'AAPL', 'BTC-USD', 'MSFT').
    start : str or datetime, optional
        Start date. Defaults to 1 year ago.
    end : str or datetime, optional
        End date. Defaults to today.
    interval : str, default '1d'
        Bar interval: '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'.
        Intraday intervals limited to last 60 days by Yahoo.

    Returns
    -------
    pd.DataFrame
        Normalized OHLCV DataFrame with lowercase columns and DatetimeIndex.

    Raises
    ------
    ImportError
        If yfinance is not installed.
    ValueError
        If no data is returned for the given ticker/date range.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for equity data. Install with: pip install yfinance"
        )

    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=str(start), end=str(end), interval=interval, progress=False)

    if data.empty:
        raise ValueError(
            f"No data returned for ticker='{ticker}', start={start}, end={end}, "
            f"interval={interval}. Check the ticker symbol and date range."
        )

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Normalize columns
    df = data.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
    })

    # Keep OHLCV only
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df.index.name = "datetime"

    # Ensure UTC-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_ccxt(
    symbol: str,
    exchange: str = "binance",
    *,
    timeframe: str = "1h",
    since: str | datetime | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Download OHLCV data from a crypto exchange via ccxt.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. 'BTC/USDT', 'ETH/USDT').
    exchange : str, default 'binance'
        Exchange name (any ccxt-supported exchange).
    timeframe : str, default '1h'
        Bar interval: '1m', '5m', '15m', '1h', '4h', '1d'.
    since : str or datetime, optional
        Start date. Defaults to ``limit`` bars ago.
    limit : int, default 1000
        Maximum number of bars to fetch.

    Returns
    -------
    pd.DataFrame
        Normalized OHLCV DataFrame with lowercase columns and DatetimeIndex.

    Raises
    ------
    ImportError
        If ccxt is not installed.
    ValueError
        If no data is returned.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt is required for crypto data. Install with: pip install ccxt"
        )

    exchange_class = getattr(ccxt, exchange, None)
    if exchange_class is None:
        raise ValueError(
            f"Unknown exchange: '{exchange}'. "
            f"Available: {', '.join(ccxt.exchanges[:20])}..."
        )

    ex = exchange_class({"enableRateLimit": True})

    since_ms = None
    if since is not None:
        if isinstance(since, str):
            since = pd.Timestamp(since)
        since_ms = int(since.timestamp() * 1000)

    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

    if not ohlcv:
        raise ValueError(
            f"No data returned for {symbol} on {exchange} "
            f"(timeframe={timeframe}, limit={limit})"
        )

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def search_tickers(query: str, source: str = "yfinance", limit: int = 10) -> list[dict]:
    """Search for ticker symbols matching a query.

    Parameters
    ----------
    query : str
        Search term (company name, symbol, etc.).
    source : str, default 'yfinance'
        'yfinance' for equities or 'ccxt' for crypto.
    limit : int, default 10
        Max results.

    Returns
    -------
    list[dict]
        List of matches with 'symbol' and 'name' keys.
    """
    results = []

    if source == "yfinance":
        try:
            import yfinance as yf
            ticker = yf.Ticker(query)
            info = ticker.info
            if info and info.get("symbol"):
                results.append({
                    "symbol": info.get("symbol", query),
                    "name": info.get("longName", info.get("shortName", "")),
                    "exchange": info.get("exchange", ""),
                })
        except Exception:
            results.append({"symbol": query.upper(), "name": "(unverified)", "exchange": ""})

    elif source == "ccxt":
        try:
            import ccxt
            ex = ccxt.binance({"enableRateLimit": True})
            ex.load_markets()
            q = query.upper()
            for sym in ex.symbols:
                if q in sym.upper():
                    results.append({"symbol": sym, "name": sym, "exchange": "binance"})
                    if len(results) >= limit:
                        break
        except ImportError:
            pass

    return results[:limit]
