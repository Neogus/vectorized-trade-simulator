"""Performance metrics for the Backtest Core (Req 3.4, 13.9).

One clean, individually-callable function per metric, each documenting its
value range so the property suite (Property 10 / Req 13.9) can assert those
ranges directly. These functions are extracted and cleaned from the Old
Project's ``Indicators2.get_metrics3`` / ``get_metrics4``:

* Positional ``[-1]`` / ``[0]`` element access is replaced with explicit
  ``.iloc`` positional indexing so the metrics are robust to non-integer
  (for example datetime) indexes.
* The bare ``except:`` that wrote ``error.csv`` as a side effect is removed;
  degenerate / too-short inputs now return a documented ``NaN`` instead of
  crashing or writing files.
* The small numerical guards from the old code are kept and documented: the
  ``1e-10`` drawdown denominator guard, and ``ddof=1`` (sample) standard
  deviation used consistently across the Sharpe variants.

All functions accept a ``returns`` series of per-trade (or per-period) simple
returns ``r`` (a value of ``0.01`` meaning ``+1%``). Inputs may be a
:class:`pandas.Series`, a numpy array, or any array-like; they are never
mutated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd

__all__ = [
    "total_return",
    "max_drawdown",
    "capm_beta",
    "alpha",
    "sharpe",
    "accuracy",
]

ArrayLike = Union[pd.Series, np.ndarray, Sequence[float]]

# Drawdown denominator guard carried over from the Old Project's get_metrics*:
# prevents division-by-zero when the running-max cumulative return is 0.
_DD_EPS: float = 1e-10


def _as_float_array(returns: ArrayLike) -> np.ndarray:
    """Coerce an array-like of returns to a 1-D float ``ndarray`` (no copy of caller state)."""
    arr = np.asarray(returns, dtype=float)
    return arr.reshape(-1)


def total_return(returns: ArrayLike) -> float:
    """Cumulative simple return of a return series.

    Computes ``cumprod(1 + r) - 1`` and returns the final value, i.e. the total
    compounded return over the whole series.

    Value range: ``>= -1.0`` (a total loss is ``-1.0``; there is no upper
    bound). Returns ``NaN`` for an empty input.

    Parameters
    ----------
    returns : array-like
        Per-trade / per-period simple returns.

    Returns
    -------
    float
        The compounded total return, or ``NaN`` if ``returns`` is empty.
    """
    r = _as_float_array(returns)
    if r.size == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


def max_drawdown(returns: ArrayLike) -> float:
    """Maximum drawdown of the cumulative return path.

    Reproduces the Old Project drawdown: build the cumulative wealth curve
    ``c = cumprod(1 + r)``, its running maximum ``r_max = cummax(c)``, and the
    per-point drawdown ``(r_max - c) / (r_max + 1e-10)``. The ``1e-10`` guard
    (``_DD_EPS``) avoids division by zero when the running max is 0. The result
    is the worst (largest) drawdown, returned as a non-positive number.

    Value range: ``[-1.0, 0.0]``. ``0.0`` means the curve never dropped below a
    prior peak; ``-1.0`` is a total loss. The result is clipped into this range
    to remain well-defined even for pathological inputs (for example a return
    below ``-100%`` that drives the wealth curve negative). Returns ``NaN`` for
    an empty input.

    Parameters
    ----------
    returns : array-like
        Per-trade / per-period simple returns.

    Returns
    -------
    float
        The maximum drawdown in ``[-1.0, 0.0]``, or ``NaN`` if ``returns`` is empty.
    """
    r = _as_float_array(returns)
    if r.size == 0:
        return float("nan")
    cum = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cum)
    dd_series = (running_max - cum) / (running_max + _DD_EPS)
    dd = float(np.max(dd_series)) * -1.0
    # Clip to the documented [-1, 0] range for numerical robustness.
    return float(min(0.0, max(-1.0, dd)))


def _aligned_pair(returns: ArrayLike, market_returns: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return positionally-aligned float arrays for ``returns`` and ``market_returns``.

    Raises ``ValueError`` if the two series differ in length, since a length
    mismatch indicates the caller has not aligned strategy and market returns.
    """
    r = _as_float_array(returns)
    m = _as_float_array(market_returns)
    if r.size != m.size:
        raise ValueError(
            "returns and market_returns must have equal length; got "
            f"{r.size} and {m.size}"
        )
    return r, m


def capm_beta(returns: ArrayLike, market_returns: ArrayLike) -> float:
    """CAPM beta of a strategy relative to the market.

    Computes ``cov(returns, market_returns) / var(market_returns)`` using the
    sample covariance (``ddof=1`` via :func:`numpy.cov`). Beta measures the
    strategy's sensitivity to market moves.

    Value range: any finite real number (unbounded). Returns ``NaN`` when the
    inputs are too short (fewer than 2 aligned observations) or when the market
    variance is ~0 (a flat market has no defined beta).

    Parameters
    ----------
    returns : array-like
        Strategy per-period simple returns.
    market_returns : array-like
        Market per-period simple returns, aligned 1:1 with ``returns``.

    Returns
    -------
    float
        The CAPM beta, or ``NaN`` if it is undefined for the given inputs.

    Raises
    ------
    ValueError
        If ``returns`` and ``market_returns`` differ in length.
    """
    r, m = _aligned_pair(returns, market_returns)
    if r.size < 2:
        return float("nan")
    cov_matrix = np.cov(r, m)  # ddof=1 by default
    market_var = cov_matrix[1, 1]
    if not np.isfinite(market_var) or abs(market_var) < _DD_EPS:
        return float("nan")
    return float(cov_matrix[0, 1] / market_var)


def alpha(returns: ArrayLike, market_returns: ArrayLike, rf: float = 0.0) -> float:
    """CAPM alpha: total return in excess of the CAPM-expected return.

    Reproduces ``get_metrics4``'s CAPM alpha, replacing the hardcoded
    ``0.0141`` risk-free rate with the caller-supplied ``rf`` (default ``0.0``):

    * ``beta = capm_beta(returns, market_returns)``
    * ``expected = rf + beta * (mean(market_returns) - rf)``
    * ``alpha = total_return(returns) - expected``

    Value range: any finite real number (unbounded). Returns ``NaN`` when beta
    is undefined (see :func:`capm_beta`).

    Parameters
    ----------
    returns : array-like
        Strategy per-period simple returns.
    market_returns : array-like
        Market per-period simple returns, aligned 1:1 with ``returns``.
    rf : float, optional
        Risk-free rate over the evaluation horizon. Defaults to ``0.0``.

    Returns
    -------
    float
        The CAPM alpha, or ``NaN`` if it is undefined for the given inputs.

    Raises
    ------
    ValueError
        If ``returns`` and ``market_returns`` differ in length.
    """
    r, m = _aligned_pair(returns, market_returns)
    beta = capm_beta(r, m)
    if not np.isfinite(beta):
        return float("nan")
    expected_ret = rf + beta * (float(np.mean(m)) - rf)
    return float(total_return(r) - expected_ret)


def _std(x: np.ndarray) -> float:
    """Sample standard deviation (``ddof=1``); ``NaN`` if fewer than 2 points."""
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def sharpe(
    returns: ArrayLike,
    variant: str = "excess",
    rf: float = 0.0,
    market_returns: ArrayLike | None = None,
) -> float:
    """Sharpe-style ratio with several documented variants.

    All variants use the sample standard deviation (``ddof=1``) and return
    ``NaN`` when the ratio is undefined (fewer than 2 observations, or a zero
    denominator). Values are otherwise finite but unbounded.

    Variants
    --------
    ``"excess"`` (default)
        Classic excess-return Sharpe: ``(mean(r) - rf) / std(r)``. Needs only
        ``returns`` and ``rf``. With ``rf=0.0`` this coincides with ``"bot"``.
    ``"bot"``
        The Old Project ``bot_sharpe``: ``mean(r) / std(r)`` (mean over standard
        deviation of the strategy returns).
    ``"market_excess"``
        The Old Project ``get_metrics3`` Sharpe: ``bot_sharpe - market_sharpe``
        where ``market_sharpe = mean(m) / std(m)``. Requires ``market_returns``.
    ``"drawdown_adjusted_excess"``
        The Old Project ``get_metrics4`` excess-return Sharpe over the
        drawdown-adjusted excess series ``dra``: with strategy wealth
        ``sc = cumprod(1 + r)`` and market wealth ``mc = cumprod(1 + m)``,
        ``dra = (sc - mc) / 2`` and the ratio is ``mean(dra) / std(dra)``.
        Requires ``market_returns``.

    Parameters
    ----------
    returns : array-like
        Strategy per-period simple returns.
    variant : str, optional
        One of ``"excess"``, ``"bot"``, ``"market_excess"``,
        ``"drawdown_adjusted_excess"``. Defaults to ``"excess"``.
    rf : float, optional
        Risk-free rate, used only by the ``"excess"`` variant. Defaults to ``0.0``.
    market_returns : array-like, optional
        Market per-period simple returns, aligned 1:1 with ``returns``. Required
        by the ``"market_excess"`` and ``"drawdown_adjusted_excess"`` variants.

    Returns
    -------
    float
        The requested Sharpe variant, or ``NaN`` if it is undefined.

    Raises
    ------
    ValueError
        If ``variant`` is unknown, if a market-dependent variant is requested
        without ``market_returns``, or if ``returns`` and ``market_returns``
        differ in length.
    """
    r = _as_float_array(returns)

    if variant == "excess":
        denom = _std(r)
        if not np.isfinite(denom) or denom == 0.0:
            return float("nan")
        return float((np.mean(r) - rf) / denom)

    if variant == "bot":
        denom = _std(r)
        if not np.isfinite(denom) or denom == 0.0:
            return float("nan")
        return float(np.mean(r) / denom)

    if variant in ("market_excess", "drawdown_adjusted_excess"):
        if market_returns is None:
            raise ValueError(
                f"sharpe variant {variant!r} requires market_returns"
            )
        r, m = _aligned_pair(r, market_returns)

        if variant == "market_excess":
            r_std = _std(r)
            m_std = _std(m)
            if (
                not np.isfinite(r_std)
                or r_std == 0.0
                or not np.isfinite(m_std)
                or m_std == 0.0
            ):
                return float("nan")
            bot_sharpe = float(np.mean(r) / r_std)
            market_sharpe = float(np.mean(m) / m_std)
            return bot_sharpe - market_sharpe

        # drawdown_adjusted_excess
        strat_cum = np.cumprod(1.0 + r)
        market_cum = np.cumprod(1.0 + m)
        dra = (strat_cum - market_cum) / 2.0
        denom = _std(dra)
        if not np.isfinite(denom) or denom == 0.0:
            return float("nan")
        return float(np.mean(dra) / denom)

    raise ValueError(
        "unknown sharpe variant "
        f"{variant!r}; expected one of 'excess', 'bot', 'market_excess', "
        "'drawdown_adjusted_excess'"
    )


def accuracy(returns: ArrayLike) -> float:
    """Fraction of trades / periods with a strictly positive return (win rate).

    Value range: ``[0.0, 1.0]``. Returns ``NaN`` for an empty input.

    Parameters
    ----------
    returns : array-like
        Per-trade / per-period simple returns.

    Returns
    -------
    float
        The fraction of returns ``> 0`` in ``[0.0, 1.0]``, or ``NaN`` if
        ``returns`` is empty.
    """
    r = _as_float_array(returns)
    if r.size == 0:
        return float("nan")
    return float(np.count_nonzero(r > 0.0) / r.size)
