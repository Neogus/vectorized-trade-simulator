"""Data-decay adjustment for the Trading Toolkit backtest core.

This module ports the exponential recency weighting from the Old Project
``Indicators2.adjust_data_decay`` into a clean, parameterized function. Two
changes are made relative to the original:

* ``min_coeff`` / ``max_coeff`` become caller-supplied keyword parameters
  instead of hardcoded constants (Req 3.7). The original fixed ``min_coeff=0.2``
  and ``max_coeff=1.0``; those remain the defaults.
* The degenerate case where every timestamp is equal (a single-timestamp
  series, or a series whose index min equals its index max) is handled with
  uniform weights instead of dividing by zero. The Old Project computed
  ``(index - min) / (max - min)`` unconditionally, which produces ``NaN``/``inf``
  when ``max == min``.

The weighting ramps exponentially from ``min_coeff`` at the earliest timestamp
to ``max_coeff`` at the latest::

    time_norm = (index - index.min()) / (index.max() - index.min())   # 0..1
    b         = ln(max_coeff / min_coeff)
    coeff     = min_coeff * exp(b * time_norm)

so ``time_norm == 0`` gives ``coeff == min_coeff`` and ``time_norm == 1`` gives
``coeff == min_coeff * exp(ln(max_coeff/min_coeff)) == max_coeff``.

Public import path (single import per component -- Req 12.4)::

    from simulator.decay import adjust_data_decay
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["adjust_data_decay"]


def adjust_data_decay(
    returns: pd.Series,
    *,
    min_coeff: float = 0.2,
    max_coeff: float = 1.0,
) -> pd.Series:
    """Weight a returns series by recency using exponential decay.

    Each observation is multiplied by a coefficient that ramps exponentially
    from ``min_coeff`` (earliest timestamp) to ``max_coeff`` (latest timestamp),
    based on its normalized position in time.

    Parameters
    ----------
    returns:
        A returns :class:`pandas.Series` indexed by a datetime-like index. The
        index is used to compute each observation's normalized time position;
        it does not need to be sorted.
    min_coeff:
        Weight applied to the earliest observation. Default ``0.2``.
    max_coeff:
        Weight applied to the latest observation. Default ``1.0``.

    Returns
    -------
    pandas.Series
        ``returns`` multiplied elementwise by the recency coefficients, with the
        same index and dtype-compatible values.

    Notes
    -----
    When the index has zero span (a single timestamp, or all timestamps equal),
    the normalization ``(index - min) / (max - min)`` would divide by zero. In
    that case every observation is assigned the uniform weight ``max_coeff``
    (the most-recent weight), and no division is performed.
    """
    first_time = returns.index.min()
    last_time = returns.index.max()

    # Degenerate case: empty, single timestamp, or all-equal index -> no time
    # span. Use uniform weights (max_coeff) rather than dividing by zero.
    if len(returns) == 0 or first_time == last_time:
        return returns * max_coeff

    span = last_time - first_time
    time_norm = (returns.index - first_time) / span
    time_norm = np.asarray(time_norm, dtype=float)

    # b solves min_coeff * exp(b * 1) == max_coeff.
    b = np.log(max_coeff / min_coeff)
    coeff = min_coeff * np.exp(b * time_norm)

    return returns * coeff
