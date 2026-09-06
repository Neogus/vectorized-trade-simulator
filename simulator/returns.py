"""Position-aware sequential trade simulation for the Backtest Core.

This module implements :func:`simulate_trades` (and its detail variant
:func:`simulate_trades_detail`), the single, correct replacement for the Old
Project's ``Indicators2.get_ret`` + ``exit_loop`` pair *and* for the earlier
``triple_barrier_returns`` resolver that this module supersedes (design section
"2a. Exit model / return generation", Requirement 3).

Why the previous resolver was wrong
-----------------------------------
The prior ``triple_barrier_returns`` resolved **every** non-zero entry signal as
an *independent* trade with free overlap: it tracked no position state, so it
allowed a new entry while a position was already open and even allowed a long
and a short trade to be open simultaneously. That does not model a real single-
account strategy and inflates the trade count / return statistics.

New semantics: a single position at a time
------------------------------------------
:func:`simulate_trades` runs a genuinely **sequential state machine** over the
bars with exactly one of three states -- ``flat``, ``long``, ``short``:

* **Entering.** When *flat* and ``entries[i] != 0`` and ``ATR[i]`` is valid
  (not NaN), a position is opened in that direction at ``close[i]`` (signals are
  assumed already ``shift(1)``-ed upstream, so entering at the signal bar's
  close introduces no look-ahead).
* **Holding.** While a position is open, **all** entry signals are ignored. Each
  subsequent bar ``j`` is checked for an intrabar first touch of the ATR stop-
  loss / take-profit levels. On a touch the trade is closed at that barrier's
  price and the machine returns to *flat* at bar ``j``; the *next* bar may open a
  new position.
* **Ambiguity.** When both the SL and TP levels are touched in the same bar the
  intrabar price path is unknown, so the outcome is decided by the ``intrabar``
  policy: ``"sl_first"`` (default, conservative -- the stop wins) or
  ``"tp_first"``.
* **Optional timeout.** ``max_hold`` (default ``None``) optionally force-closes a
  position at the close of bar ``i + max_hold`` when no barrier is hit, bounding
  how long a position can stay open. With ``max_hold=None`` a position is held
  until a barrier is touched; if no barrier is ever touched before the end of the
  data the final open position is **left unclosed and not counted** as a
  completed trade.

ATR-based barriers (SL/TP only)
-------------------------------
Exits are ATR stop-loss / take-profit **only** -- the Old signal-based exits, the
horizon-as-timeout exit, the fixed-percent TP/SL model, and the ``exit_signals``
parameter are all dropped. ATR is Wilder-smoothed (``wilder_rma`` over the true
range, matching :meth:`tecana.Tecana.atr`) with a caller
``atr_window`` (default 14):

* **Long** opened at bar ``i``: ``entry = close[i]``, ``a = ATR[i]``,
  ``tp_level = entry + tp_mult*a``, ``sl_level = entry - sl_mult*a``.
* **Short** (mirrored): ``tp_level = entry - tp_mult*a``,
  ``sl_level = entry + sl_mult*a``.

First-touch check per subsequent bar ``j``:

* **Long** closes if ``low[j] <= sl_level`` (SL) or ``high[j] >= tp_level`` (TP).
* **Short** closes if ``high[j] >= sl_level`` (SL) or ``low[j] <= tp_level`` (TP).

Guarantees
----------
Because the scan advances ``i`` past a trade's ``exit_idx`` before it can open
again, the engine guarantees:

* **no overlapping trades** -- each trade's ``exit_idx`` is strictly less than the
  next trade's ``entry_idx``;
* **never a long and a short open at the same time** -- there is a single
  position variable; and
* an entry signal occurring **while a position is open never creates a trade** --
  such signals are ignored until the machine is flat again.

Realized return (net of explicit costs)
----------------------------------------
Round-trip cost is ``2*fee + 2*slippage`` (entry side + exit side):

* long:  ``exit / entry - 1 - cost``
* short: ``entry / exit - 1 - cost``

Compiled kernel
---------------
The sequential scan is a genuinely sequential computation (Req 9.6 permits a
compiled kernel for such cases). It is implemented as a ``numba`` ``@njit``
kernel operating on plain ``float64``/``int`` arrays, with an **identical**
pure-Python/numpy fallback selected automatically when numba is unavailable.
The kernel returns parallel arrays ``(entry_idx, exit_idx, direction,
entry_price, exit_price, ret)`` so it stays numba-friendly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulator._helpers import wilder_rma
from simulator._helpers import prepare_df

__all__ = ["simulate_trades", "simulate_trades_detail", "compute_atr_barriers"]

#: Columns of the OHLCV frame consumed by the engine.
_REQUIRED_COLS = ("high", "low", "close")

#: Accepted intrabar SL-vs-TP ambiguity policies.
_INTRABAR_POLICIES = ("sl_first", "tp_first")


# --------------------------------------------------------------------------- #
# Sequential scan kernel (numba-friendly; identical pure-Python fallback)
# --------------------------------------------------------------------------- #
def _simulate_scan_impl(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    tp_level_long: np.ndarray,
    sl_level_long: np.ndarray,
    tp_level_short: np.ndarray,
    sl_level_short: np.ndarray,
    entries: np.ndarray,
    fee: float,
    slippage: float,
    sl_first: bool,
    max_hold: int,
):
    """Sequential position-aware scan producing completed-trade arrays.

    This is the single source of truth for the state machine. It is written in
    a numba-friendly style (explicit scalar loop over ``float64``/``int`` arrays)
    so the *exact same body* can be JIT-compiled by numba (production path) or
    run as plain Python (fallback). Both paths produce identical results.

    The per-bar ATR barrier arithmetic (``close ± mult*ATR``) is **not**
    performed here: it is precomputed once for the whole array by
    :func:`compute_atr_barriers` (see also the module-level ``_compute_barrier_levels``
    helper) before this scan runs. This function only *indexes* the precomputed
    ``tp_level_*``/``sl_level_*`` arrays at each candidate entry bar, which keeps
    the sequential loop free of any per-visited-bar multiplication -- important
    because this loop is re-run across a very large number of strategy
    backtests.

    Parameters
    ----------
    close, high, low : numpy.ndarray
        1-D ``float64`` price arrays aligned index-for-index.
    tp_level_long, sl_level_long, tp_level_short, sl_level_short : numpy.ndarray
        1-D ``float64`` arrays, precomputed once for the whole dataset by
        :func:`compute_atr_barriers`: the take-profit/stop-loss level that would
        apply *if* a long/short position were opened at that bar. They carry
        ``NaN`` wherever the underlying ATR was still warming up; a position is
        only opened where the relevant level is finite.
    entries : numpy.ndarray
        1-D ``int64`` array in ``{-1, 0, +1}`` aligned to the price arrays.
    fee, slippage : float
        Per-side costs.
    sl_first : bool
        ``True`` -> SL wins a same-bar SL/TP tie; ``False`` -> TP wins.
    max_hold : int
        Force-close after this many bars when positive; ``-1`` means "no limit"
        (hold until a barrier is touched).

    Returns
    -------
    count : int
        Number of completed trades written into the leading slots of the arrays.
    entry_idx, exit_idx, direction : numpy.ndarray
        ``int64`` arrays; positional entry/exit bar indices and trade direction
        (``+1`` long / ``-1`` short).
    entry_price, exit_price, ret : numpy.ndarray
        ``float64`` arrays; fill prices and net realized return per trade.
    """
    n = close.shape[0]

    entry_idx = np.empty(n, dtype=np.int64)
    exit_idx = np.empty(n, dtype=np.int64)
    direction = np.empty(n, dtype=np.int64)
    entry_price = np.empty(n, dtype=np.float64)
    exit_price = np.empty(n, dtype=np.float64)
    ret = np.empty(n, dtype=np.float64)
    count = 0

    cost = 2.0 * fee + 2.0 * slippage

    i = 0
    while i < n:
        d = entries[i]
        # Only open when FLAT, a signal is present, and the precomputed level
        # for that direction is valid (finite -> ATR warm-up has completed).
        if d == 1:
            tp_level = tp_level_long[i]
            sl_level = sl_level_long[i]
        else:  # d == -1 or d == 0; only read when d != 0 below
            tp_level = tp_level_short[i]
            sl_level = sl_level_short[i]

        if d != 0 and not np.isnan(tp_level):
            entry = close[i]

            closed = False
            exit_bar = -1
            exit_p = np.nan

            j = i + 1
            while j < n:
                if d == 1:  # long
                    sl_touch = low[j] <= sl_level
                    tp_touch = high[j] >= tp_level
                else:  # short
                    sl_touch = high[j] >= sl_level
                    tp_touch = low[j] <= tp_level

                if sl_touch and tp_touch:
                    # Same-bar ambiguity resolved by policy.
                    exit_p = sl_level if sl_first else tp_level
                    exit_bar = j
                    closed = True
                    break
                if sl_touch:
                    exit_p = sl_level
                    exit_bar = j
                    closed = True
                    break
                if tp_touch:
                    exit_p = tp_level
                    exit_bar = j
                    closed = True
                    break

                # No barrier this bar: honor the optional max-hold timeout.
                if max_hold > 0 and (j - i) >= max_hold:
                    exit_p = close[j]
                    exit_bar = j
                    closed = True
                    break

                j += 1

            if closed:
                if d == 1:
                    r = exit_p / entry - 1.0 - cost
                else:
                    r = entry / exit_p - 1.0 - cost
                entry_idx[count] = i
                exit_idx[count] = exit_bar
                direction[count] = d
                entry_price[count] = entry
                exit_price[count] = exit_p
                ret[count] = r
                count += 1
                # The bar AFTER the exit may open a new position.
                i = exit_bar + 1
                continue
            else:
                # Never closed before end of data: the final open position is
                # left unclosed and NOT counted. No later bar can open a new
                # position, so the scan is finished.
                break

        i += 1

    return count, entry_idx, exit_idx, direction, entry_price, exit_price, ret


# --- Select the active backend ---------------------------------------------
#
# Primary/production path: numba @njit compiled kernel. Numba is imported
# lazily and guarded so this module still imports where numba is absent (this
# venv). The fallback runs the identical function body as plain Python.
try:  # pragma: no cover - exercised only when numba is installed
    from numba import njit

    _SIMULATE_SCAN_BACKEND = njit(cache=True)(_simulate_scan_impl)
    NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import/JIT failure falls back safely
    _SIMULATE_SCAN_BACKEND = _simulate_scan_impl
    NUMBA_AVAILABLE = False


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Wilder-smoothed ATR over the true range (matches ``Tecana.atr``).

    ``TR = max(high-low, |high - prev_close|, |low - prev_close|)`` with the
    first bar's previous close undefined (NaN), smoothed by :func:`wilder_rma`.
    Returns a ``float64`` array aligned to the inputs, ``NaN`` during warm-up.
    """
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    tr = pd.Series(true_range)
    atr = wilder_rma(tr, window).to_numpy(dtype=float)
    return atr


def compute_atr_barriers(
    df: pd.DataFrame,
    *,
    sl_mult: float,
    tp_mult: float,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Precompute the ATR SL/TP barrier levels for **every bar** up front.

    ``sl_mult``/``tp_mult`` are scalar strategy parameters (the same role as
    any other strategy/indicator parameter randomized by the future search
    layer) -- they are not per-bar. ATR itself, however, is a per-bar rolling
    indicator, so the arithmetic ``close ± mult*ATR`` can be fully vectorized
    across the whole array with numpy rather than recomputed one bar at a time
    inside the sequential scan (Req 9.4). This is a pure hoist: it does not
    change the state machine, it only moves the multiply/add out of the loop
    so the scan can index into ready-made arrays.

    For each bar ``i`` this computes the level that would apply *if* a
    position were opened at that bar's close:

    * ``tp_level_long[i]  = close[i] + tp_mult * ATR[i]``
    * ``sl_level_long[i]  = close[i] - sl_mult * ATR[i]``
    * ``tp_level_short[i] = close[i] - tp_mult * ATR[i]``
    * ``sl_level_short[i] = close[i] + sl_mult * ATR[i]``

    Parameters
    ----------
    df : pandas.DataFrame
        Lowercase OHLCV market data; only ``high``, ``low``, ``close`` are
        consumed. The caller's frame is never mutated.
    sl_mult : float, keyword-only
        Stop-loss ATR multiple. Must be ``> 0``.
    tp_mult : float, keyword-only
        Take-profit ATR multiple. Must be ``> 0``.
    atr_window : int, keyword-only, default 14
        Wilder ATR smoothing period. Must be ``>= 1``.

    Returns
    -------
    pandas.DataFrame
        Indexed like the prepared frame, with float64 columns
        ``tp_level_long``, ``sl_level_long``, ``tp_level_short``,
        ``sl_level_short``. All four columns carry ``NaN`` during the ATR
        warm-up period, matching where a position could not yet be opened.

    Raises
    ------
    ValueError
        If ``sl_mult``/``tp_mult`` are not ``> 0``, if ``atr_window`` is not a
        positive integer, or if a required OHLCV column is missing.
    """
    if not sl_mult > 0:
        raise ValueError(f"sl_mult must be > 0; got {sl_mult}")
    if not tp_mult > 0:
        raise ValueError(f"tp_mult must be > 0; got {tp_mult}")
    if not isinstance(atr_window, (int, np.integer)) or isinstance(atr_window, bool):
        raise ValueError(
            f"atr_window must be an integer; got {type(atr_window).__name__}"
        )
    atr_window = int(atr_window)
    if atr_window < 1:
        raise ValueError(f"atr_window must be >= 1; got {atr_window}")

    prepared = prepare_df(df, _REQUIRED_COLS)
    high = prepared["high"].to_numpy(dtype=float)
    low = prepared["low"].to_numpy(dtype=float)
    close = prepared["close"].to_numpy(dtype=float)
    atr = _compute_atr(high, low, close, atr_window)

    tp_level_long, sl_level_long, tp_level_short, sl_level_short = _compute_barrier_levels(
        close, atr, float(sl_mult), float(tp_mult)
    )

    return pd.DataFrame(
        {
            "tp_level_long": tp_level_long,
            "sl_level_long": sl_level_long,
            "tp_level_short": tp_level_short,
            "sl_level_short": sl_level_short,
        },
        index=prepared.index,
    )


def _compute_barrier_levels(
    close: np.ndarray,
    atr: np.ndarray,
    sl_mult: float,
    tp_mult: float,
):
    """Vectorized ``close ± mult*ATR`` for every bar (shared by public/internal callers).

    Returns ``(tp_level_long, sl_level_long, tp_level_short, sl_level_short)``,
    each a ``float64`` array the same length as ``close``/``atr``, carrying
    ``NaN`` wherever ``atr`` is ``NaN`` (ATR warm-up).
    """
    tp_level_long = close + tp_mult * atr
    sl_level_long = close - sl_mult * atr
    tp_level_short = close - tp_mult * atr
    sl_level_short = close + sl_mult * atr
    return tp_level_long, sl_level_long, tp_level_short, sl_level_short


def _simulate(
    df: pd.DataFrame,
    entries,
    *,
    sl_mult: float,
    tp_mult: float,
    atr_window: int,
    fee: float,
    slippage: float,
    intrabar: str,
    max_hold,
):
    """Validate inputs, compute ATR, run the scan, and return trade arrays.

    Shared core for :func:`simulate_trades` and :func:`simulate_trades_detail`.

    Returns
    -------
    tuple
        ``(index, entry_idx, exit_idx, direction, entry_price, exit_price,
        ret)`` where ``index`` is the prepared frame's index and the remaining
        entries are the completed-trade arrays (already sliced to the trade
        count).
    """
    # --- Parameter validation ------------------------------------------------
    if intrabar not in _INTRABAR_POLICIES:
        raise ValueError(
            f"intrabar must be one of {_INTRABAR_POLICIES}; got {intrabar!r}"
        )
    if not sl_mult > 0:
        raise ValueError(f"sl_mult must be > 0; got {sl_mult}")
    if not tp_mult > 0:
        raise ValueError(f"tp_mult must be > 0; got {tp_mult}")
    if not isinstance(atr_window, (int, np.integer)) or isinstance(atr_window, bool):
        raise ValueError(
            f"atr_window must be an integer; got {type(atr_window).__name__}"
        )
    atr_window = int(atr_window)
    if atr_window < 1:
        raise ValueError(f"atr_window must be >= 1; got {atr_window}")

    if max_hold is None:
        max_hold_val = -1
    else:
        if not isinstance(max_hold, (int, np.integer)) or isinstance(max_hold, bool):
            raise ValueError(
                f"max_hold must be an integer or None; got {type(max_hold).__name__}"
            )
        max_hold_val = int(max_hold)
        if max_hold_val < 1:
            raise ValueError(f"max_hold must be >= 1 or None; got {max_hold_val}")

    prepared = prepare_df(df, _REQUIRED_COLS)
    if not isinstance(prepared.index, pd.DatetimeIndex):
        raise ValueError(
            "simulate_trades requires a DatetimeIndex; got index of type "
            f"{type(prepared.index).__name__}"
        )

    n = len(prepared)

    # --- Insufficient rows (Req 3.9) ----------------------------------------
    # ATR needs a warm-up of ``atr_window`` bars and the engine needs at least
    # one bar after the earliest openable entry to resolve a barrier, so the
    # minimum well-defined frame has ``atr_window + 2`` rows.
    min_rows = atr_window + 2
    if n < min_rows:
        raise ValueError(
            "simulate_trades requires at least "
            f"{min_rows} rows (atr_window + 2) for atr_window={atr_window}; "
            f"received {n}"
        )

    # --- Entry signal validation --------------------------------------------
    ent = np.asarray(entries)
    if ent.ndim != 1:
        raise ValueError(f"entries must be 1-D; got {ent.ndim} dimensions")
    if ent.shape[0] != n:
        raise ValueError(
            f"entries length ({ent.shape[0]}) must match df row count ({n})"
        )
    ent = ent.astype(np.int64, copy=False)
    if ent.size and not np.isin(ent, (-1, 0, 1)).all():
        bad = np.unique(ent[~np.isin(ent, (-1, 0, 1))]).tolist()
        raise ValueError(
            f"entries must be in {{-1, 0, +1}}; got out-of-range values {bad}"
        )

    high = prepared["high"].to_numpy(dtype=float)
    low = prepared["low"].to_numpy(dtype=float)
    close = prepared["close"].to_numpy(dtype=float)
    atr = _compute_atr(high, low, close, atr_window)

    # Precompute the ATR barrier levels for every bar up front (Req 9.4): the
    # scan below only indexes into these arrays at the entry bar instead of
    # recomputing entry ± mult*atr per trade.
    tp_level_long, sl_level_long, tp_level_short, sl_level_short = _compute_barrier_levels(
        close, atr, float(sl_mult), float(tp_mult)
    )

    (
        cnt,
        entry_idx,
        exit_idx,
        direction,
        entry_price,
        exit_price,
        ret,
    ) = _SIMULATE_SCAN_BACKEND(
        np.ascontiguousarray(close, dtype=np.float64),
        np.ascontiguousarray(high, dtype=np.float64),
        np.ascontiguousarray(low, dtype=np.float64),
        np.ascontiguousarray(tp_level_long, dtype=np.float64),
        np.ascontiguousarray(sl_level_long, dtype=np.float64),
        np.ascontiguousarray(tp_level_short, dtype=np.float64),
        np.ascontiguousarray(sl_level_short, dtype=np.float64),
        np.ascontiguousarray(ent, dtype=np.int64),
        float(fee),
        float(slippage),
        intrabar == "sl_first",
        int(max_hold_val),
    )

    count = int(cnt)
    return (
        prepared.index,
        np.asarray(entry_idx[:count], dtype=np.int64),
        np.asarray(exit_idx[:count], dtype=np.int64),
        np.asarray(direction[:count], dtype=np.int64),
        np.asarray(entry_price[:count], dtype=float),
        np.asarray(exit_price[:count], dtype=float),
        np.asarray(ret[:count], dtype=float),
    )


def simulate_trades(
    df: pd.DataFrame,
    entries,
    *,
    sl_mult: float,
    tp_mult: float,
    atr_window: int = 14,
    fee: float = 0.0,
    slippage: float = 0.0,
    intrabar: str = "sl_first",
    max_hold: int | None = None,
) -> pd.Series:
    """Simulate a single-position sequential strategy, ATR SL/TP exits only.

    Runs the position-aware state machine described in the module docstring: at
    most one position is open at any time, entry signals during an open position
    are ignored, and exits are ATR-based stop-loss / take-profit (with an
    optional ``max_hold`` timeout).

    Parameters
    ----------
    df : pandas.DataFrame
        Lowercase OHLCV market data indexed by a :class:`~pandas.DatetimeIndex`.
        Only ``high``, ``low`` and ``close`` are consumed; column names are
        lowercased and validated. The caller's frame is never mutated.
    entries : array_like
        Per-bar entry signals in ``{-1, 0, +1}`` using the canonical
        :class:`~simulator.signals.Signal` convention (``+1`` enter long,
        ``-1`` enter short, ``0`` no entry), positionally aligned to ``df``.
        Signals are assumed already ``shift(1)``-ed upstream (no look-ahead); a
        position is entered at the signal bar's close.
    sl_mult : float, keyword-only
        Stop-loss ATR multiple. Must be ``> 0``. Long stop is
        ``entry - sl_mult*ATR`` (short mirrored).
    tp_mult : float, keyword-only
        Take-profit ATR multiple. Must be ``> 0``. Long target is
        ``entry + tp_mult*ATR`` (short mirrored).
    atr_window : int, keyword-only, default 14
        Wilder ATR smoothing period. Must be ``>= 1``.
    fee : float, keyword-only, default 0.0
        Explicit per-side fee fraction (charged on entry and exit).
    slippage : float, keyword-only, default 0.0
        Explicit per-side slippage fraction (charged on entry and exit).
    intrabar : {"sl_first", "tp_first"}, keyword-only, default "sl_first"
        Policy for the case where SL and TP are both touched in the *same* bar.
        ``"sl_first"`` (default, conservative) assumes the stop hit first;
        ``"tp_first"`` assumes the target hit first.
    max_hold : int or None, keyword-only, default None
        When set, force-close a position at the close of bar ``i + max_hold`` if
        no barrier is hit, bounding open positions. ``None`` (default) holds
        until a barrier is touched; a position never touched by the end of the
        data is left unclosed and is **not** counted as a completed trade.

    Returns
    -------
    pandas.Series
        Net realized per-trade returns (``float64``), indexed by the **entry
        timestamps** and named ``"ret"``. Empty (with the frame's index dtype)
        when no trade completes.

    Raises
    ------
    ValueError
        If ``df`` is not indexed by a :class:`~pandas.DatetimeIndex`; if a
        required OHLCV column is missing; if ``sl_mult``/``tp_mult`` are not
        ``> 0``; if ``atr_window`` is not a positive integer; if ``max_hold`` is
        not a positive integer or ``None``; if ``intrabar`` is not a recognized
        policy; if ``entries`` length does not match ``df`` or contains
        out-of-range values; or if ``df`` has fewer than ``atr_window + 2`` rows
        (message states both the minimum required and the received count).
    """
    index, entry_idx, _exit_idx, _direction, _entry_p, _exit_p, ret = _simulate(
        df,
        entries,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        atr_window=atr_window,
        fee=fee,
        slippage=slippage,
        intrabar=intrabar,
        max_hold=max_hold,
    )
    return pd.Series(ret, index=index[entry_idx], dtype=float, name="ret")


def simulate_trades_detail(
    df: pd.DataFrame,
    entries,
    *,
    sl_mult: float,
    tp_mult: float,
    atr_window: int = 14,
    fee: float = 0.0,
    slippage: float = 0.0,
    intrabar: str = "sl_first",
    max_hold: int | None = None,
) -> pd.DataFrame:
    """Per-trade detail variant of :func:`simulate_trades`.

    Same engine and parameters as :func:`simulate_trades`, but returns one row
    per completed trade with the full context rather than just the return.

    Returns
    -------
    pandas.DataFrame
        One row per completed trade with columns ``entry_time``, ``exit_time``,
        ``direction`` (``+1`` long / ``-1`` short), ``entry_price``,
        ``exit_price`` and ``ret`` (net realized return). Empty (with those
        columns) when no trade completes. See :func:`simulate_trades` for the
        parameter and ``ValueError`` semantics.
    """
    index, entry_idx, exit_idx, direction, entry_p, exit_p, ret = _simulate(
        df,
        entries,
        sl_mult=sl_mult,
        tp_mult=tp_mult,
        atr_window=atr_window,
        fee=fee,
        slippage=slippage,
        intrabar=intrabar,
        max_hold=max_hold,
    )
    return pd.DataFrame(
        {
            "entry_time": index[entry_idx],
            "exit_time": index[exit_idx],
            "direction": direction.astype(np.int8),
            "entry_price": entry_p,
            "exit_price": exit_p,
            "ret": ret,
        }
    )
