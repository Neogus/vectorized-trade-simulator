"""Compact signal encoding, aggregation, and the signal registry.

This module implements the Backtest_Core signal layer (design sections
"2b. Compact signal encoding", "2c. Signal registry") and satisfies:

* **Req 11.1** -- an integer encoding that distinguishes every required signal
  state, including a *distinct* no-action state, a long-entry state and a
  short-entry state (``SHORT = -1``, ``NONE = 0``, ``LONG = +1``).
* **Req 11.2** -- signal values are stored in the smallest suitable integer
  dtype (``int8``), not a 64-bit integer type.
* **Req 11.3** -- aggregation correctness is preserved under the compact
  encoding (all-agree long / all-agree short / otherwise no-action).
* **Req 3.8** -- a signal-registry mechanism (:data:`SIGNAL_REGISTRY`,
  :func:`register_signal`, :func:`get_signal`) that maps each signal *name* to
  its generating function so a strategy can be composed by referencing names;
  an unknown name raises :class:`ValueError` naming it.

Sign-convention reconciliation
-------------------------------
The ported :class:`~tecana.Tecana` signal methods faithfully
kept the Old Project's numeric convention: the directional families emit
**+1 on the sell-side crossover** and **-1 on the buy-side crossover**
(e.g. ``rsi_m`` sets ``+1`` when RSI crosses *down* out of overbought and
``-1`` when it crosses *up* out of oversold), with ``0`` = no-signal. That is
the **opposite** sign association from the canonical :class:`Signal` enum used
by the backtest core, where ``LONG = +1`` and ``SHORT = -1``.

**Chosen reconciliation -- approach (a): normalize at registry time.**
Rather than teach every downstream consumer about the inverted raw convention,
the registry callables *negate* each raw directional stream as it enters the
signal matrix, so that a "buy/long" indication becomes :data:`Signal.LONG`
(``+1``) and a "sell/short" indication becomes :data:`Signal.SHORT` (``-1``).
This means the entire backtest core -- ``aggregate_signals``,
``triple_barrier_returns``, the signal matrix -- consistently uses
``LONG = +1`` / ``SHORT = -1`` and never has to reason about the historical
sign. Negation preserves the no-signal state exactly (``-0 == 0``).

Volatility / regime flags (the ``*_v`` family) are **not** directional -- they
emit ``{0, 1}`` where ``1`` means "condition active" (e.g. high volatility).
They are *not* negated; they are passed through unchanged so a flag stays a
flag. Only the directional ``*_m`` / ``*_z`` / ``*_t`` families are normalized.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum

import numpy as np

__all__ = [
    "Signal",
    "SIGNAL_DTYPE",
    "normalize_direction",
    "encode_entries",
    "aggregate_signals",
    "SIGNAL_REGISTRY",
    "register_signal",
    "get_signal",
]

#: The smallest suitable integer dtype for signal storage (Req 11.2).
SIGNAL_DTYPE = np.int8

#: Directional signal families whose raw Tecana output uses the inverted
#: (+1 = sell, -1 = buy) convention and must be negated to canonical.
_DIRECTIONAL_SUFFIXES = ("_m", "_z", "_t")

#: Non-directional flag families ({0, 1}); passed through unchanged.
_FLAG_SUFFIXES = ("_v",)


class Signal(IntEnum):
    """Canonical compact signal encoding, stored as :data:`SIGNAL_DTYPE` (int8).

    A single byte distinguishes the three required states (Req 11.1):

    ============  =====  ==========================================
    Member        Value  Meaning
    ============  =====  ==========================================
    ``SHORT``      -1    short entry (canonical short direction)
    ``NONE``        0    no action / no signal (distinct state)
    ``LONG``       +1    long entry (canonical long direction)
    ============  =====  ==========================================

    The values coincide with a signed integer so that aggregation across
    streams is a natural signed comparison, and the whole set fits in ``int8``
    (Req 11.2).
    """

    SHORT = -1
    NONE = 0
    LONG = 1


def normalize_direction(raw) -> np.ndarray:
    """Reconcile a raw Tecana directional stream to the canonical convention.

    The raw ``*_m`` / ``*_z`` / ``*_t`` streams use ``+1`` for the sell-side and
    ``-1`` for the buy-side crossover. This negates them so a buy/long
    indication becomes :data:`Signal.LONG` (``+1``) and a sell/short indication
    becomes :data:`Signal.SHORT` (``-1``). The ``0`` no-signal state is
    preserved (``-0 == 0``).

    Parameters
    ----------
    raw : array_like
        Raw directional signal values in ``{-1, 0, +1}``.

    Returns
    -------
    numpy.ndarray
        A new ``int8`` array with each value negated (canonical convention).
    """
    arr = np.asarray(raw, dtype=SIGNAL_DTYPE)
    return np.negative(arr).astype(SIGNAL_DTYPE, copy=False)


def encode_entries(streams, *, normalize: bool = False) -> np.ndarray:
    """Stack per-stream signal values into a compact ``int8`` matrix.

    Builds the ``(rows, k)`` signal matrix consumed by
    :func:`aggregate_signals`. Each stream must contain only values in the
    encoded set (``{-1, 0, +1}`` for directional streams, ``{0, 1}`` for
    flags -- both are subsets of ``{-1, 0, +1}``), so any out-of-range value is
    a programming error and raises :class:`ValueError`.

    Parameters
    ----------
    streams : array_like
        Either a single 2-D ``(rows, k)`` array, or a sequence of ``k`` equal
        length 1-D arrays (one per signal stream). A single 1-D array is
        treated as one stream and returned as an ``(rows, 1)`` matrix.
    normalize : bool, default False
        If ``True``, negate every stream via :func:`normalize_direction` to
        reconcile raw Tecana directional output to the canonical convention.
        Leave ``False`` when the inputs are already canonical (the registry
        callables normalize at their source, so matrices built from the
        registry are already canonical).

    Returns
    -------
    numpy.ndarray
        A C-contiguous ``(rows, k)`` ``int8`` matrix.

    Raises
    ------
    ValueError
        If the streams have mismatched lengths, produce more than 2 dimensions,
        or contain a value outside ``{-1, 0, +1}``.
    """
    arr = _as_matrix(streams)

    if normalize:
        arr = np.negative(arr)

    arr = arr.astype(SIGNAL_DTYPE, copy=False)

    if arr.size and not np.isin(arr, (-1, 0, 1)).all():
        bad = np.unique(arr[~np.isin(arr, (-1, 0, 1))])
        raise ValueError(
            f"signal values must be in {{-1, 0, +1}}; got out-of-range values {bad.tolist()}"
        )
    return np.ascontiguousarray(arr, dtype=SIGNAL_DTYPE)


def _as_matrix(streams) -> np.ndarray:
    """Coerce ``streams`` into a 2-D ``(rows, k)`` float/int array (pre-cast)."""
    arr = np.asarray(streams)

    # A sequence of 1-D arrays of equal length becomes 2-D via np.asarray, but a
    # ragged sequence yields an object array -- detect and report that clearly.
    if arr.dtype == object:
        raise ValueError("signal streams must all have the same length")

    if arr.ndim == 1:
        # Single stream -> column vector (rows, 1).
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        # A sequence of k streams comes in as (k, rows); a caller-supplied
        # (rows, k) matrix comes in as-is. We standardize on "columns are
        # streams". When given a list/tuple of 1-D streams, transpose so each
        # stream is a column.
        if isinstance(streams, (list, tuple)):
            return arr.T
        return arr
    raise ValueError(f"signal streams must be 1-D or 2-D; got {arr.ndim} dimensions")


def aggregate_signals(mat: np.ndarray) -> np.ndarray:
    """Aggregate a ``(rows, k)`` signal matrix into a single entry vector.

    Aggregation semantics (design section 2b, Req 11.3): using the canonical
    encoding (``LONG = +1`` / ``SHORT = -1`` / ``NONE = 0``),

    * a **LONG** entry (``+1``) is produced at every row where *all* ``k``
      required streams agree long,
    * a **SHORT** entry (``-1``) is produced at every row where *all* ``k``
      streams agree short,
    * otherwise the row is **NONE** (``0``).

    Because the matrix is already canonical, "all agree long" is exactly
    ``(mat == +1).all(axis=1)`` and "all agree short" is
    ``(mat == -1).all(axis=1)`` -- pure vectorized integer ops, no per-row loop.

    Parameters
    ----------
    mat : numpy.ndarray
        A ``(rows, k)`` array of per-stream canonical signals in
        ``{-1, 0, +1}``. A 1-D ``(rows,)`` array is treated as ``k == 1``.

    Returns
    -------
    numpy.ndarray
        A ``(rows,)`` ``int8`` array in ``{-1, 0, +1}`` using the
        :class:`Signal` convention.
    """
    arr = np.asarray(mat)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"signal matrix must be 2-D (rows, k); got {arr.ndim} dimensions")

    rows, k = arr.shape
    out = np.zeros(rows, dtype=SIGNAL_DTYPE)

    # With zero streams there is nothing to agree on -> all no-action. Guard
    # explicitly because ``.all(axis=1)`` over an empty axis returns True.
    if k == 0:
        return out

    all_long = (arr == Signal.LONG.value).all(axis=1)
    all_short = (arr == Signal.SHORT.value).all(axis=1)

    out[all_long] = Signal.LONG.value
    out[all_short] = Signal.SHORT.value
    return out


# --------------------------------------------------------------------------- #
# Signal registry (Req 3.8) -- lazily populated (Req 9/10)
# --------------------------------------------------------------------------- #
#
# **Why lazy (Req 9/10 -- worker import overhead).** Building the registry means
# importing the whole :class:`~tecana.Tecana` indicator stack and
# constructing a callable for every ``*_m`` / ``*_z`` / ``*_t`` / ``*_v`` method
# (124 of them). Doing that at ``import simulator`` would make every freshly
# *spawned* worker pay the Tecana import cost even when it never touches a signal
# (e.g. a strategy-evaluation worker that only reads precomputed signal columns
# from the matrix). We therefore populate the registry exactly once, on first
# access, and expose it via PEP 562 module-level ``__getattr__`` so that
# ``from simulator.signals import SIGNAL_REGISTRY`` still returns the fully
# populated mapping while ``import simulator`` stays cheap.

#: The backing store for the signal registry. Private on purpose: public access
#: goes through the module-level ``SIGNAL_REGISTRY`` name (PEP 562
#: :func:`__getattr__`), which forces one-time population before handing back
#: this exact dict. ``register_signal`` / ``get_signal`` operate on it directly
#: after calling :func:`_ensure_populated`.
_SIGNAL_REGISTRY: dict[str, Callable[..., np.ndarray]] = {}

#: One-time guard for :func:`_ensure_populated`.
_registry_populated: bool = False


def _ensure_populated() -> None:
    """Populate :data:`_SIGNAL_REGISTRY` exactly once (idempotent).

    Called at the top of :func:`get_signal` / :func:`register_signal` and by the
    module-level :func:`__getattr__` when ``SIGNAL_REGISTRY`` is first accessed.
    The guard flag is flipped *before* populating so the populate step -- which
    imports the Tecana stack -- can never recurse back into this function.
    """
    global _registry_populated
    if not _registry_populated:
        _registry_populated = True
        _populate_registry(_SIGNAL_REGISTRY)


def __getattr__(name: str):
    """PEP 562 hook exposing ``SIGNAL_REGISTRY`` as a lazily-populated dict.

    ``from simulator.signals import SIGNAL_REGISTRY`` (or attribute access
    ``signals.SIGNAL_REGISTRY``) triggers this hook -- because no module-level
    ``SIGNAL_REGISTRY`` variable is defined -- which populates the registry from
    the Tecana signal methods on first use and returns the fully-populated dict.
    """
    if name == "SIGNAL_REGISTRY":
        _ensure_populated()
        return _SIGNAL_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_signal(name: str) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Decorator registering ``func`` under ``name`` in the signal registry.

    Forces one-time population first so hand-registered signals coexist with the
    introspected Tecana signals under a single, unambiguous name space.

    Parameters
    ----------
    name : str
        The signal name a strategy will reference (e.g. ``"rsi_m"``).

    Returns
    -------
    Callable
        A decorator that stores the function and returns it unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered (registration must be unambiguous).
    """
    _ensure_populated()

    def _decorator(func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        if name in _SIGNAL_REGISTRY:
            raise ValueError(f"signal {name!r} is already registered")
        _SIGNAL_REGISTRY[name] = func
        return func

    return _decorator


def get_signal(name: str) -> Callable[..., np.ndarray]:
    """Return the generating function registered under ``name``.

    Forces one-time population first (see :func:`_ensure_populated`).

    Parameters
    ----------
    name : str
        A registered signal name.

    Returns
    -------
    Callable
        The exact callable registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not registered. The message names the unknown signal
        (a plain :class:`KeyError` is deliberately *not* raised -- Req 3.8).
    """
    _ensure_populated()
    try:
        return _SIGNAL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown signal: {name!r}. Registered signals: "
            f"{sorted(_SIGNAL_REGISTRY)}"
        ) from None


def _make_registry_callable(method_name: str, negate: bool) -> Callable[..., np.ndarray]:
    """Build a registry callable wrapping a :class:`Tecana` signal method.

    The returned function computes the raw signal column via the named Tecana
    method, extracts it as an ``int8`` stream, and (for directional families)
    negates it into the canonical convention. The actual per-signal indicator
    computation is deferred until the callable is invoked.
    """

    def _signal(df, *args, **kwargs) -> np.ndarray:
        
        out = getattr(Tecana(), method_name)(df, *args, **kwargs)
        stream = np.asarray(out[method_name], dtype=SIGNAL_DTYPE)
        if negate:
            return normalize_direction(stream)
        return np.ascontiguousarray(stream, dtype=SIGNAL_DTYPE)

    _signal.__name__ = f"{method_name}_signal"
    _signal.__qualname__ = _signal.__name__
    kind = "directional (normalized to canonical LONG=+1/SHORT=-1)" if negate else "flag ({0, 1})"
    _signal.__doc__ = (
        f"Compute the {method_name!r} signal stream ({kind}).\n\n"
        f"Calls ``Tecana().{method_name}(df, *args, **kwargs)`` and returns the "
        f"resulting column as a compact int8 numpy array."
    )
    return _signal


def _populate_registry(registry: dict) -> None:
    """Populate ``registry`` from the tecana library (PyPI package).

    Introspects ``tecana.Tecana`` for public methods whose name ends in
    ``_m`` / ``_z`` / ``_t`` (directional) or ``_v`` (flag). Falls back
    gracefully if tecana is not installed.
    """
    try:
        from tecana import Tecana  # PyPI: pip install tecana
    except ImportError:
        return  # tecana not installed — registry stays empty

    for method_name in dir(Tecana):
        if method_name.startswith("_"):
            continue
        attr = getattr(Tecana, method_name, None)
        if not callable(attr):
            continue
        if method_name.endswith(_DIRECTIONAL_SUFFIXES):
            negate = True
        elif method_name.endswith(_FLAG_SUFFIXES):
            negate = False
        else:
            continue
        if not dict.__contains__(registry, method_name):
            dict.__setitem__(
                registry, method_name, _make_registry_callable(method_name, negate)
            )
