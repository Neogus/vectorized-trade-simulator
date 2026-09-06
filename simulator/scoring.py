"""Strategy scoring formulas for the Backtest Core (Req 3.5).

The Old Project encoded its scoring logic as a single ``get_score(...)``
function with an ``if v == 'A' .. 'G'`` ladder over positional arguments. This
module replaces that ladder with a small registry of callables:

* :data:`SCORE_FORMULAS` maps a formula name to its callable.
* :func:`register_score` is a decorator that adds a formula to the registry so
  callers can extend it with their own strategies.
* :func:`get_score` looks a formula up by name (raising :class:`ValueError` for
  an unknown name).

Every formula takes its aggregate metrics as **keyword-only** arguments and
tolerates a superset via ``**_``. This keeps the weighting/aggregate inputs
caller-supplied (Req 3.5): callers assemble one metric dictionary and hand it to
whichever formula they select without having to trim it to an exact signature.

The formulas themselves are ported verbatim from the Old
``Indicators2.get_score`` ladder (A..G); only their calling convention changed.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["SCORE_FORMULAS", "register_score", "get_score"]

#: Registry mapping a formula name to its scoring callable.
SCORE_FORMULAS: dict[str, Callable[..., float]] = {}


def register_score(name: str) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Return a decorator that registers a scoring formula under ``name``.

    Parameters
    ----------
    name : str
        The key under which the decorated callable is stored in
        :data:`SCORE_FORMULAS`. Registering a name that already exists
        overwrites the previous formula.

    Returns
    -------
    Callable
        A decorator that stores the callable it wraps and returns it unchanged.
    """

    def _decorator(func: Callable[..., float]) -> Callable[..., float]:
        SCORE_FORMULAS[name] = func
        return func

    return _decorator


def get_score(name: str) -> Callable[..., float]:
    """Look up a registered scoring formula by name.

    Parameters
    ----------
    name : str
        The registered formula name (for example ``"A"``).

    Returns
    -------
    Callable
        The scoring callable registered under ``name``.

    Raises
    ------
    ValueError
        If ``name`` is not present in :data:`SCORE_FORMULAS`.
    """
    try:
        return SCORE_FORMULAS[name]
    except KeyError:
        known = ", ".join(sorted(SCORE_FORMULAS)) or "<none>"
        raise ValueError(
            f"unknown score formula {name!r}; registered formulas: {known}"
        ) from None


@register_score("A")
def _score_a(
    *,
    sharpe_avg: float,
    sharpe_dev: float,
    dd_avg: float,
    dd_dev: float,
    alpha_avg: float,
    ret_dev_avg: float,
    **_: object,
) -> float:
    """Sharpe-, drawdown- and alpha-weighted composite score."""
    return (
        (sharpe_avg - sharpe_dev)
        * max(0.0, 1 + (dd_avg - dd_dev / 2))
        * (alpha_avg - ret_dev_avg)
    )


@register_score("B")
def _score_b(*, sharpe_avg: float, sharpe_dev: float, **_: object) -> float:
    """Sharpe minus its dispersion."""
    return sharpe_avg - sharpe_dev


@register_score("C")
def _score_c(*, sharpe_avg: float, sharpe_dev: float, **_: object) -> float:
    """Sharpe minus its dispersion (Old ladder duplicate of B)."""
    return sharpe_avg - sharpe_dev


@register_score("D")
def _score_d(
    *,
    tacc_avg: float,
    tacc_dev: float,
    dd_avg: float,
    dd_dev: float,
    **_: object,
) -> float:
    """Trade-accuracy score weighted by the drawdown factor."""
    return (tacc_avg - tacc_dev) * max(0.0, 1 + (dd_avg - dd_dev / 2))


@register_score("E")
def _score_e(*, alpha_avg: float, ret_dev_avg: float, **_: object) -> float:
    """Alpha minus return dispersion."""
    return alpha_avg - ret_dev_avg


@register_score("F")
def _score_f(*, alpha_avg: float, ret_dev_avg: float, **_: object) -> float:
    """Alpha minus return dispersion (Old ladder duplicate of E)."""
    return alpha_avg - ret_dev_avg


@register_score("G")
def _score_g(*, alpha_avg: float, ret_dev_avg: float, **_: object) -> float:
    """Alpha minus return dispersion (Old ladder duplicate of E)."""
    return alpha_avg - ret_dev_avg


@register_score("H")
def _score_h(
    *,
    sharpe_avg: float,
    sharpe_dev: float,
    dd_avg: float,
    dd_dev: float,
    alpha_avg: float,
    ret_dev_avg: float,
    tacc_avg: float,
    tacc_dev: float,
    **_: object,
) -> float:
    """Balanced composite: rewards high risk-adjusted returns, penalizes
    inconsistency and large drawdowns.

    Components:
    - Sharpe consistency: (sharpe_avg - sharpe_dev) — higher avg, lower variance
    - Drawdown factor: max(0, 1 + dd_avg) — penalizes deep drawdowns (dd_avg is negative)
    - Alpha factor: (alpha_avg - ret_dev_avg) — excess return over market minus volatility
    - Accuracy bonus: max(0, tacc_avg - 0.5) — reward strategies above 50% hit rate

    Score = sharpe_consistency × drawdown_factor × (1 + alpha_factor + accuracy_bonus)
    """
    sharpe_consistency = sharpe_avg - sharpe_dev
    drawdown_factor = max(0.0, 1.0 + dd_avg - dd_dev / 2)
    alpha_factor = alpha_avg - ret_dev_avg
    accuracy_bonus = max(0.0, tacc_avg - 0.5)
    return sharpe_consistency * drawdown_factor * (1.0 + alpha_factor + accuracy_bonus)
