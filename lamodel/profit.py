"""Profit-test measures and a root solver shared by both products."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def npv(cash_flows: np.ndarray, rate: float) -> float:
    """PV at time 0 of cash flows at t = 0, 1, 2, ..."""
    cf = np.asarray(cash_flows, dtype=float)
    return float(np.sum(cf / (1.0 + rate) ** np.arange(len(cf))))


def irr(cash_flows: np.ndarray, lo: float = -0.99, hi: float = 10.0) -> float:
    """Internal rate of return.

    Returns NaN unless the cash flows change sign exactly once (Descartes' rule
    then guarantees a unique root).  Spread products with little issue strain
    often produce later negative earnings, where an IRR is not meaningful and
    the PV of earnings at the hurdle rate should be used instead.
    """
    cf = np.asarray(cash_flows, dtype=float)
    signs = np.sign(cf[np.abs(cf) > 1e-9])
    if np.count_nonzero(np.diff(signs)) != 1:
        return float("nan")
    f = lambda r: npv(cf, r)
    if np.sign(f(lo)) == np.sign(f(hi)):
        return float("nan")
    return solve(f, lo, hi)


def solve(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-10) -> float:
    """Bisection: robust for the monotone pricing problems used here."""
    f_lo = f(lo)
    if np.sign(f_lo) == np.sign(f(hi)):
        raise ValueError("root not bracketed")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(hi - lo) < tol or f_mid == 0:
            return mid
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def breakeven_year(cash_flows: np.ndarray) -> int | None:
    """First year in which cumulative undiscounted cash flow turns non-negative."""
    cum = np.cumsum(cash_flows)
    hits = np.nonzero(cum >= 0)[0]
    return int(hits[0]) if len(hits) else None
