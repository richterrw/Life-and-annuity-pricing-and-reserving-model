"""Core life-contingency math on annual, curtate vectors.

All functions take a vector ``q`` of one-year death probabilities starting at
the life's current age and an effective annual interest rate ``i`` (scalar or a
vector of year-by-year forward rates).  Deaths are paid at the end of the year
of death, premiums and annuity payments at the start of the year, which is the
standard ALTAM / CRVM annual framework.
"""

from __future__ import annotations

import numpy as np


def survival(q: np.ndarray) -> np.ndarray:
    """``tp_x`` for t = 0..n (length n+1)."""
    return np.concatenate([[1.0], np.cumprod(1.0 - np.asarray(q))])


def discount(i, n: int) -> np.ndarray:
    """``v^t`` for t = 0..n.  ``i`` may be a scalar or a length-n vector of forward rates."""
    rates = np.broadcast_to(np.asarray(i, dtype=float), (n,))
    return np.concatenate([[1.0], np.cumprod(1.0 / (1.0 + rates))])


def life_table(q: np.ndarray, radix: float = 100_000.0) -> np.ndarray:
    """``l_x`` column for the ages covered by ``q``."""
    return radix * survival(q)


def annuity_due(q, i, n: int | None = None) -> float:
    """n-year temporary life annuity-due, ``a-due_{x:n}`` (whole life if n is None)."""
    q = np.asarray(q)
    n = len(q) if n is None else min(n, len(q))
    p = survival(q)[:n]
    v = discount(i, n)[:n]
    return float(np.sum(v * p))


def deferred_annuity_due(q, i, defer: int, n: int | None = None) -> float:
    """``m| a-due_x``: annuity-due starting after ``defer`` years."""
    q = np.asarray(q)
    total = len(q) if n is None else min(defer + n, len(q))
    p = survival(q)[:total]
    v = discount(i, total)[:total]
    return float(np.sum((v * p)[defer:]))


def term_insurance(q, i, n: int | None = None) -> float:
    """n-year term insurance paying 1 at end of year of death, ``A^1_{x:n}``."""
    q = np.asarray(q)
    n = len(q) if n is None else min(n, len(q))
    p = survival(q)[:n]
    v = discount(i, n + 1)[1 : n + 1]
    return float(np.sum(v * p * q[:n]))


def whole_life_insurance(q, i) -> float:
    return term_insurance(q, i)


def pure_endowment(q, i, n: int) -> float:
    """``nE_x``."""
    return float(discount(i, n)[n] * survival(q)[n])


def endowment_insurance(q, i, n: int) -> float:
    return term_insurance(q, i, n) + pure_endowment(q, i, n)


def curtate_expectation(q) -> float:
    """``e_x`` (curtate)."""
    return float(np.sum(survival(q)[1:]))
