"""Statutory-style reserve formulas.

Reserves are per unit of benefit, per policy in force at the start of the
policy year, and are evaluated on the *valuation* basis (statutory table and
valuation interest rate), never on the pricing basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import actuarial as act


@dataclass
class TermReserveFactors:
    """Terminal reserve factors per $1 of face for t = 0..n."""

    nlp: np.ndarray  # net level premium reserve
    crvm: np.ndarray  # Commissioners Reserve Valuation Method
    deficiency: np.ndarray  # deficiency reserve (gross premium < CRVM net premium)
    nlp_premium: float
    alpha: float  # CRVM first-year net premium
    beta: float  # CRVM renewal net premium
    method: str  # "FPT" when the 20-pay-life expense allowance cap does not bind

    @property
    def total(self) -> np.ndarray:
        return self.crvm + self.deficiency


def term_reserves(q_val: np.ndarray, i_val: float, n: int, gross_premium_per_unit: float = np.inf):
    """NLP and CRVM terminal reserves for n-year level-premium term insurance.

    ``q_val`` is the valuation mortality for the issue age (select from issue,
    running to the end of the table).  CRVM uses an expense allowance of
    ``beta - alpha`` limited to the 19-payment life renewal net premium at age
    x+1 less ``c_x`` (the Standard Valuation Law cap).  For level term this cap
    rarely binds, so CRVM equals Full Preliminary Term.
    """
    q_val = np.asarray(q_val)
    a = lambda t, m: act.annuity_due(q_val[t:], i_val, m)
    A = lambda t, m: act.term_insurance(q_val[t:], i_val, m)

    nlp_p = A(0, n) / a(0, n)
    c_x = q_val[0] / (1.0 + i_val)
    beta_fpt = A(1, n - 1) / a(1, n - 1) if n > 1 else 0.0
    beta_19pl = act.whole_life_insurance(q_val[1:], i_val) / a(1, 19)
    allowance = min(beta_fpt - c_x, beta_19pl - c_x)
    method = "FPT" if beta_fpt <= beta_19pl else "CRVM (20-pay life cap)"
    beta = nlp_p + allowance / a(0, n)
    alpha = beta - allowance

    nlp = np.zeros(n + 1)
    crvm = np.zeros(n + 1)
    defi = np.zeros(n + 1)
    for t in range(1, n):
        m = n - t
        nlp[t] = A(t, m) - nlp_p * a(t, m)
        crvm[t] = A(t, m) - beta * a(t, m)
        if gross_premium_per_unit < beta:
            defi[t] = (beta - gross_premium_per_unit) * a(t, m)
    return TermReserveFactors(
        nlp=np.maximum(nlp, 0.0),
        crvm=np.maximum(crvm, 0.0),
        deficiency=defi,
        nlp_premium=nlp_p,
        alpha=alpha,
        beta=beta,
        method=method,
    )


def deferred_income_annuity_reserves(
    q_val: np.ndarray,
    i_val: float,
    deferral: int,
    income: float,
    death_benefit: float,
) -> np.ndarray:
    """CARVM terminal reserves for a single-premium deferred income annuity.

    With no cash surrender value, the only benefit stream the contract holder
    can elect is the contractual one, so CARVM reduces to the PV of future
    guaranteed benefits: the return-of-premium death benefit during deferral
    plus the life income (annuity-due) from the end of deferral.  Returned for
    t = 0..len(q_val), per contract in force at time t.
    """
    q_val = np.asarray(q_val)
    n = len(q_val)
    res = np.zeros(n + 1)
    for t in range(n):
        q_t = q_val[t:]
        defer = max(deferral - t, 0)
        pv_income = income * act.deferred_annuity_due(q_t, i_val, defer)
        pv_death = death_benefit * act.term_insurance(q_t, i_val, defer) if defer else 0.0
        res[t] = pv_income + pv_death
    return res
