"""20-year level term life: pricing, cash-flow projection, statutory reserves.

The projection is annual, for one policy issued at time 0 (results scale
linearly with the number of policies).  Within each policy year:

* start of year: gross premium received, commission, premium tax and
  expenses paid;
* through the year: investment income earned on (opening reserve + premium
  - start-of-year outgo) at the earned rate;
* end of year: death claims paid, then voluntary lapses (no cash value);
* closing reserve is the statutory terminal reserve x policies surviving.

Pre-tax statutory gain is taxed at the corporate rate.  Required capital is
held as a simplified RBC-style factor on net amount at risk and reserves;
distributable earnings (DE) = after-tax gain - increase in required capital +
after-tax interest on capital.  The gross premium is solved so that the DE
IRR equals the hurdle rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import profit, reserves
from .tables import MortalityBasis, cso_2017


def _default_lapse() -> list[float]:
    # policy years 1..20
    return [0.08, 0.06, 0.05] + [0.04] * 7 + [0.03] * 10


@dataclass
class TermAssumptions:
    issue_age: int = 35
    sex: str = "M"
    term: int = 20
    face: float = 500_000.0

    # pricing (best-estimate) basis
    mortality_pct: float = 0.70  # of 2017 Loaded CSO NS select & ultimate
    lapse: list[float] = field(default_factory=_default_lapse)
    lapse_multiplier: float = 1.0
    earned_rate: float = 0.050

    # expenses
    commission_fy: float = 0.90
    commission_ren_2_10: float = 0.05
    commission_ren_11_plus: float = 0.02
    premium_tax: float = 0.020
    acq_per_policy: float = 300.0
    acq_per_1000: float = 1.00
    maint_per_policy: float = 75.0
    expense_inflation: float = 0.025
    expense_multiplier: float = 1.0

    # valuation basis
    valuation_rate: float = 0.035
    valuation_mortality_pct: float = 1.0  # of 2017 Loaded CSO NS select & ultimate

    # capital / tax / target
    tax_rate: float = 0.21
    rc_nar_factor: float = 0.0010  # required capital per $ of net amount at risk
    rc_reserve_factor: float = 0.02  # required capital per $ of reserve
    hurdle_rate: float = 0.10

    def with_(self, **kw) -> "TermAssumptions":
        return replace(self, **kw)


@dataclass
class TermResult:
    assumptions: TermAssumptions
    gross_premium: float
    projection: pd.DataFrame
    reserve_factors: reserves.TermReserveFactors
    metrics: dict


class TermLife:
    def __init__(self, a: TermAssumptions, table: MortalityBasis | None = None):
        self.a = a
        self.table = table or cso_2017(a.sex)

    # -- bases ---------------------------------------------------------------
    def pricing_q(self) -> np.ndarray:
        return self.table.scaled(self.a.mortality_pct).q(self.a.issue_age, self.a.term)

    def valuation_q(self) -> np.ndarray:
        return self.table.scaled(self.a.valuation_mortality_pct).q(self.a.issue_age)

    def reserve_factors(self, gross_premium: float) -> reserves.TermReserveFactors:
        return reserves.term_reserves(
            self.valuation_q(), self.a.valuation_rate, self.a.term, gross_premium / self.a.face
        )

    # -- projection ----------------------------------------------------------
    def project(self, gross_premium: float) -> TermResult:
        a = self.a
        n = a.term
        t = np.arange(1, n + 1)
        q = self.pricing_q()
        w = np.minimum(np.asarray(a.lapse[:n]) * a.lapse_multiplier, 1.0)
        w[-1] = 0.0  # contract expires at the end of the level period

        inforce_bop = np.empty(n)
        inforce_bop[0] = 1.0
        for k in range(1, n):
            inforce_bop[k] = inforce_bop[k - 1] * (1 - q[k - 1]) * (1 - w[k - 1])
        deaths = inforce_bop * q
        lapses = inforce_bop * (1 - q) * w
        inforce_eop = inforce_bop - deaths - lapses
        inforce_eop[-1] = 0.0  # expiries

        rf = self.reserve_factors(gross_premium)
        res_factor = rf.total * a.face
        reserve_bop = res_factor[:-1] * inforce_bop
        reserve_eop = res_factor[1:] * inforce_eop

        premium = gross_premium * inforce_bop
        comm_rate = np.where(t == 1, a.commission_fy, np.where(t <= 10, a.commission_ren_2_10, a.commission_ren_11_plus))
        commission = comm_rate * premium
        prem_tax = a.premium_tax * premium
        infl = (1 + a.expense_inflation) ** (t - 1)
        expenses = a.expense_multiplier * (
            np.where(t == 1, a.acq_per_policy + a.acq_per_1000 * a.face / 1000, 0.0)
            + a.maint_per_policy * infl
        ) * inforce_bop
        claims = deaths * a.face

        inv_income = (reserve_bop + premium - commission - prem_tax - expenses) * a.earned_rate
        pretax = premium + inv_income - commission - prem_tax - expenses - claims - (reserve_eop - reserve_bop)
        tax = a.tax_rate * pretax
        after_tax = pretax - tax

        nar_eop = np.maximum(a.face * inforce_eop - reserve_eop, 0.0)
        rc_eop = a.rc_nar_factor * nar_eop + a.rc_reserve_factor * reserve_eop
        rc_0 = a.rc_nar_factor * a.face
        rc_bop = np.concatenate([[rc_0], rc_eop[:-1]])
        int_on_rc = rc_bop * a.earned_rate * (1 - a.tax_rate)
        de = after_tax + int_on_rc - (rc_eop - rc_bop)

        df = pd.DataFrame(
            {
                "year": t,
                "attained_age": a.issue_age + t - 1,
                "q_pricing": q,
                "lapse_rate": w,
                "inforce_bop": inforce_bop,
                "deaths": deaths,
                "lapses": lapses,
                "inforce_eop": inforce_eop,
                "premium": premium,
                "commission": commission,
                "premium_tax": prem_tax,
                "expenses": expenses,
                "death_claims": claims,
                "investment_income": inv_income,
                "reserve_bop": reserve_bop,
                "reserve_eop": reserve_eop,
                "increase_in_reserve": reserve_eop - reserve_bop,
                "pretax_stat_gain": pretax,
                "tax": tax,
                "after_tax_gain": after_tax,
                "required_capital_eop": rc_eop,
                "distributable_earnings": de,
                "crvm_factor_per_1000": rf.crvm[1:] * 1000,
                "nlp_factor_per_1000": rf.nlp[1:] * 1000,
                "deficiency_factor_per_1000": rf.deficiency[1:] * 1000,
            }
        )
        de_full = np.concatenate([[-rc_0], de])
        pv_prem = profit.npv(np.concatenate([premium, [0.0]]), a.hurdle_rate)
        pv_de = profit.npv(de_full, a.hurdle_rate)
        metrics = {
            "gross_annual_premium": gross_premium,
            "premium_per_1000": gross_premium / a.face * 1000,
            "irr": profit.irr(de_full),
            "pv_de_at_hurdle": pv_de,
            "profit_margin": pv_de / pv_prem,
            "breakeven_year": profit.breakeven_year(de_full),
            "peak_reserve_per_1000": float(rf.total.max() * 1000),
            "crvm_method": rf.method,
            "valuation_net_premium_per_1000_nlp": rf.nlp_premium * 1000,
            "crvm_alpha_per_1000": rf.alpha * 1000,
            "crvm_beta_per_1000": rf.beta * 1000,
            "deficiency_reserve": bool(rf.deficiency.any()),
        }
        return TermResult(a, gross_premium, df, rf, metrics)

    # -- pricing -------------------------------------------------------------
    def price(self) -> TermResult:
        """Gross annual premium at which the DE IRR equals the hurdle rate."""
        f = lambda g: self.project(g).metrics["pv_de_at_hurdle"]
        g = profit.solve(f, 1.0, self.a.face * 0.05, tol=1e-6)
        return self.project(round(g, 2))
