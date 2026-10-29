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

    def with_(self, **kw) -> TermAssumptions:
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
    def lapse_rates(self) -> np.ndarray:
        w = np.minimum(np.asarray(self.a.lapse[: self.a.term]) * self.a.lapse_multiplier, 1.0)
        w[-1] = 0.0  # contract expires at the end of the level period
        return w

    def decrements(self, q: np.ndarray, w: np.ndarray):
        """Expected in-force, deaths and lapses per policy issued."""
        n = self.a.term
        inforce_bop = np.concatenate([[1.0], np.cumprod((1 - q) * (1 - w))[: n - 1]])
        deaths = inforce_bop * q
        lapses = inforce_bop * (1 - q) * w
        return inforce_bop, deaths, lapses

    def financials(self, gross_premium, inforce_bop, deaths, lapses, earned_rate=None) -> dict:
        """Statutory financials from decrements.

        Works on a single projection (shape ``(n,)``) or many at once (shape
        ``(sims, n)``), e.g. stochastic mortality or interest scenarios.
        ``earned_rate`` may be a scalar or a path broadcastable to that shape.
        """
        a = self.a
        n = a.term
        t = np.arange(1, n + 1)
        i_earn = a.earned_rate if earned_rate is None else np.asarray(earned_rate)
        inforce_eop = inforce_bop - deaths - lapses
        inforce_eop[..., -1] = 0.0  # expiries

        rf = self.reserve_factors(gross_premium)
        res_factor = rf.total * a.face
        reserve_bop = res_factor[:-1] * inforce_bop
        reserve_eop = res_factor[1:] * inforce_eop

        premium = gross_premium * inforce_bop
        comm_rate = np.where(
            t == 1, a.commission_fy, np.where(t <= 10, a.commission_ren_2_10, a.commission_ren_11_plus)
        )
        commission = comm_rate * premium
        prem_tax = a.premium_tax * premium
        infl = (1 + a.expense_inflation) ** (t - 1)
        expenses = (
            a.expense_multiplier
            * (np.where(t == 1, a.acq_per_policy + a.acq_per_1000 * a.face / 1000, 0.0) + a.maint_per_policy * infl)
            * inforce_bop
        )
        claims = deaths * a.face

        inv_income = (reserve_bop + premium - commission - prem_tax - expenses) * i_earn
        pretax = premium + inv_income - commission - prem_tax - expenses - claims - (reserve_eop - reserve_bop)
        tax = a.tax_rate * pretax
        after_tax = pretax - tax

        nar_eop = np.maximum(a.face * inforce_eop - reserve_eop, 0.0)
        rc_eop = a.rc_nar_factor * nar_eop + a.rc_reserve_factor * reserve_eop
        rc_0 = a.rc_nar_factor * a.face * inforce_bop[..., :1]
        rc_bop = np.concatenate([rc_0, rc_eop[..., :-1]], axis=-1)
        int_on_rc = rc_bop * i_earn * (1 - a.tax_rate)
        de = after_tax + int_on_rc - (rc_eop - rc_bop)
        return {
            "t": t,
            "rf": rf,
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
            "pretax": pretax,
            "tax": tax,
            "after_tax": after_tax,
            "rc_0": rc_0[..., 0],
            "rc_eop": rc_eop,
            "int_on_rc": int_on_rc,
            "de": de,
            "de_full": np.concatenate([-rc_0, de], axis=-1),
        }

    def project(self, gross_premium: float, earned_rate=None) -> TermResult:
        a = self.a
        q = self.pricing_q()
        w = self.lapse_rates()
        f = self.financials(gross_premium, *self.decrements(q, w), earned_rate=earned_rate)
        t, rf = f["t"], f["rf"]
        inforce_bop, deaths, lapses, inforce_eop = f["inforce_bop"], f["deaths"], f["lapses"], f["inforce_eop"]
        premium, commission, prem_tax, expenses = f["premium"], f["commission"], f["premium_tax"], f["expenses"]
        claims, inv_income, reserve_bop, reserve_eop = (
            f["death_claims"],
            f["investment_income"],
            f["reserve_bop"],
            f["reserve_eop"],
        )
        pretax, tax, after_tax, rc_eop, de = f["pretax"], f["tax"], f["after_tax"], f["rc_eop"], f["de"]

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
        de_full = f["de_full"]
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

    def liability_cash_flows(
        self, gross_premium: float, from_year: int = 0, mortality_margin: float = 0.0, lapse_multiplier: float = 1.0
    ):
        """Net liability outflows per policy in force at the end of ``from_year``.

        Returns (start-of-year outflow, end-of-year outflow) for the remaining
        policy years: commission + premium tax + expenses - premium at the
        start, death claims at the end.  Margins give a prudent-estimate basis.
        """
        a = self.a
        n = a.term
        t = np.arange(1, n + 1)[from_year:]
        q = np.clip(self.pricing_q()[from_year:] * (1 + mortality_margin), 0, 1)
        w = np.minimum(self.lapse_rates()[from_year:] * lapse_multiplier, 1.0)
        inforce = np.concatenate([[1.0], np.cumprod((1 - q) * (1 - w))[:-1]])
        premium = gross_premium * inforce
        comm_rate = np.where(
            t == 1, a.commission_fy, np.where(t <= 10, a.commission_ren_2_10, a.commission_ren_11_plus)
        )
        infl = (1 + a.expense_inflation) ** (t - 1)
        expenses = (
            a.expense_multiplier
            * (np.where(t == 1, a.acq_per_policy + a.acq_per_1000 * a.face / 1000, 0.0) + a.maint_per_policy * infl)
            * inforce
        )
        bop = (comm_rate + a.premium_tax) * premium + expenses - premium
        eop = inforce * q * a.face
        return bop, eop

    # -- pricing -------------------------------------------------------------
    def price(self) -> TermResult:
        """Gross annual premium at which the DE IRR equals the hurdle rate."""
        f = lambda g: self.project(g).metrics["pv_de_at_hurdle"]
        g = profit.solve(f, 1.0, self.a.face * 0.05, tol=1e-6)
        return self.project(round(g, 2))
