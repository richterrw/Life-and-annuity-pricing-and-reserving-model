"""Single-premium deferred income annuity (DIA): pricing, projection, CARVM reserves.

Product: single premium at issue (age 55 by default), level annual life
income payable at the start of each year from the end of the deferral period
(age 65), return-of-premium death benefit during deferral, no cash surrender
value.  This is the deferred annuity of ALTAM chapter 5 / 6 with the expense,
tax and capital layers a pricing actuary adds.

Timing within each policy year: start of year - premium (year 1), commission,
expenses and any income payment; through the year - investment income on
(opening reserve + premium - start-of-year outgo); end of year - death
benefits during deferral.  The annual income per $ of premium (payout rate) is
solved so that the distributable-earnings IRR equals the hurdle rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import profit, reserves
from .tables import MortalityBasis, iar_2012


@dataclass
class DIAAssumptions:
    issue_age: int = 55
    sex: str = "M"
    issue_year: int = 2026
    deferral: int = 10
    premium: float = 100_000.0
    return_of_premium: bool = True

    # pricing basis
    mortality_pct: float = 1.00  # of 2012 IAM basic, projected generationally
    improvement_multiplier: float = 1.0  # x Scale G2
    earned_rate: float = 0.0525

    # expenses
    commission: float = 0.03
    acq_per_policy: float = 350.0
    maint_per_policy: float = 60.0
    expense_inflation: float = 0.025
    expense_multiplier: float = 1.0

    # valuation basis (2012 IAR)
    valuation_rate: float = 0.0475

    # capital / tax / target
    tax_rate: float = 0.21
    rc_reserve_factor: float = 0.04  # C-1 / C-3 style factor on reserves
    hurdle_rate: float = 0.10

    def with_(self, **kw) -> "DIAAssumptions":
        return replace(self, **kw)


@dataclass
class DIAResult:
    assumptions: DIAAssumptions
    annual_income: float
    projection: pd.DataFrame
    metrics: dict


class DeferredIncomeAnnuity:
    def __init__(self, a: DIAAssumptions, pricing_table: MortalityBasis | None = None):
        self.a = a
        self.pricing_table = pricing_table or iar_2012(
            a.sex, a.issue_year, a.improvement_multiplier
        ).scaled(a.mortality_pct)
        # statutory valuation table is always 100% 2012 IAR with full G2
        self.valuation_table = iar_2012(a.sex, a.issue_year)

    def project(self, annual_income: float) -> DIAResult:
        a = self.a
        q = self.pricing_table.q(a.issue_age)
        n = len(q)
        t = np.arange(1, n + 1)
        in_payout = t > a.deferral
        db = a.premium if a.return_of_premium else 0.0

        inforce_bop = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])
        deaths = inforce_bop * q
        inforce_eop = inforce_bop - deaths

        res_factor = reserves.deferred_income_annuity_reserves(
            self.valuation_table.q(a.issue_age), a.valuation_rate, a.deferral, annual_income, db
        )
        # reserve at the start of year t (after the premium is received, before
        # payments) equals the terminal reserve at the end of year t-1
        reserve_bop = res_factor[:-1] * inforce_bop
        reserve_eop = res_factor[1:] * inforce_eop
        reserve_bop[0] = 0.0  # nothing held before the premium is received
        initial_reserve = res_factor[0]

        premium = np.where(t == 1, a.premium, 0.0)
        commission = a.commission * premium
        infl = (1 + a.expense_inflation) ** (t - 1)
        expenses = a.expense_multiplier * (
            np.where(t == 1, a.acq_per_policy, 0.0) + a.maint_per_policy * infl
        ) * inforce_bop
        income = np.where(in_payout, annual_income, 0.0) * inforce_bop
        death_benefits = np.where(in_payout, 0.0, db) * deaths

        inv_income = (reserve_bop + premium - commission - expenses - income) * a.earned_rate
        pretax = (
            premium + inv_income - commission - expenses - income - death_benefits
            - (reserve_eop - reserve_bop)
        )
        tax = a.tax_rate * pretax
        after_tax = pretax - tax

        rc_eop = a.rc_reserve_factor * reserve_eop
        rc_0 = a.rc_reserve_factor * initial_reserve
        rc_bop = np.concatenate([[rc_0], rc_eop[:-1]])
        de = after_tax + rc_bop * a.earned_rate * (1 - a.tax_rate) - (rc_eop - rc_bop)

        df = pd.DataFrame(
            {
                "year": t,
                "attained_age": a.issue_age + t - 1,
                "q_pricing": q,
                "inforce_bop": inforce_bop,
                "deaths": deaths,
                "inforce_eop": inforce_eop,
                "premium": premium,
                "commission": commission,
                "expenses": expenses,
                "annuity_income": income,
                "death_benefits": death_benefits,
                "investment_income": inv_income,
                "reserve_bop": reserve_bop,
                "reserve_eop": reserve_eop,
                "increase_in_reserve": reserve_eop - reserve_bop,
                "pretax_stat_gain": pretax,
                "tax": tax,
                "after_tax_gain": after_tax,
                "required_capital_eop": rc_eop,
                "distributable_earnings": de,
                "carvm_reserve_per_contract": res_factor[1:],
            }
        )
        de_full = np.concatenate([[-rc_0], de])
        pv_de = profit.npv(de_full, a.hurdle_rate)
        q_val = self.valuation_table.q(a.issue_age)
        metrics = {
            "annual_income": annual_income,
            "payout_rate": annual_income / a.premium,
            "irr": profit.irr(de_full),
            "pv_de_at_hurdle": pv_de,
            "profit_margin": pv_de / a.premium,
            "breakeven_year": profit.breakeven_year(de_full),
            "initial_reserve": float(initial_reserve),
            "initial_reserve_pct_premium": float(initial_reserve / a.premium),
            "peak_reserve_per_contract": float(res_factor.max()),
            "life_expectancy_at_issue_pricing": float(np.sum(np.cumprod(1 - q))),
            "life_expectancy_at_issue_valuation": float(np.sum(np.cumprod(1 - q_val))),
        }
        return DIAResult(a, annual_income, df, metrics)

    def price(self) -> DIAResult:
        """Annual income per contract at which the DE IRR equals the hurdle rate."""
        f = lambda inc: self.project(inc).metrics["pv_de_at_hurdle"]
        inc = profit.solve(f, 1.0, self.a.premium, tol=1e-6)
        return self.project(round(inc, 2))
