"""Interest, mortality, lapse and expense sensitivities.

Each scenario is run two ways:

* **fixed price** - the base premium (term) or payout rate (DIA) is held and the
  shocked assumptions flow through to IRR and PV of distributable earnings;
* **repriced** - the premium / payout rate is re-solved to hit the hurdle,
  showing what the shock would cost the policyholder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .deferred_annuity import DeferredIncomeAnnuity, DIAAssumptions
from .reserves import deferred_income_annuity_reserves
from .tables import iar_2012
from .term_life import TermAssumptions, TermLife


def term_scenarios(base: TermAssumptions) -> dict[str, TermAssumptions]:
    return {
        "Base": base,
        "Earned rate -100bp": base.with_(earned_rate=base.earned_rate - 0.01),
        "Earned rate +100bp": base.with_(earned_rate=base.earned_rate + 0.01),
        "Valuation rate -50bp": base.with_(valuation_rate=base.valuation_rate - 0.005),
        "Valuation rate +50bp": base.with_(valuation_rate=base.valuation_rate + 0.005),
        "Mortality 90%": base.with_(mortality_pct=base.mortality_pct * 0.9),
        "Mortality 110%": base.with_(mortality_pct=base.mortality_pct * 1.1),
        "Mortality 120%": base.with_(mortality_pct=base.mortality_pct * 1.2),
        "Lapses 75%": base.with_(lapse_multiplier=0.75),
        "Lapses 125%": base.with_(lapse_multiplier=1.25),
        "Expenses 110%": base.with_(expense_multiplier=1.10),
    }


def dia_scenarios(base: DIAAssumptions) -> dict[str, DIAAssumptions]:
    return {
        "Base": base,
        "Earned rate -100bp": base.with_(earned_rate=base.earned_rate - 0.01),
        "Earned rate +100bp": base.with_(earned_rate=base.earned_rate + 0.01),
        "Valuation rate -50bp": base.with_(valuation_rate=base.valuation_rate - 0.005),
        "Valuation rate +50bp": base.with_(valuation_rate=base.valuation_rate + 0.005),
        "Mortality 90% (longevity)": base.with_(mortality_pct=base.mortality_pct * 0.9),
        "Mortality 110%": base.with_(mortality_pct=base.mortality_pct * 1.1),
        "No mortality improvement": base.with_(improvement_multiplier=0.0),
        "150% Scale G2 improvement": base.with_(improvement_multiplier=1.5),
        "Expenses 110%": base.with_(expense_multiplier=1.10),
    }


def run_term(base: TermAssumptions) -> pd.DataFrame:
    base_res = TermLife(base).price()
    g = base_res.gross_premium
    rows = []
    for name, a in term_scenarios(base).items():
        model = TermLife(a)
        fixed = model.project(g).metrics
        repriced = model.price().metrics
        rows.append(
            {
                "scenario": name,
                "irr_at_base_premium": fixed["irr"],
                "pv_de_at_base_premium": fixed["pv_de_at_hurdle"],
                "profit_margin_at_base_premium": fixed["profit_margin"],
                "peak_reserve_per_1000": fixed["peak_reserve_per_1000"],
                "repriced_annual_premium": repriced["gross_annual_premium"],
                "premium_change_pct": repriced["gross_annual_premium"] / g - 1,
            }
        )
    return pd.DataFrame(rows)


def run_dia(base: DIAAssumptions) -> pd.DataFrame:
    base_res = DeferredIncomeAnnuity(base).price()
    inc = base_res.annual_income
    rows = []
    for name, a in dia_scenarios(base).items():
        model = DeferredIncomeAnnuity(a)
        fixed = model.project(inc).metrics
        repriced = model.price().metrics
        rows.append(
            {
                "scenario": name,
                "irr_at_base_payout": fixed["irr"],
                "pv_de_at_base_payout": fixed["pv_de_at_hurdle"],
                "profit_margin_at_base_payout": fixed["profit_margin"],
                "initial_reserve_pct_premium": fixed["initial_reserve_pct_premium"],
                "repriced_payout_rate": repriced["payout_rate"],
                "payout_rate_change_bp": (repriced["payout_rate"] - base_res.metrics["payout_rate"]) * 1e4,
            }
        )
    return pd.DataFrame(rows)


def irr_vs_earned_rate(term: TermAssumptions, dia: DIAAssumptions, rates) -> pd.DataFrame:
    """IRR and profit margin at the base price across a range of earned rates."""
    g = TermLife(term).price().gross_premium
    inc = DeferredIncomeAnnuity(dia).price().annual_income
    rows = []
    for r in rates:
        tm = TermLife(term.with_(earned_rate=r)).project(g).metrics
        dm = DeferredIncomeAnnuity(dia.with_(earned_rate=r)).project(inc).metrics
        rows.append(
            {
                "earned_rate": r,
                "term_irr": tm["irr"],
                "term_margin": tm["profit_margin"],
                "dia_irr": dm["irr"],
                "dia_margin": dm["profit_margin"],
            }
        )
    return pd.DataFrame(rows)


def term_reserve_grid(base: TermAssumptions, val_rates, mort_pcts) -> pd.DataFrame:
    """CRVM reserve per 1000 by duration across valuation interest / mortality."""
    out = {}
    g = TermLife(base).price().gross_premium
    for r in val_rates:
        for m in mort_pcts:
            rf = TermLife(base.with_(valuation_rate=r, valuation_mortality_pct=m)).reserve_factors(g)
            out[f"i={r:.2%}, q={m:.0%} CSO"] = rf.total * 1000
    df = pd.DataFrame(out)
    df.index.name = "duration"
    return df


def dia_reserve_grid(base: DIAAssumptions, val_rates, improvement_mults) -> pd.DataFrame:
    """CARVM reserve at issue as % of premium across valuation rate and improvement."""
    inc = DeferredIncomeAnnuity(base).price().annual_income
    db = base.premium if base.return_of_premium else 0.0
    rows = []
    for r in val_rates:
        row = {"valuation_rate": r}
        for m in improvement_mults:
            q = iar_2012(base.sex, base.issue_year, m).q(base.issue_age)
            res = deferred_income_annuity_reserves(q, r, base.deferral, inc, db)
            row[f"{m:.0%} G2"] = res[0] / base.premium
        rows.append(row)
    return pd.DataFrame(rows)


def default_rate_grid() -> np.ndarray:
    return np.round(np.arange(0.03, 0.0701, 0.0025), 4)
