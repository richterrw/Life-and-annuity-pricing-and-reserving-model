"""Pricing and risk analyses built on the product models.

* Rate cards: term premium per $1,000 by issue age and sex; DIA payout rates
  by issue age and deferral period.
* Premium build-up: where each dollar of premium goes, as PV at the hurdle
  rate.  Because the price is solved so PV of distributable earnings is zero,
  the components add exactly to 100% of PV premium (or are left as profit).
* Stochastic studies: profit distributions and stochastic reserves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stochastic as st
from .deferred_annuity import DeferredIncomeAnnuity, DIAAssumptions
from .term_life import TermAssumptions, TermLife

# ---------------------------------------------------------------------------
# Rate cards
# ---------------------------------------------------------------------------


def term_rate_card(base: TermAssumptions, ages=range(25, 66, 5), sexes=("M", "F")) -> pd.DataFrame:
    rows = []
    for x in ages:
        row = {"issue_age": x}
        for s in sexes:
            r = TermLife(base.with_(issue_age=x, sex=s)).price()
            row[f"{s} annual premium"] = r.gross_premium
            row[f"{s} per $1,000"] = r.metrics["premium_per_1000"]
        rows.append(row)
    return pd.DataFrame(rows)


def dia_rate_card(base: DIAAssumptions, ages=range(45, 71, 5), deferrals=(5, 10, 15, 20), sex="M") -> pd.DataFrame:
    rows = []
    for x in ages:
        row = {"issue_age": x}
        for m in deferrals:
            if x + m > 85:
                row[f"defer {m}"] = np.nan
                continue
            r = DeferredIncomeAnnuity(base.with_(issue_age=x, deferral=m, sex=sex)).price()
            row[f"defer {m}"] = r.metrics["payout_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Premium build-up
# ---------------------------------------------------------------------------


def _pv(x, rate):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * (1 + rate) ** -np.arange(1, len(x) + 1)))


def term_premium_buildup(result) -> pd.DataFrame:
    """Decompose PV premium (at the hurdle rate) into its uses."""
    p, a = result.projection, result.assumptions
    h = a.hurdle_rate
    model = TermLife(a)
    f = model.financials(result.gross_premium, *model.decrements(model.pricing_q(), model.lapse_rates()))
    rc0 = float(f["rc_0"])
    rc_bop = np.concatenate([[rc0], f["rc_eop"][:-1]])
    parts = {
        "Death claims": _pv(p.death_claims, h),
        "Commission": _pv(p.commission, h),
        "Premium tax": _pv(p.premium_tax, h),
        "Acquisition & maintenance expense": _pv(p.expenses, h),
        "Reserve funding (increase less interest)": _pv(p.increase_in_reserve - p.investment_income, h),
        "Income tax": _pv(p.tax, h),
        "Cost of required capital": _pv(f["rc_eop"] - rc_bop - f["int_on_rc"], h) + rc0,
        "Profit (PV of DE at hurdle)": _pv(p.distributable_earnings, h) - rc0,
    }
    total = _pv(p.premium, h)
    df = pd.DataFrame({"component": list(parts), "pv": list(parts.values())})
    df["pct_of_premium"] = df.pv / total
    return pd.concat(
        [df, pd.DataFrame([{"component": "Gross premium", "pv": total, "pct_of_premium": 1.0}])], ignore_index=True
    )


def dia_premium_buildup(result) -> pd.DataFrame:
    """Decompose the single premium on the annuity pricing view.

    Cash flows are discounted at the portfolio earned rate with their actual
    timing (income and expenses at the start of the year, death benefits at the
    end).  What remains of the premium is the pricing margin, which has to pay
    for income tax, the cost of statutory reserves and capital above the earned
    rate, and the profit that delivers the hurdle return.
    """
    p, a = result.projection, result.assumptions
    i = a.earned_rate
    v_bop = (1 + i) ** -np.arange(len(p))
    v_eop = v_bop / (1 + i)
    parts = {
        "Life income payments": float(p.annuity_income @ v_bop),
        "Return-of-premium death benefits": float(p.death_benefits @ v_eop),
        "Commission": float(p.commission @ v_bop),
        "Acquisition & maintenance expense": float(p.expenses @ v_bop),
    }
    parts["Margin for tax, capital cost and profit"] = a.premium - sum(parts.values())
    df = pd.DataFrame({"component": list(parts), "pv": list(parts.values())})
    df["pct_of_premium"] = df.pv / a.premium
    return pd.concat(
        [df, pd.DataFrame([{"component": "Single premium", "pv": a.premium, "pct_of_premium": 1.0}])], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Stochastic studies
# ---------------------------------------------------------------------------


def interest_scenarios(
    initial_yield: float, n_years: int, n_scen: int, seed: int = 2026, model: st.VasicekModel | None = None
):
    model = model or st.VasicekModel()
    return st.PortfolioYield(initial_yield).earned_rates(model, n_years, n_scen, seed)


def dia_stochastic_profit(result, earned: np.ndarray) -> np.ndarray:
    """PV at hurdle of DE for each interest scenario, price held at base."""
    a = result.assumptions
    model = DeferredIncomeAnnuity(a)
    disc = (1 + a.hurdle_rate) ** -np.arange(earned.shape[1] + 1)
    rc0 = a.rc_reserve_factor * result.metrics["initial_reserve"]
    out = np.empty(len(earned))
    for s, path in enumerate(earned):
        de = model.project(result.annual_income, earned_rate=path).projection.distributable_earnings.values
        out[s] = np.concatenate([[-rc0], de]) @ disc
    return out


def dia_reserve_comparison(
    result, earned: np.ndarray, mortality_margin: float = -0.10, years=range(0, 41)
) -> pd.DataFrame:
    """CARVM vs deterministic (DR-style) vs stochastic CTE70 (SR-style) reserve by duration."""
    a = result.assumptions
    model = DeferredIncomeAnnuity(a)
    carvm = result.projection.carvm_reserve_per_contract.values
    rows = []
    for tau in years:
        bop, eop = model.liability_cash_flows(result.annual_income, tau, mortality_margin)
        n = len(bop)
        g = st.gpvad(bop, eop, earned[:, :n])
        dr = st.gpvad(bop, eop, np.full((1, n), a.earned_rate))[0]
        rows.append(
            {
                "duration": tau,
                "attained_age": a.issue_age + tau,
                "carvm": result.metrics["initial_reserve"] if tau == 0 else carvm[tau - 1],
                "deterministic": dr,
                "stochastic_cte70": st.cte(g, 0.70),
                "stochastic_cte98": st.cte(g, 0.98),
            }
        )
    return pd.DataFrame(rows)


def term_reserve_comparison(
    result, earned: np.ndarray, mortality_margin: float = 0.10, lapse_multiplier: float = 0.75
) -> pd.DataFrame:
    """CRVM vs deterministic vs stochastic CTE70 reserve per policy by duration."""
    a = result.assumptions
    model = TermLife(a)
    crvm = model.reserve_factors(result.gross_premium).total * a.face
    rows = []
    for tau in range(1, a.term):  # valuation dates after the first premium
        bop, eop = model.liability_cash_flows(result.gross_premium, tau, mortality_margin, lapse_multiplier)
        n = len(bop)
        g = st.gpvad(bop, eop, earned[:, :n])
        dr = st.gpvad(bop, eop, np.full((1, n), a.earned_rate))[0]
        rows.append({"duration": tau, "crvm": crvm[tau], "deterministic": dr, "stochastic_cte70": st.cte(g, 0.70)})
    return pd.DataFrame(rows)


def term_mortality_risk(
    result, block_sizes=(1_000, 10_000, 100_000), n_sims: int = 5_000, param_sd: float = 0.10, seed: int = 2026
) -> pd.DataFrame:
    """Process vs parameter risk in PV of profit for blocks of different sizes."""
    model = TermLife(result.assumptions)
    rows = []
    for n in block_sizes:
        for label, sd in [("Process only", 0.0), (f"Process + parameter ({param_sd:.0%} level sd)", param_sd)]:
            sim = st.simulate_term_block(model, result.gross_premium, n, n_sims, sd, seed)
            s = st.summarise(sim["pv_de"])
            rows.append(
                {
                    "policies": n,
                    "risk": label,
                    "pv_premium": sim["pv_premium"],
                    "sd_pct_premium": s["std"] / sim["pv_premium"],
                    "var995_pct_premium": s["var_995_loss"] / sim["pv_premium"],
                    "cte98_pct_premium": s["cte98_loss"] / sim["pv_premium"],
                    "prob_loss": s["prob_loss"],
                }
            )
    return pd.DataFrame(rows)
