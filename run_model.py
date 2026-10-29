"""Run the full model: validate, price, project, reserve, stress, simulate.

python run_model.py                      # full run, writes ./outputs
python run_model.py --quick              # fewer scenarios / simulations
python run_model.py --scenarios 2000 --sims 10000 --seed 7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from lamodel import analysis as an
from lamodel import charts
from lamodel import sensitivity as sens
from lamodel import stochastic as st
from lamodel.deferred_annuity import DeferredIncomeAnnuity, DIAAssumptions
from lamodel.excel import write_workbook
from lamodel.term_life import TermAssumptions, TermLife
from lamodel.validation import susm_check

OUT = Path(__file__).resolve().parent / "outputs"


def md(df: pd.DataFrame, pct=(), money=(), dp=4) -> str:
    """Markdown table with readable number formatting."""
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        elif c in money:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
        elif d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{dp}f}")
    return d.to_markdown(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=int, default=1000, help="interest-rate scenarios")
    ap.add_argument("--sims", type=int, default=5000, help="mortality simulations per block size")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--quick", action="store_true", help="200 scenarios / 1,000 simulations")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.quick:
        args.scenarios, args.sims = 200, 1000
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.1f}s] {msg}")

    term_a, dia_a = TermAssumptions(), DIAAssumptions()

    # --- deterministic ----------------------------------------------------------
    log("validating engine against ALTAM SUSM")
    val = susm_check()
    log("pricing 20-year term and deferred income annuity")
    term = TermLife(term_a).price()
    dia = DeferredIncomeAnnuity(dia_a).price()
    term_build = an.term_premium_buildup(term)
    dia_build = an.dia_premium_buildup(dia)

    log("sensitivities and rate cards")
    term_sens = sens.run_term(term_a)
    dia_sens = sens.run_dia(dia_a)
    curve = sens.irr_vs_earned_rate(term_a, dia_a, sens.default_rate_grid())
    term_grid = sens.term_reserve_grid(term_a, [0.03, 0.035, 0.04], [0.9, 1.0, 1.1])
    dia_grid = sens.dia_reserve_grid(dia_a, [0.0375, 0.0425, 0.0475, 0.0525, 0.0575], [0.0, 1.0, 1.5])
    term_card = an.term_rate_card(term_a)
    dia_card = an.dia_rate_card(dia_a)

    # --- stochastic -------------------------------------------------------------
    log(f"generating {args.scenarios:,} interest scenarios")
    n_dia = len(dia.projection)
    dia_earned, dia_new_money = an.interest_scenarios(dia_a.earned_rate, n_dia, args.scenarios, args.seed)
    term_earned, _ = an.interest_scenarios(term_a.earned_rate, term_a.term, args.scenarios, args.seed)

    log("DIA profit distribution across interest scenarios")
    dia_pv = an.dia_stochastic_profit(dia, dia_earned)
    dia_dist = st.summarise(dia_pv)
    log("stochastic vs formula reserves")
    dia_res_cmp = an.dia_reserve_comparison(dia, dia_earned)
    term_res_cmp = an.term_reserve_comparison(term, term_earned)
    log(f"term mortality simulation ({args.sims:,} sims per block size)")
    mort_risk = an.term_mortality_risk(term, n_sims=args.sims, seed=args.seed)

    stoch_dist = pd.DataFrame(
        [
            {
                "product": "Deferred income annuity (interest)",
                **{k: v / dia_a.premium for k, v in dia_dist.items() if k != "prob_loss"},
                "prob_loss": dia_dist["prob_loss"],
            }
        ]
    )

    # --- outputs ----------------------------------------------------------------
    log("writing CSVs, workbook, charts and summary")
    csvs = {
        "term_projection": term.projection,
        "dia_projection": dia.projection,
        "term_sensitivity": term_sens,
        "dia_sensitivity": dia_sens,
        "margin_vs_earned_rate": curve,
        "term_rate_card": term_card,
        "dia_rate_card": dia_card,
        "term_premium_buildup": term_build,
        "dia_premium_buildup": dia_build,
        "term_reserve_comparison": term_res_cmp,
        "dia_reserve_comparison": dia_res_cmp,
        "term_mortality_risk": mort_risk,
    }
    for name, df in csvs.items():
        df.to_csv(out / f"{name}.csv", index=False)

    tm, dm = term.metrics, dia.metrics
    sr0 = dia_res_cmp.iloc[0]
    mr = mort_risk.set_index(["policies", "risk"])
    summary = [
        (
            "20-year level term: male 35 nonsmoker, $500,000",
            [
                ("Gross annual premium", tm["gross_annual_premium"], '"$"#,##0.00'),
                ("Premium per $1,000", tm["premium_per_1000"], '"$"0.000'),
                ("IRR on distributable earnings", tm["irr"], "0.00%"),
                ("Breakeven year", tm["breakeven_year"], "0"),
                ("CRVM method", tm["crvm_method"], None),
                ("Peak CRVM reserve per $1,000", tm["peak_reserve_per_1000"], "0.000"),
                ("Deficiency reserve", "Yes" if tm["deficiency_reserve"] else "No", None),
                (
                    "IRR, mortality +10%",
                    term_sens.set_index("scenario").loc["Mortality 110%", "irr_at_base_premium"],
                    "0.00%",
                ),
                (
                    "IRR, earned rate -100bp",
                    term_sens.set_index("scenario").loc["Earned rate -100bp", "irr_at_base_premium"],
                    "0.00%",
                ),
            ],
        ),
        (
            "Deferred income annuity: male 55, $100,000, income from 65",
            [
                ("Annual income", dm["annual_income"], '"$"#,##0.00'),
                ("Payout rate", dm["payout_rate"], "0.00%"),
                ("Pricing margin (tax, capital, profit), % premium", dia_build.iloc[4].pct_of_premium, "0.00%"),
                ("CARVM reserve at issue, % premium", dm["initial_reserve_pct_premium"], "0.00%"),
                ("Stochastic reserve (CTE70) at issue, % premium", sr0.stochastic_cte70 / dia_a.premium, "0.00%"),
                (
                    "PV profit, earned rate -100bp, % premium",
                    dia_sens.set_index("scenario").loc["Earned rate -100bp", "profit_margin_at_base_payout"],
                    "0.00%",
                ),
            ],
        ),
        (
            f"Stochastic results ({args.scenarios:,} interest scenarios; {args.sims:,} mortality sims)",
            [
                ("DIA: probability PV profit < 0", dia_dist["prob_loss"], "0.0%"),
                ("DIA: CTE98 of PV loss, % premium", dia_dist["cte98_loss"] / dia_a.premium, "0.00%"),
                (
                    "Term 10k block: 99.5% VaR, process only, % PV premium",
                    mr.loc[(10_000, "Process only"), "var995_pct_premium"],
                    "0.00%",
                ),
                (
                    "Term 10k block: 99.5% VaR, process + parameter, % PV premium",
                    mr.loc[10_000].iloc[1].var995_pct_premium,
                    "0.00%",
                ),
            ],
        ),
    ]
    sheets = {
        "ALTAM validation": val,
        "Term assumptions": pd.DataFrame([vars(term_a)])
        .T.reset_index()
        .set_axis(["assumption", "value"], axis=1)
        .astype(str),
        "Term projection": term.projection,
        "Term premium build-up": term_build,
        "Term rate card": term_card,
        "Term sensitivity": term_sens,
        "Term reserve grid": term_grid.reset_index(),
        "Term reserve vs stochastic": term_res_cmp,
        "Term mortality risk": mort_risk,
        "DIA assumptions": pd.DataFrame([vars(dia_a)])
        .T.reset_index()
        .set_axis(["assumption", "value"], axis=1)
        .astype(str),
        "DIA projection": dia.projection,
        "DIA premium build-up": dia_build,
        "DIA rate card": dia_card,
        "DIA sensitivity": dia_sens,
        "DIA reserve grid": dia_grid,
        "DIA reserve vs stochastic": dia_res_cmp,
        "DIA stochastic profit": stoch_dist,
        "Margin vs earned rate": curve,
    }
    write_workbook(out / "life_annuity_model.xlsx", summary, sheets)

    charts.term_reserves(term, out / "term_reserves.png")
    charts.term_distributable_earnings(term, out / "term_distributable_earnings.png")
    charts.term_rate_card(term_card, out / "term_rate_card.png")
    charts.premium_buildup(
        term_build,
        out / "term_premium_buildup.png",
        f"20-year term: where the ${tm['gross_annual_premium']:,.0f} premium goes (PV at 10%)",
    )
    charts.tornado(
        term_sens,
        "irr_at_base_premium",
        "IRR",
        f"20-year term: sensitivity of IRR (base {tm['irr']:.1%})",
        out / "term_tornado.png",
    )
    charts.term_reserve_comparison(term_res_cmp, out / "term_reserve_comparison.png")
    charts.mortality_risk(mort_risk, out / "term_mortality_risk.png")
    charts.dia_reserves(dia, out / "dia_reserves.png")
    charts.dia_reserve_sensitivity(dia_grid, out / "dia_reserve_sensitivity.png")
    charts.dia_rate_card(dia_card, out / "dia_rate_card.png")
    charts.premium_buildup(
        dia_build, out / "dia_premium_buildup.png", "DIA: where the $100,000 premium goes (PV at the 5.25% earned rate)"
    )
    charts.tornado(
        dia_sens,
        "profit_margin_at_base_payout",
        "profit margin",
        "DIA: sensitivity of PV profit, % of premium (base 0.0% at the 10% hurdle)",
        out / "dia_tornado.png",
    )
    charts.interest_fan(dia_earned, dia_new_money, out / "interest_scenarios.png")
    charts.dia_profit_distribution(dia_pv, dia_a.premium, out / "dia_profit_distribution.png")
    charts.dia_reserve_comparison(dia_res_cmp, out / "dia_reserve_comparison.png")
    charts.margin_vs_earned_rate(curve, out / "margin_vs_earned_rate.png")

    # --- markdown summary -------------------------------------------------------------
    ta, da = term_a, dia_a
    pct_cols = {
        "irr_at_base_premium",
        "profit_margin_at_base_premium",
        "premium_change_pct",
        "irr_at_base_payout",
        "profit_margin_at_base_payout",
        "initial_reserve_pct_premium",
        "repriced_payout_rate",
        "pct_of_premium",
        "sd_pct_premium",
        "var995_pct_premium",
        "cte98_pct_premium",
        "prob_loss",
        "earned_rate",
        "term_irr",
        "term_margin",
        "dia_irr",
        "dia_margin",
        "valuation_rate",
        "0% G2",
        "100% G2",
        "150% G2",
        "defer 5",
        "defer 10",
        "defer 15",
        "defer 20",
    }
    lines = [
        "# Model results",
        "",
        f"Generated by `python run_model.py` ({args.scenarios:,} interest scenarios, {args.sims:,} mortality "
        f"simulations per block size, seed {args.seed}).",
        "",
        "## Engine validation: ALTAM Standard Ultimate Survival Model, i = 5%",
        "",
        md(val, dp=5),
        "",
        f"## 20-year level term: male {ta.issue_age} nonsmoker, ${ta.face:,.0f} face",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Gross annual premium | ${tm['gross_annual_premium']:,.2f} (${tm['premium_per_1000']:.3f} per $1,000) |",
        f"| IRR on distributable earnings | {tm['irr']:.2%} |",
        f"| Breakeven year (cumulative DE) | {tm['breakeven_year']} |",
        f"| Valuation basis | 2017 CSO NS male select & ultimate, {ta.valuation_rate:.2%} |",
        f"| NLP valuation net premium per $1,000 | {tm['valuation_net_premium_per_1000_nlp']:.4f} |",
        f"| CRVM alpha / beta per $1,000 | {tm['crvm_alpha_per_1000']:.4f} / {tm['crvm_beta_per_1000']:.4f} ({tm['crvm_method']}) |",
        f"| Peak CRVM reserve per $1,000 | {tm['peak_reserve_per_1000']:.3f} |",
        f"| Deficiency reserve required | {'Yes' if tm['deficiency_reserve'] else 'No (gross premium > CRVM net premium)'} |",
        "",
        "### Premium build-up (PV at the 10% hurdle)",
        "",
        md(term_build, pct={"pct_of_premium"}, money={"pv"}),
        "",
        "### Rate card",
        "",
        md(term_card, dp=3),
        "",
        "### Sensitivities",
        "",
        md(term_sens, pct=pct_cols, dp=2),
        "",
        "### Formula vs. stochastic reserve per policy",
        "",
        md(term_res_cmp, money={"crvm", "deterministic", "stochastic_cte70"}),
        "",
        "### Mortality risk (PV of block distributable earnings)",
        "",
        md(mort_risk, pct=pct_cols, money={"pv_premium"}),
        "",
        f"## Deferred income annuity: male {da.issue_age}, ${da.premium:,.0f} single premium, income from age {da.issue_age + da.deferral}",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Annual life income | ${dm['annual_income']:,.2f} (payout rate {dm['payout_rate']:.2%}) |",
        f"| IRR on distributable earnings | {dm['irr']:.2%} |",
        f"| Breakeven year (cumulative DE) | {dm['breakeven_year']} |",
        f"| Valuation basis | 2012 IAR (2012 IAM x Scale G2 generational), {da.valuation_rate:.2%} |",
        f"| CARVM reserve at issue | ${dm['initial_reserve']:,.0f} ({dm['initial_reserve_pct_premium']:.1%} of premium) |",
        f"| Stochastic reserve at issue (CTE70) | ${sr0.stochastic_cte70:,.0f} ({sr0.stochastic_cte70 / da.premium:.1%} of premium) |",
        f"| Peak CARVM reserve | ${dm['peak_reserve_per_contract']:,.0f} |",
        f"| Curtate life expectancy at issue (2012 IAR) | {dm['life_expectancy_at_issue_valuation']:.1f} years |",
        "",
        "### Premium build-up (PV at the earned rate)",
        "",
        md(dia_build, pct={"pct_of_premium"}, money={"pv"}),
        "",
        "### Rate card (payout rate, male)",
        "",
        md(dia_card, pct=pct_cols),
        "",
        "### Sensitivities",
        "",
        md(dia_sens, pct=pct_cols, dp=2),
        "",
        "### CARVM reserve at issue (% of premium) by valuation rate and improvement scale",
        "",
        md(dia_grid, pct=pct_cols),
        "",
        "### Profit distribution across interest scenarios (% of premium)",
        "",
        md(stoch_dist, pct=set(stoch_dist.columns) - {"product"}),
        "",
        "### Formula vs. stochastic reserve per contract (every 5 years)",
        "",
        md(
            dia_res_cmp[dia_res_cmp.duration % 5 == 0],
            money={"carvm", "deterministic", "stochastic_cte70", "stochastic_cte98"},
        ),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
