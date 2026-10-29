"""Run the full model: validate, price, project, reserve, stress.

    python run_model.py            # writes everything to ./outputs
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lamodel import sensitivity as sens
from lamodel.deferred_annuity import DIAAssumptions, DeferredIncomeAnnuity
from lamodel.term_life import TermAssumptions, TermLife
from lamodel.validation import susm_check

OUT = Path(__file__).resolve().parent / "outputs"

# chart styling: reference categorical palette, recessive axes, thin marks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def fmt_pct(x, d=2):
    return f"{x:.{d}%}"


def charts(term, dia, term_sens, dia_sens, irr_curve, dia_grid):
    # 1. Term: statutory reserve by duration (CRVM vs NLP)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    p = term.projection
    ax.plot(p.year, p.nlp_factor_per_1000, color=ORANGE, label="Net level premium")
    ax.plot(p.year, p.crvm_factor_per_1000, color=BLUE, label="CRVM (full preliminary term)")
    ax.set(xlabel="Policy year (end)", ylabel="Terminal reserve per $1,000 face",
           title="20-year term: statutory reserve, 2017 CSO NS male select, 3.5%")
    ax.legend(loc="upper left")
    ax.set_xticks(range(0, 21, 2))
    fig.tight_layout(); fig.savefig(OUT / "term_reserves.png"); plt.close(fig)

    # 2. DIA: CARVM reserve at issue across valuation rate and improvement scale
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for col, c in zip(dia_grid.columns[1:], [ORANGE, BLUE, AQUA]):
        ax.plot(dia_grid.valuation_rate * 100, dia_grid[col] * 100, color=c, marker="o", markersize=4,
                label=f"2012 IAM x {col}")
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(dia_grid.valuation_rate.max() * 100, 101, "premium received", color=MUTED, ha="right", va="bottom")
    ax.set(xlabel="Valuation interest rate (%)", ylabel="CARVM reserve at issue (% of premium)",
           title="Deferred income annuity: issue reserve vs. valuation rate and improvement")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "dia_reserve_sensitivity.png"); plt.close(fig)

    # 3. Term: distributable earnings by year
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(p.year, p.distributable_earnings, color=np.where(p.distributable_earnings < 0, ORANGE, BLUE),
           width=0.7, edgecolor="white", linewidth=1)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set(xlabel="Policy year", ylabel="Distributable earnings per policy issued ($)",
           title=f"20-year term: distributable earnings (IRR {fmt_pct(term.metrics['irr'], 1)})")
    ax.set_xticks(range(1, 21))
    fig.tight_layout(); fig.savefig(OUT / "term_distributable_earnings.png"); plt.close(fig)

    # 4. DIA: CARVM reserve run-off per contract issued
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    d = dia.projection.iloc[:55]
    ax.plot(d.attained_age + 1, d.reserve_eop, color=BLUE, label="CARVM reserve (2012 IAR, 4.75%)")
    ax.set(xlabel="Attained age (end of year)", ylabel="Reserve per contract issued ($)",
           title="Deferred income annuity: statutory reserve run-off")
    ax.axvline(65, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(65.8, 10_000, "income starts at 65", color=MUTED)
    ax.legend(loc="upper right")
    ax.set_xlim(55, 110)
    fig.tight_layout(); fig.savefig(OUT / "dia_reserves.png"); plt.close(fig)

    # 5. Profit margin vs earned rate (comparable across products; IRR is not
    #    unique for the annuity once later-year earnings turn negative)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(irr_curve.earned_rate * 100, irr_curve.term_margin * 100, color=BLUE, marker="o", markersize=4, label="20-year term (% of PV premium)")
    ax.plot(irr_curve.earned_rate * 100, irr_curve.dia_margin * 100, color=ORANGE, marker="o", markersize=4, label="Deferred income annuity (% of premium)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set(xlabel="Portfolio earned rate (%)", ylabel="PV of distributable earnings at 10% (%)",
           title="Interest-rate sensitivity: profit margin at the priced premium / payout")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "margin_vs_earned_rate.png"); plt.close(fig)

    # 6/7. Tornado charts: change from base with price held fixed
    for name, df, col, metric, fname in [
        ("20-year term", term_sens, "irr_at_base_premium", "IRR", "term_tornado.png"),
        ("Deferred income annuity", dia_sens, "profit_margin_at_base_payout", "profit margin", "dia_tornado.png"),
    ]:
        base = df.loc[df.scenario == "Base", col].iloc[0]
        s = df[df.scenario != "Base"].assign(delta=lambda x: (x[col] - base) * 1e4)
        s = s.reindex(s.delta.abs().sort_values().index)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.barh(s.scenario, s.delta, color=np.where(s.delta < 0, ORANGE, BLUE), height=0.6,
                edgecolor="white", linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        lim = s.delta.abs().max() * 1.35
        for y, v in enumerate(s.delta):
            ax.text(v + (lim * 0.02 if v >= 0 else -lim * 0.02), y, f"{v:+.0f}", va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="y", visible=False)
        ax.set(xlabel=f"Change in {metric} vs. base (bp), price held fixed",
               title=f"{name}: sensitivity of {metric} (base {fmt_pct(base, 1)})")
        fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)


def summary_md(term, dia, val, term_sens, dia_sens, dia_grid) -> str:
    tm, dm = term.metrics, dia.metrics
    ta, da = term.assumptions, dia.assumptions
    lines = [
        "# Model results",
        "",
        "Generated by `python run_model.py`.",
        "",
        "## Engine validation: ALTAM Standard Ultimate Survival Model, i = 5%",
        "",
        val.to_markdown(index=False, floatfmt=".5f"),
        "",
        f"## 20-year level term: male {ta.issue_age} nonsmoker, ${ta.face:,.0f} face",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Gross annual premium | ${tm['gross_annual_premium']:,.2f} (${tm['premium_per_1000']:.3f} per $1,000) |",
        f"| IRR on distributable earnings | {fmt_pct(tm['irr'])} |",
        f"| Breakeven year (cumulative DE) | {tm['breakeven_year']} |",
        f"| Valuation basis | 2017 CSO NS male select & ultimate, {fmt_pct(ta.valuation_rate)} |",
        f"| NLP valuation net premium per $1,000 | {tm['valuation_net_premium_per_1000_nlp']:.4f} |",
        f"| CRVM alpha / beta per $1,000 | {tm['crvm_alpha_per_1000']:.4f} / {tm['crvm_beta_per_1000']:.4f} ({tm['crvm_method']}) |",
        f"| Peak CRVM reserve per $1,000 | {tm['peak_reserve_per_1000']:.3f} |",
        f"| Deficiency reserve required | {'Yes' if tm['deficiency_reserve'] else 'No (gross premium > CRVM net premium)'} |",
        "",
        "### Sensitivities",
        "",
        term_sens.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Deferred income annuity: male {da.issue_age}, ${da.premium:,.0f} single premium, income from age {da.issue_age + da.deferral}",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Annual life income | ${dm['annual_income']:,.2f} (payout rate {fmt_pct(dm['payout_rate'])}) |",
        f"| IRR on distributable earnings | {fmt_pct(dm['irr'])} |",
        f"| Breakeven year (cumulative DE) | {dm['breakeven_year']} |",
        f"| Valuation basis | 2012 IAR (2012 IAM x Scale G2 generational), {fmt_pct(da.valuation_rate)} |",
        f"| CARVM reserve at issue | ${dm['initial_reserve']:,.0f} ({fmt_pct(dm['initial_reserve_pct_premium'], 1)} of premium) |",
        f"| Peak CARVM reserve | ${dm['peak_reserve_per_contract']:,.0f} |",
        f"| Curtate life expectancy at issue (2012 IAR) | {dm['life_expectancy_at_issue_valuation']:.1f} years |",
        "",
        "### Sensitivities",
        "",
        dia_sens.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### CARVM reserve at issue (% of premium) by valuation rate and improvement scale",
        "",
        dia_grid.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    term_a, dia_a = TermAssumptions(), DIAAssumptions()

    val = susm_check()
    term = TermLife(term_a).price()
    dia = DeferredIncomeAnnuity(dia_a).price()
    term_sens = sens.run_term(term_a)
    dia_sens = sens.run_dia(dia_a)
    irr_curve = sens.irr_vs_earned_rate(term_a, dia_a, sens.default_rate_grid())
    term_grid = sens.term_reserve_grid(term_a, [0.03, 0.035, 0.04], [1.0])
    term_grid_mort = sens.term_reserve_grid(term_a, [0.035], [0.9, 1.1])
    dia_grid = sens.dia_reserve_grid(dia_a, [0.0375, 0.0425, 0.0475, 0.0525, 0.0575], [0.0, 1.0, 1.5])

    term.projection.to_csv(OUT / "term_projection.csv", index=False)
    dia.projection.to_csv(OUT / "dia_projection.csv", index=False)
    term_sens.to_csv(OUT / "term_sensitivity.csv", index=False)
    dia_sens.to_csv(OUT / "dia_sensitivity.csv", index=False)
    irr_curve.to_csv(OUT / "margin_vs_earned_rate.csv", index=False)

    with pd.ExcelWriter(OUT / "life_annuity_model.xlsx") as xw:
        val.to_excel(xw, sheet_name="ALTAM validation", index=False)
        pd.DataFrame([term.metrics]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="Term summary")
        pd.DataFrame([vars(term_a)]).T.rename(columns={0: "assumption"}).astype(str).to_excel(xw, sheet_name="Term assumptions")
        term.projection.to_excel(xw, sheet_name="Term projection", index=False)
        term_sens.to_excel(xw, sheet_name="Term sensitivity", index=False)
        pd.concat([term_grid, term_grid_mort], axis=1).to_excel(xw, sheet_name="Term reserve grid")
        pd.DataFrame([dia.metrics]).T.rename(columns={0: "value"}).to_excel(xw, sheet_name="DIA summary")
        pd.DataFrame([vars(dia_a)]).T.rename(columns={0: "assumption"}).astype(str).to_excel(xw, sheet_name="DIA assumptions")
        dia.projection.to_excel(xw, sheet_name="DIA projection", index=False)
        dia_sens.to_excel(xw, sheet_name="DIA sensitivity", index=False)
        dia_grid.to_excel(xw, sheet_name="DIA reserve grid", index=False)
        irr_curve.to_excel(xw, sheet_name="Margin vs earned rate", index=False)

    charts(term, dia, term_sens, dia_sens, irr_curve, dia_grid)
    (OUT / "summary.md").write_text(summary_md(term, dia, val, term_sens, dia_sens, dia_grid))
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
