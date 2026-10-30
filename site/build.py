"""Build the project website from the model's outputs.

    python run_model.py          # refresh outputs/ first if assumptions changed
    python site/build.py         # writes site/_build/ (index.html + assets/)

Every number on the page is read from ``outputs/`` or recomputed with
``lamodel``, so the site can never drift from the model.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import string
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lamodel.deferred_annuity import DeferredIncomeAnnuity, DIAAssumptions  # noqa: E402
from lamodel.term_life import TermAssumptions, TermLife  # noqa: E402
from lamodel.validation import susm_check  # noqa: E402

OUT = ROOT / "outputs"
BUILD = ROOT / "site" / "_build"
REPO = "https://github.com/richterrw/Life-and-annuity-pricing-and-reserving-model"
CONTACT = {"github": "https://github.com/richterrw", "email": "robbierichter15@gmail.com"}


class Page(string.Template):
    delimiter = "@@"


def money(v, dp=0):
    return f"${v:,.{dp}f}"


def pct(v, dp=1):
    return f"{v:.{dp}%}"


def bp(v):
    return f"{v * 1e4:+.0f}bp"


def count_tests() -> int:
    """Number of tests as pytest collects them (parametrized cases included)."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return sum("::" in line for line in res.stdout.splitlines())


def main():
    term = TermLife(TermAssumptions()).price()
    dia = DeferredIncomeAnnuity(DIAAssumptions()).price()
    tm, dm = term.metrics, dia.metrics

    tsens = pd.read_csv(OUT / "term_sensitivity.csv").set_index("scenario")
    dsens = pd.read_csv(OUT / "dia_sensitivity.csv").set_index("scenario")
    dres = pd.read_csv(OUT / "dia_reserve_comparison.csv")
    dbuild = pd.read_csv(OUT / "dia_premium_buildup.csv").set_index("component")
    mrisk = pd.read_csv(OUT / "term_mortality_risk.csv")
    dstoch = pd.read_csv(OUT / "dia_stochastic_profit.csv").iloc[0]
    tcard = pd.read_csv(OUT / "term_rate_card.csv")
    dcard = pd.read_csv(OUT / "dia_rate_card.csv")

    def var(n, risk_prefix):
        row = mrisk[(mrisk.policies == n) & mrisk.risk.str.startswith(risk_prefix)].iloc[0]
        return pct(row.var995_pct_premium)

    base_irr = tsens.loc["Base", "irr_at_base_premium"]
    irr_mort = tsens.loc["Mortality 110%", "irr_at_base_premium"]
    irr_exp = tsens.loc["Expenses 110%", "irr_at_base_premium"]
    dmarg = lambda s: dsens.loc[s, "profit_margin_at_base_payout"]
    reprice = dsens.loc["Earned rate -100bp", "payout_rate_change_bp"]

    altam_rows = "".join(
        f"<tr><td>{int(r.age)}</td><td>{r['a-due_x model']:.4f}</td><td>{r['a-due_x ALTAM']:.4f}</td>"
        f"<td>{r['A_x model']:.5f}</td><td>{r['A_x ALTAM']:.5f}</td></tr>"
        for _, r in susm_check().iterrows()
    )
    term_json = json.dumps(
        [{"age": int(r.issue_age), "M": r["M annual premium"], "F": r["F annual premium"]} for _, r in tcard.iterrows()]
    )
    dia_json = json.dumps(
        {
            int(r.issue_age): {
                int(c.split()[1]): (None if pd.isna(r[c]) else round(r[c], 6)) for c in dcard.columns[1:]
            }
            for _, r in dcard.iterrows()
        }
    )

    values = {
        "repo": REPO,
        "github": CONTACT["github"],
        "email": CONTACT["email"],
        "n_tests": str(count_tests()),
        "built": dt.date.today().strftime("%B %Y"),
        # term
        "term_premium": money(tm["gross_annual_premium"], 2),
        "term_per1000": money(tm["premium_per_1000"], 3),
        "term_irr": pct(tm["irr"], 2),
        "term_breakeven": str(tm["breakeven_year"]),
        "term_alpha": money(tm["crvm_alpha_per_1000"], 3),
        "term_beta": money(tm["crvm_beta_per_1000"], 3),
        "term_peak": money(tm["peak_reserve_per_1000"], 2),
        "term_irr_mort": pct(irr_mort, 2),
        "term_irr_mort_bp": bp(irr_mort - base_irr),
        "term_irr_mort_bp_abs": f"{abs(irr_mort - base_irr) * 1e4:.0f}bp",
        "term_exp_bp_abs": f"{abs(irr_exp - base_irr) * 1e4:.0f}bp",
        "term_var10k": var(10_000, "Process +"),
        # annuity
        "dia_income": money(dm["annual_income"]),
        "dia_payout": pct(dm["payout_rate"], 2),
        "dia_margin": pct(dbuild.loc["Margin for tax, capital cost and profit", "pct_of_premium"]),
        "dia_carvm_pct": pct(dm["initial_reserve_pct_premium"]),
        "dia_sr_pct": pct(dres.iloc[0].stochastic_cte70 / DIAAssumptions().premium),
        "dia_peak": money(dm["peak_reserve_per_contract"]),
        "dia_rate_down": pct(dmarg("Earned rate -100bp")),
        "dia_rate_down_abs": pct(abs(dmarg("Earned rate -100bp"))),
        "dia_reprice_bp": f"{reprice:+.0f}bp",
        "dia_reprice_bp_abs": f"{abs(reprice):.0f}bp",
        "dia_cte98": pct(dstoch.cte98_loss),
        "dia_noimp": f"{dmarg('No mortality improvement') * 1e4:.0f}bp",
        "dia_g2x15": f"{abs(dmarg('150% Scale G2 improvement')) * 1e4:.0f}bp",
        "dia_mort90": f"{abs(dmarg('Mortality 90% (longevity)')) * 1e4:.0f}bp",
        # risk
        "var1k_proc": var(1_000, "Process only"),
        "var100k_proc": var(100_000, "Process only"),
        "var100k_param": var(100_000, "Process +"),
        # tables and data
        "altam_rows": altam_rows,
        "term_json": term_json,
        "dia_json": dia_json,
    }

    html = Page((ROOT / "site" / "template.html").read_text()).substitute(values)

    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "assets").mkdir(parents=True)
    (BUILD / "index.html").write_text(html)
    (BUILD / ".nojekyll").write_text("")
    for f in list(OUT.glob("*.png")) + [OUT / "life_annuity_model.xlsx"]:
        shutil.copy2(f, BUILD / "assets" / f.name)
    print(f"built {BUILD / 'index.html'}")


if __name__ == "__main__":
    main()
