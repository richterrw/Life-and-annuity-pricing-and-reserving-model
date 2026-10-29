"""Rebuild and execute notebooks/walkthrough.ipynb.  Run from the repository root:

python notebooks/build_walkthrough.py
"""

import nbformat as nbf
from nbclient import NotebookClient

cells = []
M = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

M("""
# Life & annuity pricing and reserving: walkthrough

This notebook walks through the model end to end, the way a pricing actuary would build it up:

1. **Mortality**: load SOA tables (2017 CSO select & ultimate, 2012 IAM + Scale G2)
2. **Validate the engine** against ALTAM's Standard Ultimate Survival Model
3. **Price a 20-year term policy** to a 10% IRR and compute its CRVM reserves
4. **Price a deferred income annuity** and compute its CARVM reserve
5. **Stress-test** interest, mortality, lapse and expense assumptions
6. **Go stochastic**: interest scenarios, a stochastic reserve, and mortality risk for a block

Every number below comes from the `lamodel` package in this repository.
""")
C("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from lamodel import actuarial as act, analysis as an, charts, sensitivity as sens, stochastic as st
from lamodel.tables import cso_2017, iar_2012
from lamodel.term_life import TermAssumptions, TermLife
from lamodel.deferred_annuity import DIAAssumptions, DeferredIncomeAnnuity
from lamodel.validation import susm_check
%matplotlib inline

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from pathlib import Path
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
""")
M("""
## 1. Mortality tables

The 2017 CSO is a **select and ultimate** table: a newly underwritten life has lighter mortality than the ultimate
rate for the same attained age, and the advantage wears off over the 25-year select period. Every basis in the model
reduces to one vector of one-year death rates from the issue age.
""")
C("""
cso = cso_2017("M")
q_select = cso.q(35, 40)
q_ult = cso_2017("M", use_select=False).q(35, 40)
iar = iar_2012("M", issue_year=2026).q(55)
iar_static = iar_2012("M", 2026, improvement_multiplier=0).q(55)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
ax = axes[0]
ax.plot(35 + np.arange(40), q_select * 1000, color=charts.BLUE, label="Select (issued at 35)")
ax.plot(35 + np.arange(40), q_ult * 1000, color=charts.ORANGE, label="Ultimate")
ax.set(title="2017 CSO NS male: select vs. ultimate", xlabel="Attained age", ylabel="q per 1,000"); ax.legend()
ax = axes[1]
ax.plot(55 + np.arange(50), iar[:50] * 1000, color=charts.BLUE, label="2012 IAR (G2 generational), issued 2026")
ax.plot(55 + np.arange(50), iar_static[:50] * 1000, color=charts.ORANGE, label="2012 IAM basic (no improvement)")
ax.set(title="Annuitant mortality: male issued at 55", xlabel="Attained age", ylabel="q per 1,000", yscale="log"); ax.legend()
plt.tight_layout(); plt.show()
""")
M("""
## 2. Engine validation against ALTAM

Before pricing anything, the actuarial functions are checked against ALTAM's published Standard Ultimate Survival
Model tables (Makeham: A = 0.00022, B = 2.7e-6, c = 1.124, i = 5%).
""")
C("susm_check()")
M("""
## 3. 20-year level term

The premium is solved so that distributable earnings (statutory profit after tax, less the increase in required
capital) earn exactly the 10% hurdle rate. Reserves are on the statutory basis: 100% 2017 CSO select at 3.5%.
""")
C("""
term = TermLife(TermAssumptions()).price()
pd.Series(term.metrics).to_frame("value")
""")
C("""
cols = ["year", "inforce_bop", "premium", "commission", "expenses", "death_claims", "investment_income",
        "increase_in_reserve", "pretax_stat_gain", "distributable_earnings", "crvm_factor_per_1000"]
term.projection[cols].round(2).head(10)
""")
M("""
**Where the premium goes.** Since the price is solved to a zero PV of profit at the hurdle, the components below
add to exactly 100% of PV premium. Holding reserves and capital is a *cost*: they earn 5% while the shareholder
requires 10%.
""")
C("""
build = an.term_premium_buildup(term)
charts.premium_buildup(build, FIG / "term_buildup.png", "20-year term: where the premium goes (PV at 10%)")
display(Image(FIG / "term_buildup.png"))
""")
M("""
**CRVM.** For level term, the expense allowance cap (the 19-payment life premium) does not bind, so CRVM equals
full preliminary term: the first-year net premium is just the one-year term cost and the reserve at the end of year 1
is zero. The recursion $({}_tV + P)(1+i) = q\\,b + p\\,{}_{t+1}V$ holds year by year (see `tests/`).
""")
C("""
rf = term.reserve_factors
print(f"NLP premium per $1,000:  {rf.nlp_premium * 1000:.4f}")
print(f"CRVM alpha / beta:       {rf.alpha * 1000:.4f} / {rf.beta * 1000:.4f}   ({rf.method})")
display(Image("../outputs/term_reserves.png"))
""")
M("""
## 4. Deferred income annuity

A single premium of $100,000 at 55 buys level life income from 65, with return of premium on death before 65.
""")
C("""
dia = DeferredIncomeAnnuity(DIAAssumptions()).price()
pd.Series(dia.metrics).to_frame("value")
""")
C("an.dia_premium_buildup(dia)")
C("display(Image('../outputs/dia_reserves.png'))")
M("""
## 5. Sensitivities

Each shock is applied with the price held at base (impact on profitability) and with the price re-solved
(what the shock would cost the customer).
""")
C(
    "sens.run_term(TermAssumptions())[['scenario', 'irr_at_base_premium', 'repriced_annual_premium', 'premium_change_pct']]"
)
C(
    "sens.run_dia(DIAAssumptions())[['scenario', 'profit_margin_at_base_payout', 'initial_reserve_pct_premium', 'repriced_payout_rate']]"
)
M("""
Term life is a **mortality and expense** business, and the deferred annuity is a **spread** business: 100bp of earned
rate moves annuity profit by about 9% of premium, roughly ten times any mortality shock.

Note that several annuity IRRs come back blank: when later-year earnings turn negative the cash flows change sign
more than once and the IRR is not unique, so the annuity is judged on PV of profit instead.
""")
M("""
## 6. Stochastic analysis

### Interest scenarios
A Vasicek short-rate model drives the 10-year new-money yield; the portfolio earned rate rolls toward new money as
10% of assets reinvest each year.
""")
C("""
earned, new_money = an.interest_scenarios(0.0525, len(dia.projection), 500, seed=2026)
charts.interest_fan(earned, new_money, FIG / "fan.png")
display(Image(FIG / "fan.png"))
""")
C("""
pv = an.dia_stochastic_profit(dia, earned)
pd.Series(st.summarise(pv / 1e5)).to_frame("% of premium")
""")
M("""
### Stochastic reserve
The **greatest present value of accumulated deficiencies** (GPVAD) is the starting asset amount that keeps the
asset balance non-negative throughout a scenario. Its CTE70 across scenarios is the stochastic reserve, the mechanics
behind VM-20 and VM-22. For the annuity it sits **above** the formula CARVM reserve, because reinvestment risk is
real. For term it sits well **below** CRVM: the formula basis is conservative, which is why principles-based
reserving released term reserves.
""")
C("""
cmp_dia = an.dia_reserve_comparison(dia, earned, years=range(0, 41, 5))
cmp_dia.round(0)
""")
M("""
### Mortality risk for a block of term policies
Process risk (random timing of deaths) diversifies as the block grows, roughly as $1/\\sqrt{N}$. Parameter risk
(the true mortality level differs from the assumption) does not, so it dominates for large blocks. That is why
experience studies and credibility matter more than block size.
""")
C("""
risk = an.term_mortality_risk(term, block_sizes=(1_000, 10_000, 100_000), n_sims=2_000)
risk[["policies", "risk", "sd_pct_premium", "var995_pct_premium"]]
""")
C("""
charts.mortality_risk(risk, FIG / "mort.png")
display(Image(FIG / "mort.png"))
""")
M("""
---
The full run (`python run_model.py`) writes every table here, plus rate cards and sensitivity grids, to `outputs/`
as CSVs, a formatted Excel workbook and charts. See `docs/methodology.md` for the formulas.
""")

nb = nbf.v4.new_notebook(
    cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()
nbf.write(nb, "notebooks/walkthrough.ipynb")
print("ok")
