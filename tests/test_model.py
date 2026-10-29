import numpy as np
import pytest

from lamodel import actuarial as act
from lamodel import profit, reserves
from lamodel.deferred_annuity import DeferredIncomeAnnuity, DIAAssumptions
from lamodel.tables import cso_2017, iar_2012, read_xtbml, susm
from lamodel.term_life import TermAssumptions, TermLife
from lamodel.validation import ALTAM_SUSM_5PCT

# --- tables -----------------------------------------------------------------


def test_cso_2017_select_and_ultimate():
    cso = cso_2017("M")
    assert cso.select_period == 25
    q = cso.q(35)
    assert q[0] == pytest.approx(0.00018)  # select, issue 35, duration 1
    assert q[25] == cso.ultimate[60]  # ultimate after the 25-year select period
    assert q[-1] == 1.0 and len(q) == 120 - 35 + 1


def test_2012_iar_is_generational_iam_times_g2():
    iam = read_xtbml(2581).tables[0]["values"]
    g2 = read_xtbml(2583).tables[0]["values"]
    q = iar_2012("M", issue_year=2026).q(65)
    assert iam[65] == pytest.approx(0.009007)
    assert q[0] == pytest.approx(iam[65] * (1 - g2[65]) ** 14)
    assert q[1] == pytest.approx(iam[66] * (1 - g2[66]) ** 15)
    static = iar_2012("M", 2026, improvement_multiplier=0.0).q(65)
    assert static[0] == pytest.approx(iam[65])


# --- engine vs ALTAM ----------------------------------------------------------


@pytest.mark.parametrize("age", sorted(ALTAM_SUSM_5PCT))
def test_susm_matches_altam(age):
    l_ref, a_ref, A_ref = ALTAM_SUSM_5PCT[age]
    lx = act.life_table(susm().q(20))
    q = susm().q(age)
    assert lx[age - 20] == pytest.approx(l_ref, abs=0.05)
    assert act.annuity_due(q, 0.05) == pytest.approx(a_ref, abs=5e-5)
    assert act.whole_life_insurance(q, 0.05) == pytest.approx(A_ref, abs=5e-6)


def test_insurance_annuity_identity():
    q, i = susm().q(40), 0.04
    d = i / (1 + i)
    assert act.endowment_insurance(q, i, 25) == pytest.approx(1 - d * act.annuity_due(q, i, 25))


# --- term reserves --------------------------------------------------------------


def test_term_reserve_recursions():
    q, i, n = cso_2017("M").q(35), 0.035, 20
    rf = reserves.term_reserves(q, i, n)
    for t in range(n):
        p = rf.nlp_premium
        assert (rf.nlp[t] + p) * (1 + i) == pytest.approx(q[t] + (1 - q[t]) * rf.nlp[t + 1], abs=1e-12)
        prem = rf.alpha if t == 0 else rf.beta
        assert (rf.crvm[t] + prem) * (1 + i) == pytest.approx(q[t] + (1 - q[t]) * rf.crvm[t + 1], abs=1e-12)
    assert rf.method == "FPT"
    assert rf.alpha == pytest.approx(q[0] / (1 + i))
    assert rf.crvm[1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(rf.crvm <= rf.nlp + 1e-15)


def test_deficiency_reserve_when_premium_below_net_premium():
    q, i, n = cso_2017("M").q(35), 0.035, 20
    rf = reserves.term_reserves(q, i, n, gross_premium_per_unit=0.5 / 1000)
    assert rf.deficiency[5] == pytest.approx((rf.beta - 0.5 / 1000) * act.annuity_due(q[5:], i, 15))


# --- DIA reserves ---------------------------------------------------------------


def test_dia_reserve_recursion():
    q, i = iar_2012("M", 2026).q(55), 0.0475
    inc, db, m = 10_000.0, 100_000.0, 10
    V = reserves.deferred_income_annuity_reserves(q, i, m, inc, db)
    for t in range(40):
        pay = inc if t >= m else 0.0
        death = db if t < m else 0.0
        assert (V[t] - pay) * (1 + i) == pytest.approx(q[t] * death + (1 - q[t]) * V[t + 1], rel=1e-10)


# --- pricing ------------------------------------------------------------------


def test_term_price_hits_hurdle():
    r = TermLife(TermAssumptions()).price()
    assert r.metrics["irr"] == pytest.approx(0.10, abs=1e-4)
    assert 0.5 < r.metrics["premium_per_1000"] < 3.0


def test_term_premium_rises_with_mortality():
    base = TermAssumptions()
    lo = TermLife(base.with_(mortality_pct=0.6)).price().gross_premium
    hi = TermLife(base.with_(mortality_pct=0.8)).price().gross_premium
    assert hi > lo


def test_dia_price_hits_hurdle_and_responds_to_longevity():
    base = DIAAssumptions()
    r = DeferredIncomeAnnuity(base).price()
    assert r.metrics["pv_de_at_hurdle"] == pytest.approx(0.0, abs=1.0)
    more_improvement = DeferredIncomeAnnuity(base.with_(improvement_multiplier=1.5)).price()
    assert more_improvement.metrics["payout_rate"] < r.metrics["payout_rate"]


def test_irr_is_nan_when_not_unique():
    assert np.isnan(profit.irr([-100, 230, -132]))
    assert profit.irr([-100, 110]) == pytest.approx(0.10)
