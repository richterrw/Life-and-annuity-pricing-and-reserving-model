import numpy as np
import pytest

from lamodel import analysis as an
from lamodel import stochastic as st
from lamodel.deferred_annuity import DeferredIncomeAnnuity, DIAAssumptions
from lamodel.reserves import deferred_income_annuity_reserves
from lamodel.term_life import TermAssumptions, TermLife


@pytest.fixture(scope="module")
def term():
    return TermLife(TermAssumptions()).price()


@pytest.fixture(scope="module")
def dia():
    return DeferredIncomeAnnuity(DIAAssumptions()).price()


# --- scenario generator -------------------------------------------------------


def test_vasicek_moments():
    m = st.VasicekModel()
    r = m.short_rates(30, 20_000, seed=1)
    e = np.exp(-m.a * 30)
    mean = m.b + (m.r0 - m.b) * e
    sd = m.sigma * np.sqrt((1 - e**2) / (2 * m.a))
    assert r[:, -1].mean() == pytest.approx(mean, abs=4 * sd / np.sqrt(20_000))
    assert r[:, -1].std() == pytest.approx(sd, rel=0.03)


def test_zero_yield_with_no_volatility_is_flat():
    m = st.VasicekModel(r0=0.04, b=0.04, sigma=1e-12)
    assert m.zero_yield(np.array(0.04)) == pytest.approx(np.expm1(0.04), abs=1e-9)


def test_portfolio_yield_starts_at_book_yield_and_stays_central():
    earned, _ = st.PortfolioYield(0.05).earned_rates(st.VasicekModel(), 40, 4000, seed=3)
    assert np.all(earned[:, 0] == 0.05)
    assert earned[:, -1].mean() == pytest.approx(0.05, abs=0.002)


# --- risk measures and GPVAD ----------------------------------------------------


def test_cte_and_var():
    x = np.arange(1, 101, dtype=float)
    assert st.cte(x, 0.90) == pytest.approx(np.mean(np.arange(91, 101)))
    assert st.var(x, 0.5) == pytest.approx(50.5)


def test_gpvad_equals_pv_when_no_inflows():
    # pure outflows: the greatest PV of accumulated deficiency is the total PV
    bop, eop = np.array([1.0, 1.0, 1.0]), np.array([2.0, 0.0, 3.0])
    rates = np.full((1, 3), 0.05)
    v = 1.05 ** -np.arange(4)
    assert st.gpvad(bop, eop, rates)[0] == pytest.approx(bop @ v[:3] + eop @ v[1:])


def test_gpvad_reproduces_carvm_on_the_valuation_basis(dia):
    # with valuation mortality and a flat path at the valuation rate, GPVAD of the
    # DIA's benefit outflows is exactly the CARVM PV-of-benefits reserve
    a = dia.assumptions
    model = DeferredIncomeAnnuity(a.with_(maint_per_policy=0.0))
    model.pricing_table = model.valuation_table
    bop, eop = model.liability_cash_flows(dia.annual_income)
    g = st.gpvad(bop, eop, np.full((1, len(bop)), a.valuation_rate))[0]
    q = model.valuation_table.q(a.issue_age)
    carvm = deferred_income_annuity_reserves(q, a.valuation_rate, a.deferral, dia.annual_income, a.premium)[0]
    assert g == pytest.approx(carvm, rel=1e-10)


# --- product-level stochastic results ------------------------------------------------


def test_flat_scenarios_reproduce_deterministic_profit(dia):
    flat = np.full((3, len(dia.projection)), dia.assumptions.earned_rate)
    pv = an.dia_stochastic_profit(dia, flat)
    assert pv == pytest.approx(np.full(3, dia.metrics["pv_de_at_hurdle"]), abs=1e-6)


def test_vectorised_term_financials_match_single_projection(term):
    model = TermLife(term.assumptions)
    dec = model.decrements(model.pricing_q(), model.lapse_rates())
    stacked = model.financials(term.gross_premium, *[np.vstack([d, d]) for d in dec])
    assert stacked["de"][1] == pytest.approx(term.projection.distributable_earnings.values)


def test_mortality_simulation_mean_and_diversification(term):
    model = TermLife(term.assumptions)
    small = st.simulate_term_block(model, term.gross_premium, 1_000, 4_000, seed=5)
    large = st.simulate_term_block(model, term.gross_premium, 100_000, 4_000, seed=5)
    se = small["pv_de"].std() / np.sqrt(4_000)
    assert small["pv_de"].mean() == pytest.approx(small["expected_pv_de"], abs=4 * se)
    cv = lambda s: s["pv_de"].std() / s["pv_premium"]
    assert cv(large) == pytest.approx(cv(small) / 10, rel=0.15)  # process risk ~ 1/sqrt(N)


def test_premium_buildups_add_up(term, dia):
    tb, db = an.term_premium_buildup(term), an.dia_premium_buildup(dia)
    assert tb.pv.iloc[:-1].sum() == pytest.approx(tb.pv.iloc[-1], rel=1e-10)
    assert db.pv.iloc[:-1].sum() == pytest.approx(db.pv.iloc[-1], rel=1e-10)
    assert abs(tb.set_index("component").loc["Profit (PV of DE at hurdle)", "pv"]) < 1.0


def test_rate_cards_are_monotone():
    card = an.term_rate_card(TermAssumptions(), ages=(30, 45, 60), sexes=("M",))
    assert card["M per $1,000"].is_monotonic_increasing
    dcard = an.dia_rate_card(DIAAssumptions(), ages=(50, 60), deferrals=(5, 10))
    assert (dcard["defer 10"] > dcard["defer 5"]).all()
