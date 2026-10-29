"""Stochastic interest and mortality.

Interest
--------
``VasicekModel`` generates real-world short-rate paths with exact annual
discretisation and prices the 10-year zero-coupon yield analytically on each
path.  New money is invested at that yield plus a credit spread, and the
portfolio earned rate rolls toward new money as a fixed share of assets
matures each year.  This captures the reinvestment risk that dominates spread
products.

Stochastic reserve
------------------
For each scenario the *greatest present value of accumulated deficiencies*
(GPVAD) is the smallest starting asset amount that keeps the projected asset
balance non-negative at every future date, when net liability cash flows are
paid out of it and the balance earns the scenario's portfolio rate.  The
stochastic reserve is the CTE70 of GPVAD across scenarios.  These are the
mechanics of the VM-20 / VM-22 stochastic reserve, simplified: there is no
asset model beyond the portfolio-yield roll, no tax, and the margins are
illustrative.  It is a teaching implementation, not a regulatory one.

Mortality
---------
``simulate_term_block`` draws binomial deaths and lapses for a block of
policies (process risk), optionally scaled by a lognormal level multiplier
drawn once per simulation (parameter risk: the true mortality level differs
from the pricing assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Interest scenarios
# ---------------------------------------------------------------------------


@dataclass
class VasicekModel:
    """dr = a (b - r) dt + sigma dW, with continuously compounded rates."""

    r0: float = 0.040
    a: float = 0.12  # speed of mean reversion
    b: float = 0.040  # long-run mean short rate
    sigma: float = 0.011  # annual volatility of the short rate
    maturity: float = 10.0  # maturity of the new-money benchmark

    def short_rates(self, n_years: int, n_scen: int, seed: int | None = 2026) -> np.ndarray:
        """Short-rate paths, shape (n_scen, n_years + 1), starting at r0."""
        rng = np.random.default_rng(seed)
        e = np.exp(-self.a)
        sd = self.sigma * np.sqrt((1 - e**2) / (2 * self.a))
        r = np.empty((n_scen, n_years + 1))
        r[:, 0] = self.r0
        z = rng.standard_normal((n_scen, n_years))
        for t in range(n_years):
            r[:, t + 1] = self.b + (r[:, t] - self.b) * e + sd * z[:, t]
        return r

    def zero_yield(self, r: np.ndarray, tau: float | None = None) -> np.ndarray:
        """Annual-effective yield of a tau-year zero-coupon bond given short rate r."""
        tau = self.maturity if tau is None else tau
        B = (1 - np.exp(-self.a * tau)) / self.a
        lnA = (self.b - self.sigma**2 / (2 * self.a**2)) * (B - tau) - self.sigma**2 * B**2 / (4 * self.a)
        cont = -(lnA - B * r) / tau
        return np.expm1(cont)


@dataclass
class PortfolioYield:
    """Earned rate on a portfolio that reinvests a share of assets each year at new money."""

    initial_yield: float  # book yield of the assets backing the business at issue
    turnover: float = 0.10  # share of the portfolio reinvested each year
    spread: float | None = None  # new money = benchmark yield + spread; None => calibrate

    def earned_rates(self, model: VasicekModel, n_years: int, n_scen: int, seed: int | None = 2026):
        """Earned-rate paths (n_scen, n_years) and the new-money paths behind them."""
        r = model.short_rates(n_years, n_scen, seed)
        benchmark = model.zero_yield(r)
        spread = self.spread
        if spread is None:  # long-run new money equals today's book yield
            spread = self.initial_yield - model.zero_yield(np.array(model.b))
        new_money = benchmark + spread
        earned = np.empty((n_scen, n_years))
        y = np.full(n_scen, self.initial_yield)
        for t in range(n_years):
            earned[:, t] = y
            y = (1 - self.turnover) * y + self.turnover * new_money[:, t + 1]
        return earned, new_money


# ---------------------------------------------------------------------------
# Risk measures
# ---------------------------------------------------------------------------


def cte(x: np.ndarray, level: float) -> float:
    """Conditional tail expectation: mean of the worst (1 - level) share of x (high = bad)."""
    x = np.sort(np.asarray(x))[::-1]
    k = max(int(round(len(x) * (1 - level))), 1)
    return float(x[:k].mean())


def var(x: np.ndarray, level: float) -> float:
    return float(np.quantile(np.asarray(x), level))


def gpvad(outflow_bop: np.ndarray, outflow_eop: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Greatest PV of accumulated deficiencies per scenario.

    ``outflow_bop`` / ``outflow_eop`` are net liability outflows (benefits +
    expenses - premiums) at the start / end of each year, shape (n,).
    ``rates`` are scenario earned rates, shape (S, n).  Returns shape (S,),
    floored at zero.
    """
    disc_eop = np.cumprod(1.0 / (1.0 + rates), axis=1)
    disc_bop = np.concatenate([np.ones((rates.shape[0], 1)), disc_eop[:, :-1]], axis=1)
    pv_bop = outflow_bop * disc_bop
    pv_eop = outflow_eop * disc_eop
    # accumulated PV after each start-of-year payment and after each end-of-year payment
    after_bop = np.cumsum(pv_bop + pv_eop, axis=1) - pv_eop
    after_eop = np.cumsum(pv_bop + pv_eop, axis=1)
    worst = np.maximum(after_bop.max(axis=1), after_eop.max(axis=1))
    return np.maximum(worst, 0.0)


# ---------------------------------------------------------------------------
# Stochastic mortality: term block
# ---------------------------------------------------------------------------


def simulate_term_block(
    model, gross_premium: float, n_policies: int, n_sims: int, param_sd: float = 0.0, seed: int | None = 2026
) -> dict:
    """Simulate a block of ``n_policies`` identical term policies.

    Returns the PV at the hurdle rate of block distributable earnings for each
    simulation, plus the deterministic (expected) value for comparison.
    """
    rng = np.random.default_rng(seed)
    a = model.a
    q = model.pricing_q()
    w = model.lapse_rates()
    n = a.term
    level = np.exp(param_sd * rng.standard_normal(n_sims) - 0.5 * param_sd**2) if param_sd else np.ones(n_sims)

    inforce = np.empty((n_sims, n))
    deaths = np.empty((n_sims, n))
    lapses = np.empty((n_sims, n))
    alive = np.full(n_sims, n_policies, dtype=np.int64)
    for t in range(n):
        inforce[:, t] = alive
        d = rng.binomial(alive, np.clip(q[t] * level, 0, 1))
        lap = rng.binomial(alive - d, w[t])
        deaths[:, t], lapses[:, t] = d, lap
        alive = alive - d - lap

    f = model.financials(gross_premium, inforce, deaths, lapses)
    disc = (1 + a.hurdle_rate) ** -np.arange(n + 1)
    pv = f["de_full"] @ disc
    expected = model.financials(gross_premium, *[x * n_policies for x in model.decrements(q, w)])
    return {
        "pv_de": pv,
        "expected_pv_de": float(expected["de_full"] @ disc),
        "pv_premium": float(expected["premium"] @ disc[:-1]),
        "level": level,
    }


def summarise(pv: np.ndarray) -> dict:
    """Distribution summary of PV of profit (loss = -pv)."""
    loss = -np.asarray(pv)
    return {
        "mean": float(np.mean(pv)),
        "std": float(np.std(pv)),
        "p05": float(np.quantile(pv, 0.05)),
        "p50": float(np.quantile(pv, 0.50)),
        "p95": float(np.quantile(pv, 0.95)),
        "prob_loss": float(np.mean(pv < 0)),
        "var_995_loss": var(loss, 0.995),
        "cte70_loss": cte(loss, 0.70),
        "cte98_loss": cte(loss, 0.98),
    }
