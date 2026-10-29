"""Static charts for the report (matplotlib, PNG).

Styling: a fixed categorical order (blue, orange, aqua), a single-hue blue
ramp for fan charts, thin 2px lines, recessive grid and axes, and text in ink
colours rather than series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#dbe8f8", "#a9c8ee", "#6fa3e3"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
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

dollars = FuncFormatter(lambda x, _: f"${x:,.0f}")
dollars_k = FuncFormatter(lambda x, _: f"${x / 1e3:,.0f}k")
dollars_m = FuncFormatter(lambda x, _: f"${x / 1e6:,.1f}M")


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig(w=6.4, h=3.6):
    return plt.subplots(figsize=(w, h))


# --- term -------------------------------------------------------------------


def term_reserves(term, path):
    p = term.projection
    fig, ax = _fig()
    ax.plot(p.year, p.nlp_factor_per_1000, color=ORANGE, label="Net level premium")
    ax.plot(p.year, p.crvm_factor_per_1000, color=BLUE, label="CRVM (full preliminary term)")
    ax.set(
        xlabel="End of policy year",
        ylabel="Terminal reserve per $1,000 face",
        title="20-year term: formula reserves (2017 CSO NS male select, 3.5%)",
    )
    ax.set_xticks(range(0, 21, 2))
    ax.legend(loc="upper left")
    _save(fig, path)


def term_distributable_earnings(term, path):
    p = term.projection
    fig, ax = _fig()
    ax.bar(
        p.year,
        p.distributable_earnings,
        color=np.where(p.distributable_earnings < 0, ORANGE, BLUE),
        width=0.7,
        edgecolor="white",
        linewidth=1,
    )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.yaxis.set_major_formatter(dollars)
    ax.set_xticks(range(1, 21))
    ax.set(
        xlabel="Policy year",
        ylabel="Distributable earnings per policy issued",
        title=f"20-year term: distributable earnings (IRR {term.metrics['irr']:.1%})",
    )
    _save(fig, path)


def term_rate_card(card, path):
    fig, ax = _fig()
    ax.plot(card.issue_age, card["M per $1,000"], color=BLUE, marker="o", markersize=4, label="Male")
    ax.plot(card.issue_age, card["F per $1,000"], color=ORANGE, marker="o", markersize=4, label="Female")
    for col in ("M per $1,000", "F per $1,000"):
        last = card.iloc[-1]
        ax.annotate(
            f"${last[col]:.2f}",
            (last.issue_age, last[col]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set(
        xlabel="Issue age",
        ylabel="Annual premium per $1,000 face",
        title="20-year term rate card: premium at a 10% IRR, $500k face, nonsmoker",
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.legend(loc="upper left")
    _save(fig, path)


def premium_buildup(df, path, title):
    d = df.iloc[:-1].copy()
    d = d[d.pct_of_premium.abs() > 1e-4]
    fig, ax = _fig(6.4, 0.42 * len(d) + 1.2)
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.pct_of_premium * 100, color=BLUE, height=0.6, edgecolor="white", linewidth=1)
    for yy, v in zip(y, d.pct_of_premium):
        ax.text(v * 100 + 0.6, yy, f"{v:.1%}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, d.component)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlim(0, d.pct_of_premium.max() * 100 * 1.18)
    ax.set(xlabel="Share of PV premium", title=title)
    _save(fig, path)


def tornado(df, col, metric, title, path):
    base = df.loc[df.scenario == "Base", col].iloc[0]
    s = df[df.scenario != "Base"].assign(delta=lambda x: (x[col] - base) * 1e4)
    s = s.reindex(s.delta.abs().sort_values().index)
    fig, ax = _fig(6.4, 3.8)
    ax.barh(s.scenario, s.delta, color=np.where(s.delta < 0, ORANGE, BLUE), height=0.6, edgecolor="white", linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    lim = s.delta.abs().max() * 1.35
    for y, v in enumerate(s.delta):
        ax.text(
            v + (lim * 0.02 if v >= 0 else -lim * 0.02),
            y,
            f"{v:+.0f}",
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK,
            fontsize=8,
        )
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=f"Change in {metric} vs. base (bp), price held fixed", title=title)
    _save(fig, path)


def term_reserve_comparison(df, path):
    fig, ax = _fig()
    ax.plot(df.duration, df.crvm, color=BLUE, label="CRVM formula reserve")
    ax.plot(df.duration, df.stochastic_cte70, color=ORANGE, label="Stochastic reserve (CTE70 of GPVAD)")
    ax.plot(df.duration, df.deterministic, color=AQUA, linestyle="--", label="Deterministic (level 5% path)")
    ax.yaxis.set_major_formatter(dollars)
    ax.set_xticks(range(0, 20, 2))
    ax.set(
        xlabel="Valuation date (end of policy year)",
        ylabel="Reserve per policy in force ($500k face)",
        title="Term: formula vs. modelled reserve (prudent mortality +10%, lapses 75%)",
    )
    ax.legend(loc="upper left")
    _save(fig, path)


def mortality_risk(df, path):
    fig, ax = _fig()
    for (label, g), c in zip(df.groupby("risk", sort=False), [BLUE, ORANGE]):
        ax.plot(g.policies, g.var995_pct_premium * 100, color=c, marker="o", markersize=5, label=label)
        last = g.iloc[-1]
        ax.annotate(
            f"{last.var995_pct_premium:.1%}",
            (last.policies, last.var995_pct_premium * 100),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xticks(df.policies.unique(), [f"{n:,}" for n in df.policies.unique()])
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set(
        xlabel="Policies in block (log scale)",
        ylabel="99.5% VaR of PV loss, % of PV premium",
        title="Term mortality risk: process risk diversifies, parameter risk does not",
    )
    ax.legend(loc="upper right")
    _save(fig, path)


# --- annuity ------------------------------------------------------------------


def dia_reserves(dia, path):
    d = dia.projection.iloc[:55]
    fig, ax = _fig()
    ax.plot(d.attained_age + 1, d.reserve_eop, color=BLUE, label="CARVM reserve (2012 IAR, 4.75%)")
    ax.axvline(dia.assumptions.issue_age + dia.assumptions.deferral, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(65.8, 10_000, "income starts at 65", color=MUTED)
    ax.yaxis.set_major_formatter(dollars_k)
    ax.set_xlim(55, 110)
    ax.set(
        xlabel="Attained age (end of year)",
        ylabel="Reserve per contract issued",
        title="Deferred income annuity: statutory reserve run-off",
    )
    ax.legend(loc="upper right")
    _save(fig, path)


def dia_reserve_sensitivity(grid, path):
    fig, ax = _fig()
    for col, c in zip(grid.columns[1:], [ORANGE, BLUE, AQUA]):
        ax.plot(
            grid.valuation_rate * 100, grid[col] * 100, color=c, marker="o", markersize=4, label=f"2012 IAM x {col}"
        )
    ax.axhline(100, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(grid.valuation_rate.max() * 100, 101, "premium received", color=MUTED, ha="right", va="bottom")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set(
        xlabel="Valuation interest rate (%)",
        ylabel="CARVM reserve at issue, % of premium",
        title="DIA: issue reserve vs. valuation rate and mortality improvement",
    )
    ax.legend(loc="lower left")
    _save(fig, path)


def dia_rate_card(card, path):
    fig, ax = _fig()
    cols = [c for c in card.columns if c.startswith("defer")]
    colors = [BLUE, ORANGE, AQUA, "#eda100"]
    for col, c in zip(cols, colors):
        ax.plot(
            card.issue_age, card[col] * 100, color=c, marker="o", markersize=4, label=f"{col.split()[1]}-year deferral"
        )
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set(
        xlabel="Issue age",
        ylabel="Annual income, % of single premium",
        title="Deferred income annuity rate card: payout rate at a 10% IRR, male",
    )
    ax.legend(loc="upper left")
    _save(fig, path)


def interest_fan(earned, new_money, path):
    years = np.arange(earned.shape[1])[:40]
    fig, ax = _fig()
    for (lo, hi), c in zip([(5, 95), (25, 75)], BLUE_RAMP[:2]):
        ax.fill_between(
            years,
            np.percentile(new_money[:, years + 1], lo, axis=0) * 100,
            np.percentile(new_money[:, years + 1], hi, axis=0) * 100,
            color=c,
            linewidth=0,
            label=f"New-money rate, {lo}th-{hi}th pct",
        )
    ax.plot(years, np.median(earned[:, years], axis=0) * 100, color=BLUE, label="Portfolio earned rate, median")
    ax.plot(
        years,
        np.percentile(earned[:, years], 5, axis=0) * 100,
        color=ORANGE,
        linewidth=1.5,
        label="Portfolio earned rate, 5th pct",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set(
        xlabel="Projection year",
        ylabel="Annual rate",
        title=f"Interest scenarios: {earned.shape[0]:,} Vasicek paths, 10% annual reinvestment",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    _save(fig, path)


def dia_profit_distribution(pv, premium, path):
    x = pv / premium * 100
    loss = -x
    var995 = -np.quantile(loss, 0.995)
    cte70 = -np.mean(np.sort(loss)[::-1][: int(len(loss) * 0.30)])
    fig, ax = _fig()
    ax.hist(x, bins=50, color=BLUE, edgecolor="white", linewidth=0.8)
    for v, lbl in [(cte70, "CTE70"), (var995, "99.5% VaR")]:
        ax.axvline(v, color=ORANGE, linewidth=1.2, linestyle="--")
        ax.text(v, ax.get_ylim()[1] * 0.95, f" {lbl}\n {v:.1f}%", color=INK, fontsize=8, va="top")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set(
        xlabel="PV of distributable earnings at 10%, % of premium",
        ylabel="Scenarios",
        title="DIA: profit distribution across interest scenarios (payout rate held)",
    )
    _save(fig, path)


def dia_reserve_comparison(df, path):
    fig, ax = _fig()
    ax.plot(df.attained_age, df.carvm, color=BLUE, label="CARVM formula reserve")
    ax.plot(df.attained_age, df.stochastic_cte70, color=ORANGE, label="Stochastic reserve (CTE70 of GPVAD)")
    ax.plot(df.attained_age, df.deterministic, color=AQUA, linestyle="--", label="Deterministic (level 5.25% path)")
    ax.yaxis.set_major_formatter(dollars_k)
    ax.set(
        xlabel="Attained age at valuation",
        ylabel="Reserve per contract in force",
        title="DIA: formula vs. modelled reserve (prudent mortality -10%)",
    )
    ax.legend(loc="upper right")
    _save(fig, path)


def margin_vs_earned_rate(curve, path):
    fig, ax = _fig()
    ax.plot(
        curve.earned_rate * 100,
        curve.term_margin * 100,
        color=BLUE,
        marker="o",
        markersize=4,
        label="20-year term (% of PV premium)",
    )
    ax.plot(
        curve.earned_rate * 100,
        curve.dia_margin * 100,
        color=ORANGE,
        marker="o",
        markersize=4,
        label="Deferred income annuity (% of premium)",
    )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set(
        xlabel="Portfolio earned rate (%)",
        ylabel="PV of distributable earnings at 10%",
        title="Interest sensitivity: profit margin at the priced premium / payout",
    )
    ax.legend(loc="upper left")
    _save(fig, path)
