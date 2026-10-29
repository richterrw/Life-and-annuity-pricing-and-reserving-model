"""Mortality tables.

Three sources of mortality are supported:

* SOA XTbML files (the format published at mort.soa.org).  The repository ships
  the 2017 CSO smoker-distinct select & ultimate tables and the 2012 IAM basic
  tables with Projection Scale G2, in ``data/soa``.
* The 2012 IAR generational table (2012 IAM basic x Scale G2), the statutory
  valuation basis for individual annuities.
* The Standard Ultimate Survival Model (SUSM) from *Actuarial Mathematics for
  Life Contingent Risks* (ALTAM), used to validate the actuarial engine against
  published textbook values.

Every table is reduced to the same thing: a vector of one-year death
probabilities ``q`` for a life of a given issue age, starting at policy
duration 1.  The product models only ever consume that vector, so swapping the
basis (pricing vs. valuation vs. sensitivity) is a one-line change.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soa"

# SOA table ids shipped in data/soa
CSO2017_NS_MALE = 3291  # 2017 Loaded CSO Smoker Distinct Nonsmoker Male ANB, S&U
CSO2017_NS_FEMALE = 3292  # 2017 Loaded CSO Smoker Distinct Nonsmoker Female ANB, S&U
IAM2012_MALE = 2581  # 2012 IAM Basic Table - Male, ANB
IAM2012_FEMALE = 2582  # 2012 IAM Basic Table - Female, ANB
G2_MALE = 2583  # Projection Scale G2 - Male, ANB
G2_FEMALE = 2584  # Projection Scale G2 - Female, ANB


# ---------------------------------------------------------------------------
# XTbML parsing
# ---------------------------------------------------------------------------


@dataclass
class XTbMLTable:
    table_id: int
    name: str
    reference: str
    tables: list[dict]  # each: {"description", "values": {age: {dur: q}} or {age: q}}


def _text(node, path: str) -> str:
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else ""


@cache
def read_xtbml(table_id: int, data_dir: Path = DATA_DIR) -> XTbMLTable:
    """Parse an SOA XTbML file into nested dicts.

    One-dimensional tables become ``{age: rate}``; select tables become
    ``{issue_age: {duration: rate}}``.
    """
    path = Path(data_dir) / f"t{table_id}.xml"
    root = ET.parse(path).getroot()
    cc = root.find("ContentClassification")
    parsed = []
    for tbl in root.findall("Table"):
        values_node = tbl.find("Values")
        outer = values_node.findall("Axis")
        if len(outer) == 1 and "t" not in outer[0].attrib:
            values = {int(y.attrib["t"]): float(y.text) for y in outer[0].findall("Y")}
        else:
            values = {
                int(ax.attrib["t"]): {int(y.attrib["t"]): float(y.text) for y in ax.find("Axis").findall("Y")}
                for ax in outer
            }
        parsed.append({"description": _text(tbl, "MetaData/TableDescription"), "values": values})
    return XTbMLTable(
        table_id=table_id,
        name=_text(cc, "TableName"),
        reference=_text(cc, "TableReference"),
        tables=parsed,
    )


# ---------------------------------------------------------------------------
# Mortality bases
# ---------------------------------------------------------------------------


class MortalityBasis:
    """Interface: one-year death probabilities for a life issued at ``issue_age``."""

    name: str = "mortality"
    max_age: int = 120

    def q(self, issue_age: int, n_years: int | None = None) -> np.ndarray:
        raise NotImplementedError

    def scaled(self, factor: float) -> ScaledBasis:
        return ScaledBasis(self, factor)


class SelectUltimateTable(MortalityBasis):
    """Select & ultimate table such as the 2017 CSO (25-year select period)."""

    def __init__(self, table_id: int, use_select: bool = True):
        raw = read_xtbml(table_id)
        self.name = raw.name.strip() + ("" if use_select else " (ultimate only)")
        self.reference = raw.reference
        self.select = raw.tables[0]["values"]
        self.ultimate = raw.tables[1]["values"]
        self.select_period = max(next(iter(self.select.values())).keys())
        self.max_age = max(self.ultimate)
        self.use_select = use_select

    def q(self, issue_age: int, n_years: int | None = None) -> np.ndarray:
        n_max = self.max_age - issue_age + 1
        n = n_max if n_years is None else min(n_years, n_max)
        out = np.empty(n)
        sel = self.select.get(issue_age) if self.use_select else None
        for t in range(n):
            dur = t + 1
            if sel is not None and dur in sel:
                out[t] = sel[dur]
            else:
                out[t] = self.ultimate[issue_age + t]
        if n == n_max:
            out[-1] = 1.0
        return out


class GenerationalTable(MortalityBasis):
    """Base table projected generationally with an improvement scale.

    ``q(x, year) = q_base(x) * (1 - AA(x)) ** (year - base_year)``.  With the
    2012 IAM basic table and Scale G2 this is the 2012 IAR valuation table.
    ``improvement_multiplier`` scales the improvement rates (e.g. 0 = static
    table, 1.5 = 150% of G2) for longevity sensitivity testing.
    """

    def __init__(
        self,
        base_table_id: int,
        scale_table_id: int,
        issue_year: int,
        base_year: int = 2012,
        improvement_multiplier: float = 1.0,
        name: str | None = None,
    ):
        base = read_xtbml(base_table_id).tables[0]["values"]
        scale = read_xtbml(scale_table_id).tables[0]["values"]
        self.max_age = max(base)
        self.base_q = np.array([base[a] for a in range(self.max_age + 1)])
        last_scale = scale[max(scale)]
        self.scale = np.array([scale.get(a, last_scale) for a in range(self.max_age + 1)])
        self.issue_year = issue_year
        self.base_year = base_year
        self.improvement_multiplier = improvement_multiplier
        self.name = name or (
            f"{read_xtbml(base_table_id).name} x {improvement_multiplier:.0%} "
            f"{read_xtbml(scale_table_id).name} (generational, issue {issue_year})"
        )

    def q(self, issue_age: int, n_years: int | None = None) -> np.ndarray:
        n_max = self.max_age - issue_age + 1
        n = n_max if n_years is None else min(n_years, n_max)
        ages = issue_age + np.arange(n)
        years = self.issue_year + np.arange(n) - self.base_year
        aa = np.clip(self.scale[ages] * self.improvement_multiplier, None, 1.0)
        out = self.base_q[ages] * (1.0 - aa) ** years
        out = np.clip(out, 0.0, 1.0)
        if n == n_max:
            out[-1] = 1.0
        return out


class MakehamTable(MortalityBasis):
    """Makeham's law, mu_x = A + B c^x, integrated exactly over each year.

    The default parameters are ALTAM's Standard Ultimate Survival Model
    (A = 0.00022, B = 2.7e-6, c = 1.124).
    """

    def __init__(self, A=0.00022, B=2.7e-6, c=1.124, max_age=130, name=None):
        self.A, self.B, self.c = A, B, c
        self.max_age = max_age
        self.name = name or "ALTAM Standard Ultimate Survival Model (Makeham)"

    def p(self, x, t=1.0):
        """t-year survival probability from age x."""
        lnc = np.log(self.c)
        return np.exp(-self.A * t - self.B / lnc * self.c**x * (self.c**t - 1.0))

    def q(self, issue_age: int, n_years: int | None = None) -> np.ndarray:
        n_max = self.max_age - issue_age + 1
        n = n_max if n_years is None else min(n_years, n_max)
        ages = issue_age + np.arange(n)
        out = 1.0 - self.p(ages)
        if n == n_max:
            out[-1] = 1.0
        return out


class ScaledBasis(MortalityBasis):
    """A multiplicative adjustment, e.g. 80% of 2017 CSO for pricing."""

    def __init__(self, base: MortalityBasis, factor: float):
        self.base = base
        self.factor = factor
        self.max_age = base.max_age
        self.name = f"{factor:.0%} x {base.name}"

    def q(self, issue_age: int, n_years: int | None = None) -> np.ndarray:
        q = self.base.q(issue_age, n_years)
        out = np.clip(q * self.factor, 0.0, 1.0)
        if n_years is None or len(q) == self.max_age - issue_age + 1:
            out[-1] = q[-1]
        return out


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def cso_2017(sex: str = "M", use_select: bool = True) -> SelectUltimateTable:
    """2017 Loaded CSO smoker-distinct nonsmoker, ANB, select & ultimate."""
    return SelectUltimateTable(CSO2017_NS_MALE if sex.upper() == "M" else CSO2017_NS_FEMALE, use_select)


def iar_2012(sex: str = "M", issue_year: int = 2026, improvement_multiplier: float = 1.0) -> GenerationalTable:
    """2012 IAR: 2012 IAM basic projected generationally with Scale G2."""
    male = sex.upper() == "M"
    return GenerationalTable(
        IAM2012_MALE if male else IAM2012_FEMALE,
        G2_MALE if male else G2_FEMALE,
        issue_year=issue_year,
        improvement_multiplier=improvement_multiplier,
        name=(
            f"2012 IAR {'Male' if male else 'Female'} (2012 IAM basic x "
            f"{improvement_multiplier:.0%} Scale G2, issue {issue_year})"
        ),
    )


def susm() -> MakehamTable:
    return MakehamTable()
