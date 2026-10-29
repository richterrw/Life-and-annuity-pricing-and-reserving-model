"""Validate the actuarial engine against published ALTAM values.

Reference values are from *Actuarial Mathematics for Life Contingent Risks*
(Dickson, Hardy & Waters), Standard Ultimate Survival Model tables at i = 5%
(l_20 = 100,000).
"""

from __future__ import annotations

import pandas as pd

from . import actuarial as act
from .tables import susm

ALTAM_SUSM_5PCT = {
    # age: (l_x, a-due_x, A_x)
    20: (100_000.0, 19.9664, 0.04922),
    50: (98_576.4, 17.0245, 0.18931),
    60: (96_634.1, 14.9041, 0.29028),
    65: (94_579.7, 13.5498, 0.35477),
}


def susm_check(i: float = 0.05) -> pd.DataFrame:
    table = susm()
    lx = act.life_table(table.q(20))
    rows = []
    for age, (l_ref, a_ref, A_ref) in ALTAM_SUSM_5PCT.items():
        q = table.q(age)
        rows.append(
            {
                "age": age,
                "l_x model": lx[age - 20],
                "l_x ALTAM": l_ref,
                "a-due_x model": act.annuity_due(q, i),
                "a-due_x ALTAM": a_ref,
                "A_x model": act.whole_life_insurance(q, i),
                "A_x ALTAM": A_ref,
            }
        )
    return pd.DataFrame(rows)
