"""Replication of March (1991), "Exploration and Exploitation in Organizational Learning".

Organization Science 2(1):71-87. Model as described in the paper:

* reality: m dimensions, each +1 or -1 with equal probability;
* n individuals hold beliefs in {+1, 0, -1} per dimension, uniformly at random;
* the organisational code starts at 0 on every dimension;
* each period, every individual adopts the code's (non-zero) belief on each
  dimension where they differ, with probability p1 (socialisation rate);
* the code learns from the *superior group* - individuals whose beliefs match
  reality on more dimensions than the code does. On each dimension it adopts
  the superior group's dominant belief with probability 1 - (1 - p2)^k, where
  k is the number of superior individuals holding that belief minus the number
  holding the code's current belief (code learning rate p2);
* both updates use the state at the start of the period;
* the system stops at equilibrium: when no further change is possible (every
  individual matches the code wherever the code holds a belief, and no
  dimension is eligible for a code change).

Where the paper is ambiguous (tie-breaking, simultaneity, the exact stopping
rule) the choices above are our interpretation; they are documented here so
results can be checked.
"""
from __future__ import annotations

import numpy as np


def simulate(p1, p2: float, m: int = 30, n: int = 50, seed: int = 0,
             max_periods: int = 20_000, turnover: float = 0.0, periods: int | None = None) -> dict:
    """Run one organisation.

    ``p1`` may be a scalar or a length-n array (heterogeneous socialisation rates).
    With ``turnover`` > 0 there is no absorbing equilibrium, so pass ``periods``
    to stop after a fixed number of periods.
    """
    rng = np.random.default_rng(seed)
    reality = rng.choice(np.array([-1, 1], dtype=np.int8), size=m)
    beliefs = rng.integers(-1, 2, size=(n, m)).astype(np.int8)
    code = np.zeros(m, dtype=np.int8)
    p1 = np.broadcast_to(np.asarray(p1, dtype=float), (n,))[:, None]
    limit = periods if periods is not None else max_periods

    reached_equilibrium = False
    t = 0
    while t < limit:
        old = beliefs

        # Which code dimensions could change, given the superior group?
        know_ind = (old == reality).sum(axis=1)
        superior = old[know_ind > int((code == reality).sum())]
        if len(superior):
            dominant = np.sign(superior.sum(axis=0, dtype=np.int64)).astype(np.int8)
            k = (superior == dominant).sum(axis=0) - (superior == code).sum(axis=0)
            code_eligible = (dominant != 0) & (dominant != code) & (k > 0)
        else:
            code_eligible = np.zeros(m, dtype=bool)
        adopt_eligible = (code != 0) & (old != code)

        if periods is None and not code_eligible.any() and not adopt_eligible.any():
            reached_equilibrium = True
            break
        t += 1

        new_code = code.copy()
        if code_eligible.any():
            prob = 1.0 - (1.0 - p2) ** np.maximum(k, 0)
            change = code_eligible & (rng.random(m) < prob)
            new_code[change] = dominant[change]

        adopt = adopt_eligible & (rng.random((n, m)) < p1)
        beliefs = np.where(adopt, code, old).astype(np.int8)
        code = new_code

        if turnover > 0:
            leave = rng.random(n) < turnover
            beliefs[leave] = rng.integers(-1, 2, size=(int(leave.sum()), m))

    return {
        "code_knowledge": float((code == reality).mean()),
        "mean_individual_knowledge": float((beliefs == reality).mean()),
        "periods": t,
        "equilibrium": reached_equilibrium,
    }
