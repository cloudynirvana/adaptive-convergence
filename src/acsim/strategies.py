"""Research-allocation strategies.

All non-oracle strategies see the same :class:`~acsim.sim.Observation` and
share the same value estimates, so differences between them come only from
*how* they turn beliefs into allocations.
"""
from __future__ import annotations

import numpy as np


class Strategy:
    name = "base"
    privileged = False

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def allocate(self, obs, amount: float) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _all_to(i: int, n: int, amount: float) -> np.ndarray:
    a = np.zeros(n)
    a[i] = amount
    return a


def _argmax_available(score: np.ndarray, available: np.ndarray) -> int:
    return int(np.argmax(np.where(available, score, -np.inf)))


class GreedyMyopic(Strategy):
    """Pure convergence: everything into the best immediate value-per-cost."""

    name = "Greedy (myopic)"

    def allocate(self, obs, amount):
        score = obs.direct_value() / obs.expected_remaining(obs.mu, obs.sd)
        return _all_to(_argmax_available(score, obs.available), obs.n, amount)


class GreedyOption(Strategy):
    """Convergence on a structure-aware objective: value includes what a technology unlocks."""

    name = "Greedy + option value"

    def allocate(self, obs, amount):
        D = obs.direct_value()
        score = (D + obs.option_value(D)) / obs.expected_remaining(obs.mu, obs.sd)
        return _all_to(_argmax_available(score, obs.available), obs.n, amount)


class UniformDivergent(Strategy):
    """Pure divergence: spread the budget evenly over everything available."""

    name = "Uniform (divergent)"

    def allocate(self, obs, amount):
        return obs.available * (amount / obs.available.sum())


class FixedSplit(Strategy):
    """A fixed research/deployment split: (1-f) to the best option, f spread over the rest."""

    def __init__(self, explore: float = 0.2):
        self.explore = explore
        self.name = f"Fixed split ({round(100 * (1 - explore))}/{round(100 * explore)})"

    def allocate(self, obs, amount):
        D = obs.direct_value()
        score = (D + obs.option_value(D)) / obs.expected_remaining(obs.mu, obs.sd)
        best = _argmax_available(score, obs.available)
        others = obs.available.copy()
        others[best] = False
        if not others.any():
            return _all_to(best, obs.n, amount)
        a = others * (amount * self.explore / others.sum())
        a[best] = amount * (1 - self.explore)
        return a


class Thompson(Strategy):
    """Thompson sampling over cost beliefs (standard bandit baseline)."""

    name = "Thompson sampling"

    def allocate(self, obs, amount):
        D = obs.direct_value()
        score = (D + obs.option_value(D)) / obs.sample_remaining(self.rng)
        return _all_to(_argmax_available(score, obs.available), obs.n, amount)


class AdaptiveConvergence(Strategy):
    """Commit to the best option, probe uncertain alternatives, and stop confirmed failures.

    * option value: credit technologies for what they unlock;
    * optimism (beta): rank by an optimistic cost bound, mu - beta * sd;
    * probing (kappa): send a fraction ``min(max_explore, kappa * mean_sd)`` of the
      budget to the runners-up, weighted by score x uncertainty - so the
      strategy spreads out when beliefs are uncertain and converges as they sharpen;
    * kill rule: drop a technology once the posterior is confident its cost
      exceeds ``kill_factor`` times the prior estimate.
    """

    def __init__(self, beta=1.0, kappa=0.5, max_explore=0.5, k_max=5, kill_factor=3.0,
                 use_option=True, name="Adaptive convergence"):
        self.beta, self.kappa, self.max_explore = beta, kappa, max_explore
        self.k_max, self.kill_factor, self.use_option = k_max, kill_factor, use_option
        self.name = name

    def allocate(self, obs, amount):
        D = obs.direct_value()
        V = D + obs.option_value(D) if self.use_option else D
        score = V / obs.expected_remaining(obs.mu - self.beta * obs.sd, obs.sd)

        avail = obs.available
        if self.kill_factor:
            killed = avail & (obs.mu - 2 * obs.sd > obs.prior_log_cost + np.log(self.kill_factor))
            if (avail & ~killed).any():
                avail = avail & ~killed

        score = np.where(avail, score, -np.inf)
        cand = [i for i in np.argsort(-score)[: self.k_max] if np.isfinite(score[i])]
        top, others = cand[0], cand[1:]
        explore = min(self.max_explore, self.kappa * float(np.mean(obs.sd[cand])))
        if not others or explore <= 0:
            return _all_to(top, obs.n, amount)
        w = score[others] * obs.sd[others]
        a = np.zeros(obs.n)
        a[top] = amount * (1 - explore)
        a[others] = amount * explore * w / w.sum()
        return a


class OracleGreedy(Strategy):
    """Reference only: greedy + option value with perfect knowledge of the hidden truth."""

    name = "Oracle (perfect information)"
    privileged = True

    def allocate(self, obs, amount):
        steps_left = obs.horizon - obs.t
        vpc = np.median(obs.true_value) / np.median(obs.true_cost)
        D = obs.true_value + obs.true_mult * obs.base_budget * steps_left * vpc
        D = np.where(obs.is_dead, 0.0, D)  # dead ends unlock nothing
        rem = np.maximum(obs.true_cost - obs.progress, 1e-9)
        score = (D + obs.option_value(D)) / rem
        return _all_to(_argmax_available(score, obs.available), obs.n, amount)
