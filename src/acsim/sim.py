"""Simulation engine: a civilisation allocates a research budget over time.

Each step the civilisation receives a budget that grows with the research
multipliers of the technologies it has completed (recursive capability).
A strategy splits the budget across currently *available* technologies.
Spending makes progress and also yields a noisy observation of the
technology's true cost, which updates a Gaussian belief over log-cost.
Unused money from a completed technology is returned to the pool and the
strategy is asked again, so no strategy is penalised for overshooting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr, ndtri

from .world import MILESTONES, World


@dataclass(frozen=True)
class SimParams:
    horizon: int = 200
    base_budget: float = 10.0
    obs_noise: float = 1.0      # log-space noise of one "unit" of information
    info_unit: float = 25.0     # spend that buys one unit of information
    rounds_per_step: int = 8
    min_prior_sd: float = 0.05
    shift_at: int | None = None  # step at which hidden costs change (regime shift)
    shift_sigma: float = 0.0


class Observation:
    """Everything a (non-privileged) strategy is allowed to see."""

    def __init__(self, world: World, state: "_State", t: int, sp: SimParams):
        self.t = t
        self.horizon = sp.horizon
        self.base_budget = sp.base_budget
        self.n = world.n
        self.layer = world.layer
        self.dist = world.dist
        self.is_milestone = world.milestone >= 0
        self.prior_log_cost = world.prior_log_cost
        self.available = state.available.copy()
        self.done = state.done.copy()
        self.progress = state.progress.copy()
        self.mu = state.mu.copy()
        self.sd = 1.0 / np.sqrt(state.prec)
        self.est_value = state.est_value.copy()
        self.est_mult = state.est_mult.copy()
        self._W_cache: dict[float, np.ndarray] = state.w_cache

    def expected_remaining(self, loc: np.ndarray, sd: np.ndarray) -> np.ndarray:
        """Expected remaining cost under a log-normal belief, conditioned on cost > progress.

        Money already spent without finishing is evidence the cost is higher:
        E[C - p | C > p] for log C ~ N(loc, sd^2).
        """
        s = np.maximum(sd, 1e-6)
        a = np.minimum(_log_progress(self.progress) - loc, 1e300) / s
        log_total = loc + s**2 / 2 + log_ndtr(s - a) - log_ndtr(-a)
        return np.maximum(np.exp(log_total) - self.progress, 1e-6 * np.exp(loc))

    def sample_remaining(self, rng: np.random.Generator) -> np.ndarray:
        """Draw remaining cost from the posterior truncated to cost > progress (Thompson sampling)."""
        a = np.minimum((_log_progress(self.progress) - self.mu) / self.sd, 30.0)
        z = -ndtri(np.exp(log_ndtr(-a)) * rng.random(self.n))
        total = np.exp(self.mu + self.sd * z)
        return np.maximum(total - self.progress, 1e-6 * np.exp(self.mu))

    def direct_value(self) -> np.ndarray:
        """Estimated value of completing each technology now.

        A research multiplier m is worth m * base_budget extra budget for every
        remaining step, converted to value at the prior value-per-cost rate.
        """
        steps_left = self.horizon - self.t
        value_per_cost = np.median(self.est_value) / np.median(np.exp(self.prior_log_cost))
        return self.est_value + self.est_mult * self.base_budget * steps_left * value_per_cost

    def option_value(self, direct: np.ndarray, gamma: float = 0.5) -> np.ndarray:
        """Discounted value of not-yet-completed descendants each technology leads to."""
        W = self._W_cache.get(gamma)
        if W is None:
            W = np.where(np.isfinite(self.dist), gamma ** np.where(np.isfinite(self.dist), self.dist, 0), 0.0)
            np.fill_diagonal(W, 0.0)
            self._W_cache[gamma] = W
        return W @ (direct * ~self.done)


def _log_progress(progress: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(progress)  # -inf where nothing has been spent


class PrivilegedObservation(Observation):
    """Observation that also exposes ground truth; used only by reference oracles."""

    def __init__(self, world: World, state: "_State", t: int, sp: SimParams):
        super().__init__(world, state, t, sp)
        self.true_cost = state.true_cost.copy()
        self.true_value = world.true_value
        self.true_mult = world.true_mult
        self.is_dead = world.is_dead


@dataclass
class _State:
    true_cost: np.ndarray
    progress: np.ndarray
    done: np.ndarray
    available: np.ndarray
    mu: np.ndarray
    prec: np.ndarray
    est_value: np.ndarray
    est_mult: np.ndarray
    w_cache: dict = field(default_factory=dict)


@dataclass
class RunResult:
    strategy: str
    seed: int
    capability: np.ndarray          # true capability (sum of completed values) per step
    done_time: np.ndarray           # completion step per technology, inf if never
    milestone_time: dict[str, float]
    total_budget: float
    idle_budget: float              # budget that could not be spent (nothing available)
    spend_dead_ends: float
    spend_unfinished: float         # sunk into technologies not finished by the horizon

    @property
    def auc(self) -> float:
        return float(self.capability.mean())

    @property
    def final(self) -> float:
        return float(self.capability[-1])

    @property
    def n_done(self) -> int:
        return int(np.isfinite(self.done_time).sum())


def _available(world: World, done: np.ndarray) -> np.ndarray:
    av = np.zeros(world.n, dtype=bool)
    for i in range(world.n):
        if not done[i]:
            av[i] = any(all(done[q] for q in r) for r in world.routes[i])
    return av


def run(world: World, strategy, sp: SimParams = SimParams(), seed: int = 0) -> RunResult:
    obs_rng = np.random.default_rng([seed, 1])
    strategy.reset(np.random.default_rng([seed, 2]))
    n = world.n
    sd0 = max(world.params.sigma_est, sp.min_prior_sd)
    st = _State(
        true_cost=world.true_cost.copy(), progress=np.zeros(n), done=np.zeros(n, dtype=bool),
        available=np.zeros(n, dtype=bool), mu=world.prior_log_cost.copy(),
        prec=np.full(n, 1.0 / sd0**2), est_value=world.est_value.copy(), est_mult=world.est_mult.copy(),
    )
    done_time = np.full(n, np.inf)
    capability = np.zeros(sp.horizon)
    total_budget = idle = 0.0
    obs_cls = PrivilegedObservation if getattr(strategy, "privileged", False) else Observation

    for t in range(sp.horizon):
        if sp.shift_at is not None and t == sp.shift_at and sp.shift_sigma > 0:
            open_ = ~st.done
            st.true_cost[open_] *= np.exp(sp.shift_sigma * obs_rng.standard_normal(open_.sum()))
            st.true_cost = np.maximum(st.true_cost, st.progress + 1.0)

        pool = sp.base_budget * (1.0 + world.true_mult[st.done].sum())
        total_budget += pool
        for _ in range(sp.rounds_per_step):
            st.available = _available(world, st.done)
            if pool <= 1e-9 or not st.available.any():
                break
            alloc = np.asarray(strategy.allocate(obs_cls(world, st, t, sp), pool), dtype=float)
            alloc = _validate(alloc, st.available, pool, strategy)
            if alloc.sum() <= 1e-12:
                break
            for i in np.flatnonzero(alloc > 0):
                used = min(alloc[i], st.true_cost[i] - st.progress[i])
                st.progress[i] += used
                pool -= used
                info = used / (sp.info_unit * sp.obs_noise**2)
                y = np.log(st.true_cost[i]) + obs_rng.standard_normal() / np.sqrt(info)
                st.mu[i] = (st.prec[i] * st.mu[i] + info * y) / (st.prec[i] + info)
                st.prec[i] += info
                if st.progress[i] >= st.true_cost[i] - 1e-9:
                    st.done[i] = True
                    done_time[i] = t
                    st.mu[i] = np.log(st.true_cost[i])
                    st.est_value[i] = world.true_value[i]
                    st.est_mult[i] = world.true_mult[i]
        idle += pool
        capability[t] = world.true_value[st.done].sum()

    unfinished = ~st.done
    return RunResult(
        strategy=strategy.name, seed=world.seed, capability=capability, done_time=done_time,
        milestone_time={name: float(done_time[world.milestone_index(name)]) for name, _ in MILESTONES},
        total_budget=total_budget, idle_budget=idle,
        spend_dead_ends=float(st.progress[world.is_dead].sum()),
        spend_unfinished=float(st.progress[unfinished].sum()),
    )


def _validate(alloc: np.ndarray, available: np.ndarray, pool: float, strategy) -> np.ndarray:
    if alloc.shape != available.shape or not np.all(np.isfinite(alloc)):
        raise ValueError(f"{strategy.name}: malformed allocation")
    if (alloc < -1e-9).any():
        raise ValueError(f"{strategy.name}: negative allocation")
    alloc = np.clip(alloc, 0.0, None)
    if (alloc[~available] > 1e-9).any():
        raise ValueError(f"{strategy.name}: funded an unavailable technology")
    if alloc.sum() > pool * (1 + 1e-9) + 1e-9:
        raise ValueError(f"{strategy.name}: allocated more than the budget")
    return alloc
