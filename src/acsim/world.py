"""Procedurally generated technology worlds.

A world is a layered directed acyclic graph of technologies. Each technology
can be reached by one or more alternative *routes* (sets of prerequisite
technologies), modelling competing architectures. The world holds hidden
ground truth (true costs, values, dead ends) and the civilisation's noisy
prior beliefs about it. Strategies never see the ground truth; they only see
an :class:`acsim.sim.Observation`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Milestones are placed at fixed fractional depths of the graph.
MILESTONES = (
    ("lunar_industry", 0.5),
    ("mars_settlement", 0.8),
    ("interstellar_probe", 1.0),
)


@dataclass(frozen=True)
class WorldParams:
    n_layers: int = 8
    width: int = 8
    p_two_routes: float = 0.6        # chance a technology has an alternative route
    p_second_prereq: float = 0.5     # chance a route needs two prerequisites
    base_cost: float = 30.0          # median cost of a layer-0 technology
    cost_depth_growth: float = 0.6   # median cost grows linearly with depth
    cost_sigma: float = 0.5          # log-normal spread of true costs
    p_dead_end: float = 0.12         # chance a technology is practically infeasible
    dead_end_cost_factor: float = 25.0
    base_value: float = 10.0
    depth_value_growth: float = 0.4
    value_sigma: float = 0.6
    p_enabler: float = 0.2           # chance a technology raises research budget
    enabler_mult_low: float = 0.05
    enabler_mult_high: float = 0.20
    milestone_value: float = 200.0
    sigma_est: float = 0.7           # log-space noise of the civilisation's priors
    p_underrated: float = 0.15       # chance a technology's value is systematically underrated
    underrated_factor: float = 0.25


@dataclass
class World:
    params: WorldParams
    seed: int
    layer: np.ndarray
    routes: list[list[tuple[int, ...]]]
    dist: np.ndarray                 # shortest prerequisite-path distance i -> j (inf if none)
    milestone: np.ndarray            # index into MILESTONES, or -1
    # hidden ground truth
    true_cost: np.ndarray
    true_value: np.ndarray
    true_mult: np.ndarray
    is_dead: np.ndarray
    is_underrated: np.ndarray
    # public priors
    prior_log_cost: np.ndarray
    est_value: np.ndarray
    est_mult: np.ndarray

    @property
    def n(self) -> int:
        return len(self.layer)

    def milestone_index(self, name: str) -> int:
        k = [m[0] for m in MILESTONES].index(name)
        return int(np.flatnonzero(self.milestone == k)[0])


def generate_world(seed: int, params: WorldParams = WorldParams()) -> World:
    """Generate a world whose milestones are all truly reachable."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        world = _try_generate(rng, params, seed)
        if world is not None:
            return world
    raise RuntimeError(f"could not generate a reachable world for seed {seed}")


def _try_generate(rng: np.random.Generator, p: WorldParams, seed: int) -> World | None:
    L, W = p.n_layers, p.width
    n = L * W
    layer = np.repeat(np.arange(L), W)

    routes: list[list[tuple[int, ...]]] = []
    for i in range(n):
        l = layer[i]
        if l == 0:
            routes.append([()])  # the empty route: always available
            continue
        k = 2 if rng.random() < p.p_two_routes else 1
        found: set[tuple[int, ...]] = set()
        for _ in range(50):
            if len(found) == k:
                break
            prereqs = {int(rng.integers((l - 1) * W, l * W))}
            if rng.random() < p.p_second_prereq:
                prereqs.add(int(rng.integers(max(0, l - 2) * W, l * W)))
            found.add(tuple(sorted(prereqs)))
        routes.append(sorted(found))

    median_cost = p.base_cost * (1.0 + p.cost_depth_growth * layer)
    log_cost_if_feasible = np.log(median_cost) + p.cost_sigma * rng.standard_normal(n)
    is_dead = rng.random(n) < p.p_dead_end
    true_cost = np.exp(log_cost_if_feasible)
    true_cost[is_dead] *= p.dead_end_cost_factor

    true_value = (p.base_value * (1.0 + p.depth_value_growth * layer)
                  * np.exp(p.value_sigma * rng.standard_normal(n) - p.value_sigma**2 / 2))
    is_enabler = rng.random(n) < p.p_enabler
    true_mult = np.where(is_enabler, rng.uniform(p.enabler_mult_low, p.enabler_mult_high, n), 0.0)

    # Which technologies are actually achievable given the dead ends?
    reach = np.zeros(n, dtype=bool)
    for i in range(n):
        reach[i] = (not is_dead[i]) and any(all(reach[q] for q in r) for r in routes[i])

    milestone = np.full(n, -1)
    for k, (_, frac) in enumerate(MILESTONES):
        lm = int(round(frac * (L - 1)))
        candidates = np.flatnonzero((layer == lm) & reach & (milestone == -1))
        if len(candidates) == 0:
            return None
        milestone[rng.choice(candidates)] = k
    true_value[milestone >= 0] += p.milestone_value

    # Priors. Dead ends look like ordinary technologies: their prior is built
    # from the cost they *would* have had - the failure lies outside the model.
    prior_log_cost = log_cost_if_feasible + p.sigma_est * rng.standard_normal(n)
    est_value = true_value * np.exp(p.sigma_est * rng.standard_normal(n))
    est_mult = true_mult * np.exp(p.sigma_est * rng.standard_normal(n))
    is_underrated = (rng.random(n) < p.p_underrated) & (milestone < 0)
    est_value[is_underrated] *= p.underrated_factor
    est_mult[is_underrated] *= p.underrated_factor

    return World(
        params=p, seed=seed, layer=layer, routes=routes, dist=_distances(routes, n),
        milestone=milestone, true_cost=true_cost, true_value=true_value,
        true_mult=true_mult, is_dead=is_dead, is_underrated=is_underrated,
        prior_log_cost=prior_log_cost, est_value=est_value, est_mult=est_mult,
    )


def _distances(routes: list[list[tuple[int, ...]]], n: int) -> np.ndarray:
    children: list[list[int]] = [[] for _ in range(n)]
    for j, rs in enumerate(routes):
        for q in {q for r in rs for q in r}:
            children[q].append(j)
    dist = np.full((n, n), np.inf)
    for s in range(n):
        dist[s, s] = 0.0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in children[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = dist[s, u] + 1
                    queue.append(v)
    return dist
