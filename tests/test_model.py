import numpy as np
import pytest

from acsim import strategies as S
from acsim.march1991 import simulate as march
from acsim.sim import Observation, SimParams, run
from acsim.stats import holm, paired_comparison
from acsim.world import MILESTONES, WorldParams, generate_world

ALL = [S.GreedyMyopic, S.GreedyOption, S.UniformDivergent, S.FixedSplit, S.Thompson,
       S.AdaptiveConvergence, S.OracleGreedy]


@pytest.fixture(scope="module")
def world():
    return generate_world(123)


def test_world_is_deterministic():
    a, b = generate_world(7), generate_world(7)
    assert np.array_equal(a.true_cost, b.true_cost)
    assert a.routes == b.routes
    assert not np.array_equal(a.true_cost, generate_world(8).true_cost)


def test_routes_point_to_earlier_layers(world):
    for i, rs in enumerate(world.routes):
        for r in rs:
            assert all(world.layer[q] < world.layer[i] for q in r)


def test_milestones_are_feasible_and_reachable():
    for seed in range(20):
        w = generate_world(seed)
        reach = np.zeros(w.n, dtype=bool)
        for i in range(w.n):
            reach[i] = (not w.is_dead[i]) and any(all(reach[q] for q in r) for r in w.routes[i])
        for name, _ in MILESTONES:
            assert reach[w.milestone_index(name)]


def test_observation_hides_ground_truth(world):
    seen = {}

    class Spy(S.Strategy):
        name = "spy"

        def allocate(self, obs, amount):
            seen["obs"] = obs
            return obs.available * (amount / obs.available.sum())

    run(world, Spy(), SimParams(horizon=3))
    obs = seen["obs"]
    assert type(obs) is Observation
    for attr in ("true_cost", "true_value", "true_mult", "is_dead", "is_underrated"):
        assert not hasattr(obs, attr)


@pytest.mark.parametrize("cls", ALL)
def test_budget_and_prerequisites_are_respected(world, cls):
    r = run(world, cls(), SimParams(), seed=1)
    finished = np.isfinite(r.done_time)
    # each finished technology had one complete route before (or in the same step as) it finished
    for i in np.flatnonzero(finished):
        assert any(all(r.done_time[q] <= r.done_time[i] for q in route) for route in world.routes[i])
    assert not finished[world.is_dead].any()
    spent = world.true_cost[finished].sum() + r.spend_unfinished
    assert spent <= r.total_budget + 1e-6
    assert np.all(np.diff(r.capability) >= -1e-9)


def test_strategy_cannot_overspend(world):
    class Cheat(S.Strategy):
        name = "cheat"

        def allocate(self, obs, amount):
            return obs.available * amount

    with pytest.raises(ValueError):
        run(world, Cheat(), SimParams(horizon=2))


def test_beliefs_converge_to_true_cost(world):
    seen = {}
    target = int(np.flatnonzero(world.layer == 0)[0])

    class Focus(S.Strategy):
        name = "focus"

        def allocate(self, obs, amount):
            seen["mu"], seen["sd"] = obs.mu[target], obs.sd[target]
            a = np.zeros(obs.n)
            a[target if obs.available[target] else int(np.argmax(obs.available))] = amount
            return a

    run(world, Focus(), SimParams(horizon=1, base_budget=world.true_cost[target] * 0.999,
                                  rounds_per_step=1, info_unit=0.01))
    # One round of heavy information gathering: belief before it was the prior.
    assert seen["sd"] == pytest.approx(world.params.sigma_est)
    run(world, Focus(), SimParams(horizon=2, base_budget=world.true_cost[target] * 0.49,
                                    rounds_per_step=1, info_unit=0.01))
    assert abs(seen["mu"] - np.log(world.true_cost[target])) < 0.05


def test_adaptive_without_exploration_matches_greedy_option(world):
    a = run(world, S.AdaptiveConvergence(beta=0, kappa=0, kill_factor=None), SimParams(), seed=3)
    g = run(world, S.GreedyOption(), SimParams(), seed=3)
    assert np.allclose(a.capability, g.capability)


def test_oracle_beats_random_baseline_on_average():
    o = np.mean([run(generate_world(s), S.OracleGreedy(), SimParams(), s).auc for s in range(5)])
    u = np.mean([run(generate_world(s), S.UniformDivergent(), SimParams(), s).auc for s in range(5)])
    assert o > u


def test_regime_shift_changes_outcomes(world):
    a = run(world, S.GreedyOption(), SimParams(), seed=0)
    b = run(world, S.GreedyOption(), SimParams(shift_at=50, shift_sigma=1.0), seed=0)
    assert not np.allclose(a.capability, b.capability)


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_comparison_detects_clear_gain():
    rng = np.random.default_rng(0)
    ref = rng.uniform(50, 100, 200)
    c = paired_comparison(ref * 1.1, ref)
    assert c["ratio"] == pytest.approx(1.1)
    assert c["win_rate"] == 1.0 and c["p_value"] < 1e-10


# ----------------------------------------------------------------- March (1991)

def test_march_reaches_equilibrium_with_valid_knowledge():
    r = march(0.5, 0.5, seed=0)
    assert r["equilibrium"]
    assert 0.0 <= r["code_knowledge"] <= 1.0


def test_march_code_never_learns_when_p2_is_zero():
    r = march(0.5, 0.0, seed=0)
    assert r["code_knowledge"] == 0.0  # code stays all-zero


def test_march_slow_socialisation_yields_more_knowledge():
    slow = np.mean([march(0.1, 0.5, seed=s)["code_knowledge"] for s in range(30)])
    fast = np.mean([march(0.9, 0.5, seed=s)["code_knowledge"] for s in range(30)])
    assert slow > fast
