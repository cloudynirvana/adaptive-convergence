"""Run every experiment, write results/ (JSON + Markdown) and figures/ (PNG).

    python experiments/run_experiments.py            # full run (~10-20 min on 4 cores)
    python experiments/run_experiments.py --quick    # smoke run with few seeds
    python experiments/run_experiments.py --only e1 e2

Seeds 0-99 are used only for tuning strategy hyperparameters. All reported
results use disjoint evaluation seeds starting at 10_000.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from acsim import strategies as S  # noqa: E402
from acsim.march1991 import simulate as march_simulate  # noqa: E402
from acsim.sim import SimParams, run  # noqa: E402
from acsim.stats import bootstrap_ci, holm, paired_comparison  # noqa: E402
from acsim.world import MILESTONES, WorldParams, generate_world  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
TUNE_SEEDS = range(0, 100)
EVAL_SEED0 = 10_000
REFERENCE = "Greedy + option value"

COLORS = {
    "Greedy (myopic)": "#8c564b",
    "Greedy + option value": "#1f77b4",
    "Uniform (divergent)": "#7f7f7f",
    "Fixed split": "#ff7f0e",
    "Thompson sampling": "#9467bd",
    "Adaptive convergence": "#2ca02c",
    "Oracle (perfect information)": "#000000",
}


def color(name: str) -> str:
    return next((c for k, c in COLORS.items() if name.startswith(k)), "#17becf")


# ----------------------------------------------------------------------------- running

def _job(args):
    seed, wp, sp, specs, keep_curve = args
    world = generate_world(seed, WorldParams(**wp))
    out = []
    for cls, kwargs in specs:
        r = run(world, getattr(S, cls)(**kwargs), SimParams(**sp), seed)
        row = {"seed": seed, "strategy": r.strategy, "auc": r.auc, "final": r.final,
               "n_done": r.n_done, "dead_share": r.spend_dead_ends / r.total_budget,
               "unfinished_share": r.spend_unfinished / r.total_budget,
               "milestones": r.milestone_time}
        if keep_curve:
            row["curve"] = r.capability.tolist()
        out.append(row)
    return out


def run_grid(seeds, specs, wp=WorldParams(), sp=SimParams(), keep_curve=False, pool=None):
    jobs = [(s, asdict(wp), asdict(sp), specs, keep_curve) for s in seeds]
    rows = [r for batch in pool.map(_job, jobs, chunksize=4) for r in batch]
    by = {}
    for r in rows:
        by.setdefault(r["strategy"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["seed"])
    return by


def metric(by, name, key="auc"):
    return np.array([r[key] for r in by[name]])


# ----------------------------------------------------------------------------- strategies

def spec(cls, **kw):
    return (cls, kw)


def tuned_specs():
    t = json.loads((RESULTS / "tuning.json").read_text())
    return [
        spec("GreedyMyopic"),
        spec("GreedyOption"),
        spec("UniformDivergent"),
        spec("FixedSplit", explore=t["fixed_split"]["explore"]),
        spec("Thompson"),
        spec("AdaptiveConvergence", **t["adaptive"]),
        spec("OracleGreedy"),
    ]


def adaptive_name():
    return "Adaptive convergence"


# ----------------------------------------------------------------------------- experiments

def tune(pool, n_seeds):
    seeds = list(TUNE_SEEDS)[:n_seeds]
    grid = [dict(beta=b, kappa=k, kill_factor=kf, name=f"A b={b} k={k} kill={kf}")
            for b in (0.0, 0.5, 1.0) for k in (0.0, 0.1, 0.25, 0.5, 1.0) for kf in (None, 2.0, 3.0, 5.0)]
    splits = (0.1, 0.2, 0.3, 0.5)
    specs = ([spec("GreedyOption")] + [spec("AdaptiveConvergence", **g) for g in grid]
             + [spec("FixedSplit", explore=f) for f in splits])
    by = run_grid(seeds, specs, pool=pool)
    ref = metric(by, REFERENCE)
    score = {name: float(np.exp(np.mean(np.log(metric(by, name)) - np.log(ref))))
             for name in by if name != REFERENCE}
    best = max(grid, key=lambda g: score[g["name"]])
    best_split = max(splits, key=lambda f: score[S.FixedSplit(f).name])
    out = {
        "seeds": [seeds[0], seeds[-1]],
        "adaptive": {k: v for k, v in best.items() if k != "name"},
        "fixed_split": {"explore": best_split},
        "scores_vs_reference": dict(sorted(score.items(), key=lambda kv: -kv[1])),
    }
    (RESULTS / "tuning.json").write_text(json.dumps(out, indent=2))
    return out


def comparison_table(by, names, ref=REFERENCE):
    rows, pvals = [], []
    for name in names:
        if name == ref:
            continue
        c = paired_comparison(metric(by, name), metric(by, ref))
        rows.append((name, c))
        pvals.append(c["p_value"])
    for (_, c), p in zip(rows, holm(pvals)):
        c["p_holm"] = p
    return rows


def e1_main(pool, n):
    seeds = range(EVAL_SEED0, EVAL_SEED0 + n)
    by = run_grid(seeds, tuned_specs(), keep_curve=True, pool=pool)
    names = list(by)
    summary = {}
    for name in names:
        auc = metric(by, name)
        lo, hi = bootstrap_ci(auc)
        ms = {m: float(np.mean([np.isfinite(r["milestones"][m]) for r in by[name]])) for m, _ in MILESTONES}
        summary[name] = {
            "auc_mean": float(auc.mean()), "auc_ci": (lo, hi),
            "techs_completed": float(metric(by, name, "n_done").mean()),
            "budget_share_dead_ends": float(metric(by, name, "dead_share").mean()),
            "budget_share_unfinished": float(metric(by, name, "unfinished_share").mean()),
            "milestone_reached_rate": ms,
        }
    comps = comparison_table(by, [n_ for n_ in names if not n_.startswith("Oracle")])
    (RESULTS / "e1_main.json").write_text(json.dumps({"n_worlds": n, "summary": summary,
                                                       "vs_reference": dict(comps)}, indent=2))

    # Figure: distribution of AUC ratio vs reference
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ref = np.log(metric(by, REFERENCE))
    order = [nm for nm in names if nm != REFERENCE]
    data = [np.exp(np.log(metric(by, nm)) - ref) for nm in order]
    bp = axes[0].boxplot(data, orientation="horizontal", tick_labels=order, showfliers=False, patch_artist=True)
    for patch, nm in zip(bp["boxes"], order):
        patch.set_facecolor(color(nm)); patch.set_alpha(0.55)
    axes[0].axvline(1.0, color="k", lw=1, ls="--")
    axes[0].set_xlabel(f"Cumulative capability relative to '{REFERENCE}' (per world)")
    axes[0].set_title(f"E1 - Strategy performance across {n} worlds")
    for nm in names:
        curves = np.array([r["curve"] for r in by[nm]])
        axes[1].plot(curves.mean(0), label=nm, color=color(nm),
                     ls="--" if nm.startswith("Oracle") else "-")
    axes[1].set_xlabel("Time step"); axes[1].set_ylabel("Mean capability (true value completed)")
    axes[1].set_title("Capability over time"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "e1_main.png", dpi=150); plt.close(fig)

    # Figure: milestone attainment curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    H = SimParams().horizon
    for ax, (m, _) in zip(axes, MILESTONES):
        for nm in names:
            t = np.array([r["milestones"][m] for r in by[nm]])
            ax.plot(np.arange(H), [(t <= h).mean() for h in range(H)], color=color(nm), label=nm,
                    ls="--" if nm.startswith("Oracle") else "-")
        ax.set_title(m.replace("_", " ").title()); ax.set_xlabel("Time step")
    axes[0].set_ylabel("Share of worlds where milestone reached"); axes[2].legend(fontsize=7)
    fig.suptitle("E1 - Milestone attainment (cumulative)")
    fig.tight_layout(); fig.savefig(FIGURES / "e1_milestones.png", dpi=150); plt.close(fig)
    return summary, comps


def e2_uncertainty(pool, n):
    levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    specs = [s for s in tuned_specs()]
    seeds = range(EVAL_SEED0, EVAL_SEED0 + n)
    out = {}
    for lv in levels:
        by = run_grid(seeds, specs, wp=WorldParams(sigma_est=lv), pool=pool)
        ref = np.log(metric(by, REFERENCE))
        out[lv] = {}
        for nm in by:
            lr = np.log(metric(by, nm)) - ref
            lo, hi = bootstrap_ci(lr)
            out[lv][nm] = {"ratio": float(np.exp(lr.mean())), "ci": (float(np.exp(lo)), float(np.exp(hi)))}
    (RESULTS / "e2_uncertainty.json").write_text(json.dumps({"n_worlds_per_level": n, "results": out}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for nm in out[levels[0]]:
        r = np.array([out[lv][nm]["ratio"] for lv in levels])
        ci = np.array([out[lv][nm]["ci"] for lv in levels])
        ax.plot(levels, r, marker="o", color=color(nm), label=nm, ls="--" if nm.startswith("Oracle") else "-")
        ax.fill_between(levels, ci[:, 0], ci[:, 1], color=color(nm), alpha=0.12)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlabel("Prior uncertainty  σ_est  (log-space error of the civilisation's estimates)")
    ax.set_ylabel(f"Capability relative to '{REFERENCE}'")
    ax.set_title("E2 - When does exploring pay? Performance vs. uncertainty")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "e2_uncertainty.png", dpi=150); plt.close(fig)
    return out


def e3_heatmap(pool, n):
    sig = [0.0, 0.35, 0.7, 1.05, 1.4]
    dead = [0.0, 0.1, 0.2, 0.3]
    t = json.loads((RESULTS / "tuning.json").read_text())
    specs = [spec("GreedyOption"), spec("AdaptiveConvergence", **t["adaptive"])]
    seeds = range(EVAL_SEED0, EVAL_SEED0 + n)
    grid = np.zeros((len(dead), len(sig)))
    win = np.zeros_like(grid)
    for i, d in enumerate(dead):
        for j, s in enumerate(sig):
            by = run_grid(seeds, specs, wp=WorldParams(sigma_est=s, p_dead_end=d), pool=pool)
            lr = np.log(metric(by, adaptive_name())) - np.log(metric(by, REFERENCE))
            grid[i, j] = 100 * (np.exp(lr.mean()) - 1)
            win[i, j] = (lr > 0).mean()
    (RESULTS / "e3_heatmap.json").write_text(json.dumps(
        {"n_worlds_per_cell": n, "sigma_est": sig, "p_dead_end": dead,
         "adaptive_advantage_pct": grid.tolist(), "adaptive_win_rate": win.tolist()}, indent=2))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    lim = max(1.0, np.abs(grid).max())
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    for i in range(len(dead)):
        for j in range(len(sig)):
            ax.text(j, i, f"{grid[i, j]:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(sig)), [str(s) for s in sig]); ax.set_yticks(range(len(dead)), [str(d) for d in dead])
    ax.set_xlabel("Prior uncertainty σ_est"); ax.set_ylabel("Share of dead-end technologies")
    ax.set_title("E3 - Adaptive convergence vs. greedy+option (capability gain)")
    fig.colorbar(im, label="% difference (geometric mean)")
    fig.tight_layout(); fig.savefig(FIGURES / "e3_heatmap.png", dpi=150); plt.close(fig)
    return grid


def e4_shift(pool, n):
    seeds = range(EVAL_SEED0, EVAL_SEED0 + n)
    H = SimParams().horizon
    conds = {"stable": SimParams(), "regime shift at t=100": SimParams(shift_at=H // 2, shift_sigma=1.0)}
    out = {}
    for label, sp in conds.items():
        by = run_grid(seeds, tuned_specs(), sp=sp, pool=pool)
        out[label] = {nm: c for nm, c in comparison_table(by, [x for x in by if not x.startswith("Oracle")])}
        out[label]["_auc_mean"] = {nm: float(metric(by, nm).mean()) for nm in by}
    (RESULTS / "e4_shift.json").write_text(json.dumps({"n_worlds": n, "results": out}, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [k for k in out["stable"] if not k.startswith("_")]
    x = np.arange(len(names))
    for k, (label, res) in enumerate(out.items()):
        r = [res[nm]["ratio"] for nm in names]
        err = np.array([[res[nm]["ratio"] - res[nm]["ratio_ci"][0], res[nm]["ratio_ci"][1] - res[nm]["ratio"]] for nm in names]).T
        ax.bar(x + (k - 0.5) * 0.38, r, 0.38, yerr=err, label=label, alpha=0.8, capsize=3)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"Capability relative to '{REFERENCE}'")
    ax.set_title("E4 - Robustness to a mid-run regime shift in hidden costs")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "e4_shift.png", dpi=150); plt.close(fig)
    return out


def e5_ablation(pool, n):
    t = json.loads((RESULTS / "tuning.json").read_text())["adaptive"]
    # Toggle each component relative to the tuned configuration.
    variants = {
        "Adaptive (tuned)": t,
        "- option value": {**t, "use_option": False},
        ("- optimism (beta=0)" if t["beta"] else "+ optimism (beta=1)"): {**t, "beta": 0.0 if t["beta"] else 1.0},
        ("- probing (kappa=0)" if t["kappa"] else "+ probing (kappa=0.25)"): {**t, "kappa": 0.0 if t["kappa"] else 0.25},
        "- kill rule": {**t, "kill_factor": None},
    }
    specs = [spec("GreedyOption")] + [spec("AdaptiveConvergence", **v, name=k) for k, v in variants.items()]
    by = run_grid(range(EVAL_SEED0, EVAL_SEED0 + n), specs, pool=pool)
    full = metric(by, "Adaptive (tuned)")
    out = {}
    for k in list(variants)[1:] + [REFERENCE]:
        out[k] = paired_comparison(metric(by, k), full)
        out[k]["dead_share"] = float(metric(by, k, "dead_share").mean())
    pv = holm([v["p_value"] for v in out.values()])
    for v, p in zip(out.values(), pv):
        v["p_holm"] = p
    (RESULTS / "e5_ablation.json").write_text(json.dumps({"n_worlds": n, "full_config": t,
                                                           "relative_to_full": out}, indent=2))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = list(out)
    r = [out[k]["ratio"] for k in names]
    err = np.array([[out[k]["ratio"] - out[k]["ratio_ci"][0], out[k]["ratio_ci"][1] - out[k]["ratio"]] for k in names]).T
    ax.barh(names, [100 * (v - 1) for v in r], xerr=100 * err, color="#2ca02c", alpha=0.7, capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("% change in capability vs. tuned adaptive strategy (95% CI)")
    ax.set_title("E5 - Ablation: toggling each component")
    fig.tight_layout(); fig.savefig(FIGURES / "e5_ablation.png", dpi=150); plt.close(fig)
    return out


def _march_job(args):
    p1, p2, seed = args
    return march_simulate(p1, p2, seed=seed)


def m1_march(pool, n):
    p1s = [round(0.1 * i, 1) for i in range(1, 10)]
    p2s = [0.1, 0.5, 0.9]
    out = {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p2, c in zip(p2s, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        means, cis = [], []
        for p1 in p1s:
            res = list(pool.map(_march_job, [(p1, p2, s) for s in range(n)], chunksize=8))
            k = np.array([r["code_knowledge"] for r in res])
            means.append(float(k.mean())); cis.append(bootstrap_ci(k))
            out[f"p1={p1},p2={p2}"] = {"code_knowledge": means[-1], "ci": cis[-1],
                                       "equilibrium_rate": float(np.mean([r["equilibrium"] for r in res]))}
        cis = np.array(cis)
        ax.plot(p1s, means, marker="o", color=c, label=f"p2 = {p2}")
        ax.fill_between(p1s, cis[:, 0], cis[:, 1], color=c, alpha=0.15)
    ax.set_xlabel("Socialisation rate p1 (speed of convergence to the code)")
    ax.set_ylabel("Equilibrium code knowledge")
    ax.set_title(f"M1 - Replication of March (1991), Fig. 1  ({n} runs/point)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "m1_march_fig1.png", dpi=150); plt.close(fig)

    # Mixed learners: 50 individuals, some fast (p1=0.9) and the rest slow (p1=0.1), p2=0.5
    mixed = {}
    for n_slow in range(0, 51, 5):
        p1 = np.array([0.1] * n_slow + [0.9] * (50 - n_slow))
        res = list(pool.map(_march_mixed_job, [(p1, s) for s in range(n)], chunksize=8))
        k = np.array([r["code_knowledge"] for r in res])
        mixed[n_slow] = {"code_knowledge": float(k.mean()), "ci": bootstrap_ci(k)}
    out["mixed_learners_p2_0.5"] = mixed
    (RESULTS / "m1_march.json").write_text(json.dumps({"runs_per_point": n, "results": out}, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = list(mixed)
    ys = [mixed[x]["code_knowledge"] for x in xs]
    ci = np.array([mixed[x]["ci"] for x in xs])
    ax.plot(xs, ys, marker="o", color="#2ca02c"); ax.fill_between(xs, ci[:, 0], ci[:, 1], color="#2ca02c", alpha=0.15)
    ax.set_xlabel("Number of slow learners (p1=0.1) among 50; rest fast (p1=0.9)")
    ax.set_ylabel("Equilibrium code knowledge")
    ax.set_title("M2 - Mixing fast (p1=0.9) and slow (p1=0.1) learners, p2=0.5")
    fig.tight_layout(); fig.savefig(FIGURES / "m2_march_mixed.png", dpi=150); plt.close(fig)
    return out


def _march_mixed_job(args):
    p1, seed = args
    return march_simulate(p1, 0.5, seed=seed)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    q = a.quick
    plan = {
        "tune": lambda p: tune(p, 12 if q else 100),
        "e1": lambda p: e1_main(p, 12 if q else 400),
        "e2": lambda p: e2_uncertainty(p, 8 if q else 200),
        "e3": lambda p: e3_heatmap(p, 6 if q else 120),
        "e4": lambda p: e4_shift(p, 8 if q else 300),
        "e5": lambda p: e5_ablation(p, 8 if q else 400),
        "m1": lambda p: m1_march(p, 10 if q else 80),
    }
    todo = a.only or list(plan)
    with ProcessPoolExecutor(a.workers) as pool:
        for name in todo:
            t0 = time.time()
            plan[name](pool)
            print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
