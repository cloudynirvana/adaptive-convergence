"""Paired statistics for comparing strategies evaluated on the same worlds."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Compare a candidate to a reference on matched worlds.

    Reports the geometric-mean ratio (scale-free across worlds), its bootstrap
    CI, the share of worlds where the candidate wins, and a two-sided Wilcoxon
    signed-rank p-value on the log ratios.
    """
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    log_ratio = np.log(candidate) - np.log(reference)
    lo, hi = bootstrap_ci(log_ratio)
    p = float(wilcoxon(log_ratio).pvalue) if np.any(log_ratio != 0) else 1.0
    return {
        "ratio": float(np.exp(log_ratio.mean())),
        "ratio_ci": (float(np.exp(lo)), float(np.exp(hi))),
        "win_rate": float((log_ratio > 0).mean()),
        "p_value": p,
        "n": int(len(log_ratio)),
    }
