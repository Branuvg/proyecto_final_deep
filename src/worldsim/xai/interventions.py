"""Interventions on planet factors, evaluated on the model and on the simulator (ground truth).

An intervention sets factor j of planet i to its *baseline value* (the same reference used by SHAP/IG/TreeSHAP:
the mean train planet, i.e. z = 0 in standardised space; geometric mean for log-sampled factors). The simulator
estimates P(survive | factors) with R Monte Carlo replicates; common random numbers (same seed and row layout)
are used for the original and the intervened planets so that their difference has low variance.
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import Preprocessor
from ..simulator.handy import SimConfig, simulate


def baseline_raw(pre: Preprocessor) -> np.ndarray:
    """Baseline planet in raw factor units (exp of the mean log for log-sampled factors)."""
    b = pre.s_mu.copy()
    b[pre._log_cols] = np.exp(b[pre._log_cols])
    return b


def sim_survival(X_raw: np.ndarray, names: list[str], R: int, seed: int, chunk_planets: int = 2000) -> np.ndarray:
    """Monte Carlo P(survive) for each planet (rows of X_raw) with R replicates; shape (n,)."""
    n = len(X_raw); out = np.empty(n)
    for ci, i in enumerate(range(0, n, chunk_planets)):
        blk = X_raw[i:i + chunk_planets]
        params = {nm: np.repeat(blk[:, j], R) for j, nm in enumerate(names)}
        r = simulate(params, SimConfig(horizon=600, record_every=100, seed=seed + ci))
        out[i:i + len(blk)] = (~r.collapsed).reshape(len(blk), R).mean(1)
    return out


def single_factor_interventions(X_raw: np.ndarray, names: list[str], base: np.ndarray, R: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (P_true (n,), P_after (n, d)) where P_after[:, j] is P(survive) after setting factor j to its baseline."""
    p0 = sim_survival(X_raw, names, R, seed)
    after = np.empty((len(X_raw), len(names)))
    for j in range(len(names)):
        Xj = X_raw.copy(); Xj[:, j] = base[j]
        after[:, j] = sim_survival(Xj, names, R, seed)
    return p0, after


def deletion_curve_sim(X_raw: np.ndarray, order: np.ndarray, names: list[str], base: np.ndarray, R: int, seed: int) -> np.ndarray:
    """P(survive) after resetting the first k factors of `order` (n, d) to baseline, k = 0..d. Returns (n, d+1)."""
    n, d = X_raw.shape
    curve = np.empty((n, d + 1))
    curve[:, 0] = sim_survival(X_raw, names, R, seed)
    for k in range(1, d):
        Xk = X_raw.copy()
        for i in range(d):            # factors among the first k of each planet's ordering go to baseline
            m = (order[:, :k] == i).any(1)
            Xk[m, i] = base[i]
        curve[:, k] = sim_survival(Xk, names, R, seed)
    curve[:, d] = sim_survival(np.tile(base, (n, 1)), names, R, seed)
    return curve
