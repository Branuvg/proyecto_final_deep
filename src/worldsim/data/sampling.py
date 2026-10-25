"""Latin Hypercube sampling of planet factors."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from ..simulator.factors import FACTORS


def sample_params(n: int, seed: int, ranges: dict[str, tuple[float, float]] | None = None) -> dict[str, np.ndarray]:
    """LHS in [0,1]^d mapped to each factor's range (log-uniform for factors flagged `log`)."""
    ranges = ranges or {}
    u = qmc.LatinHypercube(d=len(FACTORS), seed=seed).random(n)
    out = {}
    for j, f in enumerate(FACTORS):
        lo, hi = ranges.get(f.name, (f.low, f.high))
        out[f.name] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if f.log else lo + u[:, j] * (hi - lo)
    return out
