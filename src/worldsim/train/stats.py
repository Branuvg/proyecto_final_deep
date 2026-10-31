"""Uncertainty helpers: mean/std across seeds and paired bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def mean_std(per_seed: list[dict]) -> tuple[dict, dict]:
    """Aggregate a list of flat metric dicts: returns (mean, std) over seeds (ddof=1 when >1 seed)."""
    keys = [k for k, v in per_seed[0].items() if isinstance(v, float)]
    arr = np.array([[d[k] for k in keys] for d in per_seed])
    sd = arr.std(0, ddof=1) if len(per_seed) > 1 else np.zeros(len(keys))
    mean = {k: float(m) for k, m in zip(keys, arr.mean(0))}
    mean.update({k: v for k, v in per_seed[0].items() if isinstance(v, int)})
    return mean, {k: float(s) for k, s in zip(keys, sd)}


def paired_bootstrap(y, preds: dict[str, np.ndarray], pairs: list[tuple[str, str]], B: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap over test planets. Returns, for each model, the 95% CI of ROC-AUC and Brier, and for each
    (a, b) pair the 95% CI of metric(a) - metric(b) (positive AUC diff = a better; negative Brier diff = a better)."""
    y = np.asarray(y).astype(int); n = len(y)
    rng = np.random.default_rng(seed)
    auc = {k: np.empty(B) for k in preds}; bri = {k: np.empty(B) for k in preds}
    for b in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            i = rng.integers(0, n, n)
        for k, p in preds.items():
            auc[k][b] = roc_auc_score(y[i], p[i]); bri[k][b] = brier_score_loss(y[i], p[i])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.mean(v)), float(np.percentile(v, 97.5))]
    out = {"models": {k: {"roc_auc": ci(auc[k]), "brier": ci(bri[k])} for k in preds}, "pairs": {}}
    for a, b_ in pairs:
        out["pairs"][f"{a}_minus_{b_}"] = {"roc_auc": ci(auc[a] - auc[b_]), "brier": ci(bri[a] - bri[b_])}
    return out
