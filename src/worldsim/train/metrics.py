"""Classification and calibration metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = ids == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def calibration_curve(y, p, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1); ids = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = ids == b
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def classification_metrics(y, p) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=np.float64)
    out = dict(n=len(y), pos_rate=float(y.mean()))
    if 0 < y.sum() < len(y):
        out.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))
    out.update(f1=f1_score(y, p >= 0.5, zero_division=0), brier=brier_score_loss(y, p), ece=ece(y, p))
    return {k: (v if isinstance(v, int) else float(v)) for k, v in out.items()}
