"""Fidelity metrics: do attributions agree with the effect of intervening each factor?

Two reference effects are compared with an attribution matrix phi (n planets x d factors):
  * the *model* effect  (how the model's output changes when factor j is reset to baseline), and
  * the *simulator* effect (how the true P(survive) changes), i.e. the ground truth.
Sign convention: phi_ij > 0 means "factor j pushes towards survival relative to the baseline", so resetting j to
baseline should *lower* survival: effect_ij = output(x) - output(x with j at baseline) > 0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def row_spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of A with the same row of B (nan for constant rows)."""
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ra * rb).sum(1) / den


def informative(effect: np.ndarray, thr: float) -> np.ndarray:
    """Planets whose largest single-factor effect exceeds `thr` (otherwise there is nothing to explain)."""
    return np.abs(effect).max(1) >= thr


def agreement(phi: np.ndarray, effect: np.ndarray, thr: float, B: int = 1000, seed: int = 0) -> dict:
    """Agreement metrics between attributions and an intervention effect, on informative planets only."""
    m = informative(effect, thr)
    phi, eff = phi[m], effect[m]
    rho = row_spearman(phi, eff); ok = ~np.isnan(rho)
    sig = np.abs(eff) >= thr
    sign_agree = float((np.sign(phi)[sig] == np.sign(eff)[sig]).mean()) if sig.any() else float("nan")
    top1 = float((np.abs(phi).argmax(1) == np.abs(eff).argmax(1)).mean())
    t3 = lambda a: np.argsort(-np.abs(a), axis=1)[:, :3]
    p3, e3 = t3(phi), t3(eff)
    top3 = float(np.mean([len(set(p3[i]) & set(e3[i])) / 3 for i in range(len(phi))]))
    rng = np.random.default_rng(seed); r = rho[ok]
    boots = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(B)]
    return dict(n_planets=int(m.sum()), share_informative=float(m.mean()), spearman=float(r.mean()),
                spearman_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                sign_agreement=sign_agree, top1=top1, top3_overlap=top3, n_signed_pairs=int(sig.sum()))


def deletion_orders(phi: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Orderings of factors (most 'supportive of survival' first) for each strategy."""
    n, d = phi.shape
    return {"attribution": np.argsort(-phi, axis=1),
            "random": np.array([rng.permutation(d) for _ in range(n)]),
            "global_importance": np.tile(np.argsort(-np.abs(phi).mean(0)), (n, 1))}


def deletion_curve_model(f, z: np.ndarray, order: np.ndarray) -> np.ndarray:
    """f maps standardised (n, d) -> output (n,). Resets the first k factors of `order` to z = 0 (baseline)."""
    n, d = z.shape
    out = np.empty((n, d + 1)); out[:, 0] = f(z)
    rows = np.arange(n)[:, None]
    for k in range(1, d + 1):
        zk = z.copy(); zk[rows, order[:, :k]] = 0.0
        out[:, k] = f(zk)
    return out


def deletion_auc(curve: np.ndarray) -> float:
    """Mean change of the output along the deletion path (more negative = the removed factors mattered more)."""
    return float((curve[:, 1:-1] - curve[:, [0]]).mean())
