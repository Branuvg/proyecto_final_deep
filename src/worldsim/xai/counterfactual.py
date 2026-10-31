"""Counterfactual explanations over *actionable* factors only (Wachter et al., 2018 style).

Question answered: "what is the smallest change to the levers we control (depletion, inequality, consumption,
water efficiency) that makes this planet survive?". The search is gradient-based on the MLP, in standardised space,
restricted to actionable factors and to the sampling ranges. Whether the counterfactual *really* works is then
checked with the simulator (see run_counterfactuals.py).
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import Preprocessor
from ..simulator.factors import ACTIONABLE, FACTORS


def z_bounds(pre: Preprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Sampling-range bounds of every factor expressed in standardised space."""
    lo = np.array([f.low for f in FACTORS], float); hi = np.array([f.high for f in FACTORS], float)
    lo, hi = lo.copy(), hi.copy()
    lo[pre._log_cols], hi[pre._log_cols] = np.log(lo[pre._log_cols]), np.log(hi[pre._log_cols])
    return ((lo - pre.s_mu) / pre.s_sd).astype(np.float32), ((hi - pre.s_mu) / pre.s_sd).astype(np.float32)


def z_to_raw(pre: Preprocessor, Z: np.ndarray) -> np.ndarray:
    X = Z * pre.s_sd + pre.s_mu
    X[:, pre._log_cols] = np.exp(X[:, pre._log_cols])
    return X


def gradient_counterfactual(net, Z: np.ndarray, pre: Preprocessor, names: list[str], target_p: float = 0.8,
                            lam: float = 0.5, steps: int = 600, lr: float = 0.03, device="cpu") -> np.ndarray:
    """Minimise  hinge(logit_target - f(z')) + lam * L1(z' - z)/range  over actionable dims only."""
    lo, hi = z_bounds(pre)
    mask = torch.as_tensor([float(ACTIONABLE[n]) for n in names], device=device)
    lo_t, hi_t = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)
    rng_t = hi_t - lo_t
    z0 = torch.as_tensor(Z, dtype=torch.float32, device=device)
    delta = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    target = float(np.log(target_p / (1 - target_p)))
    net.eval()
    for _ in range(steps):
        z = torch.minimum(torch.maximum(z0 + delta * mask, lo_t), hi_t)
        loss = torch.relu(target - net(z)).sum() + lam * ((z - z0).abs() / rng_t).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = torch.minimum(torch.maximum(z0 + delta * mask, lo_t), hi_t)
        small = ((z - z0).abs() / rng_t) < 0.01               # drop negligible changes (sparsity)
        z = torch.where(small, z0, z)
    return z.cpu().numpy()


def single_factor_counterfactual(net, Z: np.ndarray, pre: Preprocessor, names: list[str], target_p: float = 0.8,
                                 grid: int = 201, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """For each planet, the smallest change of ONE actionable factor reaching `target_p` (grid search).
    Returns (Z_cf (n, d), solved (n,) bool); unsolved rows are returned unchanged."""
    lo, hi = z_bounds(pre)
    target = float(np.log(target_p / (1 - target_p)))
    out = Z.copy(); best = np.full(len(Z), np.inf); solved = np.zeros(len(Z), bool)
    for j, n in enumerate(names):
        if not ACTIONABLE[n]:
            continue
        g = np.linspace(lo[j], hi[j], grid, dtype=np.float32)
        for i in range(0, len(Z), 256):
            zb = Z[i:i + 256]
            cand = np.repeat(zb[:, None, :], grid, axis=1); cand[:, :, j] = g[None, :]
            with torch.no_grad():
                f = net(torch.as_tensor(cand.reshape(-1, Z.shape[1]), device=device)).cpu().numpy().reshape(len(zb), grid)
            dist = np.abs(g[None, :] - zb[:, [j]]) / (hi[j] - lo[j]); dist[f < target] = np.inf
            k = dist.argmin(1); d = dist[np.arange(len(zb)), k]
            better = d < best[i:i + 256]
            idx = np.flatnonzero(better) + i
            best[idx] = d[better]; solved[idx] = True
            out[idx] = Z[idx]; out[idx, j] = g[k[better]]
    return out, solved
