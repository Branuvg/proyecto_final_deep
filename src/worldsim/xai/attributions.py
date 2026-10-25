"""Attribution methods for the task (a) models, all in the same output space (logit of survival)
and with the same reference point (baseline = mean train planet = zero vector in standardised space).

Saved attributions (outputs/xai/attributions.npz) are indexed [planet, factor], so Phase 2 can compare
them with the effect of intervening each factor in the model and in the simulator.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import torch


def exact_shapley(f, X: np.ndarray, baseline: np.ndarray, device, chunk: int = 256) -> np.ndarray:
    """Exact baseline (single-reference) Shapley values for d <= ~14 features.
    v(S) = f(x_S, b_rest). `f` maps a (n, d) float32 tensor to (n,) logits. Returns (N, d)."""
    n, d = X.shape
    masks = ((np.arange(2 ** d)[:, None] >> np.arange(d)) & 1).astype(bool)          # (2^d, d)
    size = masks.sum(1)
    w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    b = torch.as_tensor(baseline, dtype=torch.float32, device=device)
    M = torch.as_tensor(masks, device=device)
    phi = np.zeros((n, d))
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = Xt[i:i + chunk]                                                         # (c, d)
            xin = torch.where(M[None], x[:, None, :], b[None, None, :])                 # (c, 2^d, d)
            v = f(xin.reshape(-1, d)).reshape(len(x), -1).cpu().numpy()                 # (c, 2^d)
            for j in range(d):
                without = np.flatnonzero(~masks[:, j])
                with_j = without + (1 << j)
                phi[i:i + chunk, j] = (v[:, with_j] - v[:, without]) @ w[size[without]]
    return phi


def integrated_gradients(model, X: np.ndarray, baseline: np.ndarray, device, n_steps: int = 256):
    """Captum IG on the logit; returns (attributions (N,d), convergence delta (N,))."""
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model)
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    b = torch.as_tensor(baseline, dtype=torch.float32, device=device).expand_as(x)
    attr, delta = ig.attribute(x, baselines=b, n_steps=n_steps, return_convergence_delta=True, internal_batch_size=512)
    return attr.detach().cpu().numpy(), delta.detach().cpu().numpy()


def tree_shap(gbm, X: np.ndarray, baseline: np.ndarray):
    """TreeSHAP (interventional, same single-point reference) on the LightGBM raw (log-odds) output."""
    import shap
    ex = shap.TreeExplainer(gbm, data=baseline.reshape(1, -1), feature_perturbation="interventional", model_output="raw")
    return ex.shap_values(X), float(np.ravel(ex.expected_value)[0])


def mean_abs(phi: np.ndarray) -> np.ndarray:
    return np.abs(phi).mean(0)
