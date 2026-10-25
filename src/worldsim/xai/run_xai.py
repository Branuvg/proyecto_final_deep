"""Compute and save attributions on the task (a) test set. Run: python -m worldsim.xai.run_xai"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.dataset import Preprocessor, load_worlds
from ..models.nets import MLP
from ..simulator.factors import ACTIONABLE
from ..train.loop import get_device, set_seed
from .attributions import exact_shapley, integrated_gradients, tree_shap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/xai.yaml")
    c = yaml.safe_load(open(ap.parse_args().config, encoding="utf8"))
    set_seed(c["seed"]); dev = get_device()
    w = load_worlds(c["data"]); pre = Preprocessor(w)
    te = w.idx("test"); Xs = pre.static(w.X)
    ck = torch.load(Path(c["task_a_dir"]) / "mlp_a.pt", map_location="cpu"); hp = ck["hp"]
    net = MLP(ck["d_in"], tuple(hp["hidden"]), hp["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    gbm = pickle.load(open(Path(c["task_a_dir"]) / "lgbm_a.pkl", "rb"))

    # Baseline definition (same for every method): the mean train planet = zero vector in standardised space.
    base = np.zeros(Xs.shape[1], dtype=np.float32)
    X = Xs[te]
    with torch.no_grad():
        f = lambda t: net(t)
        logit = f(torch.as_tensor(X, device=dev)).cpu().numpy()
        base_logit = float(f(torch.as_tensor(base[None], device=dev)).item())

    phi_shap = exact_shapley(f, X, base, dev)
    phi_ig, ig_delta = integrated_gradients(net, X, base, dev, c["ig_steps"])
    phi_tree, tree_base = tree_shap(gbm, X, base)

    # Sanity checks: efficiency (completeness) of every method
    eff = dict(shap_mlp=float(np.abs(phi_shap.sum(1) - (logit - base_logit)).max()),
               ig_mlp=float(np.abs(phi_ig.sum(1) - (logit - base_logit)).max()),
               ig_convergence_delta_max=float(np.abs(ig_delta).max()),
               treeshap=float(np.abs(phi_tree.sum(1) + tree_base - gbm.predict(X, raw_score=True)).max()))
    # Cross-check exact Shapley vs the shap library KernelExplainer on a few planets
    import shap
    ke = shap.KernelExplainer(lambda a: net(torch.as_tensor(a, dtype=torch.float32, device=dev)).detach().cpu().numpy(),
                              base[None], silent=True)
    sub = X[: c["kernel_check_n"]]
    kv = ke.shap_values(sub, nsamples=2 ** 11 - 2, silent=True, l1_reg=False)
    eff["exact_vs_kernelshap_max_abs_diff"] = float(np.abs(kv - phi_shap[: len(sub)]).max())
    print(json.dumps(eff, indent=1))

    out = Path(c["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "attributions.npz", planet_idx=te, X_raw=w.X[te], X_std=X, y=w.y[te], factor_names=np.array(w.names),
             actionable=np.array([ACTIONABLE[n] for n in w.names]), phi_shap_mlp=phi_shap, phi_ig_mlp=phi_ig,
             phi_treeshap_gb=phi_tree, logit_mlp=logit, base_logit_mlp=base_logit, base_logit_gb=tree_base,
             mc_idx=w.mc_idx, mc_p_true=w.mc_p_true,
             baseline_definition=np.array("baseline = mean train planet (zero vector in standardised space; log for delta,kappa); "
                                          "output space = logit of P(survive); SHAP exact with single reference; IG straight path; "
                                          "TreeSHAP interventional with the same reference"))
    json.dump(dict(checks=eff), open(out / "checks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
