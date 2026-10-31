"""Counterfactuals on actionable factors, validated with the simulator.
Run: python -m worldsim.xai.run_counterfactuals [--config configs/counterfactual.yaml]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.dataset import Preprocessor, load_worlds
from ..models.nets import MLP
from ..simulator.factors import ACTIONABLE, FACTORS
from ..train.loop import get_device, set_seed
from .counterfactual import gradient_counterfactual, single_factor_counterfactual, z_bounds, z_to_raw
from .interventions import sim_survival


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/counterfactual.yaml")
    c = yaml.safe_load(open(ap.parse_args().config, encoding="utf8"))
    set_seed(c["seed"]); dev = get_device()
    w = load_worlds(c["data"]); pre = Preprocessor(w); names = w.names
    A = np.load(Path(c["xai_dir"]) / "attributions.npz", allow_pickle=True)
    ck = torch.load(Path(c["task_a_dir"]) / "mlp_a.pt", map_location="cpu"); hp = ck["hp"]
    net = MLP(ck["d_in"], tuple(hp["hidden"]), hp["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    f = lambda z: net(torch.as_tensor(z, dtype=torch.float32, device=dev)).detach().cpu().numpy()
    sig = lambda v: 1 / (1 + np.exp(-v))

    mc_rows = np.searchsorted(A["planet_idx"], A["mc_idx"])
    Z_all, X_all = A["X_std"][mc_rows], A["X_raw"][mc_rows]
    p_model = sig(f(Z_all)); p_true = sim_survival(X_all, names, c["R"], c["seed"])      # MC ground truth with R replicates
    todo = np.flatnonzero(p_model < 0.5)                                                 # planets the model says will collapse
    Z, X0 = Z_all[todo], X_all[todo]
    act = np.array([ACTIONABLE[n] for n in names]); lo, hi = z_bounds(pre); rng_z = hi - lo

    Zg = gradient_counterfactual(net, Z, pre, names, c["target_p"], c["lam"], c["steps"], c["lr"], dev)
    Zs, solved = single_factor_counterfactual(net, Z, pre, names, c["target_p"], device=dev)
    res = dict(n_mc=len(mc_rows), n_predicted_collapse=int(len(todo)),
               n_truly_collapse=int((p_true[todo] < 0.5).sum()), n_model_error=int((p_true[todo] >= 0.5).sum()))
    arrays = dict(todo=todo, Z=Z, X0=X0, p_true_before=p_true[todo], p_model_before=p_model[todo])
    for tag, Zc, ok in (("gradient", Zg, np.ones(len(todo), bool)), ("single_factor", Zs, solved)):
        Xc = z_to_raw(pre, Zc.copy())
        p_after_model = sig(f(Zc)); p_after_true = sim_survival(Xc, names, c["R"], c["seed"] + 1)
        change = np.abs(Zc - Z) / rng_z                                           # fraction of the factor range moved
        n_changed = (change > 0.01).sum(1)
        true_collapse = p_true[todo] < 0.5
        res[tag] = dict(
            model_valid=float((p_after_model[ok] >= c["target_p"] - 1e-3).mean()),
            simulator_valid=float((p_after_true >= 0.5)[ok & true_collapse].mean()) if (ok & true_collapse).any() else None,
            simulator_valid_all=float((p_after_true >= 0.5)[ok].mean()),
            solved_share=float(ok.mean()),
            mean_true_p_before=float(p_true[todo][true_collapse].mean()),
            mean_true_p_after=float(p_after_true[ok & true_collapse].mean()),
            mean_model_p_after=float(p_after_model[ok].mean()),
            mean_factors_changed=float(n_changed[ok].mean()),
            mean_range_fraction_moved=float(change.sum(1)[ok].mean()),
            changed_share_by_factor={n: float((change[ok, j] > 0.01).mean()) for j, n in enumerate(names) if ACTIONABLE[n]},
            nonactionable_changed=float(change[:, ~act].max()) if (~act).any() else 0.0)
        arrays[f"X_cf_{tag}"] = Xc; arrays[f"p_true_after_{tag}"] = p_after_true; arrays[f"p_model_after_{tag}"] = p_after_model
        print(tag, json.dumps({k: v for k, v in res[tag].items() if not isinstance(v, dict)}, indent=0), flush=True)
    od = Path(c["out_dir"]); od.mkdir(parents=True, exist_ok=True)
    json.dump(dict(config=c, **res), open(od / "results.json", "w"), indent=1)
    np.savez_compressed(od / "arrays.npz", names=np.array(names), **arrays)


if __name__ == "__main__":
    main()
