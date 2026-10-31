"""Fidelity of XAI to the model and to the simulator. Run: python -m worldsim.xai.run_fidelity [--config configs/fidelity.yaml]

For a set of test planets it computes
  1. single-factor effects in the model (logit) and in the simulator (P(survive), Monte Carlo with common random numbers),
  2. agreement of SHAP / IG / TreeSHAP with both effects (Spearman, sign, top-k),
  3. deletion curves in the model and in the simulator (attribution order vs random vs global importance),
  4. a randomised-weights sanity check (Adebayo et al., 2018) on the SHAP of the MLP.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.dataset import Preprocessor, load_worlds
from ..models.nets import MLP
from ..train.loop import get_device, set_seed
from .attributions import exact_shapley
from .fidelity import agreement, deletion_auc, deletion_curve_model, deletion_orders, row_spearman
from .interventions import baseline_raw, deletion_curve_sim, single_factor_interventions, sim_survival


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/fidelity.yaml")
    c = yaml.safe_load(open(ap.parse_args().config, encoding="utf8"))
    set_seed(c["seed"]); dev = get_device(); rng = np.random.default_rng(c["seed"])
    w = load_worlds(c["data"]); pre = Preprocessor(w); names = w.names
    A = np.load(Path(c["xai_dir"]) / "attributions.npz", allow_pickle=True)
    te = A["planet_idx"]; base = baseline_raw(pre)
    ck = torch.load(Path(c["task_a_dir"]) / "mlp_a.pt", map_location="cpu"); hp = ck["hp"]
    net = MLP(ck["d_in"], tuple(hp["hidden"]), hp["dropout"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    gbm = pickle.load(open(Path(c["task_a_dir"]) / "lgbm_a.pkl", "rb"))
    f_mlp = lambda z: net(torch.as_tensor(z, dtype=torch.float32, device=dev)).detach().cpu().numpy()
    f_gb = lambda z: gbm.predict(z, raw_score=True)
    sig = lambda v: 1 / (1 + np.exp(-v))

    # ---- planet sets: MC planets (random sample) + planets near the decision boundary of the model ----
    mc_rows = np.searchsorted(te, A["mc_idx"])
    p_model = sig(A["logit_mlp"])
    rest = np.setdiff1d(np.flatnonzero((p_model > 0.05) & (p_model < 0.95)), mc_rows)
    bnd_rows = rng.choice(rest, size=min(c["n_boundary"], len(rest)), replace=False)
    groups = {"mc": mc_rows, "boundary": bnd_rows}
    methods = {"SHAP (MLP)": ("phi_shap_mlp", "mlp"), "IG (MLP)": ("phi_ig_mlp", "mlp"), "TreeSHAP (LightGBM)": ("phi_treeshap_gb", "gbm")}
    f_of = {"mlp": f_mlp, "gbm": f_gb}
    out = dict(config=c, groups={}, deletion={}, thr=c["effect_thr"])

    arrays = {}
    for g, rows in groups.items():
        t0 = time.time()
        X_raw, Z = A["X_raw"][rows], A["X_std"][rows]
        p0, p_after = single_factor_interventions(X_raw, names, base, c["R"], c["seed"] * 7)
        d_sim = p0[:, None] - p_after                                     # true effect of each factor (probability)
        d_model = {}
        for key, f in f_of.items():
            logit0 = f(Z); la = np.empty_like(Z)
            for j in range(Z.shape[1]):
                Zj = Z.copy(); Zj[:, j] = 0.0
                la[:, j] = logit0 - f(Zj)
            d_model[key] = la                                             # model effect (logit)
        res = dict(n=len(rows), sim_seconds=time.time() - t0, p_true_mean=float(p0.mean()),
                   share_planets_true_p_intermediate=float(((p0 > .05) & (p0 < .95)).mean()), methods={})
        for mname, (key, mk) in methods.items():
            phi = A[key][rows]
            res["methods"][mname] = dict(vs_model=agreement(phi, d_model[mk], c["effect_thr_logit"], seed=c["seed"]),
                                         vs_simulator=agreement(phi, d_sim, c["effect_thr"], seed=c["seed"]))
        # does the MODEL itself respond like the simulator? (separates model error from explanation error)
        d_model_p = sig(f_mlp(Z))[:, None] - sig(np.stack([f_mlp(np.where(np.arange(Z.shape[1]) == j, 0.0, Z)) for j in range(Z.shape[1])], 1))
        res["model_vs_simulator_effect"] = agreement(d_model_p, d_sim, c["effect_thr"], seed=c["seed"])
        out["groups"][g] = res
        arrays[g] = dict(rows=rows, p_true=p0, p_after=p_after, d_sim=d_sim, d_model_mlp=d_model["mlp"], d_model_gbm=d_model["gbm"],
                         phi_shap=A["phi_shap_mlp"][rows], phi_ig=A["phi_ig_mlp"][rows], phi_tree=A["phi_treeshap_gb"][rows])
        print(g, "done", f"{time.time()-t0:.0f}s", {m: round(v["vs_simulator"]["spearman"], 3) for m, v in res["methods"].items()}, flush=True)

    # ---- deletion curves on the MC planets (model: all methods; simulator: R_del replicates) ----
    rows = mc_rows; Z = A["X_std"][rows]; X_raw = A["X_raw"][rows]
    for mname, (key, mk) in methods.items():
        orders = deletion_orders(A[key][rows], rng)
        out["deletion"][mname] = {}
        for oname, order in orders.items():
            cm = deletion_curve_model(f_of[mk], Z, order)
            cs = deletion_curve_sim(X_raw, order, names, base, c["R_deletion"], c["seed"] * 11)
            out["deletion"][mname][oname] = dict(model_auc=deletion_auc(cm), sim_auc=deletion_auc(cs),
                                                 model_curve=cm.mean(0).tolist(), sim_curve=cs.mean(0).tolist())
        print("deletion", mname, {o: round(v["sim_auc"], 4) for o, v in out["deletion"][mname].items()}, flush=True)

    # ---- sanity check: randomised weights ----
    set_seed(c["seed"] + 99)
    rnd = MLP(ck["d_in"], tuple(hp["hidden"]), hp["dropout"]).to(dev).eval()
    f_rnd = lambda t: rnd(t)
    sub = np.concatenate([mc_rows, bnd_rows])
    phi_rnd = exact_shapley(f_rnd, A["X_std"][sub], np.zeros(Z.shape[1], dtype=np.float32), dev)
    phi_tr = A["phi_shap_mlp"][sub]
    rho = row_spearman(phi_rnd, phi_tr)
    d_sim_all = np.concatenate([arrays["mc"]["d_sim"], arrays["boundary"]["d_sim"]])
    out["sanity_randomised_model"] = dict(
        mean_spearman_trained_vs_randomised=float(np.nanmean(rho)),
        vs_simulator_randomised=agreement(phi_rnd, d_sim_all, c["effect_thr"], seed=c["seed"]),
        vs_simulator_trained=agreement(phi_tr, d_sim_all, c["effect_thr"], seed=c["seed"]))

    # ---- global importance: attribution vs true effect (mean |.| per factor) ----
    allrows = np.concatenate([arrays["mc"]["rows"], arrays["boundary"]["rows"]])
    out["global_importance"] = dict(factors=names,
                                    sim_mean_abs_effect=np.abs(d_sim_all).mean(0).tolist(),
                                    shap_mean_abs=np.abs(A["phi_shap_mlp"][allrows]).mean(0).tolist(),
                                    ig_mean_abs=np.abs(A["phi_ig_mlp"][allrows]).mean(0).tolist(),
                                    tree_mean_abs=np.abs(A["phi_treeshap_gb"][allrows]).mean(0).tolist())
    od = Path(c["out_dir"]); od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(od / "arrays.npz", names=np.array(names), base=base,
                        **{f"{g}_{k}": v for g, d in arrays.items() for k, v in d.items()})
    print("saved", od)


if __name__ == "__main__":
    main()
