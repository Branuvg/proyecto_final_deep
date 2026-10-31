"""Task (a): initial conditions -> survival. Baselines (logistic, LightGBM) + MLP with a small random search.

The search runs once (seed 0, validation BCE); the best configuration is then retrained with every seed in
`seeds` so that results are reported as mean +- std, plus paired bootstrap CIs on the test set (seed 0).

Run: python -m worldsim.train.run_task_a --config configs/task_a.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from ..data.dataset import Preprocessor, load_worlds
from ..models.nets import MLP
from .loop import fit, get_device, set_seed
from .metrics import calibration_curve, classification_metrics
from .stats import mean_std, paired_bootstrap


def _predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, device=device))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/task_a.yaml")
    c = yaml.safe_load(open(ap.parse_args().config, encoding="utf8"))
    seeds = c["seeds"]; set_seed(seeds[0]); dev = get_device()
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    w = load_worlds(c["data"]); pre = Preprocessor(w)
    tr, va, te = w.idx("train"), w.idx("val"), w.idx("test")
    Xs = pre.static(w.X); y = w.y
    out_dir = Path(c["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    mc_rows = np.searchsorted(te, w.mc_idx)  # positions of MC planets inside the test split

    def evaluate(p_test):
        r = classification_metrics(y[te], p_test)
        r["mc_mae"] = float(np.abs(p_test[mc_rows] - w.mc_p_true).mean())
        return r

    per_seed: dict[str, list[dict]] = {"logistic": [], "lightgbm": [], "mlp": []}
    preds0: dict[str, np.ndarray] = {}

    # --- Logistic regression (deterministic) ---
    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr]); p = lr.predict_proba(Xs[te])[:, 1]
    per_seed["logistic"] = [evaluate(p)]; preds0["logistic"] = p

    # --- MLP random search (validation BCE, seed 0) ---
    rng = np.random.default_rng(seeds[0]); Xt = torch.as_tensor(Xs, device=dev); yt = torch.as_tensor(y, device=dev)
    trt = torch.as_tensor(tr, device=dev); Xva, yva = Xt[va], yt[va]
    bce = torch.nn.BCEWithLogitsLoss()

    def train_mlp(hp, seed):
        set_seed(seed)
        m = MLP(Xs.shape[1], tuple(hp["hidden"]), hp["dropout"])
        bf = lambda i: ((Xt[trt[i.to(dev)]],), yt[trt[i.to(dev)]])
        m, hist = fit(m, bf, len(tr), lambda: bce(m(Xva), yva).item(), lr=hp["lr"], weight_decay=hp["weight_decay"],
                      batch_size=c["batch_size"], max_epochs=c["max_epochs"], patience=c["patience"], device=dev)
        return m, hist

    trials = []
    for t in range(c["search"]["n_trials"]):
        hp = {k: v[rng.integers(len(v))] for k, v in c["search"]["space"].items()}
        t0 = time.time(); _, hist = train_mlp(hp, seeds[0] + t)
        v = min(h[1] for h in hist)
        trials.append(dict(hp=hp, val_bce=v, epochs=len(hist), sec=time.time() - t0)); print("trial", t, hp, round(v, 4))
    best_hp = min(trials, key=lambda x: x["val_bce"])["hp"]

    # --- Retrain best configuration (MLP) and LightGBM for every seed ---
    for si, seed in enumerate(seeds):
        m, _ = train_mlp(best_hp, seed)
        p = _predict(m, Xs[te], dev); per_seed["mlp"].append(evaluate(p))
        torch.save(dict(state={k: v.cpu() for k, v in m.state_dict().items()}, hp=best_hp, d_in=Xs.shape[1], seed=seed),
                   out_dir / ("mlp_a.pt" if si == 0 else f"mlp_a_seed{seed}.pt"))
        gb = lgb.LGBMClassifier(**c["lgbm"], random_state=seed, verbose=-1)
        gb.fit(Xs[tr], y[tr], eval_set=[(Xs[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        pg = gb.predict_proba(Xs[te])[:, 1]; per_seed["lightgbm"].append(evaluate(pg))
        if si == 0:
            preds0["mlp"], preds0["lightgbm"] = p, pg
            pickle.dump(gb, open(out_dir / "lgbm_a.pkl", "wb"))
        print(f"seed {seed}: mlp auc {per_seed['mlp'][-1]['roc_auc']:.4f}  lgbm auc {per_seed['lightgbm'][-1]['roc_auc']:.4f}")

    # --- Aggregate ---
    results, stds = {}, {}
    for k, lst in per_seed.items():
        results[k], stds[k] = mean_std(lst)
        xs, ys, ns = calibration_curve(y[te], preds0[k]); results[k]["calibration"] = dict(pred=xs.tolist(), true=ys.tolist(), n=ns.tolist())
        print(k, {a: f"{results[k][a]:.4f}±{stds[k][a]:.4f}" for a in ("roc_auc", "pr_auc", "f1", "brier", "ece", "mc_mae")})
    boot = paired_bootstrap(y[te], preds0, [("mlp", "lightgbm"), ("mlp", "logistic")], B=c["bootstrap"], seed=seeds[0])
    print("bootstrap MLP-LGBM", boot["pairs"]["mlp_minus_lightgbm"])
    json.dump(dict(results=results, std=stds, per_seed=per_seed, seeds=seeds, bootstrap=boot, trials=trials, best_hp=best_hp,
                   n_train=len(tr), n_val=len(va), n_test=len(te)),
              open(out_dir / "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez(out_dir / "preds_test.npz", idx=te, y=y[te], mc_rows=mc_rows, mc_p_true=w.mc_p_true, **preds0)


if __name__ == "__main__":
    main()
