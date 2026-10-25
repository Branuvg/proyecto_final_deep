"""Task (a): initial conditions -> survival. Baselines (logistic, LightGBM) + MLP with a small random search.

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


def _predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, device=device))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/task_a.yaml")
    c = yaml.safe_load(open(ap.parse_args().config, encoding="utf8"))
    set_seed(c["seed"]); dev = get_device()
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    w = load_worlds(c["data"]); pre = Preprocessor(w)
    tr, va, te = w.idx("train"), w.idx("val"), w.idx("test")
    Xs = pre.static(w.X); y = w.y
    out_dir = Path(c["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    mc_rows = np.searchsorted(te, w.mc_idx)  # positions of MC planets inside the test split
    results, preds = {}, {}

    def evaluate(name, p_test):
        r = classification_metrics(y[te], p_test)
        r["mc_mae"] = float(np.abs(p_test[mc_rows] - w.mc_p_true).mean())
        results[name] = r; preds[name] = p_test
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})

    # --- Baselines ---
    lr = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr]); evaluate("logistic", lr.predict_proba(Xs[te])[:, 1])
    gb = lgb.LGBMClassifier(**c["lgbm"], random_state=c["seed"], verbose=-1)
    gb.fit(Xs[tr], y[tr], eval_set=[(Xs[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    evaluate("lightgbm", gb.predict_proba(Xs[te])[:, 1])
    pickle.dump(gb, open(out_dir / "lgbm_a.pkl", "wb"))

    # --- MLP random search (validation BCE) ---
    rng = np.random.default_rng(c["seed"]); Xt = torch.as_tensor(Xs, device=dev); yt = torch.as_tensor(y, device=dev)
    trt = torch.as_tensor(tr, device=dev); Xva, yva = Xt[va], yt[va]
    bce = torch.nn.BCEWithLogitsLoss()
    trials = []
    for t in range(c["search"]["n_trials"]):
        hp = {k: v[rng.integers(len(v))] for k, v in c["search"]["space"].items()}
        set_seed(c["seed"] + t)
        m = MLP(Xs.shape[1], tuple(hp["hidden"]), hp["dropout"])
        bf = lambda i: ((Xt[trt[i.to(dev)]],), yt[trt[i.to(dev)]])
        t0 = time.time()
        m, hist = fit(m, bf, len(tr), lambda: bce(m(Xva), yva).item(), lr=hp["lr"], weight_decay=hp["weight_decay"],
                      batch_size=c["batch_size"], max_epochs=c["max_epochs"], patience=c["patience"], device=dev)
        v = min(h[1] for h in hist)
        trials.append(dict(hp=hp, val_bce=v, epochs=len(hist), sec=time.time() - t0)); print("trial", t, hp, round(v, 4))
        if t == 0 or v < min(x["val_bce"] for x in trials[:-1]):
            best_state, best_hp = {k: s.clone() for k, s in m.state_dict().items()}, hp
    best = MLP(Xs.shape[1], tuple(best_hp["hidden"]), best_hp["dropout"]).to(dev); best.load_state_dict(best_state)
    evaluate("mlp", _predict(best, Xs[te], dev))
    torch.save(dict(state=best_state, hp=best_hp, d_in=Xs.shape[1]), out_dir / "mlp_a.pt")

    # --- Save ---
    for k, p in preds.items():
        xs, ys, ns = calibration_curve(y[te], p); results[k]["calibration"] = dict(pred=xs.tolist(), true=ys.tolist(), n=ns.tolist())
    json.dump(dict(results=results, trials=trials, best_hp=best_hp, n_train=len(tr), n_val=len(va), n_test=len(te)),
              open(out_dir / "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez(out_dir / "preds_test.npz", idx=te, y=y[te], mc_rows=mc_rows, mc_p_true=w.mc_p_true, **preds)


if __name__ == "__main__":
    main()
