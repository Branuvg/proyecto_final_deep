"""Task (b): survival from the first k% of the trajectory (early warning).

Models: LSTM (with static factors / ablation without), LightGBM(params + window summaries),
MLP on the flattened window, and the static-only MLP of task (a) as reference.
Only planets still alive at the end of the observed window are evaluated (anti-leakage).

Run: python -m worldsim.train.run_task_b --config configs/task_b.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import yaml

from ..data.dataset import K_PERCENTS, Preprocessor, alive_at, load_worlds, summary_features
from ..models.nets import MLP, LSTMNet
from .loop import fit, get_device, set_seed
from .metrics import classification_metrics


def train_lstm(w, pre, Xs, Q, tr, va, c, static_mode, dev, seed):
    set_seed(seed)
    L = Q.shape[1]; m_max = c["lstm"]["m_max_train"]
    Qt, St, yt = torch.as_tensor(Q, device=dev), torch.as_tensor(Xs, device=dev), torch.as_tensor(w.y, device=dev)
    tcol = torch.as_tensor(w.t_collapse, device=dev); times = torch.as_tensor(w.times, device=dev)
    trt, vat = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    m_val = torch.randint(0, m_max + 1, (len(va),), generator=g).to(dev)  # fixed random truncation for validation
    bce = torch.nn.BCEWithLogitsLoss()
    net = LSTMNet(Q.shape[2], Xs.shape[1], c["lstm"]["hidden"], c["lstm"]["layers"], c["lstm"]["dropout"], static_mode)

    def batch(rows, m):
        keep = tcol[rows] > times[m]                           # still alive at the end of the window
        rows, m = rows[keep], m[keep]
        T = int(m.max()) + 1
        return (Qt[rows, :T], St[rows], m + 1), yt[rows]

    def bf(i):
        rows = trt[i.to(dev)]
        m = torch.randint(0, m_max + 1, (len(rows),), device=dev)  # random truncation of k each time
        return batch(rows, m)

    def vf():
        inp, t = batch(vat, m_val)
        return bce(net(*inp), t).item()

    net, hist = fit(net, bf, len(tr), vf, lr=c["lstm"]["lr"], weight_decay=c["lstm"]["weight_decay"],
                    batch_size=c["lstm"]["batch_size"], max_epochs=c["lstm"]["max_epochs"], patience=c["lstm"]["patience"], device=dev)
    return net, len(hist)


def lstm_predict(net, Q, Xs, rows, m, dev):
    net.eval()
    with torch.no_grad():
        q = torch.as_tensor(Q[rows, : m + 1], device=dev); s = torch.as_tensor(Xs[rows], device=dev)
        lens = torch.full((len(rows),), m + 1)
        return torch.sigmoid(net(q, s, lens)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/task_b.yaml")
    c = yaml.safe_load(open(ap.parse_args().config, encoding="utf8"))
    set_seed(c["seed"]); dev = get_device()
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    w = load_worlds(c["data"]); pre = Preprocessor(w)
    tr, va, te = w.idx("train"), w.idx("val"), w.idx("test")
    Xs = pre.static(w.X); Q = pre.seq(w.traj, w.events, w.times); y = w.y
    out_dir = Path(c["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    n_steps = len(w.times) - 1
    ms = [int(round(k / 100 * n_steps)) for k in K_PERCENTS]

    nets = {}
    for mode in ("concat", "none"):
        t0 = time.time(); nets[mode], ep = train_lstm(w, pre, Xs, Q, tr, va, c, mode, dev, c["seed"])
        print(f"LSTM static_mode={mode}: {ep} epochs, {time.time()-t0:.0f}s")
        torch.save(nets[mode].state_dict(), out_dir / f"lstm_{mode}.pt")
    mlp_a = torch.load(Path(c["task_a_dir"]) / "mlp_a.pt", map_location="cpu")
    hp = mlp_a["hp"]; net_a = MLP(mlp_a["d_in"], tuple(hp["hidden"]), hp["dropout"]).to(dev); net_a.load_state_dict(mlp_a["state"]); net_a.eval()

    results = {}
    for k, m in zip(K_PERCENTS, ms):
        a_tr, a_va, a_te = tr[alive_at(w, tr, m)], va[alive_at(w, va, m)], te[alive_at(w, te, m)]
        r = dict(n_test=len(te), n_test_alive=len(a_te), n_excluded=len(te) - len(a_te), pos_rate_alive=float(y[a_te].mean()))
        P = {}
        P["lstm"] = lstm_predict(nets["concat"], Q, Xs, a_te, m, dev)
        P["lstm_no_static"] = lstm_predict(nets["none"], Q, Xs, a_te, m, dev)
        with torch.no_grad():
            P["mlp_static_only"] = torch.sigmoid(net_a(torch.as_tensor(Xs[a_te], device=dev))).cpu().numpy()
        # gradient boosting: static + window summaries
        F = lambda rows: np.concatenate([Xs[rows], summary_features(w.traj[rows], w.events[rows], m)], 1)
        gb = lgb.LGBMClassifier(**c["lgbm"], random_state=c["seed"], verbose=-1)
        gb.fit(F(a_tr), y[a_tr], eval_set=[(F(a_va), y[a_va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        P["lightgbm_summary"] = gb.predict_proba(F(a_te))[:, 1]
        # MLP over [static, flattened window]
        Fm = lambda rows: np.concatenate([Xs[rows], Q[rows, : m + 1].reshape(len(rows), -1)], 1).astype(np.float32)
        set_seed(c["seed"])
        mw = MLP(Fm(a_tr[:1]).shape[1], (128, 64), 0.2)
        Xa, ya = torch.as_tensor(Fm(a_tr), device=dev), torch.as_tensor(y[a_tr], device=dev)
        Xv, yv = torch.as_tensor(Fm(a_va), device=dev), torch.as_tensor(y[a_va], device=dev)
        bce = torch.nn.BCEWithLogitsLoss()
        mw, _ = fit(mw, lambda i: ((Xa[i.to(dev)],), ya[i.to(dev)]), len(a_tr), lambda: bce(mw(Xv), yv).item(),
                    lr=1e-3, weight_decay=1e-2, batch_size=512, max_epochs=150, patience=12, device=dev)
        with torch.no_grad():
            P["mlp_window"] = torch.sigmoid(mw(torch.as_tensor(Fm(a_te), device=dev))).cpu().numpy()
        for name, p in P.items():
            r[name] = classification_metrics(y[a_te], p)
        results[str(k)] = r
        print(f"k={k}% alive {len(a_te)}/{len(te)}  " + "  ".join(f"{n}={r[n].get('roc_auc', float('nan')):.3f}" for n in P))
    json.dump(dict(k_percents=list(K_PERCENTS), steps=ms, results=results), open(out_dir / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
