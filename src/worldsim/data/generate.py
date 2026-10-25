"""Generate the planet dataset (python -m worldsim.data.generate --config configs/dataset.yaml)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from ..simulator.factors import NAMES
from ..simulator.handy import SimConfig, simulate
from .sampling import sample_params


def _sim_cfg(c: dict, seed: int) -> SimConfig:
    return SimConfig(horizon=c["horizon"], dt=c["dt"], record_every=c["record_every"],
                     collapse_frac=c["collapse_frac"], floor_frac=c["floor_frac"], seed=seed)


def generate(c: dict) -> dict:
    n, bs, seed = c["n_planets"], c["batch_size"], c["seed"]
    ranges = {k: tuple(v) for k, v in c["ranges"].items()}
    params = sample_params(n, seed, ranges)
    X = np.stack([params[k] for k in NAMES], 1)

    trajs, evs, cols, tcs = [], [], [], []
    for b, i in enumerate(range(0, n, bs)):
        sl = slice(i, min(i + bs, n))
        r = simulate({k: v[sl] for k, v in params.items()}, _sim_cfg(c, seed * 1000 + b))
        trajs.append(r.traj.astype(np.float32)); evs.append(r.events.astype(np.int8))
        cols.append(r.collapsed); tcs.append(r.t_collapse)
    traj, events = np.concatenate(trajs), np.concatenate(evs)
    collapsed, t_col = np.concatenate(cols), np.concatenate(tcs)

    # Split 70/15/15 by planet
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr, n_va = int(c["split"][0] * n), int(c["split"][1] * n)
    split = np.empty(n, np.int8)
    split[perm[:n_tr]], split[perm[n_tr:n_tr + n_va]], split[perm[n_tr + n_va:]] = 0, 1, 2

    # Monte Carlo subset: replicates with different event seeds -> true P(survive | params)
    test_idx = np.flatnonzero(split == 2)
    mc_idx = rng.choice(test_idx, size=c["mc_planets"], replace=False)
    R = c["mc_replicates"]
    rep_params = {k: np.tile(v[mc_idx], R) for k, v in params.items()}   # replicate-major layout
    rr = simulate(rep_params, _sim_cfg(c, seed * 1000 + 999))
    surv = (~rr.collapsed).reshape(R, -1)                                # (R, mc)
    mc_p, mc_n = surv.mean(0), R

    return dict(X=X, names=np.array(NAMES), traj=traj, events=events, y=(~collapsed).astype(np.int8),
                t_collapse=t_col.astype(np.float32), split=split, mc_idx=mc_idx,
                mc_p_true=mc_p.astype(np.float32), mc_replicates=np.array(mc_n),
                times=np.arange(traj.shape[1]) * c["record_every"] * c["dt"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/dataset.yaml")
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config, encoding="utf8"))
    t = time.time(); d = generate(c)
    out = Path(c["output"]); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"{len(d['y'])} planets in {time.time()-t:.1f}s -> {out}; survival={d['y'].mean():.3f}; "
          f"MC p_true mean={d['mc_p_true'].mean():.3f}")


if __name__ == "__main__":
    main()
