"""Loading, preprocessing (train-only statistics) and windowing of the planet dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOG_FACTORS = ("delta", "kappa")  # sampled log-uniformly -> modelled in log-space
K_PERCENTS = (0, 5, 10, 20, 40, 60)


@dataclass
class Worlds:
    X: np.ndarray; traj: np.ndarray; events: np.ndarray; y: np.ndarray; t_collapse: np.ndarray
    split: np.ndarray; mc_idx: np.ndarray; mc_p_true: np.ndarray; times: np.ndarray
    names: list[str]

    def idx(self, part: str) -> np.ndarray:
        return np.flatnonzero(self.split == {"train": 0, "val": 1, "test": 2}[part])


def load_worlds(path: str | Path = "data/worlds.npz") -> Worlds:
    d = np.load(path)
    return Worlds(d["X"], d["traj"], d["events"], d["y"].astype(np.float32), d["t_collapse"], d["split"],
                  d["mc_idx"], d["mc_p_true"], d["times"], [str(n) for n in d["names"]])


class Preprocessor:
    """Static factors -> log (for LOG_FACTORS) + z-score; sequences -> 5 channels + time, z-scored.
    All statistics are fitted on the train split only."""

    def __init__(self, w: Worlds):
        self.names = w.names
        self._log_cols = [w.names.index(n) for n in LOG_FACTORS]
        tr = w.idx("train")
        s = self._static_raw(w.X[tr]); self.s_mu, self.s_sd = s.mean(0), s.std(0) + 1e-12
        q = self._seq_raw(w.traj[tr], w.events[tr], w.times); flat = q.reshape(-1, q.shape[-1])
        self.q_mu, self.q_sd = flat.mean(0), flat.std(0) + 1e-12

    def _static_raw(self, X):
        X = X.astype(np.float64).copy(); X[:, self._log_cols] = np.log(X[:, self._log_cols]); return X

    @staticmethod
    def _seq_raw(traj, events, times):
        t = np.broadcast_to((times / times[-1])[None, :, None], traj.shape[:2] + (1,))
        ch = [np.log1p(traj[..., 0]), np.log1p(traj[..., 1]), traj[..., 2] / 100.0, np.log1p(traj[..., 3]),
              events.astype(np.float64)]
        return np.concatenate([np.stack(ch, -1), t], -1)

    def static(self, X) -> np.ndarray:
        return ((self._static_raw(X) - self.s_mu) / self.s_sd).astype(np.float32)

    def seq(self, traj, events, times) -> np.ndarray:
        return ((self._seq_raw(traj, events, times) - self.q_mu) / self.q_sd).astype(np.float32)


def alive_at(w: Worlds, idx: np.ndarray, m: int) -> np.ndarray:
    """Mask of planets (among `idx`) that have not collapsed by recorded step m (anti-leakage filter)."""
    return w.t_collapse[idx] > w.times[m]


def summary_features(traj: np.ndarray, events: np.ndarray, m: int) -> np.ndarray:
    """Hand-made window summaries for gradient boosting: last values, slopes, extrema, event count."""
    win = traj[:, : m + 1]
    pop = win[..., 0] + win[..., 1]
    back = max(m - 5, 0)
    last = win[:, -1]
    slope = (last - win[:, back]) / max(m - back, 1)
    feats = [last, slope, pop.max(1, keepdims=True), (pop[:, -1] / pop.max(1))[:, None], win[..., 2].min(1, keepdims=True),
             win[..., 3].max(1, keepdims=True), events[:, : m + 1].sum(1, keepdims=True)]
    return np.concatenate(feats, 1).astype(np.float32)
