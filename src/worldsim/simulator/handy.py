"""Vectorised HANDY simulator (Motesharrei, Rivas & Kalnay, 2014) with extensions.

Base model (equations as reproduced in Patry et al. 2024, arXiv:2407.14860, Sec. I.1; the
numbering of the original paper must be checked against the PDF, see docs/DECISIONES.md D-02):

  (a) dxC/dt = beta_C xC - alpha_C xC
  (b) dxE/dt = beta_E xE - alpha_E xE
  (c) dy/dt  = gamma y (lam - y) - delta xC y          (regenerative nature)
  (d) dw/dt  = delta xC y - C_C - C_E
  consumption  C_C = min(1, w/w_th) s xC ;  C_E = min(1, w/w_th) kappa s xE
  threshold    w_th = rho xC + kappa rho xE
  mortality    alpha_C = alpha_m + max(0, 1 - C_C/(s xC)) (alpha_M - alpha_m)
               alpha_E = alpha_m + max(0, 1 - C_E/(kappa s xE)) (alpha_M - alpha_m)

Because C_E/(kappa s xE) = C_C/(s xC) = min(1, w/w_th), both groups share the same famine term;
elites differ only through their larger share of w_th and of consumption.

Extensions (all switched off by BASELINE): temperature, water, Poisson extreme events.
State layout: (N, 4) = [commoners, elites, nature, wealth]; N planets integrated in one batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factors import CONSTANTS, EXT, NAMES

XC, XE, Y, W = range(4)


@dataclass
class SimConfig:
    horizon: int = 600          # years
    dt: float = 1.0             # RK4 step, years
    record_every: int = 6       # store one point every `record_every` steps
    collapse_frac: float = 0.1  # collapse if P < collapse_frac * running max of P ...
    floor_frac: float = 0.2     # ... or P < floor_frac * P(0) (slow decline from the start)
    seed: int = 0


@dataclass
class SimResult:
    traj: np.ndarray        # (N, n_rec, 4) states at recorded times
    events: np.ndarray      # (N, n_rec) number of extreme events in each recorded interval
    collapsed: np.ndarray   # (N,) bool
    t_collapse: np.ndarray  # (N,) years; = horizon if never collapsed
    times: np.ndarray       # (n_rec,)


def _as_arrays(params: dict) -> dict[str, np.ndarray]:
    out = {k: np.atleast_1d(np.asarray(params[k], dtype=np.float64)) for k in NAMES}
    n = max(len(v) for v in out.values())
    return {k: np.broadcast_to(v, (n,)).copy() for k, v in out.items()}


def derivatives(state: np.ndarray, p: dict[str, np.ndarray], eff: dict[str, np.ndarray]) -> np.ndarray:
    """Right-hand side of the system, vectorised over planets. `eff` holds temperature-adjusted rates."""
    xc, xe, y, w = state[:, XC], state[:, XE], state[:, Y], state[:, W]
    aM, rho, lam = CONSTANTS["alpha_M"], CONSTANTS["rho"], CONSTANTS["lam"]
    kappa, s = p["kappa"], p["s"]

    # Water extension: demand grows with population, capacity shrinks when supply is short.
    demand = (xc + xe) * (1.0 - p["water_eff"]) / CONSTANTS["p_ref"]
    lam_eff = lam * np.clip(p["water"] / np.maximum(demand, 1e-12), 0.05, 1.0)

    w_th = rho * (xc + kappa * xe)                                   # w_th
    fed = np.minimum(1.0, w / np.maximum(w_th, 1e-12))               # min(1, w/w_th)
    cc, ce = fed * s * xc, fed * kappa * s * xe                      # C_C, C_E
    famine = np.maximum(0.0, 1.0 - fed)
    a_c = eff["alpha_m"] + famine * (aM - eff["alpha_m"])            # alpha_C (= alpha_E, see module doc)

    d = np.empty_like(state)
    d[:, XC] = (p["beta"] - a_c) * xc
    d[:, XE] = (p["beta"] - a_c) * xe
    d[:, Y] = eff["gamma"] * y * (lam_eff - y) - p["delta"] * xc * y
    d[:, W] = p["delta"] * xc * y - cc - ce
    return d


def _effective_rates(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Temperature extension: heat lowers regeneration (Gaussian in anomaly) and raises baseline mortality."""
    t = p["temp"]
    return dict(
        gamma=p["gamma"] * np.exp(-((t / EXT["temp_tol"]) ** 2)),
        alpha_m=CONSTANTS["alpha_m"] * (1.0 + EXT["temp_mort"] * np.abs(t)),
    )


def rk4_step(state, p, eff, dt):
    k1 = derivatives(state, p, eff)
    k2 = derivatives(state + 0.5 * dt * k1, p, eff)
    k3 = derivatives(state + 0.5 * dt * k2, p, eff)
    k4 = derivatives(state + dt * k3, p, eff)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def simulate(params: dict, cfg: SimConfig | None = None) -> SimResult:
    """Simulate a batch of planets. `params` maps each factor name to a scalar or an (N,) array."""
    cfg = cfg or SimConfig()
    p = _as_arrays(params)
    n = len(p["delta"])
    rng = np.random.default_rng(cfg.seed)
    eff = _effective_rates(p)

    pop0 = CONSTANTS["pop0"]
    state = np.zeros((n, 4))
    state[:, XE] = pop0 * p["elite_frac"]
    state[:, XC] = pop0 - state[:, XE]
    state[:, Y] = CONSTANTS["lam"]   # nature starts at capacity
    state[:, W] = 0.0                # no accumulated wealth

    n_steps = int(round(cfg.horizon / cfg.dt))
    n_rec = n_steps // cfg.record_every + 1
    traj = np.empty((n, n_rec, 4)); traj[:, 0] = state
    events = np.zeros((n, n_rec), dtype=np.int16)
    pop_max = state[:, XC] + state[:, XE]
    collapsed = np.zeros(n, bool)
    t_col = np.full(n, float(cfg.horizon))
    ev_acc = np.zeros(n, dtype=np.int16)

    for i in range(1, n_steps + 1):
        state = rk4_step(state, p, eff, cfg.dt)
        # Extreme events: Poisson(rate*dt) arrivals; each destroys random fractions of nature and wealth.
        n_ev = rng.poisson(p["event_rate"] * cfg.dt)
        if n_ev.any():
            for m in range(int(n_ev.max())):
                hit = n_ev > m
                u = rng.random((n, 2))
                state[:, Y] *= np.where(hit, 1.0 - p["event_sev"] * u[:, 0], 1.0)
                state[:, W] *= np.where(hit, 1.0 - p["event_sev"] * u[:, 1], 1.0)
            ev_acc += n_ev.astype(np.int16)
        np.maximum(state, 0.0, out=state)  # guard against tiny negative overshoot
        pop = state[:, XC] + state[:, XE]
        newly = (~collapsed) & (pop < np.maximum(cfg.collapse_frac * pop_max, cfg.floor_frac * pop0))
        t_col[newly] = i * cfg.dt
        collapsed |= newly
        pop_max = np.maximum(pop_max, pop)
        if i % cfg.record_every == 0:
            j = i // cfg.record_every
            traj[:, j] = state
            events[:, j] = ev_acc
            ev_acc[:] = 0
    times = np.arange(n_rec) * cfg.record_every * cfg.dt
    return SimResult(traj, events, collapsed, t_col, times)
