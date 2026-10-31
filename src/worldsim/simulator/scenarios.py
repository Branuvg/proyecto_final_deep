"""Canonical HANDY scenarios used to validate the implementation (docs/DECISIONES.md D-03).

Parameters and initial conditions follow Table 1 and Sec. 5 of Motesharrei et al. (2014):
x_C(0)=100, y(0)=lambda, w(0)=0, kappa in {1, 10, 100}, x_E(0) in {0, 1e-3, 0.2, 25}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from .factors import BASELINE
from .handy import SimConfig, simulate

DELTA_OPT = 6.67e-6  # delta* of the egalitarian society, Eq. (10)

# name -> (parameter overrides, absolute initial populations or None, integrator, dtype, horizon)
SCENARIOS = {
    "Igualitario: aterrizaje suave (1x δ*)": (dict(delta=DELTA_OPT), None, "rk4", "float64", 1000),
    "Igualitario: oscilaciones (2.5x)": (dict(delta=2.5 * DELTA_OPT), None, "rk4", "float64", 1000),
    "Igualitario: ciclos de colapso (4x)": (dict(delta=4 * DELTA_OPT), None, "rk4", "float64", 1000),
    "Igualitario: Tipo-N irreversible (12x, RK4)": (dict(delta=12 * DELTA_OPT), None, "rk4", "float64", 1000),
    "Igualitario: Tipo-N irreversible (5.5x, Euler float32)": (dict(delta=5.5 * DELTA_OPT), None, "euler", "float32", 1000),
    "Equitativo (κ=1, x_E(0)=25)": (dict(delta=DELTA_OPT, kappa=1.0), dict(xc0=100.0, xe0=25.0), "rk4", "float64", 1000),
    "Desigual: Tipo-L (κ=100, x_E(0)=1e-3)": (dict(delta=DELTA_OPT, kappa=100.0), dict(xc0=100.0, xe0=1e-3), "rk4", "float64", 1500),
    "Desigual: Tipo-N (κ=100, δ=1e-4)": (dict(delta=1e-4, kappa=100.0), dict(xc0=100.0, xe0=0.2), "rk4", "float64", 1000),
}


def run_scenarios():
    out = {}
    for name, (over, init, integ, dtype, T) in SCENARIOS.items():
        cfg = SimConfig(horizon=T, record_every=5, integrator=integ, dtype=dtype)
        out[name] = simulate({**BASELINE, **over}, cfg, init=init)
    return out


def plot_scenarios(path: Path) -> dict:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run_scenarios()
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5))
    summary = {}
    for ax, (name, r) in zip(axes.ravel(), res.items()):
        xc, xe, y, w = (r.traj[0, :, i].astype(float) for i in range(4))
        ax.plot(r.times, xc, label="Plebeyos"); ax.plot(r.times, xe, label="Élites")
        ax.set_ylabel("Población"); ax.set_title(name, fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(r.times, y, "g--", label="Naturaleza"); ax2.plot(r.times, w / max(w.max(), 1e-9) * 100, "k:", label="Riqueza (norm.)")
        ax2.set_ylim(0, 105)
        if r.collapsed[0]:
            ax.axvline(r.t_collapse[0], color="r", alpha=.4)
        summary[name] = dict(collapsed=bool(r.collapsed[0]), t_collapse=float(r.t_collapse[0]),
                             pop_final=float(xc[-1] + xe[-1]), nature_final=float(y[-1]))
    for ax in axes[1]:
        ax.set_xlabel("Año")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = fig.axes[8].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return summary
