"""Canonical HANDY scenarios used to validate the implementation (docs/DECISIONES.md D-03)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from .factors import BASELINE
from .handy import SimConfig, simulate

DELTA_OPT = 6.67e-6  # optimal depletion factor of the egalitarian baseline (y* = lam/2)

SCENARIOS = {
    "Equilibrio (igualitario, 1x delta_opt)": dict(delta=DELTA_OPT),
    "Oscilaciones y equilibrio (2.5x)": dict(delta=2.5 * DELTA_OPT),
    "Ciclos de colapso (4x)": dict(delta=4 * DELTA_OPT),
    "Ciclos de colapso más severos (5.5x)": dict(delta=5.5 * DELTA_OPT),
    "Equitativo con élites (kappa=1)": dict(elite_frac=0.2),
    "Colapso irreversible por desigualdad (kappa=6)": dict(kappa=6.0, elite_frac=0.2),
}


def run_scenarios(horizon: int = 1000):
    out = {}
    for name, over in SCENARIOS.items():
        p = {**BASELINE, **over}
        out[name] = simulate(p, SimConfig(horizon=horizon, record_every=5))
    return out


def plot_scenarios(path: Path, horizon: int = 1000) -> dict:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run_scenarios(horizon)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    summary = {}
    for ax, (name, r) in zip(axes.ravel(), res.items()):
        xc, xe, y, w = (r.traj[0, :, i] for i in range(4))
        ax.plot(r.times, xc, label="Plebeyos"); ax.plot(r.times, xe, label="Élites")
        ax.set_ylabel("Población"); ax.set_title(name, fontsize=10)
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
    h2, l2 = fig.axes[6].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return summary
