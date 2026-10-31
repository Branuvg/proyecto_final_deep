"""Figures and tables for the Phase 2 results: XAI fidelity and counterfactuals (called from the notebooks)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .reporting import _save, plt

GROUP_LABEL = {"mc": "Planetas Monte Carlo (aleatorios)", "boundary": "Planetas cerca de la frontera"}


def fidelity_table(path="outputs/fidelity/results.json") -> pd.DataFrame:
    d = json.load(open(path)); rows = []
    for g, gr in d["groups"].items():
        for m, v in gr["methods"].items():
            for ref, key in (("simulador", "vs_simulator"), ("modelo", "vs_model")):
                x = v[key]
                rows.append({"Grupo": GROUP_LABEL[g], "Método": m, "Referencia": ref, "Planetas": x["n_planets"],
                             "Spearman [IC 95 %]": f"{x['spearman']:.2f} [{x['spearman_ci'][0]:.2f}, {x['spearman_ci'][1]:.2f}]",
                             "Signo": f"{x['sign_agreement']:.0%}", "Top-1": f"{x['top1']:.0%}", "Top-3": f"{x['top3_overlap']:.0%}"})
    return pd.DataFrame(rows)


def fidelity_figs(path="outputs/fidelity/results.json", arr_path="outputs/fidelity/arrays.npz") -> dict:
    d = json.load(open(path)); A = np.load(arr_path, allow_pickle=True)
    meths = list(next(iter(d["groups"].values()))["methods"])
    # (a) Spearman bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (g, gr) in zip(axes, d["groups"].items()):
        xs = np.arange(len(meths)); wd = 0.38
        for o, (key, lab, col) in enumerate((("vs_model", "vs. modelo", "tab:gray"), ("vs_simulator", "vs. simulador (verdad)", "tab:orange"))):
            v = [gr["methods"][m][key]["spearman"] for m in meths]
            err = [[gr["methods"][m][key]["spearman"] - gr["methods"][m][key]["spearman_ci"][0] for m in meths],
                   [gr["methods"][m][key]["spearman_ci"][1] - gr["methods"][m][key]["spearman"] for m in meths]]
            ax.bar(xs + (o - .5) * wd, v, wd, yerr=err, capsize=3, label=lab, color=col)
        ax.axhline(gr["model_vs_simulator_effect"]["spearman"], color="tab:blue", ls="--", lw=1, label="el propio MLP vs. simulador")
        ax.set_xticks(xs); ax.set_xticklabels([m.replace(" (", "\n(") for m in meths], fontsize=8); ax.set_title(GROUP_LABEL[g], fontsize=10)
        ax.axhline(0, color="k", lw=.5)
    axes[0].set_ylabel("Spearman por planeta (media, IC 95 %)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Fidelidad: ¿concuerdan las atribuciones con el efecto de intervenir cada factor?")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig_fidelidad_spearman.png", tight=False)
    # (b) deletion curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); ks = np.arange(12)
    for ax, key, yl in ((axes[0], "model_curve", "logit medio del MLP"), (axes[1], "sim_curve", "P(sobrevive) verdadera (simulador)")):
        for m, col in zip(meths, ("tab:blue", "tab:green", "tab:red")):
            ax.plot(ks, d["deletion"][m]["attribution"][key], "-o", ms=3, color=col, label=f"{m}: orden por atribución")
        ax.plot(ks, d["deletion"][meths[0]]["random"][key], "k--", label="orden aleatorio")
        ax.plot(ks, d["deletion"][meths[0]]["global_importance"][key], ":", color="gray", label="importancia global")
        ax.set_xlabel("factores devueltos al valor base (más favorables primero)"); ax.set_ylabel(yl)
    axes[0].set_title("Eliminación en el modelo"); axes[1].set_title("Eliminación en el simulador"); axes[0].legend(fontsize=7)
    _save(fig, "fig_fidelidad_eliminacion.png")
    # (c) global importance: true effect vs attributions
    gi = d["global_importance"]; names = gi["factors"]
    norm = lambda v: np.array(v) / np.sum(v)
    order = np.argsort(-norm(gi["sim_mean_abs_effect"]))
    fig, ax = plt.subplots(figsize=(10, 4)); xs = np.arange(len(names)); wd = 0.2
    for o, (k, lab, col) in enumerate((("sim_mean_abs_effect", "Efecto real (simulador)", "tab:orange"), ("shap_mean_abs", "SHAP (MLP)", "tab:blue"),
                                       ("ig_mean_abs", "IG (MLP)", "tab:green"), ("tree_mean_abs", "TreeSHAP", "tab:red"))):
        ax.bar(xs + (o - 1.5) * wd, norm(gi[k])[order], wd, label=lab, color=col)
    ax.set_xticks(xs); ax.set_xticklabels([names[i] for i in order], rotation=35, fontsize=8); ax.set_ylabel("importancia relativa (suma = 1)")
    ax.set_title("Importancia global: efecto real vs. atribuciones"); ax.legend(fontsize=8)
    _save(fig, "fig_fidelidad_global.png")
    # (d) scatter phi vs true effect
    phi = np.concatenate([A["boundary_phi_shap"], A["mc_phi_shap"]]); dsim = np.concatenate([A["boundary_d_sim"], A["mc_d_sim"]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2)); cm = plt.get_cmap("tab20")
    for j, n in enumerate(names):
        ax.scatter(phi[:, j], dsim[:, j], s=5, alpha=.35, color=cm(j), label=n)
    ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
    ax.set_xlabel("atribución SHAP (logit)"); ax.set_ylabel("efecto real en P(sobrevive) al devolver el factor al valor base")
    ax.legend(fontsize=6, ncol=2, markerscale=2); ax.set_title("Cada punto = (planeta, factor)")
    _save(fig, "fig_fidelidad_dispersion.png")
    return d["sanity_randomised_model"]


def cf_table(path="outputs/counterfactuals/results.json") -> pd.DataFrame:
    d = json.load(open(path)); rows = []
    for tag, lab in (("gradient", "Gradiente (varias palancas)"), ("single_factor", "Una sola palanca (rejilla)")):
        r = d[tag]
        rows.append({"Método": lab, "Resuelto": f"{r['solved_share']:.0%}", "Válido en el modelo": f"{r['model_valid']:.0%}",
                     "Válido en el simulador": f"{r['simulator_valid']:.0%}" if r["simulator_valid"] is not None else "—",
                     "P real antes → después": f"{r['mean_true_p_before']:.2f} → {r['mean_true_p_after']:.2f}",
                     "Palancas cambiadas": f"{r['mean_factors_changed']:.1f}", "Rango movido (suma)": f"{r['mean_range_fraction_moved']:.2f}"})
    return pd.DataFrame(rows)


def cf_figs(path="outputs/counterfactuals/results.json", arr_path="outputs/counterfactuals/arrays.npz"):
    d = json.load(open(path)); A = np.load(arr_path, allow_pickle=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    tags = [("gradient", "Gradiente"), ("single_factor", "Una palanca")]
    xs = np.arange(2); wd = 0.35
    ax[0].bar(xs - wd / 2, [d[t]["model_valid"] for t, _ in tags], wd, label="en el modelo", color="tab:gray")
    ax[0].bar(xs + wd / 2, [d[t]["simulator_valid"] or 0 for t, _ in tags], wd, label="en el simulador (verdad)", color="tab:orange")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([lab for _, lab in tags]); ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("fracción de contrafactuales válidos"); ax[0].set_title("¿Funcionan los contrafactuales?"); ax[0].legend()
    tc = A["p_true_before"] < 0.5
    ax[1].hist(A["p_true_before"][tc], bins=20, alpha=.6, label="antes", color="tab:red")
    ax[1].hist(A["p_true_after_gradient"][tc], bins=20, alpha=.6, label="después (gradiente)", color="tab:green")
    ax[1].set_xlabel("P(sobrevive) verdadera (simulador)"); ax[1].set_ylabel("planetas")
    ax[1].set_title("Efecto real de aplicar el contrafactual"); ax[1].legend()
    _save(fig, "fig_contrafactuales.png")
    return d["gradient"]["changed_share_by_factor"]
