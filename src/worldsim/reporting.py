"""Figure and table helpers used by the notebooks (heavy logic lives here, notebooks only call it)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("report/figures")
LABELS = {"logistic": "Regresión logística", "lightgbm": "LightGBM", "mlp": "MLP",
          "lstm": "LSTM (con parámetros)", "lstm_no_static": "LSTM (sin parámetros)",
          "lightgbm_summary": "LightGBM (parámetros + resumen)", "mlp_window": "MLP (ventana aplanada)",
          "mlp_static_only": "MLP solo parámetros (tarea a)"}


def _save(fig, name, tight=True):
    FIG.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / name, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- EDA
def eda(w) -> dict:
    y = w.y; tc = w.t_collapse; out = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(["Colapso", "Supervivencia"], [(y == 0).mean(), (y == 1).mean()], color=["tab:red", "tab:green"])
    ax[0].set_ylabel("Fracción de planetas"); ax[0].set_title("Distribución de clases")
    for sp, name in enumerate(["train", "val", "test"]):
        out[f"surv_{name}"] = float(y[w.split == sp].mean())
    ax[1].hist(tc[y == 0], bins=40, color="tab:red", alpha=.8)
    ax[1].set_xlabel("Tiempo de colapso (años)"); ax[1].set_ylabel("Planetas"); ax[1].set_title("Tiempos de colapso (solo colapsados)")
    _save(fig, "fig_eda_clases_tiempos.png")
    out["t_collapse_quantiles"] = np.percentile(tc[y == 0], [5, 25, 50, 75, 95]).tolist()

    rng = np.random.default_rng(0)
    s_idx, c_idx = rng.choice(np.flatnonzero(y == 1), 3, replace=False), rng.choice(np.flatnonzero(y == 0), 3, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, i in zip(axes[0], s_idx):
        _traj_ax(ax, w, i, "Sobrevive")
    for ax, i in zip(axes[1], c_idx):
        _traj_ax(ax, w, i, "Colapsa")
    _save(fig, "fig_eda_trayectorias.png")

    X = pd.DataFrame(w.X, columns=w.names); X["surv"] = y
    fig, axes = plt.subplots(3, 4, figsize=(14, 7)); axes = axes.ravel()
    for ax, n in zip(axes, w.names):
        q = pd.qcut(X[n], 10, duplicates="drop")
        r = X.groupby(q, observed=True)["surv"].mean()
        ax.plot(range(len(r)), r.values, "o-"); ax.set_title(n); ax.set_ylim(0, 1); ax.set_xlabel("decil")
    axes[-1].axis("off"); fig.suptitle("Tasa de supervivencia por decil de cada factor")
    _save(fig, "fig_eda_supervivencia_por_factor.png")
    out["n"] = int(len(y)); out["survival_rate"] = float(y.mean())
    out["mc_p_true_mean"] = float(w.mc_p_true.mean())
    out["mc_share_intermediate"] = float(((w.mc_p_true > 0.05) & (w.mc_p_true < 0.95)).mean())
    return out


def _traj_ax(ax, w, i, title):
    tr = w.traj[i]; pop = tr[:, 0] + tr[:, 1]
    ax.plot(w.times, pop, label="Población"); ax.set_title(f"{title} (planeta {i})", fontsize=9)
    ax2 = ax.twinx(); ax2.plot(w.times, tr[:, 2], "g--", label="Naturaleza"); ax2.set_ylim(0, 105)
    ev = np.flatnonzero(w.events[i] > 0)
    for e in ev:
        ax.axvline(w.times[e], color="orange", alpha=.4)
    if w.y[i] == 0:
        ax.axvline(w.t_collapse[i], color="r")


# ---------------------------------------------------------------- Task (a)
def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def task_a_tables(res_path="outputs/task_a/results.json", pred_path="outputs/task_a/preds_test.npz") -> pd.DataFrame:
    d = json.load(open(res_path)); r, sd = d["results"], d["std"]; p = np.load(pred_path)
    rows = []
    for k, v in r.items():
        s = sd[k]
        rows.append({"Modelo": LABELS[k], "ROC-AUC": _pm(v["roc_auc"], s["roc_auc"]), "PR-AUC": _pm(v["pr_auc"], s["pr_auc"]), "F1": _pm(v["f1"], s["f1"]),
                     "Brier": _pm(v["brier"], s["brier"], 4), "ECE": _pm(v["ece"], s["ece"], 4), "MAE vs P_verdadera (MC)": _pm(v["mc_mae"], s["mc_mae"], 4)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot([0, 1], [0, 1], "k:")
    for k, v in r.items():
        ax[0].plot(v["calibration"]["pred"], v["calibration"]["true"], "o-", label=LABELS[k])
    ax[0].set_xlabel("Probabilidad predicha"); ax[0].set_ylabel("Frecuencia observada"); ax[0].set_title("Curva de calibración (test, semilla 0)")
    ax[0].legend(fontsize=8)
    pt = p["mc_p_true"]
    for k in ("logistic", "lightgbm", "mlp"):
        ax[1].scatter(pt + np.random.default_rng(0).normal(0, .004, len(pt)), p[k][p["mc_rows"]], s=7, alpha=.5, label=LABELS[k])
    ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set_xlabel("P(sobrevive) verdadera (32 réplicas MC)"); ax[1].set_ylabel("Probabilidad predicha")
    ax[1].set_title("Predicción vs. probabilidad verdadera"); ax[1].legend(fontsize=8)
    _save(fig, "fig_tarea_a_calibracion.png")
    return df


def task_a_bootstrap(res_path="outputs/task_a/results.json") -> pd.DataFrame:
    """Paired bootstrap 95% CIs for the differences between models (seed 0, test set)."""
    b = json.load(open(res_path))["bootstrap"]["pairs"]; names = {"mlp_minus_lightgbm": "MLP − LightGBM", "mlp_minus_logistic": "MLP − logística"}
    return pd.DataFrame([{"Comparación": names[k], "Δ ROC-AUC [IC 95 %]": f"{v['roc_auc'][1]:+.4f} [{v['roc_auc'][0]:+.4f}, {v['roc_auc'][2]:+.4f}]",
                          "Δ Brier [IC 95 %]": f"{v['brier'][1]:+.4f} [{v['brier'][0]:+.4f}, {v['brier'][2]:+.4f}]"} for k, v in b.items()])


# ---------------------------------------------------------------- Task (b)
def task_b_tables(res_path="outputs/task_b/results.json"):
    d = json.load(open(res_path)); R, S = d["results"], d["std"]; ks = d["k_percents"]
    names = [n for n in LABELS if n in R[str(ks[0])]]
    idx = [f"{k}%" for k in ks]
    auc = pd.DataFrame({LABELS[n]: [_pm(R[str(k)][n]["roc_auc"], S[str(k)][n]["roc_auc"]) for k in ks] for n in names}, index=idx)
    brier = pd.DataFrame({LABELS[n]: [_pm(R[str(k)][n]["brier"], S[str(k)][n]["brier"], 4) for k in ks] for n in names}, index=idx)
    excl = pd.DataFrame({"Planetas test": [R[str(k)]["n_test"] for k in ks], "Evaluados (vivos)": [R[str(k)]["n_test_alive"] for k in ks],
                         "Excluidos": [R[str(k)]["n_excluded"] for k in ks], "Supervivencia entre vivos": [R[str(k)]["pos_rate_alive"] for k in ks]}, index=idx)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for n in names:
        ax[0].errorbar(ks, [R[str(k)][n]["roc_auc"] for k in ks], yerr=[S[str(k)][n]["roc_auc"] for k in ks], fmt="o-", capsize=2, label=LABELS[n])
        ax[1].errorbar(ks, [R[str(k)][n]["brier"] for k in ks], yerr=[S[str(k)][n]["brier"] for k in ks], fmt="o-", capsize=2, label=LABELS[n])
    ax[0].set_ylabel("ROC-AUC"); ax[1].set_ylabel("Brier"); ax[0].set_title(f"Desempeño vs. % de trayectoria observada (media ± desv., {len(d['seeds'])} semillas)")
    ax[1].set_title("Brier (menor es mejor)")
    for a in ax:
        a.set_xlabel("k (% del horizonte observado)")
    ax[0].legend(fontsize=7)
    _save(fig, "fig_tarea_b_curva_k.png")
    return auc, brier, excl


def task_b_bootstrap(res_path="outputs/task_b/results.json") -> pd.DataFrame:
    d = json.load(open(res_path)); B = d["bootstrap"]; rows = []
    lab = {"lstm_minus_lightgbm_summary": "LSTM − LightGBM (resumen)", "lstm_minus_mlp_static_only": "LSTM − MLP solo parámetros",
           "lstm_minus_lstm_no_static": "LSTM − LSTM sin parámetros", "lstm_minus_mlp_window": "LSTM − MLP ventana"}
    for k in d["k_percents"]:
        row = {"k": f"{k}%"}
        for key, name in lab.items():
            v = B[str(k)]["pairs"][key]["roc_auc"]; row[name] = f"{v[1]:+.4f} [{v[0]:+.4f}, {v[2]:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def lstm_search_table(res_path="outputs/task_b/results.json") -> pd.DataFrame:
    d = json.load(open(res_path))
    return pd.DataFrame([{**t["hp"], "BCE val": t["val_bce"], "épocas": t["epochs"]} for t in d["trials"]]).sort_values("BCE val")


# ---------------------------------------------------------------- XAI
def xai_summary(path="outputs/xai/attributions.npz") -> dict:
    import shap
    d = np.load(path, allow_pickle=True); names = [str(n) for n in d["factor_names"]]
    A = {"SHAP (MLP)": d["phi_shap_mlp"], "Integrated Gradients (MLP)": d["phi_ig_mlp"], "TreeSHAP (LightGBM)": d["phi_treeshap_gb"]}
    imp = pd.DataFrame({k: np.abs(v).mean(0) for k, v in A.items()}, index=names).sort_values("SHAP (MLP)", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5)); imp.plot.barh(ax=ax); ax.invert_yaxis()
    ax.set_xlabel("mean |atribución| (logit de supervivencia)"); ax.set_title("Importancia global por método")
    _save(fig, "fig_xai_importancia_global.png")

    ex = shap.Explanation(values=d["phi_shap_mlp"], base_values=np.full(len(d["y"]), float(d["base_logit_mlp"])),
                          data=d["X_std"], feature_names=names)  # standardised values (raw ones span orders of magnitude)
    plt.figure(figsize=(9, 5)); shap.plots.beeswarm(ex, show=False, max_display=11); plt.title("SHAP (MLP): beeswarm (color = valor estandarizado)")
    _save(plt.gcf(), "fig_xai_beeswarm.png")

    rng = np.random.default_rng(1); y = d["y"]
    pick = list(rng.choice(np.flatnonzero(y == 1), 2, replace=False)) + list(rng.choice(np.flatnonzero(y == 0), 2, replace=False))
    for j, i in enumerate(pick):
        plt.figure(figsize=(8, 4.5)); shap.plots.waterfall(ex[i], show=False, max_display=11)
        plt.title(f"Planeta {int(d['planet_idx'][i])}: {'sobrevive' if y[i] else 'colapsa'}; logit={d['logit_mlp'][i]:.2f} (valores estandarizados)", fontsize=10)
        _save(plt.gcf(), f"fig_xai_cascada_{j+1}.png")

    def per_planet_rho(a, b):
        return float(np.nanmean([spearmanr(a[i], b[i])[0] for i in range(0, len(a), 5)]))
    agree = dict(
        spearman_global_shap_vs_ig=float(spearmanr(imp.iloc[:, 0], imp.iloc[:, 1])[0]),
        spearman_global_shap_vs_tree=float(spearmanr(imp.iloc[:, 0], imp.iloc[:, 2])[0]),
        mean_local_spearman_shap_vs_ig=per_planet_rho(d["phi_shap_mlp"], d["phi_ig_mlp"]),
        mean_local_spearman_shap_mlp_vs_tree=per_planet_rho(d["phi_shap_mlp"], d["phi_treeshap_gb"]),
    )
    act = d["actionable"]
    share = {k: float(np.abs(v)[:, act].sum() / np.abs(v).sum()) for k, v in A.items()}
    return dict(importance=imp, agreement=agree, actionable_share=share)
