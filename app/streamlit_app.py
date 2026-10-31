"""Simulador de mundos habitables: interfaz interactiva.

Run (from the repo root, after `pip install -e .` and generating data + training task (a)):
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from worldsim.data.dataset import Preprocessor, load_worlds
from worldsim.models.nets import MLP
from worldsim.simulator.factors import ACTIONABLE, FACTORS
from worldsim.simulator.handy import SimConfig, simulate
from worldsim.xai.attributions import exact_shapley
from worldsim.xai.counterfactual import gradient_counterfactual, z_to_raw
from worldsim.xai.interventions import baseline_raw, sim_survival

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f.name for f in FACTORS]
LABEL = {"delta": "δ · depleción de la naturaleza", "kappa": "κ · desigualdad (consumo de élites)", "s": "s · consumo per cápita",
         "water_eff": "eficiencia hídrica", "beta": "β · natalidad", "gamma": "γ · regeneración de la naturaleza",
         "temp": "anomalía térmica (°C)", "water": "oferta de agua", "event_rate": "eventos extremos (por año)",
         "event_sev": "severidad de los eventos", "elite_frac": "fracción inicial de élites"}
PRESETS = {
    "Planeta medio": None,
    "Equilibrio (δ cerca de δ*, sin élites ni eventos)": dict(delta=6.67e-6, kappa=1.0, s=5e-4, water_eff=0.3, beta=0.03, gamma=0.01, temp=0.0, water=1.2, event_rate=0.0, event_sev=0.1, elite_frac=0.0),
    "Sobreexplotación (δ alto)": dict(delta=5e-5, kappa=1.0, s=5e-4, water_eff=0.3, beta=0.03, gamma=0.01, temp=0.0, water=1.2, event_rate=0.0, event_sev=0.1, elite_frac=0.0),
    "Desigualdad alta con élites": dict(delta=1.5e-5, kappa=3.0, s=5e-4, water_eff=0.3, beta=0.03, gamma=0.01, temp=0.0, water=1.2, event_rate=0.0, event_sev=0.1, elite_frac=0.1),
    "Mundo caliente con eventos": dict(delta=1.5e-5, kappa=1.5, s=5e-4, water_eff=0.1, beta=0.03, gamma=0.01, temp=4.5, water=0.4, event_rate=0.1, event_sev=0.8, elite_frac=0.03),
}


@st.cache_resource(show_spinner="Cargando datos y modelo…")
def load_assets():
    data, ck = ROOT / "data/worlds.npz", ROOT / "outputs/task_a/mlp_a.pt"
    if not data.exists() or not ck.exists():
        return None
    w = load_worlds(data); pre = Preprocessor(w)
    c = torch.load(ck, map_location="cpu"); hp = c["hp"]
    net = MLP(c["d_in"], tuple(hp["hidden"]), hp["dropout"]); net.load_state_dict(c["state"]); net.eval()
    return w, pre, net, baseline_raw(pre)


def main():
    st.set_page_config(page_title="Simulador de mundos habitables", layout="wide")
    st.title("🌍 Simulador de mundos habitables")
    st.caption("HANDY extendido (Motesharrei et al., 2014) · red neuronal sustituta · explicaciones verificadas con el simulador")
    assets = load_assets()
    if assets is None:
        st.error("Faltan `data/worlds.npz` o `outputs/task_a/mlp_a.pt`. Ejecuta primero: "
                 "`python -m worldsim.data.generate` y `python -m worldsim.train.run_task_a`.")
        st.stop()
    w, pre, net, base = assets

    # ---------------- sidebar: planet factors ----------------
    st.sidebar.header("Condiciones del planeta")
    preset = st.sidebar.selectbox("Escenario", list(PRESETS))
    start = dict(zip(NAMES, base)) if PRESETS[preset] is None else PRESETS[preset]
    x = {}
    for f in FACTORS:
        tag = "🔧 " if f.actionable else "🌡️ "
        x[f.name] = st.sidebar.slider(tag + LABEL[f.name], float(f.low), float(f.high), float(np.clip(start[f.name], f.low, f.high)),
                                      step=float((f.high - f.low) / 200), format="%.2e" if f.high < 0.01 else "%.3f", key=f"{preset}-{f.name}")
    st.sidebar.caption("🔧 accionable (palanca de política) · 🌡️ no accionable (dato del entorno)")
    R = st.sidebar.slider("Réplicas del simulador", 8, 128, 48, 8)
    X_raw = np.array([[x[n] for n in NAMES]])

    tab_sim, tab_xai, tab_cf = st.tabs(["Simulación", "Explicación (SHAP)", "Contrafactual"])

    # ---------------- simulation ----------------
    params = {n: np.repeat(X_raw[:, j], R) for j, n in enumerate(NAMES)}
    res = simulate(params, SimConfig(horizon=600, record_every=6, seed=1))
    p_true = float((~res.collapsed).mean())
    z = pre.static(X_raw); logit = float(net(torch.as_tensor(z)).item()); p_model = 1 / (1 + np.exp(-logit))

    with tab_sim:
        c1, c2, c3 = st.columns(3)
        c1.metric("P(sobrevivir) — simulador", f"{p_true:.0%}", help=f"Fracción de {R} réplicas que no colapsan en 600 años")
        c2.metric("P(sobrevivir) — red neuronal", f"{p_model:.0%}", delta=f"{(p_model - p_true) * 100:+.0f} pts vs. simulador")
        tc = res.t_collapse[res.collapsed]
        c3.metric("Año de colapso (mediana)", f"{np.median(tc):.0f}" if len(tc) else "—")
        pop = res.traj[:, :, 0] + res.traj[:, :, 1]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for a, data, title, color in ((ax[0], pop, "Población total", "tab:blue"), (ax[1], res.traj[:, :, 2], "Naturaleza", "tab:green")):
            a.fill_between(res.times, np.percentile(data, 10, 0), np.percentile(data, 90, 0), alpha=.25, color=color)
            a.plot(res.times, np.median(data, 0), color=color); a.set_title(title + " (mediana y percentiles 10–90)"); a.set_xlabel("año")
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Línea: mediana de {R} réplicas con distintos eventos extremos; banda: percentiles 10–90. "
                   f"Colapso = población < max(10 % del pico, 20 % de la inicial).")

    # ---------------- explanation ----------------
    with tab_xai:
        st.write("Valores de Shapley **exactos** de la red (en logits), respecto al *planeta medio de entrenamiento*. "
                 "Positivo = el factor empuja hacia la supervivencia.")
        phi = exact_shapley(lambda t: net(t), z, np.zeros(z.shape[1], dtype=np.float32), torch.device("cpu"))[0]
        order = np.argsort(np.abs(phi))
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh([("🔧 " if ACTIONABLE[NAMES[i]] else "🌡️ ") + NAMES[i] for i in order], phi[order], color=["tab:green" if phi[i] > 0 else "tab:red" for i in order])
        ax.axvline(0, color="k", lw=.6); ax.set_xlabel("atribución SHAP (logit)")
        st.pyplot(fig); plt.close(fig)
        st.info("Una explicación convincente no es necesariamente fiel. En el proyecto la comparamos con intervenciones reales en el simulador "
                "(ver resultados de fidelidad). Aquí puedes comprobarlo: mueve el factor más importante y mira cómo cambia la simulación.")

    # ---------------- counterfactual ----------------
    with tab_cf:
        st.write("Cambio mínimo en las **palancas accionables** (🔧) para que el modelo prediga ≥ 80 % de supervivencia; "
                 "después se verifica con el simulador.")
        if st.button("Buscar contrafactual"):
            zc = gradient_counterfactual(net, z.astype(np.float32), pre, NAMES, target_p=0.8)
            xc = z_to_raw(pre, zc.copy()); pc_model = 1 / (1 + np.exp(-float(net(torch.as_tensor(zc)).item())))
            pc_true = float(sim_survival(xc, NAMES, R, 7)[0])
            rows = [(n, x[n], float(xc[0, j])) for j, n in enumerate(NAMES) if ACTIONABLE[n] and abs(xc[0, j] - x[n]) > 0.01 * abs(x[n])]
            m1, m2 = st.columns(2)
            m1.metric("Red neuronal", f"{p_model:.0%} → {pc_model:.0%}"); m2.metric("Simulador (verificación)", f"{p_true:.0%} → {pc_true:.0%}")
            if rows:
                st.table({"Factor": [LABEL[n] for n, _, _ in rows], "Actual": [f"{a:.3g}" for _, a, _ in rows], "Propuesto": [f"{b:.3g}" for _, _, b in rows]})
            else:
                st.success("No hace falta cambiar nada: el modelo ya predice supervivencia.")
            if pc_true < 0.5 <= pc_model:
                st.warning("El modelo cree que el contrafactual funciona pero el simulador no lo confirma: el sustituto se equivoca fuera de su zona de confianza.")


if __name__ == "__main__":
    main()
