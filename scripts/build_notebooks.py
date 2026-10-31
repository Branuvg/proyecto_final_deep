"""Create the four notebooks from cell lists and execute them (outputs stay visible).
Run from the repo root: python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

SETUP = """import os, sys
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd())
sys.path.insert(0, "src")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.options.display.float_format = "{:.4f}".format"""

NB = {}
NB["01_simulador_validacion"] = [
    ("md", "# 01 · Validación del simulador HANDY\nReproducimos cualitativamente los escenarios canónicos de Motesharrei et al. (2014) antes de extender el modelo. "
           "Las ecuaciones están en `src/worldsim/simulator/handy.py` y las decisiones en `docs/DECISIONES.md`."),
    ("code", SETUP),
    ("code", "from pathlib import Path\nfrom worldsim.simulator.scenarios import plot_scenarios\nsummary = plot_scenarios(Path('report/figures/fig_escenarios_handy.png'))\n"
             "display(pd.DataFrame(summary).T)\ndisplay(Image('report/figures/fig_escenarios_handy.png'))"),
    ("md", "**Lectura.** Se reproducen el equilibrio, las oscilaciones, los ciclos de colapso, el colapso Tipo-L (la población muere, la naturaleza se recupera) y el Tipo-N con κ=100. "
           "El umbral del Tipo-N *igualitario* depende del integrador: RK4 desde ≈9.25×δ*, Euler (dt=1, float32, como el paper) desde ≈3.25×δ*; el paper reporta 4× reversible y 5.5× irreversible, entre ambos (ver D-03)."),
    ("code", "import subprocess\nprint(subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True).stdout[-300:])"),
]
NB["02_dataset_eda"] = [
    ("md", "# 02 · Dataset y análisis exploratorio\nSe generan 50 000 planetas (LHS) con `python -m worldsim.data.generate`. "
           "Aquí solo cargamos y exploramos, **antes de modelar**."),
    ("code", SETUP),
    ("code", "from pathlib import Path\nif not Path('data/worlds.npz').exists():\n    from worldsim.data import generate\n    import yaml\n    d = generate.generate(yaml.safe_load(open('configs/dataset.yaml', encoding='utf8'))); Path('data').mkdir(exist_ok=True); np.savez_compressed('data/worlds.npz', **d)\n"
             "from worldsim.data.dataset import load_worlds\nw = load_worlds()\nprint(w.X.shape, w.traj.shape, 'split train/val/test:', [int((w.split==i).sum()) for i in range(3)])"),
    ("code", "from worldsim.reporting import eda\nstats = eda(w)\nprint(stats)\ndisplay(Image('report/figures/fig_eda_clases_tiempos.png'))"),
    ("md", "## Trayectorias: 3 que sobreviven y 3 que colapsan\nLínea azul: población; verde discontinua: naturaleza; franjas naranjas: eventos extremos; línea roja: tiempo de colapso."),
    ("code", "display(Image('report/figures/fig_eda_trayectorias.png'))"),
    ("md", "## Frontera de decisión: supervivencia por decil de cada factor"),
    ("code", "display(Image('report/figures/fig_eda_supervivencia_por_factor.png'))"),
    ("md", "## Subconjunto Monte Carlo (probabilidad verdadera)\n500 planetas de test × 32 réplicas con distintas semillas de eventos."),
    ("code", "import matplotlib.pyplot as plt\nplt.hist(w.mc_p_true, bins=33); plt.xlabel('P(sobrevive) verdadera'); plt.ylabel('planetas'); plt.show()\n"
             "print('fracción con P en (0.05, 0.95):', stats['mc_share_intermediate'])"),
]
NB["03_models"] = [
    ("md", "# 03 · Modelos\nTarea (a): MLP vs. regresión logística y LightGBM (S1, S8). Tarea (b): LSTM de alerta temprana (S3) vs. baselines. "
           "Entrenamiento: `python -m worldsim.train.run_task_a` y `python -m worldsim.train.run_task_b`. Resultados: media ± desviación sobre 5 semillas."),
    ("code", SETUP),
    ("md", "## Tarea (a): condiciones iniciales → supervivencia (test, 7 500 planetas)"),
    ("code", "import json\nfrom worldsim.reporting import task_a_tables, task_a_bootstrap, task_b_tables, task_b_bootstrap, lstm_search_table\nta = task_a_tables(); display(ta)\n"
             "r = json.load(open('outputs/task_a/results.json')); print('semillas:', r['seeds'], '| mejores hiperparámetros MLP:', r['best_hp'])\n"
             "print('Diferencias pareadas (bootstrap, test, semilla 0):'); display(task_a_bootstrap())\n"
             "display(pd.DataFrame(r['trials']).sort_values('val_bce').head(5))\ndisplay(Image('report/figures/fig_tarea_a_calibracion.png'))"),
    ("md", "## Tarea (b): primeros k% de la trayectoria → supervivencia\nSolo se evalúan planetas aún vivos al final de la ventana; se reporta cuántos quedan fuera."),
    ("code", "auc, brier, excl = task_b_tables()\nprint('ROC-AUC'); display(auc); print('Brier'); display(brier); print('Planetas excluidos (ya colapsados en la ventana)'); display(excl)\ndisplay(Image('report/figures/fig_tarea_b_curva_k.png'))"),
    ("md", "### Diferencias pareadas con IC 95 % (bootstrap sobre los planetas de test, semilla 0) y búsqueda de hiperparámetros de la LSTM"),
    ("code", "display(task_b_bootstrap()); display(lstm_search_table())"),
]
NB["05_fidelidad_contrafactuales"] = [
    ("md", "# 05 · Fidelidad de las explicaciones y contrafactuales\nComparamos las atribuciones con el efecto de **intervenir cada factor** en el modelo y en el simulador "
           "(`python -m worldsim.xai.run_fidelity`) y validamos contrafactuales con el simulador (`python -m worldsim.xai.run_counterfactuals`)."),
    ("code", SETUP),
    ("code", "import json\nfrom worldsim.reporting_phase2 import fidelity_table, fidelity_figs, cf_table, cf_figs\nsanity = fidelity_figs()\ndisplay(fidelity_table())"),
    ("code", "display(Image('report/figures/fig_fidelidad_spearman.png')); display(Image('report/figures/fig_fidelidad_global.png'))"),
    ("md", "## Curvas de eliminación y dispersión"),
    ("code", "display(Image('report/figures/fig_fidelidad_eliminacion.png')); display(Image('report/figures/fig_fidelidad_dispersion.png', width=480))"),
    ("md", "## Prueba de cordura (pesos aleatorizados, Adebayo et al., 2018)"),
    ("code", "print('Spearman medio entre SHAP del modelo entrenado y del aleatorizado:', round(sanity['mean_spearman_trained_vs_randomised'], 3))\n"
             "print('Spearman vs. simulador — modelo entrenado:', round(sanity['vs_simulator_trained']['spearman'], 3), '| modelo aleatorizado:', round(sanity['vs_simulator_randomised']['spearman'], 3))"),
    ("md", "## Contrafactuales sobre palancas accionables, verificados con el simulador"),
    ("code", "share = cf_figs(); display(cf_table()); print('fracción de contrafactuales que modifican cada palanca:', {k: round(v, 2) for k, v in share.items()})\ndisplay(Image('report/figures/fig_contrafactuales.png'))"),
]
NB["04_xai"] = [
    ("md", "# 04 · XAI: atribuciones sobre el MLP de la tarea (a)\nBaseline único para todos los métodos: el planeta medio de train (vector cero estandarizado; log para δ y κ). "
           "Espacio de salida: logit de P(sobrevive). Cálculo: `python -m worldsim.xai.run_xai`. **La fidelidad contra el simulador queda para la Fase 2.**"),
    ("code", SETUP),
    ("code", "import json\nprint(json.dumps(json.load(open('outputs/xai/checks.json')), indent=1))\nd = np.load('outputs/xai/attributions.npz', allow_pickle=True)\nprint(str(d['baseline_definition']))\nprint({k: d[k].shape for k in d.files if k.startswith('phi')})"),
    ("code", "from worldsim.reporting import xai_summary\ns = xai_summary(); display(s['importance']); print(s['agreement']); print('fracción de |atribución| en factores accionables:', s['actionable_share'])\ndisplay(Image('report/figures/fig_xai_importancia_global.png'))"),
    ("md", "## Beeswarm (SHAP del MLP)"),
    ("code", "display(Image('report/figures/fig_xai_beeswarm.png'))"),
    ("md", "## Explicaciones locales (cascada): 2 planetas que sobreviven y 2 que colapsan"),
    ("code", "for i in range(1, 5): display(Image(f'report/figures/fig_xai_cascada_{i}.png', width=520))"),
]


def build():
    Path("notebooks").mkdir(exist_ok=True)
    for name, cells in NB.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": "notebooks"}})
        nbf.write(nb, f"notebooks/{name}.ipynb")
        print("executed", name)


if __name__ == "__main__":
    build()
