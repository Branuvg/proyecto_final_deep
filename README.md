# Simulador de mundos habitables: supervivencia con modelos sustitutos y XAI verificable

Proyecto final de Deep Learning (UVG, Sección 10). Un simulador de civilizaciones basado en **HANDY** (Motesharrei, Rivas & Kalnay, 2014), extendido con temperatura, agua y eventos extremos de Poisson, genera miles de planetas. Redes neuronales actúan como **modelos sustitutos** que predicen la probabilidad de supervivencia, y la explicabilidad (SHAP, Integrated Gradients, y contrafactuales en la Fase 2) se **verifica contra el propio simulador**, que sí ofrece verdad de referencia.

- **Tarea (a):** condiciones iniciales → supervivencia (MLP; baselines: regresión logística y LightGBM).
- **Tarea (b):** primeros k % de la trayectoria → supervivencia (LSTM, alerta temprana).

## Instalación

Requiere Python 3.12 y, para GPU, una NVIDIA con CUDA ≥ 12.8 (la RTX 5060 es Blackwell, sm_120).

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

El código elige CUDA si existe y, si no, CPU (también corre en Colab). Hay que colocar `template.docx` en la raíz solo para generar el documento (no se sube al repositorio).

## Cómo correr cada etapa

Todos los comandos se ejecutan desde la raíz del repositorio con `PYTHONPATH=src` (en PowerShell: `$env:PYTHONPATH="src"`).

| Etapa | Comando |
|---|---|
| Tests del simulador | `pytest` |
| Validación (escenarios HANDY) | `python -c "from pathlib import Path; from worldsim.simulator.scenarios import plot_scenarios; plot_scenarios(Path('report/figures/fig_escenarios_handy.png'))"` |
| Dataset (50 000 planetas, ~10 s) | `python -m worldsim.data.generate --config configs/dataset.yaml` |
| Tarea (a) | `python -m worldsim.train.run_task_a --config configs/task_a.yaml` |
| Tarea (b) | `python -m worldsim.train.run_task_b --config configs/task_b.yaml` |
| Atribuciones XAI | `python -m worldsim.xai.run_xai --config configs/xai.yaml` |
| Notebooks ejecutados | `python scripts/build_notebooks.py` |
| Documento | `python report/build_report.py` → `report/Proyecto_Final.docx` |

## Estructura

```
configs/      configuración por etapa (yaml)
src/worldsim/ simulator/ data/ models/ train/ xai/ reporting.py
notebooks/    01 validación · 02 dataset y EDA · 03 modelos · 04 XAI (ejecutados)
report/       sections/*.md, build_report.py, figures/
docs/         PLAN.md · DECISIONES.md · PROMPTS.md
tests/        pruebas unitarias del simulador
```

## Estado del proyecto (Fase 1)

**Hecho**
- [x] Simulador HANDY vectorizado (RK4) con extensiones, 11 factores marcados accionable/no accionable y definición formal de colapso; 8 tests pasan.
- [x] Validación con escenarios canónicos (equilibrio, oscilaciones, ciclos de colapso, colapso irreversible por desigualdad).
- [x] Dataset de 50 000 planetas (63.3 % de supervivencia), subconjunto Monte Carlo (500 × 32 réplicas), split 70/15/15 y normalización solo con train.
- [x] Tarea (a): logística, LightGBM y MLP con búsqueda de hiperparámetros.
- [x] Tarea (b): LSTM con truncamiento aleatorio, ablación sin parámetros y dos baselines, evaluada sin fuga.
- [x] Atribuciones SHAP (exacto), Integrated Gradients y TreeSHAP guardadas por planeta y factor.
- [x] Notebooks ejecutados y documento (secciones 1–4 completas, 5 y 6 parciales, 7 esqueleto, 8 bibliografía).

**Pendiente**
- [ ] Verificar las ecuaciones y parámetros de HANDY contra el PDF original (`[VERIFICAR]`, ver `docs/DECISIONES.md` D-02) y las referencias marcadas.
- [ ] Colapso irreversible de la naturaleza (Tipo-N) no reproducido a 5.5×δ_opt.
- [ ] Varias semillas e intervalos de confianza; búsqueda de hiperparámetros para la LSTM.
- [ ] Fase 2: validación de fidelidad de XAI contra intervenciones (modelo y simulador), contrafactuales, interfaz Streamlit, presentación y video.
- [ ] Completar la sección 7 del documento.
- [ ] Revisión visual del `.docx` en Word (no se pudo renderizar aquí).
