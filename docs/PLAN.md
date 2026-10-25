# Plan de la Fase 1

> Nota de honestidad: el plan se redactó con el diseño ya definido en `prompt_claude_code_fase1.md`; se escribió en el repositorio
> después de validar el simulador y no antes de codificar, como pedía el prompt. El orden de trabajo sí fue el indicado.

Orden de trabajo y entregables:

1. **Entorno**: venv con Python 3.12, PyTorch 2.11 + CUDA 12.8 (RTX 5060, sm_120 verificada), `requirements.txt` fijado, `.gitignore` (excluye `template.docx`, el compendio, `venv/`, datos, checkpoints).
2. **Simulador** (`src/worldsim/simulator/`): HANDY vectorizado con RK4, extensiones (temperatura, agua, eventos de Poisson), registro de 11 factores accionables/no accionables, definición formal de colapso, validación con escenarios canónicos y tests (`tests/`).
3. **Dataset** (`src/worldsim/data/`): LHS de 50 000 planetas, ajuste de rangos para 30–70 % de supervivencia, subconjunto Monte Carlo (500 × 32), split 70/15/15, notebook `02_dataset_eda`.
4. **Modelos** (`src/worldsim/models`, `train`): tarea (a) con logística, LightGBM y MLP (búsqueda aleatoria); tarea (b) con LSTM (truncamiento aleatorio de k), ablation sin parámetros, LightGBM con resúmenes y MLP sobre ventana; notebook `03_models`.
5. **XAI** (`src/worldsim/xai`): SHAP exacto, Integrated Gradients (captum) y TreeSHAP con el mismo baseline; atribuciones guardadas por planeta y factor; notebook `04_xai`.
6. **Documento** (`report/`): secciones en Markdown + `build_report.py` sobre la plantilla local.
7. **Cierre**: README con estado, `DECISIONES.md`, `PROMPTS.md` y resumen final.

Fuera de alcance (Fase 2): contrafactuales, validación de fidelidad contra el simulador, interfaz, presentación, video.
