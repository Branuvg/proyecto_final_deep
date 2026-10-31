# Simulador de mundos habitables: supervivencia con modelos sustitutos y XAI verificable

Proyecto final de Deep Learning (UVG, Sección 10). Un simulador de civilizaciones basado en **HANDY** (Motesharrei, Rivas & Kalnay, 2014), extendido con temperatura, agua y eventos extremos de Poisson, genera miles de planetas. Redes neuronales actúan como **modelos sustitutos** que predicen la probabilidad de supervivencia, y las explicaciones (SHAP, Integrated Gradients, TreeSHAP y contrafactuales) se **verifican contra el propio simulador**, que sí ofrece verdad de referencia.

- **Tarea (a):** condiciones iniciales → supervivencia (MLP; baselines: regresión logística y LightGBM).
- **Tarea (b):** primeros k % de la trayectoria → supervivencia (LSTM, alerta temprana).
- **Fidelidad de XAI:** ¿cambia el resultado como dice la explicación al intervenir un factor? Se mide en el modelo y en el simulador.
- **Contrafactuales** sobre palancas accionables, validados con el simulador, y una **interfaz Streamlit**.

## Instalación

Probado con Python 3.12 (recomendado). Funciona en Windows, Linux y macOS, **con o sin GPU NVIDIA**: el código usa CUDA si existe y, si no, CPU.

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt   # instala PyTorch para CPU (o el build por defecto de tu plataforma)
pip install -e .                  # registra el paquete worldsim (no hace falta PYTHONPATH)
```

**Con GPU NVIDIA**, instala antes PyTorch con CUDA y luego el resto (en una RTX 50xx / Blackwell hace falta CUDA >= 12.8):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Para generar el documento hay que colocar `template.docx` en la raíz (no se sube al repositorio).

**Reproducibilidad entre máquinas:** las semillas están fijas, pero CPU y GPU (o distintas versiones de librerías) dan resultados numéricos ligeramente distintos. Los números del documento corresponden a una corrida en una RTX 5060. En CPU el entrenamiento tarda bastante más (ver "Tiempos").

## Cómo correr cada etapa

Todos los comandos se ejecutan desde la raíz del repositorio (después de `pip install -e .`). Para correr todo en orden: `python scripts/run_all.py` (agrega `--skip-notebooks` para omitir los notebooks).

| Etapa | Comando |
|---|---|
| Tests | `pytest` |
| Dataset (50 000 planetas, ~10 s) | `python -m worldsim.data.generate --config configs/dataset.yaml` |
| Tarea (a) (búsqueda + 5 semillas + IC) | `python -m worldsim.train.run_task_a --config configs/task_a.yaml` |
| Tarea (b) (búsqueda + 5 semillas + IC) | `python -m worldsim.train.run_task_b --config configs/task_b.yaml` |
| Atribuciones XAI | `python -m worldsim.xai.run_xai --config configs/xai.yaml` |
| Fidelidad de XAI (modelo y simulador) | `python -m worldsim.xai.run_fidelity --config configs/fidelity.yaml` |
| Contrafactuales verificados | `python -m worldsim.xai.run_counterfactuals --config configs/counterfactual.yaml` |
| **Interfaz interactiva** | `streamlit run app/streamlit_app.py` |
| Notebooks ejecutados | `python scripts/build_notebooks.py` |
| Documento | `python report/build_report.py` → `report/Proyecto_Final.docx` |
| Presentación y guiones | `python report/build_presentation.py` y `python report/build_video_script.py` |

### Tiempos (referencia)

| Máquina | Pipeline completo (`run_all.py --skip-notebooks`) |
|---|---|
| RTX 5060 (CUDA) | ≈ 40 min medidos (dataset + tarea a ≈ 4 min; tarea (b) con búsqueda y 5 semillas ≈ 30 min; XAI, fidelidad y contrafactuales ≈ 10 min, en paralelo) |
| Solo CPU | varias horas; para probar, reduce `seeds` y `n_trials` en `configs/task_*.yaml` |

## Estructura

```
configs/      configuración por etapa (yaml)
src/worldsim/ simulator/ data/ models/ train/ xai/ reporting.py
app/          interfaz Streamlit
notebooks/    01 validación · 02 dataset y EDA · 03 modelos · 04 XAI · 05 fidelidad y contrafactuales (ejecutados)
report/       sections/*.md, build_report.py, build_presentation.py, build_video_script.py, figures/
docs/         PLAN.md · DECISIONES.md · PROMPTS.md · PRESENTACION.md · GUION_VIDEO.md
tests/        pruebas unitarias (simulador y XAI)
```

## Estado del proyecto

**Hecho**
- [x] Simulador HANDY vectorizado (RK4 y Euler) con extensiones, 11 factores marcados accionable/no accionable y definición formal de colapso.
- [x] **Ecuaciones verificadas contra el paper original** (Ecs. 3–6, Tabla 1, δ\*, χ_M). La verificación encontró un error de la Fase 1 (mortalidad de élites, Ec. 6); se corrigió con un test y se regeneró todo. *Los números de la Fase 1 quedaron obsoletos.*
- [x] Escenarios del paper reproducidos: equilibrio, oscilaciones, ciclos, colapso Tipo-L, Tipo-N con desigualdad y caso equitativo.
- [x] Dataset de 50 000 planetas (49.6 % de supervivencia), subconjunto Monte Carlo (500 × 32 réplicas).
- [x] Tarea (a) y (b) con baselines, búsqueda de hiperparámetros (también para la LSTM), **5 semillas** e **IC por bootstrap pareado**.
- [x] Atribuciones SHAP (exacto), IG y TreeSHAP; **validación de fidelidad** contra intervenciones en el modelo y en el simulador, curvas de eliminación y prueba de cordura.
- [x] Contrafactuales sobre palancas accionables, validados con el simulador.
- [x] Interfaz Streamlit (probada con `streamlit.testing`).
- [x] Documento completo, 5 notebooks ejecutados, presentación de 10 diapositivas y guiones para los tres hablantes.

**Pendiente / limitaciones conocidas**
- [ ] Umbral exacto del colapso Tipo-N igualitario del paper (4× reversible, 5.5× irreversible): con RK4 aparece desde ≈ 9.25×δ\*, con Euler dt = 1 desde ≈ 3.25×δ\*. No se igualó.
- [ ] Cotejar las referencias marcadas `[VERIFICAR]` en la bibliografía.
- [ ] Revisión visual del `.docx` y de la presentación en Word/PowerPoint (aquí solo se verificó por código).
- [ ] Grabar el video siguiendo `docs/GUION_VIDEO.md` y exportar la presentación a PDF.
- [ ] Efecto de un factor a la vez (sin interacciones); constantes de temperatura y agua sin calibración empírica.
