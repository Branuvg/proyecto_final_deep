# Proyecto Final Deep Learning — Fase 1

## Contexto

Somos un grupo de 3 estudiantes de la Universidad del Valle de Guatemala (curso Deep Learning, Sección 10). Este repo es nuestro proyecto final. Por ahora contiene:

- `Proyecto_Final_Deep_Learning.md`: enunciado y rúbrica. Léelo completo antes de empezar.
- `deep-learning-compendio.md`: resumen de todo lo visto en el curso (S1–S15). Úsalo para conectar cada técnica con la sesión donde se vio, porque el catedrático lo valora.
- `template.docx`: plantilla del documento final.

`template.docx` y `deep-learning-compendio.md` **no deben subirse al repo**; son solo contexto. Agrégalos al `.gitignore` al inicio. Incluye también ahí `venv/`, `__pycache__/`, los datos generados pesados y los checkpoints. `build_report.py` debe leer la plantilla localmente. Si no la encuentra, que falle con un mensaje claro que diga dónde colocarla.

Fechas: entrega el 1 de noviembre de 2026 y presentación el 2 de noviembre. **Esta es la Fase 1 de dos.** Quiero un avance significativo y sólido, no todo el proyecto.

### El tema (ya aprobado)

Un simulador de mundos habitables. El programa genera planetas con distintas condiciones: temperatura, agua, alimentos/recursos, eventos extremos y otros factores. Simula cómo evolucionan sus recursos y su población. Una red neuronal predice si la civilización sobrevive. Con XAI explicamos qué factores influyeron en cada predicción y **verificamos con el propio simulador** si esas explicaciones son ciertas.

### Enfoque que decidimos (respétalo)

- **Encuadre del "problema real":** las redes se usan como *modelos sustitutos (surrogates)* de sistemas dinámicos caros de simular (clima, epidemiología, modelos basados en agentes). Además, en XAI casi nunca hay ground truth. Aquí sí lo hay, porque el simulador es el mecanismo real. **El eje del proyecto es medir qué tan fieles son las explicaciones**, no solo predecir.
- **Simulador:** basado en **HANDY** (Motesharrei, Rivas & Kalnay, 2014, *Ecological Economics*). Se extiende con temperatura, agua y eventos extremos estocásticos (proceso de Poisson). Por la estocasticidad, las mismas condiciones iniciales pueden dar resultados distintos. La red predice una **probabilidad** de supervivencia.
- **Tarea (a):** supervivencia al tiempo T a partir de las condiciones iniciales del planeta (MLP).
- **Tarea (b):** supervivencia viendo los primeros *k* pasos de la trayectoria (LSTM), como alerta temprana. Se mide cómo mejora la predicción conforme ve más historia.
- **Baselines obligatorios:** regresión logística y gradient boosting. Tenemos que demostrar que la red aporta algo; si no aporta, se reporta honestamente.
- **XAI:** SHAP, Integrated Gradients y, en la Fase 2, contrafactuales. Se validan contra intervenciones reales en el simulador.

## Entorno

- **Windows**, Intel Core Ultra 9, **NVIDIA RTX 5060**. Es arquitectura Blackwell (sm_120) y necesita PyTorch con **CUDA ≥ 12.8**. Instala el PyTorch estable más reciente con wheels CUDA ≥ 12.8 según pytorch.org, por ejemplo `--index-url https://download.pytorch.org/whl/cu128` o superior. Verifica `torch.cuda.is_available()` y `torch.cuda.get_device_name(0)` antes de seguir. Si la GPU no se detecta, detente y dime qué pasó; no sigas en CPU sin avisar.
- Usa un entorno virtual (`venv`) y un `requirements.txt` con las versiones fijadas.
- El código debe ser agnóstico al dispositivo para poder correrlo también en Colab más adelante.
- Por ser Windows: usa la guarda `if __name__ == "__main__":` en los scripts con DataLoader o multiprocessing, y usa rutas con `pathlib`.
- **Idioma:** código, nombres y comentarios en **inglés**. Documentación (`README`, `docs/`) y el documento final en **español**. Con tuteo, no voseo.
- Stack: PyTorch, con las arquitecturas escritas por nosotros (`nn.Module` propios, no un modelo de alto nivel que lo resuelva todo). Además numpy/pandas, scikit-learn, `shap`, `captum`, matplotlib y `python-docx`. Agrega lo que necesites, justificándolo.
- Reproducibilidad: semillas fijas, configuración en `configs/*.yaml` y un comando por etapa.

## Alcance de esta fase

Haz esto, en este orden:

### 1. Simulador (`src/<paquete>/simulator/`)

- Implementa HANDY con sus 4 variables de estado: commoners, elites, nature y wealth, con sus ecuaciones de consumo, umbral de riqueza y mortalidad dependiente del hambre. **Verifica las ecuaciones contra el paper** y cita la ecuación correspondiente en los docstrings.
- **Validación:** reproduce cualitativamente los escenarios canónicos del paper (equilibrio, oscilaciones, ciclos de colapso y colapso irreversible) y grafícalos. Esto demuestra que la implementación es correcta antes de extenderla.
- **Extensiones** (documenta cada decisión y su justificación):
  - La temperatura afecta la regeneración de la naturaleza y la mortalidad.
  - La disponibilidad de agua limita la capacidad de carga.
  - Los eventos extremos ocurren como proceso de Poisson, con tasa y severidad como parámetros, y destruyen una fracción aleatoria de nature y/o wealth.
  - Puedes añadir 1 o 2 factores si aportan, pero mantén alrededor de 8 a 12 factores en total.
- **Marca cada factor como accionable** (por ejemplo desigualdad, tasa de agotamiento) **o no accionable** (temperatura, agua, tasa de eventos). Esto se usa en los contrafactuales de la Fase 2.
- Integración numérica estable (RK4 o equivalente) y **vectorizada para simular miles de planetas en lote**.
- **Define formalmente "colapso" y "supervivencia".** Por ejemplo: colapso si la población total cae por debajo de una fracción del máximo alcanzado o del valor inicial; supervivencia si no hay colapso hasta T. Guarda también el tiempo de colapso.
- Tests unitarios básicos: conservación y no negatividad, determinismo con la misma semilla y efecto de los eventos.

### 2. Dataset (`src/<paquete>/data/`, `notebooks/02_dataset_eda.ipynb`)

- Muestrea los parámetros con Latin Hypercube o uniforme, en rangos elegidos para que la tasa de supervivencia quede aproximadamente entre 30% y 70% y que la frontera no sea trivial. Si no queda balanceado, ajusta los rangos y documenta la iteración.
- Tamaño orientativo: unos 50k planetas. La generación debe tardar minutos, no horas.
- Guarda los parámetros, las trayectorias submuestreadas, el indicador de eventos, la etiqueta y el tiempo de colapso en un formato eficiente (parquet o npz).
- **Subconjunto Monte Carlo:** para unos 500 planetas del test, corre aproximadamente 32 réplicas con distintas semillas de eventos. Así se obtiene la **probabilidad verdadera** P(sobrevive | parámetros). Sirve para evaluar la calibración y como ground truth de XAI.
- Split 70/15/15 por planeta, con la normalización ajustada solo con train.
- **Antes de modelar**, el notebook debe mostrar la distribución de clases, la distribución de tiempos de colapso y al menos 3 trayectorias que sobreviven y 3 que colapsan.

### 3. Modelos (`src/<paquete>/models/`, `train/`, `notebooks/03_models.ipynb`)

**Tarea (a), condiciones iniciales → supervivencia:**

- Baselines: regresión logística y gradient boosting.
- MLP con regularización (dropout, AdamW, early stopping). Conéctalo con S8 en la documentación.
- Haz una búsqueda pequeña de hiperparámetros.
- Métricas: ROC-AUC, PR-AUC, F1, Brier y ECE/curva de calibración. En el subconjunto Monte Carlo, también el MAE entre la probabilidad predicha y la verdadera.

**Tarea (b), primeros k pasos → supervivencia:**

- LSTM que recibe la trayectoria parcial y los parámetros estáticos. Por ejemplo, los parámetros pueden ir como estado inicial o concatenados a cada paso; justifica la opción que elijas.
- Entrena con truncamiento aleatorio de k, para que un solo modelo sirva para cualquier k.
- Evalúa con k ∈ {0, 5, 10, 20, 40, 60}% del horizonte.
- **Evita la fuga trivial:** evalúa solo los planetas que siguen vivos al final de la ventana observada. Reporta aparte cuántos quedan fuera.
- Baselines para (b): gradient boosting con parámetros más features resumen de la ventana (últimos valores, pendientes) y un MLP sobre la ventana aplanada.
- Ablation: LSTM con y sin parámetros estáticos.
- Tabla comparativa y curva de desempeño vs. k.

### 4. XAI: cálculo de atribuciones (`src/<paquete>/xai/`, `notebooks/04_xai.ipynb`)

**Sobre el MLP de la tarea (a):**

- SHAP (KernelSHAP o GradientExplainer) e Integrated Gradients (captum).
- La baseline o el background es una decisión nuestra: documéntala y usa la misma definición en todos los métodos.
- Como comparación, TreeSHAP sobre el gradient boosting.
- Genera las explicaciones locales de algunos planetas (gráfico de cascada) y la importancia global (mean |SHAP|, beeswarm).

**La validación de fidelidad contra el simulador queda para la Fase 2.** Aun así, deja el código de XAI preparado para conectarla. Las atribuciones deben guardarse en disco por planeta y por factor, con la definición de baseline explícita. En la Fase 2 se van a comparar con el efecto de intervenir cada factor:

- en el modelo, para medir la fidelidad al modelo;
- en el simulador, con las réplicas Monte Carlo, para medir la fidelidad a la realidad.

### 5. Documento (`report/`)

- Escribe las secciones en Markdown en `report/sections/*.md`, en español. Crea un script `report/build_report.py` que genere `report/Proyecto_Final.docx` a partir de `template.docx`.
- **Carátula:** conserva exactamente el formato de la carátula de la plantilla (fuentes, tamaños, espaciado, línea). Cambia solo estos datos:
  - "Data Science" → **Deep Learning**
  - "Sección 30" → **Sección 10**
  - "Laboratorio X:" → **Proyecto Final:**
  - "Titulo" → **Simulador de mundos habitables: predicción de supervivencia con modelos sustitutos y explicabilidad verificable**. Es provisional; deja el título en una sola variable para cambiarlo fácil.
  - Integrantes, en este orden:
    - David Domínguez, 23712
    - Jonathan Díaz, 23837
    - Gabriel Bran, 23590
  - "Repo:" → déjalo con un placeholder hasta que tengamos el link.
  - Conserva "Facultad de Ingeniería" y "2026".
- **Contenido:** borra el contenido de ejemplo que viene después de la carátula (el de YouTube). Usa los mismos estilos de la plantilla: títulos de nivel 1 como "# 1. Sección", nivel 2 como "## 1.1", y el mismo estilo de tabla. Inspecciona los estilos del `.docx` con python-docx en lugar de inventarlos. Las figuras van numeradas, con pie de figura.
- **Secciones** (las pide la rúbrica):
  1. Descripción del problema
  2. Análisis
  3. Propuesta de solución
  4. Descripción de la solución
  5. Herramientas aplicadas. Incluye el historial de prompts de IA: qué funcionó, qué falló y por qué.
  6. Resultados (métricas)
  7. Conclusión
  8. Bibliografía
- **En esta fase redacta completas las secciones 1 a 4.** La sección 5 va parcial, a partir de `docs/PROMPTS.md`. La sección 6 lleva resultados preliminares **solo con números reales salidos de las corridas**; nunca inventes ni redondees a ojo. La sección 7 queda como esqueleto.
- **Bibliografía en APA, solo con referencias reales y verificables.** No inventes citas. Si no puedes verificar una referencia, márcala como `[VERIFICAR]`. Mínimo: Motesharrei et al. 2014 (HANDY), Hochreiter & Schmidhuber 1997, Lundberg & Lee 2017, Sundararajan et al. 2017, Adebayo et al. 2018, Loshchilov & Hutter 2019 y una referencia de surrogate modeling / análisis de sensibilidad (por ejemplo Saltelli).
- Conecta cada técnica con la sesión del curso donde se vio (S3 LSTM, S8 regularización, S15 XAI, etc.).

## Fuera de alcance en esta fase

No hagas presentación, slides, video, interfaz (Streamlit va en la Fase 2), contrafactuales ni validación de fidelidad de XAI.

## Forma de trabajar

1. Antes de codificar, escribe un plan breve en `docs/PLAN.md` y **después continúa sin esperar mi aprobación**.
2. Mantén actualizados, en español:
   - `docs/DECISIONES.md`: cada decisión de diseño con su justificación y las alternativas descartadas.
   - `docs/PROMPTS.md`: los prompts relevantes que te dimos y qué resultó. La rúbrica pide este historial.
3. **No hagas ningún commit ni push.** Yo te voy a indicar después cómo quiero manejar los commits.
4. Mantén el `README.md` (en español) con:
   - una descripción breve del proyecto;
   - cómo instalar el entorno y correr cada etapa;
   - una sección de **estado del proyecto** con lo que ya está hecho y lo que falta. Actualízala al terminar cada etapa.
5. Si algo no funciona, por ejemplo que la red no supere al gradient boosting, **documéntalo honestamente**. No ajustes el experimento hasta que "salga bien".
6. Los notebooks deben quedar ejecutados, con las salidas visibles. La lógica pesada va en `src/` y los notebooks solo la llaman.
7. **Al terminar**, dame un resumen con:
   - Lo que quedó hecho.
   - La tabla de métricas principales.
   - Las atribuciones XAI preliminares.
   - Los problemas encontrados.
   - Lo que propones para la Fase 2.
