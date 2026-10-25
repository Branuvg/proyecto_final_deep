# Decisiones de diseño

Cada decisión incluye su justificación y las alternativas descartadas.

## D-01 · Paquete, entorno y GPU
- Paquete `worldsim` en `src/`; Python 3.12 (la 3.14 instalada en el sistema aún tiene soporte irregular de `shap`/`torch`).
- PyTorch 2.11.0 con wheels `cu128` (necesario para Blackwell sm_120). Verificado: `torch.cuda.is_available() == True`, GPU `NVIDIA GeForce RTX 5060 Laptop GPU`.
- El código usa `get_device()` (CUDA si existe, si no CPU) para poder correr en Colab. Rutas con `pathlib`.
- **LightGBM** se agregó como gradient boosting: es rápido, compatible con TreeSHAP y evita depender de XGBoost. Alternativa descartada: `GradientBoostingClassifier` de sklearn (lento con 35 000 filas × muchas pruebas).

## D-02 · Ecuaciones de HANDY y su verificación
- Implementadas las cuatro ecuaciones de estado, consumo `C_C, C_E`, umbral `w_th = ρ x_C + κ ρ x_E` y mortalidad `α = α_m + max(0, 1 − C/(s x))(α_M − α_m)`.
- **Verificación realizada:** no pude abrir el PDF original (el sitio con la copia devolvió error de certificado). Contrasté las ecuaciones con la reproducción literal de Patry et al. (2024, arXiv:2407.14860, Sec. I.1), que las cita del paper original, y con el comportamiento esperado: con `δ_opt = 6.67e-6` el equilibrio coincide con la solución analítica (`y* = λ/2 = 50`, `x_C* = γ(λ−y*)/δ = 75 000`; test `test_equilibrium_matches_theory`).
- **Pendiente `[VERIFICAR]`:** la numeración exacta de las ecuaciones del paper de 2014 y los valores de los parámetros (β=0.03, α_m=0.01, α_M=0.07, s=5e-4, ρ=5e-3, γ=0.01, λ=100) están tomados de lo que recordamos del paper y de la reproducción de Patry et al.; deben contrastarse con el PDF antes de la entrega. Por eso los docstrings citan las ecuaciones por contenido y no por número.
- Como `C_E/(κ s x_E) = C_C/(s x_C) = min(1, w/w_th)`, ambas clases comparten el término de hambruna; la desigualdad actúa por `w_th` y por el consumo (κ).

## D-03 · Validación con escenarios canónicos (`scenarios.py`, notebook 01)
- Reproducidos: equilibrio (1×δ_opt), oscilaciones amortiguadas (2.5×), ciclos de colapso (4×) y colapso irreversible por desigualdad (κ=6 con 20 % de élites).
- **No reproducido:** colapso irreversible de la naturaleza (Tipo-N) a 5.5×δ_opt. En nuestra implementación la población colapsa pero la naturaleza se recupera y el sistema cicla (con peor amplitud). Posibles causas: umbral de irreversibilidad distinto en el paper, condiciones iniciales diferentes (el paper usa `x_E(0)=25` sobre `x_C(0)=100`; nosotros fijamos 100 de población total y repartimos con `elite_frac`). Se reporta tal cual; no se ajustó el modelo para forzarlo.
- Observación: en los escenarios con élites la riqueza inicial es 0 y la producción inicial es muy baja, por lo que con κ alto la población nunca despega (colapso en pocas décadas). Es comportamiento del modelo, no un error numérico.

## D-04 · Integración numérica
- RK4 con `dt = 1 año`, vectorizado sobre planetas (arrays NumPy float64). Las no linealidades `min/max` son continuas pero no suaves; con dt=1 y tasas ≤ 0.07 el esquema es estable (los tests de no negatividad y balance de riqueza lo verifican).
- Alternativas descartadas: `scipy.solve_ivp` (no se vectoriza en lote), simulación en GPU con PyTorch (innecesaria: 50 000 planetas × 600 pasos tardan ~8 s en CPU).
- La no negatividad se garantiza con un recorte `max(·, 0)` después de cada paso.

## D-05 · Extensiones (11 factores)
| Factor | Accionable | Rol |
|---|---|---|
| `delta` (depleción) | Sí | HANDY |
| `kappa` (desigualdad) | Sí | HANDY |
| `s` (consumo per cápita) | Sí | HANDY |
| `water_eff` (eficiencia hídrica) | Sí | extensión |
| `beta` (natalidad) | No | HANDY |
| `gamma` (regeneración) | No | HANDY |
| `temp` (anomalía térmica) | No | extensión |
| `water` (oferta de agua) | No | extensión |
| `event_rate`, `event_sev` | No | extensión (Poisson) |
| `elite_frac` (élites iniciales) | No | condición inicial |

- **Temperatura:** `γ_eff = γ · exp(−(ΔT/3)²)` y `α_m,eff = α_m (1 + 0.15 |ΔT|)`. Gaussiana: regeneración máxima en ΔT=0 y decae suavemente; el 3 °C de tolerancia y el 15 % de mortalidad por grado son supuestos nuestros, sin calibración empírica.
- **Agua:** `λ_eff = λ · clip(W / demanda, 0.05, 1)` con `demanda = (x_C+x_E)(1−water_eff)/P_ref`, `P_ref = 75 000` (población de equilibrio de referencia). Así el agua limita la capacidad de carga solo cuando la población supera lo que el agua sostiene. Alternativa descartada: reducir directamente la natalidad (no es "capacidad de carga").
- **Eventos extremos:** llegadas Poisson(`rate·dt`); cada evento destruye `sev·U1` de la naturaleza y `sev·U2` de la riqueza (U ~ Uniforme(0,1) independientes). No se destruye población directamente.
- ρ, α_m, α_M, λ y la población inicial (100) quedan fijos para no inflar la dimensión. Se descartó un factor de "tasa de natalidad de élites" distinto (β_E = β_C).
- Alcance honesto: `water`, `water_eff` y `event_*` tienen efecto pequeño en la tasa agregada de supervivencia (ver D-06 y el EDA); los eventos sobre todo reordenan el momento del colapso.

## D-06 · Colapso, supervivencia y rangos de muestreo
- **Colapso:** primer año en que la población total `P` cae por debajo de `max(0.1 · P_máx_acumulada, 0.2 · P(0))`. **Supervivencia:** no hay colapso hasta T = 600 años. Se guarda `t_collapse` (= 600 si sobrevive).
- El segundo término (suelo absoluto) se agregó porque los declives lentos desde el inicio (p. ej. κ alto) nunca bajan del 10 % del pico, que es la población inicial.
- El umbral 0.1 se eligió porque 0.3 etiquetaba como colapso las oscilaciones amortiguadas de 2.5×δ_opt (mínimo 16 % del pico), que en el paper no son colapso.
- Horizonte de 600 años (con 1000 se ve el ciclo completo, pero el primer colapso ocurre cerca del año 420).
- **Iteración de rangos (6 000 planetas de prueba):** rangos por defecto → 73.6 % de supervivencia; `water`, `event_*` casi sin efecto. Ampliar eventos (rate ≤ 0.12, sev 0.1–0.9), agua (0.05–1.2), temperatura (0–5), δ ≤ 6e-5, κ ≤ 30 → 67.5 %. Con δ ≤ 1e-4 → **63.2 %** (elegido; en el dataset final: 63.3 %). La relación con δ es no monótona (δ muy bajo también colapsa por no despegar).
- Muestreo: Latin Hypercube; δ y κ en escala logarítmica.

## D-07 · Datos y splits
- 50 000 planetas, 101 puntos por trayectoria (cada 6 años), 4 variables de estado + conteo de eventos por intervalo. Split 70/15/15 aleatorio por planeta con semilla fija.
- Monte Carlo: 500 planetas de test × 32 réplicas (distinta semilla de eventos) → `P(sobrevive | parámetros)` con resolución 1/32. **Limitación:** solo el 17.8 % de esos planetas tiene P entre 0.05 y 0.95; el resultado es mayormente determinista dado los parámetros.
- Normalización (z-score, `log` para δ y κ; `log1p` para poblaciones y riqueza) ajustada **solo con train** (`Preprocessor`).

## D-08 · Modelos
- **MLP (tarea a):** `Linear→ReLU→Dropout`, AdamW, early stopping sobre BCE de validación (S8). Búsqueda aleatoria de 12 configuraciones (capas, dropout, lr, weight decay). Se selecciona por BCE de validación, nunca por test.
- **LSTM (tarea b):** una capa, 128 unidades, `pack_padded_sequence` para tomar el último estado válido. Los parámetros estáticos se **concatenan a cada paso y a la cabeza**. Justificación: la supervivencia depende de forma fuerte de los parámetros y no del estado inicial (que es igual para todos); inyectarlos en cada paso evita que el LSTM los "olvide" en secuencias largas. Alternativa no probada: usarlos como estado oculto inicial (`h0`).
- **Truncamiento aleatorio:** durante el entrenamiento, `m ~ U{0..70}` pasos por muestra y se filtran los planetas ya colapsados antes de `m`. Un solo modelo sirve para cualquier k.
- **Evaluación sin fuga:** para cada k solo se evalúan los planetas con `t_collapse > t_k`; se reporta cuántos quedan fuera.
- Baselines de (b): LightGBM con parámetros + resúmenes (últimos valores, pendientes de 5 pasos, pico, razón población/pico, mínimo de naturaleza, riqueza máxima, eventos totales) y MLP sobre `[parámetros, ventana aplanada]`, reentrenados por cada k. Referencia adicional: el MLP estático de (a) sobre los mismos planetas vivos.
- Sin búsqueda de hiperparámetros en (b) (fase 1); limitación declarada.
- Se probó una sola semilla por modelo; **no hay intervalos de confianza** (pendiente para la Fase 2: varias semillas).

## D-09 · XAI
- **Un solo baseline para todos los métodos:** el planeta medio de train (vector cero estandarizado). **Espacio de salida:** logit de P(sobrevive), para que las atribuciones sean aditivas y comparables entre MLP y LightGBM.
- **SHAP:** con 11 factores calculamos Shapley **exacto** (2¹¹ coaliciones, referencia única) en GPU; coincide con `shap.KernelExplainer` (`l1_reg=False`) con diferencia máxima 3.8e-7, y cumple eficiencia (error 5e-6). Primera versión de la comprobación daba 0.12 de diferencia por la regularización L1 automática de KernelSHAP; se detectó y se corrigió.
- **Integrated Gradients** (captum, 1024 pasos, mismo baseline); error de completitud máximo 0.054 logits (el rango de logits es ~±12).
- **TreeSHAP** interventional con la misma referencia sobre LightGBM.
- Atribuciones guardadas para los 7 500 planetas de test (`outputs/xai/attributions.npz`, forma `[planeta, factor]`) con la definición del baseline y los índices de los 500 planetas Monte Carlo, y la marca de accionable por factor.
- Descartado: GradientExplainer (promedia varios baselines aleatorios; difícil igualar la definición con IG).
- **No** se interpretan aún como causales: la fidelidad frente al simulador queda para la Fase 2.

## D-10 · Documento
- `report/build_report.py` parte de `template.docx` (local, no se sube) y conserva la carátula tocando solo el texto de los runs. El título está en la variable `TITLE`.
- Sin imágenes ni números inventados: las tablas del informe salen de `outputs/*/results.json`.
