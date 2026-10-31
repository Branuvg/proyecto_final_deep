# Decisiones de diseño

Cada decisión incluye su justificación y las alternativas descartadas.

## D-01 · Paquete, entorno y GPU
- Paquete `worldsim` en `src/`; Python 3.12 (la 3.14 instalada en el sistema aún tiene soporte irregular de `shap`/`torch`).
- PyTorch 2.11.0 con wheels `cu128` (necesario para Blackwell sm_120). Verificado: `torch.cuda.is_available() == True`, GPU `NVIDIA GeForce RTX 5060 Laptop GPU`.
- El código usa `get_device()` (CUDA si existe, si no CPU) para poder correr en Colab. Rutas con `pathlib`.
- **LightGBM** se agregó como gradient boosting: es rápido, compatible con TreeSHAP y evita depender de XGBoost. Alternativa descartada: `GradientBoostingClassifier` de sklearn (lento con 35 000 filas × muchas pruebas).

## D-02 · Ecuaciones de HANDY y su verificación (actualizada en la Fase 2)
- Implementadas las cuatro ecuaciones de estado Ec. (3), el consumo Ec. (4), el umbral Ec. (5) y la mortalidad Ec. (6), con la numeración del paper original.
- **Verificación contra el original (Fase 2):** se obtuvo el texto de una copia del artículo alojada por los autores (la copia de una universidad no se pudo abrir por un problema de certificado, y no se desactivó la verificación; se usó otra copia con certificado válido). Se contrastaron ecuaciones, la Tabla 1 de parámetros (β = 0.03, α_m = 0.01, α_M = 0.07, s = 5e-4, ρ = 5e-3, γ = 0.01, λ = 100, κ ∈ {1, 10, 100}, x_E(0) ∈ {0, 1, 25}), δ\* = 2ηs/λ = 6.67e-6 (Ec. 10) y χ_M (Ec. 11). Todo coincide **salvo un error nuestro**:
- **Error corregido:** α_E = α_m + max(0, 1 − C_E/(s·x_E))(α_M − α_m). Nosotros habíamos usado κ·s·x_E en el denominador (suponiendo que κ se cancelaba), de modo que élites y plebeyos compartían el mismo término de hambruna. En el paper las élites solo pasan hambre si w/w_th ≤ 1/κ (Fig. 2b). Test: `test_elite_famine_uses_kappa`. **Todos los resultados de la Fase 1 quedaron obsoletos** y se regeneraron (dataset, modelos, XAI).
- El paper integra con Euler, dt = 1 año, precisión simple. Implementamos ambos esquemas (`SimConfig.integrator`, `dtype`); el dataset usa RK4.

## D-03 · Validación con escenarios canónicos (`scenarios.py`, notebook 01)
- Reproducidos (RK4): equilibrio (1×δ\*), oscilaciones (2.5×), ciclos de colapso reversibles (4×), colapso **Tipo-L** (κ = 100, x_E(0) = 1e-3: la población muere y la naturaleza se recupera), colapso **Tipo-N** irreversible con desigualdad (κ = 100, δ = 1e-4, x_E(0) = 0.2) y el caso equitativo (κ = 1, x_E(0) = 25).
- **Tipo-N igualitario (parcial):** el paper reporta 4×δ\* reversible y 5.5×δ\* irreversible. Barrido propio: con RK4 el colapso irreversible (naturaleza y población en 0) aparece desde ≈ 9.25×δ\*; con Euler dt = 1 en float32 desde ≈ 3.25×δ\*. El umbral del paper queda entre ambos y **no lo reproducimos con exactitud** (no tenemos su código). Hipótesis no verificada por separado: con Euler el paso de la naturaleza se vuelve negativo y se recorta a 0 cuando δ·x_C > 1 + γλ.
- No se ajustó ningún parámetro para forzar el resultado.
- Se descartó replicar el escenario 5.3.3 del paper (aterrizaje suave con β_E ≠ β_C) porque nuestro modelo usa una sola natalidad.

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
- Alcance honesto: según las intervenciones en el simulador (D-11), `water` y `water_eff` tienen un efecto real mínimo sobre P(sobrevive) (efecto medio |ΔP| ≈ 0.03 y 0.01, frente a ≈ 0.37 de δ); los eventos extremos pesan más (≈ 0.19 la tasa) pero sobre todo reordenan el momento del colapso.

## D-06 · Colapso, supervivencia y rangos de muestreo
- **Colapso:** primer año en que la población total `P` cae por debajo de `max(0.1 · P_máx_acumulada, 0.2 · P(0))`. **Supervivencia:** no hay colapso hasta T = 600 años. Se guarda `t_collapse` (= 600 si sobrevive).
- El segundo término (suelo absoluto) se agregó porque los declives lentos desde el inicio nunca bajan del 10 % del pico, que es la población inicial.
- El umbral 0.1 se eligió porque 0.3 etiquetaba como colapso las oscilaciones amortiguadas de 2.5×δ\*.
- **Iteración de rangos.** Fase 1 (con α_E erróneo): 73.6 % de supervivencia con rangos por defecto → 63.2 % tras ampliar δ, κ ≤ 30, eventos, agua y temperatura. **Fase 2 (α_E correcto):** con esos mismos rangos la supervivencia cayó a **15.3 %** (las élites resisten la hambruna y drenan la riqueza; κ alto la hunde). Se redujo κ a [1, 3], la fracción inicial de élites a [0, 0.1] y δ a [5e-6, 6e-5]. Con eso: 37–43 % (κ ≤ 3–4) y 49.5 % en la muestra de prueba; **49.6 %** en el dataset final.
- `configs/dataset.yaml` es la fuente de verdad de los rangos; `FACTORS` en `factors.py` debe coincidir (lo comprueba un test, porque una discrepancia hizo que los contrafactuales recortaran factores no accionables a rangos equivocados).
- Muestreo: Latin Hypercube; δ y κ en escala logarítmica.

## D-07 · Datos y splits
- 50 000 planetas, 101 puntos por trayectoria (cada 6 años), 4 variables de estado + conteo de eventos por intervalo. Split 70/15/15 aleatorio por planeta con semilla fija.
- Monte Carlo: 500 planetas de test × 32 réplicas (distinta semilla de eventos) → `P(sobrevive | parámetros)` con resolución 1/32. **Nota:** la fracción de esos planetas con P entre 0.05 y 0.95 se reporta en el EDA (notebook 02); el resultado es en buena parte determinista dados los parámetros, pero hay casos genuinamente inciertos.
- Normalización (z-score, `log` para δ y κ; `log1p` para poblaciones y riqueza) ajustada **solo con train** (`Preprocessor`).

## D-08 · Modelos (actualizada en la Fase 2)
- **MLP (tarea a):** `Linear→ReLU→Dropout`, AdamW, early stopping sobre la BCE de validación (S8). Búsqueda aleatoria de 12 configuraciones con la semilla 0; la mejor se reentrena con **5 semillas** (2026–2030). Se reporta media ± desviación y IC 95 % por *bootstrap* pareado (1000 remuestreos de los planetas de test) para las diferencias entre modelos. El modelo "principal" que usan XAI, fidelidad y contrafactuales es el de la semilla 2026.
- **LSTM (tarea b):** `pack_padded_sequence` para tomar el último estado válido. Los parámetros estáticos se concatenan a cada paso y a la cabeza de clasificación (la supervivencia depende mucho de ellos y el estado inicial es igual para todos). Alternativa no probada: usarlos como estado oculto inicial.
- **Búsqueda de hiperparámetros de la LSTM (Fase 2):** 8 configuraciones aleatorias (unidades, capas, dropout, lr, weight decay), criterio: BCE de validación con truncamiento aleatorio fijo. Los baselines de (b) siguen con configuración fija (sin búsqueda): limitación declarada.
- **Truncamiento aleatorio:** `m ~ U{0..70}` pasos por muestra y se filtran los planetas ya colapsados antes de `m`. Un solo modelo sirve para cualquier k.
- **Evaluación sin fuga:** solo planetas con `t_collapse > t_k`; se reporta cuántos quedan fuera. Las métricas entre distintos k **no son comparables** (cambia la población evaluada).
- Baselines de (b): LightGBM con parámetros + resúmenes de la ventana y MLP sobre `[parámetros, ventana aplanada]`, reentrenados por cada k y con cada semilla; referencia adicional: el MLP estático de (a).

## D-09 · XAI: atribuciones
- **Un solo baseline para todos los métodos:** el planeta medio de train (vector cero estandarizado; geometría log para δ y κ). **Espacio de salida:** logit de P(sobrevive).
- **SHAP exacto** (2¹¹ coaliciones, referencia única) en GPU; coincide con `KernelExplainer` (`l1_reg=False`) con diferencia ≈ 1e-7. (En la Fase 1 una diferencia de 0.12 se debió a la regularización L1 automática de KernelSHAP.) **IG** (captum, 1024 pasos); error de completitud pequeño, se reporta en `outputs/xai/checks.json`. **TreeSHAP** interventional con la misma referencia.
- Atribuciones guardadas para los 7 500 planetas de test (`outputs/xai/attributions.npz`).
- Descartado: GradientExplainer (promedia baselines aleatorios; difícil igualar la definición con IG).

## D-10 · Documento
- `report/build_report.py` parte de `template.docx` (local, no se sube) y conserva la carátula tocando solo el texto de los runs. El título está en la variable `TITLE`.
- Sin imágenes ni números inventados: las tablas del informe salen de `outputs/*/results.json`.

## D-11 · Validación de fidelidad (`xai/run_fidelity.py`)
- **Intervención:** devolver el factor j al valor base (mismo baseline que las atribuciones). Es una intervención de **un factor a la vez**: no captura interacciones, y por eso SHAP (que reparte las interacciones) no tiene por qué coincidir exactamente. Se declara como limitación.
- **Verdad en el simulador:** P(sobrevive) estimada con R = 64 réplicas; el planeta original y el intervenido usan **números aleatorios comunes** (misma semilla y disposición de filas) para reducir la varianza de la diferencia. Los eventos cambian el flujo del generador si la intervención toca `event_rate`/`event_sev`, lo que reduce (sin eliminar) la ventaja.
- **Conjuntos de planetas:** (i) los 500 del subconjunto Monte Carlo (muestra aleatoria del test), (ii) 1000 planetas de test cuyo P del modelo está en (0.05, 0.95) ("frontera"), donde hay más incertidumbre real que explicar.
- **Métricas:** Spearman por planeta entre φ y el efecto (media e IC 95 % por *bootstrap* sobre planetas); acuerdo de signo solo donde |efecto| ≥ umbral; top-1 y top-3. Solo planetas "informativos" (algún efecto ≥ 0.10 en probabilidad o ≥ 0.25 en logit). Umbrales fijados antes de ver resultados.
- **Curvas de eliminación:** orden por atribución vs. aleatorio vs. importancia global, en el modelo y en el simulador (R = 32). El orden aleatorio *sube* en promedio la supervivencia porque devolver al valor medio a planetas atípicos los acerca al planeta medio (que sobrevive); lo relevante es la diferencia con el orden por atribución.
- **Prueba de cordura (Adebayo et al., 2018):** SHAP exacto del MLP con pesos aleatorios (misma arquitectura); debe perder la concordancia con el simulador.
- **Separar error del modelo y de la explicación:** también se mide cuánto concuerda el *efecto que predice el propio MLP* con el simulador; es un techo aproximado para cualquier explicación del MLP.

## D-12 · Contrafactuales (`xai/counterfactual.py`)
- Descenso de gradiente sobre el MLP en el espacio estandarizado, **solo palancas accionables** (δ, κ, s, eficiencia hídrica), dentro de los rangos muestreados; pérdida = bisagra hacia P ≥ 0.8 + 0.5·L1 relativo al rango; los cambios menores al 1 % del rango se descartan (esparcidad).
- Comparación: cambio de **una sola** palanca por rejilla (201 valores).
- **Validación con el simulador** (64 réplicas): un contrafactual es válido si la supervivencia real ≥ 0.5. Se aplica a los planetas Monte Carlo que el modelo predice que colapsarán; las tasas de validez se reportan sobre los que realmente colapsan según el simulador, y los casos donde el modelo se equivocaba se cuentan aparte.
- Alternativas descartadas: DiCE / búsqueda genética (más dependencias, sin ventaja clara con 4 palancas); optimizar directamente contra el simulador (caro y no diferenciable).

## D-13 · Interfaz (`app/streamlit_app.py`)
- Streamlit: sliders de los 11 factores con los rangos del dataset, simulación con 8–128 réplicas (mediana y percentiles 10–90), probabilidad del simulador vs. la red, SHAP exacto del planeta y contrafactual verificado con el simulador. Probada con `streamlit.testing.v1.AppTest`.
- Requiere `data/worlds.npz` (para las estadísticas de normalización) y `outputs/task_a/mlp_a.pt`; si faltan, muestra el comando para generarlos.

## D-14 · Presentación y video
- `report/build_presentation.py` genera `Presentacion.pptx` (10 diapositivas, 3 hablantes, texto hablado en las notas) y `docs/PRESENTACION.md`; `report/build_video_script.py` genera `docs/GUION_VIDEO.md`. Los números salen de los JSON de resultados.
- Reparto: David (1, 2, 3, 10), Jonathan (4, 5, 6) y Gabriel (7, 8, 9), según los temas del documento; ajustable en el script.

