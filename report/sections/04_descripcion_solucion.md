# 4. Descripción de la solución

## 4.1 Arquitectura del código

El código vive en el paquete `worldsim` (`src/worldsim/`): `simulator/` (HANDY, factores, escenarios), `data/` (muestreo, generación, preprocesamiento), `models/` (redes escritas como `nn.Module` propios), `train/` (bucle de entrenamiento, métricas, estadística, scripts de cada tarea) y `xai/` (atribuciones, intervenciones, fidelidad, contrafactuales). La interfaz está en `app/`. Cada etapa tiene un comando y su configuración en `configs/*.yaml`; las semillas están fijas y `scripts/run_all.py` ejecuta todo en orden.

## 4.2 Simulador

Se integra con Runge-Kutta de cuarto orden (dt = 1 año) vectorizado sobre miles de planetas a la vez: los 50 000 planetas de 600 años se simulan en unos diez segundos. También hay un integrador de Euler en precisión simple para replicar el esquema del paper. Los eventos extremos se aplican después de cada paso: el número de llegadas es Poisson(tasa·dt) y cada evento destruye una fracción aleatoria de naturaleza y de riqueza. Las pruebas unitarias (15 en total) verifican no negatividad, determinismo con la misma semilla, el efecto de los eventos, el balance de riqueza, el equilibrio analítico, la mortalidad de las élites (Ec. 6), los escenarios Tipo-L y Tipo-N, la convergencia de Euler a RK4 con paso pequeño y la coherencia de los rangos de los factores con el dataset.

## 4.3 Dataset y preprocesamiento

Cada planeta guarda sus 11 parámetros, la trayectoria submuestreada (101 puntos, un punto cada 6 años) de las 4 variables de estado, el conteo de eventos por intervalo, la etiqueta y el tiempo de colapso. El split 70/15/15 es por planeta. La normalización (z-score; logaritmo para δ y κ, que se muestrean en escala logarítmica; `log1p` para poblaciones y riqueza) se ajusta **solo con train**. Los rangos se ajustaron en dos iteraciones documentadas (`docs/DECISIONES.md`, D-06): al corregir la mortalidad de las élites la supervivencia cayó a 15 % y se redujeron κ (≤ 3) y la fracción inicial de élites (≤ 10 %) hasta obtener {e[survival_rate]:.1%}.

## 4.4 Tarea (a): MLP

El MLP está compuesto por bloques Linear → ReLU → Dropout y una cabeza lineal que produce un logit. Se entrena con la pérdida de entropía cruzada binaria (S1, S8), el optimizador AdamW (Loshchilov & Hutter, 2019; S8), dropout como regularización (S8) y *early stopping* sobre la pérdida de validación (S8). Una búsqueda aleatoria de 12 configuraciones (capas, dropout, tasa de aprendizaje y *weight decay*) se evalúa con la pérdida de validación, nunca con test. La mejor configuración se reentrena con 5 semillas y se reportan media ± desviación estándar, más intervalos de confianza por *bootstrap* pareado sobre los planetas de test. Los baselines son regresión logística (la neurona de S1) y LightGBM (Ke et al., 2017).

## 4.5 Tarea (b): LSTM de alerta temprana

La LSTM (S3; Hochreiter & Schmidhuber, 1997) lee la trayectoria parcial con 6 canales por paso (log población de plebeyos, log población de élites, naturaleza, log riqueza, eventos en el intervalo y tiempo normalizado). Los parámetros estáticos se **concatenan a cada paso y a la cabeza de clasificación**, porque la supervivencia depende mucho de ellos y el estado inicial es igual para todos los planetas. Se usa `pack_padded_sequence` para tomar el último estado válido. Durante el entrenamiento, el número de pasos observados se sortea al azar por muestra (truncamiento aleatorio) y se descartan los planetas ya colapsados dentro de la ventana. Se evalúa con k ∈ {{0, 5, 10, 20, 40, 60}} % del horizonte. Una búsqueda aleatoria de 8 configuraciones (unidades, capas, dropout, tasa de aprendizaje, *weight decay*) elige los hiperparámetros por la pérdida de validación con truncamiento aleatorio, y la mejor se reentrena con 5 semillas.

Comparaciones: LightGBM con parámetros más resúmenes de la ventana, MLP sobre la ventana aplanada (ambos reentrenados por cada k), el MLP de la tarea (a) que no ve la trayectoria y una ablación de la LSTM sin parámetros estáticos.

## 4.6 XAI: cálculo de atribuciones

Se calculan tres métodos sobre el logit de supervivencia y con **el mismo baseline**: el planeta medio de entrenamiento (vector cero estandarizado). (i) SHAP: como son 11 factores, calculamos el valor de Shapley **exacto** (Shapley, 1953; Lundberg & Lee, 2017) sobre las 2¹¹ coaliciones, y lo verificamos contra `KernelExplainer`. (ii) Integrated Gradients con Captum (Sundararajan et al., 2017; Kokhlikyan et al., 2020). (iii) TreeSHAP sobre LightGBM (Lundberg et al., 2020). Las atribuciones de los 7 500 planetas de test se guardan por planeta y por factor, con la definición del baseline y la marca de factor accionable.

## 4.7 Validación de la fidelidad

Para un conjunto de planetas de test (los 500 del subconjunto Monte Carlo y {fx[bd_n]} planetas cercanos a la frontera de decisión del modelo) se devuelve cada factor j a su valor base, el mismo de las atribuciones, y se mide (i) el cambio del logit del modelo y (ii) el cambio de la probabilidad real de supervivencia en el simulador, estimada con 64 réplicas y números aleatorios comunes (misma semilla y disposición de filas) para que la diferencia tenga poca varianza. Las métricas son la correlación de Spearman por planeta entre φ y el efecto, el acuerdo de signo (solo en efectos mayores a 0.10 de probabilidad), la coincidencia del factor más importante (top-1) y del top-3. Se evalúan solo los planetas **informativos**: aquellos con algún efecto individual de al menos 0.10. Las curvas de eliminación devuelven los factores al valor base en el orden dado por la atribución, por la importancia global y al azar, tanto en el modelo como en el simulador (32 réplicas). La prueba de cordura reemplaza el MLP por uno con pesos aleatorios.

## 4.8 Contrafactuales

Se busca, por descenso de gradiente sobre el MLP, el cambio mínimo de las palancas accionables (δ, κ, s y eficiencia hídrica; dentro de los rangos muestreados) que lleva la predicción a una probabilidad de al menos 0.8, penalizando la distancia L1 relativa al rango. Los factores no accionables quedan fijos. Como comparación se busca el cambio de **una sola** palanca por búsqueda en rejilla. Cada contrafactual se verifica en el simulador con 64 réplicas: es válido si la supervivencia real supera 0.5. Se aplican a los planetas Monte Carlo que el modelo predice que colapsarán.

## 4.9 Interfaz

La aplicación Streamlit (`app/streamlit_app.py`) permite mover los 11 factores y ver: la simulación con bandas de réplicas, la probabilidad del simulador frente a la de la red, la explicación SHAP exacta del planeta y la búsqueda de un contrafactual verificado con el simulador.
