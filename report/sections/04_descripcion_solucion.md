# 4. Descripción de la solución

## 4.1 Arquitectura del código

El código vive en el paquete `worldsim` (`src/worldsim/`): `simulator/` (HANDY, factores, escenarios), `data/` (muestreo, generación, preprocesamiento), `models/` (redes escritas como `nn.Module` propios), `train/` (bucle de entrenamiento, métricas, scripts de cada tarea) y `xai/` (atribuciones). Cada etapa tiene un comando y su configuración en `configs/*.yaml`; las semillas están fijas.

## 4.2 Simulador

Se integra con Runge-Kutta de cuarto orden (dt = 1 año) vectorizado sobre miles de planetas a la vez: los 50 000 planetas de 600 años se simulan en menos de diez segundos. Los eventos extremos se aplican después de cada paso: el número de llegadas es Poisson(tasa·dt) y cada evento destruye una fracción aleatoria de naturaleza y de riqueza. Las pruebas unitarias verifican no negatividad, determinismo con la misma semilla, el efecto de los eventos, el balance de riqueza, el equilibrio analítico y la definición de colapso.

## 4.3 Dataset y preprocesamiento

Cada planeta guarda sus 11 parámetros, la trayectoria submuestreada (101 puntos, un punto cada 6 años) de las 4 variables de estado, el conteo de eventos por intervalo, la etiqueta y el tiempo de colapso. El split 70/15/15 es por planeta. La normalización (z-score; logaritmo para δ y κ, que se muestrean en escala logarítmica; `log1p` para poblaciones y riqueza) se ajusta **solo con train**. Los rangos de muestreo se ajustaron en una iteración documentada: con los rangos iniciales la supervivencia era 73.6 % y varios factores casi no influían; ampliándolos se llegó a 63.3 %.

## 4.4 Tarea (a): MLP

El MLP está compuesto por bloques Linear → ReLU → Dropout y una cabeza lineal que produce un logit. Se entrena con la pérdida de entropía cruzada binaria (S1, S8), el optimizador AdamW (Loshchilov & Hutter, 2019; S8), dropout como regularización (S8) y *early stopping* sobre la pérdida de validación (S8). Una búsqueda aleatoria de 12 configuraciones (capas, dropout, tasa de aprendizaje y *weight decay*) se evalúa con la pérdida de validación, nunca con test. Los baselines son regresión logística (la neurona de S1) y LightGBM (Ke et al., 2017).

## 4.5 Tarea (b): LSTM de alerta temprana

La LSTM (S3; Hochreiter & Schmidhuber, 1997) lee la trayectoria parcial con 6 canales por paso (log población de plebeyos, log población de élites, naturaleza, log riqueza, eventos en el intervalo y tiempo normalizado). Los parámetros estáticos se **concatenan a cada paso y a la cabeza de clasificación**, porque la supervivencia depende mucho de ellos y el estado inicial es igual para todos los planetas. Se usa `pack_padded_sequence` para tomar el último estado válido. Durante el entrenamiento, el número de pasos observados se sortea al azar por muestra (truncamiento aleatorio) y se descartan los planetas ya colapsados dentro de la ventana. Se evalúa con k ∈ {{0, 5, 10, 20, 40, 60}} % del horizonte.

Comparaciones: LightGBM con parámetros más resúmenes de la ventana, MLP sobre la ventana aplanada (ambos reentrenados por cada k), el MLP de la tarea (a) que no ve la trayectoria y una ablación de la LSTM sin parámetros estáticos.

## 4.6 XAI: cálculo de atribuciones

Se calculan tres métodos sobre el logit de supervivencia y con **el mismo baseline**: el planeta medio de entrenamiento (vector cero estandarizado). (i) SHAP: como son 11 factores, calculamos el valor de Shapley **exacto** (Shapley, 1953; Lundberg & Lee, 2017) sobre las 2¹¹ coaliciones, y lo verificamos contra `KernelExplainer`. (ii) Integrated Gradients con Captum (Sundararajan et al., 2017; Kokhlikyan et al., 2020). (iii) TreeSHAP sobre LightGBM (Lundberg et al., 2020). Las atribuciones de los 7 500 planetas de test se guardan por planeta y por factor, junto con la definición del baseline, los índices Monte Carlo y la marca de factor accionable, de modo que la Fase 2 pueda compararlas con intervenciones en el modelo y en el simulador (S15).
