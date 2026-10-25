# 3. Propuesta de solución

## 3.1 Idea general

Proponemos usar redes neuronales como **modelos sustitutos** del simulador y evaluar sus explicaciones **contra el propio simulador**. La propuesta tiene cuatro piezas:

1. **Un simulador de referencia** (HANDY extendido), validado con los escenarios canónicos del paper y con pruebas unitarias.
2. **Un dataset** de 50 000 planetas muestreados con Latin Hypercube (McKay et al., 1979), con rangos ajustados para que la frontera entre supervivencia y colapso no sea trivial.
3. **Dos modelos de predicción**: un MLP que mapea condiciones iniciales a probabilidad de supervivencia (tarea a) y un LSTM que lee los primeros k pasos de la trayectoria (tarea b).
4. **Explicaciones verificables**: SHAP, Integrated Gradients y TreeSHAP calculados con un baseline común, listos para compararse en la Fase 2 con el efecto de intervenir cada factor en el modelo y en el simulador.

## 3.2 Por qué una red y no solo un modelo clásico

Declaramos de antemano que la red debe **demostrar** que aporta algo. Por eso son obligatorios dos baselines, regresión logística y gradient boosting, y se reportan honestamente aunque la red no gane. La regresión logística responde si la frontera es lineal; el gradient boosting es el estándar fuerte sobre datos tabulares.

## 3.3 Por qué una LSTM para la alerta temprana

La alerta temprana consume una secuencia de longitud variable (la trayectoria parcial), que es el caso de uso de la LSTM (sesión S3; también S12 para pronóstico con ventanas). El truncamiento aleatorio de k durante el entrenamiento permite que **un solo modelo** sirva para cualquier ventana. Para no medir una ventaja trivial evaluamos solo los planetas que aún siguen vivos al final de la ventana observada.

## 3.4 Qué consideramos un resultado válido

- Un modelo es útil si supera a los baselines **en el mismo conjunto de test** y está bien calibrado (Brier, ECE y error frente a la probabilidad verdadera Monte Carlo).
- Una explicación es útil si es **fiel**: si lo que dice que importa es lo que, al intervenir, cambia el resultado. Esta medición completa es la meta de la Fase 2.
