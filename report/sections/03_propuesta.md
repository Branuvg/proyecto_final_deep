# 3. Propuesta de solución

## 3.1 Idea general

Proponemos usar redes neuronales como **modelos sustitutos** del simulador y evaluar sus explicaciones **contra el propio simulador**. La propuesta tiene cinco piezas:

1. **Un simulador de referencia** (HANDY extendido), validado con los escenarios canónicos del paper y con pruebas unitarias.
2. **Un dataset** de 50 000 planetas muestreados con Latin Hypercube (McKay et al., 1979), con rangos ajustados para que la frontera entre supervivencia y colapso no sea trivial.
3. **Dos modelos de predicción**: un MLP que mapea condiciones iniciales a probabilidad de supervivencia (tarea a) y un LSTM que lee los primeros k pasos de la trayectoria (tarea b), con baselines y varias semillas.
4. **Explicaciones y su validación**: SHAP, Integrated Gradients y TreeSHAP con un baseline común, comparados con el efecto de **intervenir cada factor** en el modelo y en el simulador.
5. **Contrafactuales accionables**: el cambio mínimo en las palancas de política que lleva a la supervivencia, verificado con el simulador, y una interfaz para explorarlos.

## 3.2 Por qué una red y no solo un modelo clásico

Declaramos de antemano que la red debe **demostrar** que aporta algo. Por eso son obligatorios dos baselines, regresión logística y gradient boosting, y se reportan honestamente aunque la red no gane. La regresión logística responde si la frontera es lineal; el gradient boosting es el estándar fuerte sobre datos tabulares.

## 3.3 Por qué una LSTM para la alerta temprana

La alerta temprana consume una secuencia de longitud variable (la trayectoria parcial), que es el caso de uso de la LSTM (sesión S3; también S12 para pronóstico con ventanas). El truncamiento aleatorio de k durante el entrenamiento permite que **un solo modelo** sirva para cualquier ventana. Para no medir una ventaja trivial evaluamos solo los planetas que aún siguen vivos al final de la ventana observada.

## 3.4 Cómo medimos la fidelidad de una explicación

Una atribución φ_ij dice cuánto contribuye el factor j a la predicción del planeta i respecto a un planeta de referencia. Si es fiel, devolver el factor j a su valor de referencia debe cambiar el resultado en el mismo sentido y con una magnitud proporcional. Comparamos φ con tres cantidades:

- el **efecto en el modelo** de la intervención (fidelidad al modelo);
- el **efecto en el simulador**, estimado con réplicas Monte Carlo y números aleatorios comunes (fidelidad a la realidad);
- **curvas de eliminación** (S15; Petsiuk et al., 2018): devolver al valor de referencia los factores en el orden que dice la atribución debe degradar el resultado más rápido que un orden aleatorio.

Como control, la prueba de cordura de Adebayo et al. (2018) con pesos aleatorizados: una explicación que sí depende del modelo debe dejar de concordar con el simulador si el modelo se destruye.

## 3.5 Qué consideramos un resultado válido

- Un modelo es útil si supera a los baselines **con intervalos de confianza** que excluyan la igualdad y está bien calibrado (Brier, ECE y error frente a la probabilidad verdadera Monte Carlo).
- Una explicación es útil si es **fiel** según las tres pruebas anteriores, y debe reportarse en cuáles falla.
- Un contrafactual es válido si el **simulador** confirma que el planeta modificado sobrevive, no solo la red.
