# 1. Descripción del problema

## 1.1 Contexto

Muchos sistemas de interés científico y de política pública (clima, epidemiología, modelos basados en agentes, dinámica de recursos) se estudian con simuladores dinámicos que son costosos de ejecutar muchas veces. Una práctica extendida es entrenar un **modelo sustituto** (*surrogate*) que aproxime la salida del simulador a partir de sus parámetros, de modo que explorar escenarios sea casi inmediato (Saltelli et al., 2008). Cuando el sustituto es una red neuronal aparece un segundo problema: ¿por qué predice lo que predice y podemos creerle a la explicación?

## 1.2 El problema que abordamos

Construimos un **simulador de mundos habitables**. Cada planeta tiene condiciones iniciales (temperatura, agua, desigualdad, tasa de agotamiento de recursos, eventos extremos y otras) y evoluciona en el tiempo con cuatro variables de estado: plebeyos, élites, naturaleza y riqueza. El modelo base es HANDY (Motesharrei et al., 2014), que explica ciclos de crecimiento y colapso por el agotamiento de la naturaleza y la desigualdad. Lo extendimos con temperatura, agua y eventos extremos estocásticos.

Sobre ese simulador planteamos dos tareas de predicción:

- **Tarea (a):** dadas las condiciones iniciales, predecir la probabilidad de que la civilización **sobreviva** hasta el año T = 600.
- **Tarea (b):** dados los primeros k % de la trayectoria, predecir la supervivencia como **alerta temprana**, y medir cómo mejora la predicción al observar más historia.

## 1.3 Por qué el problema es interesante: la explicación se puede verificar

En la mayoría de aplicaciones de explicabilidad (XAI) no existe una "explicación verdadera" con la cual comparar. Aquí sí: el simulador es el mecanismo real y podemos **intervenir** un factor y observar qué ocurre. Por eso el eje del proyecto no es solo predecir, sino **medir qué tan fieles son las explicaciones** (SHAP, Integrated Gradients y, en la Fase 2, contrafactuales) frente a la realidad del simulador. Esto conecta con la advertencia de la sesión S15 del curso: *una explicación convincente no es una explicación correcta*, y debe validarse (Adebayo et al., 2018).

## 1.4 Alcance de esta entrega (Fase 1)

Esta versión incluye simulador validado, dataset, modelos para (a) y (b) con baselines y el cálculo de atribuciones. La validación de fidelidad contra el simulador, los contrafactuales, la interfaz, la presentación y el video quedan para la Fase 2.
