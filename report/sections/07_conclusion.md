# 7. Conclusión

## 7.1 Qué logramos

Construimos un simulador de civilizaciones basado en HANDY, lo verificamos contra el artículo original (lo que reveló y permitió corregir un error nuestro en la mortalidad de las élites), generamos 50 000 planetas con una verdad de referencia Monte Carlo y entrenamos modelos sustitutos con baselines, varias semillas e intervalos de confianza. La parte distintiva es que **medimos la fidelidad de las explicaciones contra el propio simulador**, algo que casi nunca es posible, y que **verificamos los contrafactuales en el simulador** y no solo en la red.

## 7.2 Qué aportó la red frente a los baselines

- En la tarea (a) el MLP supera a la regresión logística por un amplio margen (la frontera es muy no lineal) y a LightGBM por muy poco, pero con un intervalo de confianza que excluye el cero (ΔAUC = {ba[mlp_minus_lightgbm][roc_auc][1]:+.4f}). La red está además mejor calibrada y más cerca de la probabilidad verdadera Monte Carlo.
- En la alerta temprana (tarea b), la LSTM **no** es siempre la mejor: pierde con k = 0 %, empata con el MLP estático en ventanas cortas, gana con k = 40 % y empata con LightGBM con resúmenes con k = 60 %. Un modelo clásico bien alimentado compite con la red recurrente.

## 7.3 Qué tan fieles son las explicaciones

SHAP, Integrated Gradients y TreeSHAP concuerdan bastante con el efecto real de intervenir un factor (Spearman por planeta de {fx[bd_tree_sim_spearman]:.2f} a {fx[bd_ig_sim_spearman]:.2f} en planetas de frontera), superan con holgura a un orden aleatorio en las curvas de eliminación y pasan la prueba de cordura. Pero están lejos de ser exactas: el factor más importante solo se acierta en cerca de la mitad de los planetas y parte de la brecha viene del propio modelo sustituto. **Una explicación convincente no es necesariamente una explicación correcta; con un simulador pudimos cuantificar cuánto.**

## 7.4 Contrafactuales

Para los planetas que el modelo predice que colapsarán, el método de gradiente sobre las palancas accionables encuentra cambios que el simulador confirma en el {cf[gradient][simulator_valid]:.0%} de los casos, y la supervivencia real media pasa de {cf[gradient][mean_true_p_before]:.2f} a {cf[gradient][mean_true_p_after]:.2f}. Cambiar una sola palanca no alcanza en una parte de los casos.

## 7.5 Limitaciones y trabajo futuro

- Las intervenciones de un factor a la vez no capturan interacciones: un siguiente paso es medir la fidelidad con intervenciones conjuntas.
- Las constantes de temperatura y agua son supuestos nuestros; calibrarlas con datos reales (o con un modelo climático) haría el simulador más creíble.
- No reproducimos con exactitud el umbral del colapso Tipo-N del paper; faltaría su código o un estudio de sensibilidad al integrador.
- Probar otros arreglos del sustituto (por ejemplo, una red neuronal que aproxime directamente la trayectoria) y comparar más métodos de explicación (LIME, explicaciones basadas en conceptos).
- La lección sobre el uso de la IA: acelera el trabajo, pero verificar contra la fuente (aquí, el paper original) evitó que publicáramos resultados sobre un simulador mal implementado.
