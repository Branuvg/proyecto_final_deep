# 6. Resultados (preliminares)

Todos los números de esta sección se generan desde `outputs/*/results.json`; son de **una sola corrida con una semilla**, sin intervalos de confianza. Las diferencias pequeñas entre modelos no deben interpretarse como significativas hasta repetir con varias semillas (Fase 2).

## 6.1 Tarea (a): condiciones iniciales → supervivencia

Conjunto de test: {n_test} planetas. MAE: error absoluto medio entre la probabilidad predicha y la probabilidad verdadera estimada con 32 réplicas Monte Carlo, en los 500 planetas del subconjunto.

[[tabla: Métricas de la tarea (a) en el conjunto de test.]]
<<TABLE task_a>>

La regresión logística apenas supera al azar (ROC-AUC {a[logistic][roc_auc]:.3f}): la frontera de supervivencia es fuertemente no lineal. El MLP supera al gradient boosting (ROC-AUC {a[mlp][roc_auc]:.3f} frente a {a[lightgbm][roc_auc]:.3f}; Brier {a[mlp][brier]:.4f} frente a {a[lightgbm][brier]:.4f}) y está mejor calibrado (ECE {a[mlp][ece]:.4f} frente a {a[lightgbm][ece]:.4f}). Frente a la probabilidad verdadera, el MAE del MLP es {a[mlp][mc_mae]:.3f} y el de LightGBM {a[lightgbm][mc_mae]:.3f}. Con una sola semilla no podemos afirmar que la ventaja sea estable; sí es consistente en todas las métricas. Recordemos además que el {e[mc_share_intermediate]:.1%} de los planetas Monte Carlo tiene probabilidad intermedia, por lo que la parte realmente estocástica del problema es pequeña.

![Calibración (izquierda) y predicción frente a la probabilidad verdadera Monte Carlo (derecha).](report/figures/fig_tarea_a_calibracion.png)

## 6.2 Tarea (b): primeros k pasos → supervivencia

Solo se evalúan los planetas que siguen vivos al final de la ventana; la Tabla 5 muestra cuántos quedan fuera. A mayor k quedan menos planetas y la proporción de supervivientes entre ellos sube, de modo que **las métricas entre distintos k no son directamente comparables** (cambia la población evaluada).

[[tabla: ROC-AUC según el porcentaje k de trayectoria observada.]]
<<TABLE task_b_auc>>

[[tabla: Planetas excluidos por haber colapsado dentro de la ventana observada.]]
<<TABLE task_b_excl>>

Lo que observamos:

- **Con k = 0 %** la LSTM ve solo el estado inicial (idéntico para todos) y los parámetros: llega a {b[0][lstm][roc_auc]:.3f}, por debajo del MLP de la tarea (a) ({b[0][mlp_static_only][roc_auc]:.3f}). La LSTM no es la mejor opción cuando no hay trayectoria.
- **Con k = 60 %** la LSTM ({b[60][lstm][roc_auc]:.3f}) y LightGBM con resúmenes ({b[60][lightgbm_summary][roc_auc]:.3f}) quedan prácticamente empatados, y ambos superan al MLP que solo ve los parámetros ({b[60][mlp_static_only][roc_auc]:.3f}). Su Brier baja a {b[60][lstm][brier]:.4f} y {b[60][lightgbm_summary][brier]:.4f}, frente a {b[60][mlp_static_only][brier]:.4f} del MLP estático: observar la trayectoria sí ayuda en ventanas largas.
- **En ventanas cortas la historia aporta poco**: el MLP estático ya alcanza ROC-AUC {b[5][mlp_static_only][roc_auc]:.3f} con k = 5 %, y ningún modelo con trayectoria lo supera de forma clara. La curva de desempeño frente a k es casi plana para los modelos que usan los parámetros.
- **Ablación:** sin parámetros estáticos la LSTM pasa de {b[0][lstm_no_static][roc_auc]:.3f} (k = 0 %, equivale a azar) a {b[60][lstm_no_static][roc_auc]:.3f} (k = 60 %). La trayectoria por sí sola acaba siendo muy informativa, pero con ventanas cortas los parámetros son indispensables.

![ROC-AUC y Brier frente al porcentaje de trayectoria observada.](report/figures/fig_tarea_b_curva_k.png)

## 6.3 XAI: atribuciones preliminares

Las atribuciones se calcularon sobre el logit de supervivencia, con el planeta medio de train como baseline común (Tabla 6). Las comprobaciones numéricas son: error máximo de eficiencia del SHAP exacto {x[shap_mlp]:.1e}, diferencia máxima con KernelSHAP {x[exact_vs_kernelshap_max_abs_diff]:.1e} y error máximo de completitud de Integrated Gradients {x[ig_mlp]:.3f} logits.

[[tabla: Importancia global (media del valor absoluto de la atribución, en logits) por método.]]
<<TABLE xai_importance>>

Los tres métodos coinciden en el orden global: δ (tasa de depleción), κ (desigualdad) y la fracción inicial de élites son los factores más influyentes. La correlación de Spearman entre el ranking global de SHAP y el de IG es {g[spearman_global_shap_vs_ig]:.2f}, y con TreeSHAP {g[spearman_global_shap_vs_tree]:.2f}. A nivel local el acuerdo es menor: Spearman medio por planeta {g[mean_local_spearman_shap_vs_ig]:.2f} entre SHAP e IG y {g[mean_local_spearman_shap_mlp_vs_tree]:.2f} entre el SHAP del MLP y TreeSHAP del boosting. Cerca del {act[SHAP (MLP)]:.0%} de la atribución total del SHAP del MLP recae en factores accionables.

**Advertencia:** que los métodos coincidan entre sí no prueba que sean fieles al simulador. Esa verificación (intervenir cada factor en el modelo y en el simulador) es el núcleo de la Fase 2. Las figuras 8 a 12 muestran el *beeswarm* y cuatro explicaciones locales (valores estandarizados).

![Importancia global por método.](report/figures/fig_xai_importancia_global.png)

![Beeswarm del SHAP del MLP.](report/figures/fig_xai_beeswarm.png)

![Cascada SHAP de un planeta que sobrevive.](report/figures/fig_xai_cascada_1.png)

![Cascada SHAP de otro planeta que sobrevive.](report/figures/fig_xai_cascada_2.png)

![Cascada SHAP de un planeta que colapsa.](report/figures/fig_xai_cascada_3.png)

![Cascada SHAP de otro planeta que colapsa.](report/figures/fig_xai_cascada_4.png)

## 6.4 Limitaciones conocidas

- Una semilla por modelo y sin intervalos de confianza.
- La LSTM no tuvo búsqueda de hiperparámetros; los baselines de (b) usan configuración fija.
- Las constantes de las extensiones (tolerancia térmica, mortalidad por grado, demanda de agua) son supuestos nuestros sin calibración empírica.
- Agua, eficiencia hídrica y eventos extremos tienen un efecto pequeño sobre la supervivencia agregada; los eventos sobre todo mueven el momento del colapso.
- El resultado es mayormente determinista dados los parámetros, lo que reduce el margen de mejora de cualquier modelo.
- Las ecuaciones de HANDY siguen pendientes de verificarse contra el PDF original.
