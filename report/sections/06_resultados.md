# 6. Resultados

Todos los números de esta sección se generan desde `outputs/*/results.json`. Los modelos se entrenaron con {n_seeds} semillas (media ± desviación estándar); los intervalos de confianza (IC) del 95 % salen de un *bootstrap* pareado de 1000 remuestreos de los planetas de test, con la semilla principal.

## 6.1 Tarea (a): condiciones iniciales → supervivencia

Conjunto de test: {n_test} planetas. MAE: error absoluto medio entre la probabilidad predicha y la probabilidad verdadera estimada con 32 réplicas Monte Carlo, en los 500 planetas del subconjunto.

[[tabla: Métricas de la tarea (a) en el conjunto de test (media ± desviación sobre las semillas).]]
<<TABLE task_a>>

[[tabla: Diferencias pareadas entre modelos con IC 95 % (bootstrap sobre los planetas de test).]]
<<TABLE task_a_boot>>

La regresión logística apenas llega a ROC-AUC {a[logistic][roc_auc]:.3f}: la frontera de supervivencia es fuertemente no lineal. El MLP supera al gradient boosting (ROC-AUC {a[mlp][roc_auc]:.3f} ± {sa[mlp][roc_auc]:.3f} frente a {a[lightgbm][roc_auc]:.3f} ± {sa[lightgbm][roc_auc]:.3f}; Brier {a[mlp][brier]:.4f} frente a {a[lightgbm][brier]:.4f}). La diferencia es pequeña pero **estadísticamente clara**: ΔAUC = {ba[mlp_minus_lightgbm][roc_auc][1]:+.4f} con IC 95 % [{ba[mlp_minus_lightgbm][roc_auc][0]:+.4f}, {ba[mlp_minus_lightgbm][roc_auc][2]:+.4f}], que no contiene el cero, y la desviación entre semillas es menor que la diferencia. El MLP también está mejor calibrado (ECE {a[mlp][ece]:.4f} frente a {a[lightgbm][ece]:.4f}) y su error frente a la probabilidad verdadera Monte Carlo es {a[mlp][mc_mae]:.3f}, frente a {a[lightgbm][mc_mae]:.3f} de LightGBM.

![Calibración (izquierda) y predicción frente a la probabilidad verdadera Monte Carlo (derecha).](report/figures/fig_tarea_a_calibracion.png)

## 6.2 Tarea (b): primeros k pasos → supervivencia

Solo se evalúan los planetas que siguen vivos al final de la ventana observada; la Tabla 6 muestra cuántos quedan fuera. A mayor k quedan menos planetas y la proporción de supervivientes entre ellos sube, de modo que **las métricas entre distintos k no son directamente comparables** (cambia la población evaluada). La búsqueda de hiperparámetros (Tabla 8) eligió una LSTM de 2 capas y {hp_b[hidden]} unidades; las 8 configuraciones difieren poco entre sí en la pérdida de validación.

[[tabla: ROC-AUC según el porcentaje k de trayectoria observada (media ± desviación sobre las semillas).]]
<<TABLE task_b_auc>>

[[tabla: Planetas excluidos por haber colapsado dentro de la ventana observada.]]
<<TABLE task_b_excl>>

[[tabla: Diferencias de ROC-AUC de la LSTM frente a cada alternativa, con IC 95 % (bootstrap pareado sobre los planetas de test). Un intervalo que contiene el cero indica empate.]]
<<TABLE task_b_boot>>

[[tabla: Búsqueda aleatoria de hiperparámetros de la LSTM (pérdida de validación con truncamiento aleatorio; menor es mejor).]]
<<TABLE lstm_search>>

Lo que observamos:

- **Sin trayectoria (k = 0 %) la LSTM no es la mejor opción.** Ve solo los parámetros y el estado inicial, y su ROC-AUC ({b[0][lstm][roc_auc]:.3f}) es inferior al del MLP de la tarea (a) ({b[0][mlp_static_only][roc_auc]:.3f}); la diferencia es de {bb[0][lstm_minus_mlp_static_only][roc_auc][1]:+.4f} [{bb[0][lstm_minus_mlp_static_only][roc_auc][0]:+.4f}, {bb[0][lstm_minus_mlp_static_only][roc_auc][2]:+.4f}]. Un modelo que sirve para cualquier k paga un costo cuando k es cero.
- **Con ventanas cortas (5–20 %) la historia aporta poco.** La LSTM y el MLP que solo ve los parámetros están prácticamente empatados (k = 5 %: {bb[5][lstm_minus_mlp_static_only][roc_auc][1]:+.4f} [{bb[5][lstm_minus_mlp_static_only][roc_auc][0]:+.4f}, {bb[5][lstm_minus_mlp_static_only][roc_auc][2]:+.4f}]; k = 20 %: {bb[20][lstm_minus_mlp_static_only][roc_auc][1]:+.4f} [{bb[20][lstm_minus_mlp_static_only][roc_auc][0]:+.4f}, {bb[20][lstm_minus_mlp_static_only][roc_auc][2]:+.4f}]). En esas ventanas la LSTM sí supera a LightGBM con resúmenes (≈ +0.007 de AUC, IC que excluye el cero).
- **La ventaja de observar la trayectoria aparece con ventanas largas.** Con k = 40 % la LSTM ({b[40][lstm][roc_auc]:.3f}) supera al MLP estático ({b[40][mlp_static_only][roc_auc]:.3f}; Δ = {bb[40][lstm_minus_mlp_static_only][roc_auc][1]:+.4f}) y a LightGBM con resúmenes ({b[40][lightgbm_summary][roc_auc]:.3f}; Δ = {bb[40][lstm_minus_lightgbm_summary][roc_auc][1]:+.4f}), con IC que excluyen el cero. Con k = 60 % el MLP estático cae a {b[60][mlp_static_only][roc_auc]:.3f} (la población evaluada es más difícil), mientras que la LSTM llega a {b[60][lstm][roc_auc]:.3f} y su Brier baja a {b[60][lstm][brier]:.4f}, frente a {b[60][mlp_static_only][brier]:.4f} del MLP estático.
- **A k = 60 % la LSTM y LightGBM con resúmenes empatan** (ROC-AUC {b[60][lstm][roc_auc]:.3f} frente a {b[60][lightgbm_summary][roc_auc]:.3f}; Δ = {bb[60][lstm_minus_lightgbm_summary][roc_auc][1]:+.4f} [{bb[60][lstm_minus_lightgbm_summary][roc_auc][0]:+.4f}, {bb[60][lstm_minus_lightgbm_summary][roc_auc][2]:+.4f}]). La ventaja de la LSTM sobre el MLP aplanado es pequeña ({bb[60][lstm_minus_mlp_window][roc_auc][1]:+.4f}) aunque su IC excluye el cero. Con resúmenes bien diseñados, un modelo sin recurrencia alcanza lo mismo.
- **Ablación:** sin parámetros estáticos la LSTM parte de {b[0][lstm_no_static][roc_auc]:.3f} (k = 0 %, cerca del azar) y llega a {b[60][lstm_no_static][roc_auc]:.3f} (k = 60 %), por debajo de la LSTM completa en todas las ventanas. La trayectoria sola acaba siendo muy informativa, pero con ventanas cortas los parámetros son indispensables.

![ROC-AUC y Brier frente al porcentaje de trayectoria observada (barras: desviación entre semillas).](report/figures/fig_tarea_b_curva_k.png)

## 6.3 XAI: atribuciones

Las atribuciones se calcularon sobre el logit de supervivencia, con el planeta medio de train como baseline común (Tabla 9). Comprobaciones numéricas: error máximo de eficiencia del SHAP exacto {x[shap_mlp]:.1e}, diferencia máxima con KernelSHAP {x[exact_vs_kernelshap_max_abs_diff]:.1e} y error máximo de completitud de Integrated Gradients {x[ig_mlp]:.3f} logits.

[[tabla: Importancia global (media del valor absoluto de la atribución, en logits) por método.]]
<<TABLE xai_importance>>

Los tres métodos coinciden en el orden global: δ (tasa de depleción) y κ (desigualdad) son los más influyentes. La correlación de Spearman entre el ranking global de SHAP y el de IG es {g[spearman_global_shap_vs_ig]:.2f}, y con TreeSHAP {g[spearman_global_shap_vs_tree]:.2f}. A nivel local el acuerdo es menor: Spearman medio por planeta {g[mean_local_spearman_shap_vs_ig]:.2f} entre SHAP e IG y {g[mean_local_spearman_shap_mlp_vs_tree]:.2f} entre el SHAP del MLP y TreeSHAP del boosting. Alrededor del {act[SHAP (MLP)]:.0%} de la atribución total del SHAP del MLP recae en factores accionables.

![Importancia global por método.](report/figures/fig_xai_importancia_global.png)

![Beeswarm del SHAP del MLP.](report/figures/fig_xai_beeswarm.png)

![Cascada SHAP de un planeta que sobrevive.](report/figures/fig_xai_cascada_1.png)

![Cascada SHAP de otro planeta que sobrevive.](report/figures/fig_xai_cascada_2.png)

![Cascada SHAP de un planeta que colapsa.](report/figures/fig_xai_cascada_3.png)

![Cascada SHAP de otro planeta que colapsa.](report/figures/fig_xai_cascada_4.png)

## 6.4 Fidelidad de las explicaciones frente al modelo y al simulador

Que los métodos coincidan entre sí no prueba que sean correctos, así que los comparamos con el efecto de **devolver cada factor a su valor base**, en el modelo y en el simulador. Se evalúan dos conjuntos: {fx[mc_n]} planetas del subconjunto Monte Carlo (muestra aleatoria; el {fx[mc_inter]:.0%} tiene probabilidad real intermedia) y {fx[bd_n]} planetas cercanos a la frontera de decisión del modelo (el {fx[bd_inter]:.0%} con probabilidad real intermedia). Solo cuentan los planetas con algún efecto individual de al menos {fx[thr]:.2f} en la probabilidad real.

[[tabla: Fidelidad de las atribuciones: correlación de Spearman por planeta con el efecto de la intervención (IC 95 %), acuerdo de signo, y coincidencia del factor más importante (top-1) y de los tres más importantes (top-3).]]
<<TABLE fidelity>>

**Fidelidad a la realidad (simulador).** En los planetas cercanos a la frontera, la correlación con el efecto real es {fx[bd_shap_sim_spearman]:.2f} para SHAP, {fx[bd_ig_sim_spearman]:.2f} para IG y {fx[bd_tree_sim_spearman]:.2f} para TreeSHAP; en los planetas Monte Carlo aleatorios, {fx[mc_shap_sim_spearman]:.2f}, {fx[mc_ig_sim_spearman]:.2f} y {fx[mc_tree_sim_spearman]:.2f}. El signo concuerda en cerca del {fx[bd_shap_sim_sign_agreement]:.0%} de los efectos apreciables, pero el factor más importante solo coincide en {fx[bd_shap_sim_top1]:.0%} (frontera) a {fx[mc_shap_sim_top1]:.0%} (Monte Carlo) de los planetas. Las explicaciones **capturan buena parte de la estructura causal real, pero están lejos de ser exactas**.

**Fidelidad al modelo.** Frente al efecto que la intervención tiene sobre el propio modelo, la correlación sube a {fx[bd_shap_mod_spearman]:.2f}–{fx[bd_tree_mod_spearman]:.2f} (frontera). La diferencia restante se debe sobre todo a que SHAP reparte las interacciones entre factores y la intervención de un factor a la vez no.

**Error del modelo frente a error de la explicación.** El efecto que predice el propio MLP al intervenir un factor correlaciona {fx[bd_mlp_vs_sim_spearman]:.2f} (frontera) y {fx[mc_mlp_vs_sim_spearman]:.2f} (Monte Carlo) con el efecto real; es un techo aproximado para cualquier explicación del MLP. La brecha entre la explicación y la realidad tiene por lo tanto dos fuentes: el modelo sustituto (su propio efecto no coincide del todo con el simulador) y el método de explicación (SHAP no coincide del todo con el efecto que el modelo tiene al intervenir un factor).

**Importancia global.** El orden de importancia real (simulador) y el de las atribuciones es parecido para los factores principales (Figura 14): δ es el más importante en ambos. Normalizadas para sumar 1, las atribuciones de SHAP dan más peso relativo a δ y κ que el simulador y menos a los eventos extremos (tasa y severidad).

![Spearman por planeta entre las atribuciones y el efecto de intervenir cada factor, en el modelo y en el simulador. La línea azul es el propio MLP frente al simulador.](report/figures/fig_fidelidad_spearman.png)

![Importancia global: efecto real (simulador) frente a atribuciones, normalizadas para sumar 1.](report/figures/fig_fidelidad_global.png)

**Curvas de eliminación.** Devolver los factores al valor base en el orden que dicta SHAP cambia la probabilidad real de supervivencia en {fx[del_shap_attr_sim]:+.3f} en promedio (IG: {fx[del_ig_attr_sim]:+.3f}; TreeSHAP: {fx[del_tree_attr_sim]:+.3f}), frente a {fx[del_shap_rand_sim]:+.3f} con un orden aleatorio y {fx[del_shap_glob_sim]:+.3f} con la importancia global. Las atribuciones locales son claramente mejores guías que el orden aleatorio y que la importancia global, tanto en el modelo ({fx[del_shap_attr_model]:+.2f} frente a {fx[del_shap_rand_model]:+.2f} logits) como en el simulador. (El orden aleatorio sube la supervivencia en promedio porque devolver al valor medio a los planetas atípicos los acerca al planeta medio, que sobrevive.)

![Curvas de eliminación en el modelo (izquierda) y en el simulador (derecha).](report/figures/fig_fidelidad_eliminacion.png)

**Prueba de cordura (Adebayo et al., 2018).** Con el MLP reemplazado por uno con pesos aleatorios, el SHAP correlaciona {fx[rnd_vs_trained]:.2f} con el del modelo entrenado, y su concordancia con el simulador cae de {fx[trained_sim]:.2f} a {fx[rnd_sim]:.2f} (acuerdo de signo de {fx[trained_sign]:.0%} a {fx[rnd_sign]:.0%}). Las explicaciones **dependen del modelo entrenado**, como debe ser.

![Atribución SHAP frente al efecto real de cada (planeta, factor).](report/figures/fig_fidelidad_dispersion.png)

## 6.5 Contrafactuales verificados con el simulador

Se aplicaron a los {cf[n_predicted_collapse]} planetas Monte Carlo que el modelo predice que colapsarán ({cf[n_truly_collapse]} colapsan de verdad según el simulador; en {cf[n_model_error]} el modelo se equivocaba).

[[tabla: Contrafactuales sobre palancas accionables (objetivo: P del modelo ≥ 0.8), validados con 64 réplicas del simulador.]]
<<TABLE cf>>

El método de gradiente encuentra un contrafactual para el {cf[gradient][solved_share]:.0%} de los planetas; el {cf[gradient][simulator_valid]:.0%} de los que realmente iban a colapsar sobrevive en el simulador, y su probabilidad real media sube de {cf[gradient][mean_true_p_before]:.2f} a {cf[gradient][mean_true_p_after]:.2f}, con {cf[gradient][mean_factors_changed]:.1f} palancas cambiadas en promedio. Cambiar una sola palanca resuelve el {cf[single_factor][solved_share]:.0%} de los casos y es válido en el simulador en el {cf[single_factor][simulator_valid]:.0%}. La validez medida en el modelo ({cf[gradient][model_valid]:.0%}) y en el simulador ({cf[gradient][simulator_valid]:.0%}) es parecida, y en el {cfx[disagree]:.0%} de los planetas que realmente iban a colapsar el modelo declara válido un contrafactual que el simulador no confirma: ese es el costo de confiar en el sustituto. Ningún contrafactual modifica factores no accionables (cambio máximo {cf[gradient][nonactionable_changed]:.1f}).

![Validez de los contrafactuales en el modelo y en el simulador (izquierda) y efecto real (derecha).](report/figures/fig_contrafactuales.png)

## 6.6 Limitaciones

- Cinco semillas por modelo, pero **un solo conjunto de datos y un solo simulador**; los IC de las atribuciones consideran solo la variabilidad entre planetas, no entre semillas del modelo.
- La búsqueda de hiperparámetros de la LSTM es pequeña (8 configuraciones) y los baselines de (b) usan configuración fija.
- La intervención de **un factor a la vez** no captura interacciones; por eso la fidelidad al modelo y a la realidad no puede llegar a 1.
- Las constantes de temperatura y agua son supuestos nuestros sin calibración empírica; agua y eficiencia hídrica tienen un efecto real casi nulo sobre la supervivencia (ver D-05).
- La estimación de la verdad en el simulador usa 32–64 réplicas; los efectos menores a ≈ 0.10 se ignoran por ruido.
- No reproducimos con exactitud el umbral del colapso Tipo-N igualitario del paper (sección 2.1).
- El resultado es en buena parte determinista dados los parámetros, lo que reduce el margen de mejora de cualquier modelo.
