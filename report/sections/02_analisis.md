# 2. Análisis

## 2.1 El modelo HANDY

HANDY describe una sociedad con plebeyos (x_C), élites (x_E), naturaleza (y) y riqueza acumulada (w). La naturaleza se regenera de forma logística y se agota por el trabajo de los plebeyos; la riqueza es lo producido menos lo consumido; la mortalidad sube cuando la riqueza no alcanza para el consumo mínimo (hambruna). Verificamos las ecuaciones contra el texto del artículo original (Motesharrei et al., 2014; copia del autor), con su numeración:

- Ec. (3): dx_C/dt = β_C x_C − α_C x_C; dx_E/dt = β_E x_E − α_E x_E; dy/dt = γ y (λ − y) − δ x_C y; dw/dt = δ x_C y − C_C − C_E.
- Ec. (4): C_C = min(1, w/w_th) s x_C y C_E = min(1, w/w_th) κ s x_E.
- Ec. (5): w_th = ρ x_C + κ ρ x_E.
- Ec. (6): α_C = α_m + max(0, 1 − C_C/(s x_C)) (α_M − α_m) y α_E = α_m + max(0, 1 − C_E/(s x_E)) (α_M − α_m).

Los parámetros de la Tabla 1 del paper (β = 0.03, α_m = 0.01, α_M = 0.07, s = 5·10⁻⁴, ρ = 5·10⁻³, γ = 0.01, λ = 100) coinciden con los que usamos. Como comprobación independiente, la Ec. (10) da δ\* = 2ηs/λ = 6.67·10⁻⁶ con η = (α_M − β_C)/(α_M − α_m) = 2/3, y la Ec. (11) una capacidad de carga máxima χ_M = 75 000; nuestro simulador converge exactamente a ese equilibrio (y\* = λ/2 = 50).

**Un error encontrado al verificar.** En la primera versión escribimos α_E con κ en el denominador, de modo que élites y plebeyos compartían el mismo término de hambruna. El paper usa s x_E (sin κ): las élites solo sufren hambruna cuando w/w_th ≤ 1/κ. Lo corregimos (con un test unitario) y regeneramos todo el dataset y los resultados; los números de una versión previa de este documento quedaron obsoletos.

**Escenarios del paper reproducidos (Figura 1).** Equilibrio (1×δ\*), oscilaciones amortiguadas (2.5×δ\*), ciclos de colapso reversibles (4×δ\*), colapso Tipo-L (κ = 100 con una semilla minúscula de élites: la población muere pero la naturaleza se recupera), colapso Tipo-N irreversible (κ = 100, δ = 10⁻⁴) y el caso equitativo (κ = 1, x_E(0) = 25).

**Lo que reproducimos solo a medias: el umbral del Tipo-N igualitario.** El paper reporta que 4×δ\* es reversible y 5.5×δ\* irreversible, usando Euler con paso de 1 año y precisión simple. Con RK4 nuestra naturaleza se recupera hasta ≈ 9.25×δ\*; con Euler (dt = 1) y precisión simple el colapso irreversible aparece desde ≈ 3.25×δ\*. El umbral del paper (entre 4× y 5.5×) queda entre ambos: la irreversibilidad depende de la discretización. Una explicación plausible, que no verificamos por separado, es que con Euler y dt = 1 el paso de la naturaleza se vuelve negativo cuando δ·x_C supera 1 + γλ y se recorta a cero exacto, mientras que en RK4 la naturaleza queda en valores ≈ 10⁻³⁰ desde los que puede regenerarse. Usamos RK4 en el dataset por ser más preciso y mostramos ambos esquemas en la Figura 1.

![Escenarios canónicos de HANDY reproducidos con el simulador (RK4, salvo el panel inferior izquierdo: Euler, dt = 1 año, precisión simple).](report/figures/fig_escenarios_handy.png)

## 2.2 Extensiones y factores

Se extendió el modelo con temperatura (reduce la regeneración y sube la mortalidad base), agua (limita la capacidad de carga según la demanda de la población) y eventos extremos (proceso de Poisson que destruye fracciones aleatorias de naturaleza y riqueza). El resultado son 11 factores, cada uno marcado como accionable o no accionable (Tabla 1). Los accionables son los únicos que pueden modificarse en los contrafactuales.

[[tabla: Factores del simulador, rango muestreado y carácter accionable.]]
<<TABLE factors>>

## 2.3 Definición de colapso y supervivencia

Un planeta **colapsa** en el primer año en que su población total cae por debajo del máximo entre el 10 % de su pico histórico y el 20 % de su población inicial. **Sobrevive** si no colapsa durante los 600 años simulados. Se guarda además el tiempo de colapso. Las decisiones y alternativas están en `docs/DECISIONES.md` (D-06).

## 2.4 Estocasticidad y probabilidad verdadera

Por los eventos de Poisson, las mismas condiciones iniciales pueden dar resultados distintos. Para 500 planetas de test corrimos 32 réplicas con semillas de eventos distintas y obtuvimos la probabilidad verdadera P(sobrevive | parámetros), con resolución de 1/32. Esa cantidad sirve para evaluar la calibración y como verdad de referencia para XAI. En el subconjunto Monte Carlo el {e[mc_share_intermediate]:.1%} de los planetas tiene una probabilidad entre 0.05 y 0.95: el resultado es en buena parte determinista dados los parámetros, pero hay una fracción importante de casos genuinamente inciertos.

## 2.5 Análisis exploratorio

El dataset tiene {e[n]} planetas con una tasa de supervivencia de {e[survival_rate]:.1%} (train {e[surv_train]:.1%}, validación {e[surv_val]:.1%}, test {e[surv_test]:.1%}), dentro del rango objetivo de 30 % a 70 %. Los tiempos de colapso tienen mediana de {e[t_collapse_quantiles][2]:.0f} años (percentiles 5 y 95: {e[t_collapse_quantiles][0]:.0f} y {e[t_collapse_quantiles][4]:.0f}). La supervivencia no es monótona en la tasa de depleción δ: valores muy bajos también colapsan porque la sociedad no logra acumular riqueza para despegar. Las Figuras 2 a 4 muestran las clases, las trayectorias y la tasa de supervivencia por factor.

![Distribución de clases y de tiempos de colapso.](report/figures/fig_eda_clases_tiempos.png)

![Tres trayectorias que sobreviven (arriba) y tres que colapsan (abajo). Azul: población; verde: naturaleza; naranja: eventos extremos; rojo: tiempo de colapso.](report/figures/fig_eda_trayectorias.png)

![Tasa de supervivencia por decil de cada factor.](report/figures/fig_eda_supervivencia_por_factor.png)
