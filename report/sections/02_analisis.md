# 2. Análisis

## 2.1 El modelo HANDY

HANDY describe una sociedad con plebeyos (x_C), élites (x_E), naturaleza (y) y riqueza acumulada (w). La naturaleza se regenera de forma logística y se agota por el trabajo de los plebeyos; la riqueza es lo producido menos lo consumido; la mortalidad sube cuando la riqueza no alcanza para el consumo mínimo (hambruna). Las ecuaciones implementadas son:

- Plebeyos: dx_C/dt = β x_C − α_C x_C. Élites: dx_E/dt = β x_E − α_E x_E.
- Naturaleza: dy/dt = γ y (λ − y) − δ x_C y.
- Riqueza: dw/dt = δ x_C y − C_C − C_E.
- Consumo: C_C = min(1, w/w_th) s x_C y C_E = min(1, w/w_th) κ s x_E, con umbral de riqueza w_th = ρ x_C + κ ρ x_E.
- Mortalidad: α = α_m + max(0, 1 − C/(s x)) (α_M − α_m).

La numeración de las ecuaciones del paper original está pendiente de verificar contra el PDF (`[VERIFICAR]`, ver `docs/DECISIONES.md`, D-02).

Con δ_opt = 6.67·10⁻⁶ el sistema converge al equilibrio analítico y* = λ/2 = 50 y x_C* = γ(λ − y*)/δ = 75 000, lo que usamos como prueba de la implementación. Reproducimos de forma cualitativa el equilibrio, las oscilaciones amortiguadas, los ciclos de colapso y el colapso irreversible por desigualdad (Figura 1). **No** logramos reproducir el colapso irreversible de la naturaleza a 5.5·δ_opt: en nuestra implementación la naturaleza se recupera y el sistema cicla.

![Escenarios canónicos de HANDY reproducidos con el simulador.](report/figures/fig_escenarios_handy.png)

## 2.2 Extensiones y factores

Se extendió el modelo con temperatura (reduce la regeneración y sube la mortalidad base), agua (limita la capacidad de carga según la demanda de la población) y eventos extremos (proceso de Poisson que destruye fracciones aleatorias de naturaleza y riqueza). El resultado son 11 factores, cada uno marcado como accionable o no accionable (Tabla 1). Los accionables serán los únicos que puedan modificarse en los contrafactuales de la Fase 2.

[[tabla: Factores del simulador y su carácter accionable.]]
<<TABLE factors>>

## 2.3 Definición de colapso y supervivencia

Un planeta **colapsa** en el primer año en que su población total cae por debajo del máximo entre el 10 % de su pico histórico y el 20 % de su población inicial. **Sobrevive** si no colapsa durante los 600 años simulados. Se guarda además el tiempo de colapso. Las decisiones y alternativas están en `docs/DECISIONES.md` (D-06).

## 2.4 Estocasticidad y probabilidad verdadera

Por los eventos de Poisson, las mismas condiciones iniciales pueden dar resultados distintos. Para 500 planetas de test corrimos 32 réplicas con semillas de eventos distintas y obtuvimos la probabilidad verdadera P(sobrevive | parámetros), con resolución de 1/32. Esa cantidad sirve para evaluar la calibración y, en la Fase 2, como verdad de referencia para XAI. Un hallazgo importante del análisis: solo el {e[mc_share_intermediate]:.1%} de esos planetas tiene una probabilidad entre 0.05 y 0.95; **el resultado es mayormente determinista dados los parámetros**, y eso limita cuánto puede aportar observar la trayectoria.

## 2.5 Análisis exploratorio

El dataset tiene {e[n]} planetas con una tasa de supervivencia de {e[survival_rate]:.1%} (train {e[surv_train]:.1%}, validación {e[surv_val]:.1%}, test {e[surv_test]:.1%}), dentro del rango objetivo de 30 % a 70 %. Los tiempos de colapso tienen mediana de {e[t_collapse_quantiles][2]:.0f} años (percentiles 5 y 95: {e[t_collapse_quantiles][0]:.0f} y {e[t_collapse_quantiles][4]:.0f}). La supervivencia no es monótona en la tasa de depleción δ: valores muy bajos también colapsan porque la sociedad no logra acumular riqueza para despegar. Las Figuras 2 a 4 muestran las clases, las trayectorias y la tasa de supervivencia por factor.

![Distribución de clases y de tiempos de colapso.](report/figures/fig_eda_clases_tiempos.png)

![Tres trayectorias que sobreviven (arriba) y tres que colapsan (abajo). Azul: población; verde: naturaleza; naranja: eventos extremos; rojo: tiempo de colapso.](report/figures/fig_eda_trayectorias.png)

![Tasa de supervivencia por decil de cada factor.](report/figures/fig_eda_supervivencia_por_factor.png)
