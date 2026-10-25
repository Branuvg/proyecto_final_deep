# 5. Herramientas aplicadas

> Sección parcial: se completará en la Fase 2 con las herramientas de la interfaz y los contrafactuales.

## 5.1 Herramientas y relación con el curso

[[tabla: Herramientas aplicadas y sesión del curso con la que se relacionan.]]
| Herramienta | Uso en el proyecto | Sesión del curso |
|---|---|---|
| PyTorch 2.11 (CUDA 12.8) | Redes escritas como `nn.Module` propios | S1, S3, S8 |
| Regresión logística (scikit-learn) | Baseline lineal; equivale a una red de una neurona | S1 |
| MLP con dropout y AdamW | Tarea (a) | S1, S8 |
| LSTM con secuencias empaquetadas | Tarea (b), alerta temprana | S3, S12 |
| LightGBM | Baseline de gradient boosting y modelo para TreeSHAP | no vista en clase |
| SHAP, Integrated Gradients (Captum) | Atribuciones | S15 |
| NumPy, SciPy | Simulador vectorizado y muestreo Latin Hypercube | no vista en clase |
| python-docx | Generación del documento a partir de la plantilla | no vista en clase |

**Herramientas no vistas en clase, en detalle.** *LightGBM* construye árboles de decisión de forma secuencial, cada uno corrigiendo el error del anterior, con crecimiento por hojas y histogramas para ser rápido; lo usamos por ser el estándar fuerte para datos tabulares y por ser compatible con TreeSHAP. *Latin Hypercube* es un muestreo estratificado que cubre cada factor de forma más uniforme que el azar simple con el mismo número de puntos. *Valores de Shapley*: reparten la diferencia entre la predicción y el baseline entre los factores, promediando su contribución marginal sobre todas las coaliciones; con 11 factores se calculan de forma exacta.

## 5.2 Historial de prompts de IA (resumen)

El historial completo está en `docs/PROMPTS.md`. Resumen de la Fase 1:

- **Prompt principal (Fase 1).** Pedía simulador, dataset, modelos, XAI y documento con reglas de honestidad (documentar si la red no supera al boosting, no ajustar hasta que "salga bien"). *Funcionó:* fijar de antemano el encuadre (surrogate, baselines obligatorios) y exigir pruebas y un escenario canónico.
- **Falló: verificación de ecuaciones contra el paper.** La IA no pudo abrir el PDF original y contrastó con un artículo posterior; la numeración de las ecuaciones queda `[VERIFICAR]`. *Por qué:* la herramienta de lectura no accede a ese PDF; sin el texto fuente la verificación es parcial.
- **Falló y se corrigió: umbral de colapso.** El primer umbral (30 % del pico) trataba como colapso unas oscilaciones amortiguadas; se detectó al graficar los escenarios canónicos.
- **Falló y se reportó: colapso irreversible de la naturaleza.** No se reproduce a 5.5·δ_opt; se renombró el escenario en lugar de ajustar parámetros.
- **Falló y se corrigió: comparación KernelSHAP.** Una diferencia de 0.12 se debía a la regularización L1 automática de KernelSHAP y no a un error del cálculo exacto.

**Cómo evaluamos a la IA:** pruebas unitarias, inspección visual de los escenarios, comprobaciones de eficiencia de las atribuciones y comparación entre tres métodos. **Contribución de la IA:** código y borradores; las decisiones de encuadre, la lectura de los resultados y la verificación contra el paper son del equipo [PENDIENTE de confirmar por el equipo].
