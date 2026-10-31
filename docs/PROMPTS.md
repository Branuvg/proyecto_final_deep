# Historial de prompts de IA

La rúbrica pide los prompts relevantes, qué funcionó, qué falló y por qué, y cómo evaluamos las respuestas. Esta bitácora cubre las Fases 1 y 2. El prompt de la Fase 1 (P-01) fue el primer prompt del proyecto como tal; el equipo confirmó que no hay prompts anteriores que registrar.

## P-01 · Prompt de la Fase 1 (`prompt_claude_code_fase1.md`)
**Qué pedimos:** simulador HANDY extendido, dataset de ~50 000 planetas, modelos (MLP, LSTM) con baselines, atribuciones XAI y el documento, sin commits, con documentación honesta.

**Qué funcionó**
- El prompt fijaba decisiones de diseño (surrogate, baselines obligatorios, verificación contra el simulador en la Fase 2). Eso evitó que la IA eligiera un tema "fácil" o un modelo sin comparación.
- Pedir "documenta honestamente si la red no supera al boosting" produjo una comparación limpia: el MLP sí supera a LightGBM en la tarea (a), pero solo por ~0.006 de ROC-AUC.
- Exigir `tests` y un escenario canónico sirvió para detectar problemas reales (ver abajo).

**Qué falló o necesitó corrección, y por qué**
1. *Verificación de ecuaciones contra el paper:* la IA no pudo abrir el PDF original (error de certificado del sitio y un PDF binario ilegible por la herramienta). Se contrastó contra una reproducción de las ecuaciones en un artículo posterior. **Lección:** "verifica contra el paper" requiere darle el PDF o el texto; sin eso la verificación es parcial. Quedó marcado `[VERIFICAR]`.
2. *Umbral de colapso:* el primer valor (30 % del pico) etiquetaba como colapso las oscilaciones amortiguadas. Se detectó graficando los escenarios canónicos y se bajó a 10 %, más un suelo absoluto. **Lección:** la definición formal debe probarse contra casos cuya etiqueta conocemos.
3. *Escenario de colapso irreversible de la naturaleza:* la implementación no lo reproduce a 5.5×δ_opt (cicla). La IA lo reportó y renombró el escenario en lugar de ajustar parámetros hasta que "saliera".
4. *Factores con poco efecto:* con los rangos iniciales, agua y eventos no cambiaban la supervivencia agregada. Se ampliaron los rangos y se documentó que el efecto sigue siendo pequeño.
5. *KernelSHAP vs. Shapley exacto:* la primera comparación dio 0.12 de diferencia. La causa era la regularización L1 automática de KernelSHAP, no un error del cálculo exacto; con `l1_reg=False` la diferencia fue 4e-7.
6. *Problemas de herramienta:* varios comandos `heredoc` de shell fallaron por comillas y se reescribieron con escritura directa de archivos; la instalación de PyTorch se interrumpió una vez y se relanzó.

**Cómo evaluamos las respuestas:** tests unitarios (equilibrio analítico, balance de riqueza, determinismo), inspección visual de los escenarios, comprobaciones de eficiencia de las atribuciones y comparación entre tres métodos XAI. Los números del informe salen de archivos de resultados, no del texto de la IA.

**Cómo contribuyó la IA:** escritura del código y de los borradores del informe. **Qué sigue siendo nuestro** [PENDIENTE de confirmar por el equipo]: el tema, el encuadre de surrogate/fidelidad, la revisión de las ecuaciones contra el paper y la interpretación de resultados.


## P-02 · Prompt de la Fase 2 (lista de pendientes del README)
**Qué pedimos:** (1) verificar las ecuaciones de HANDY contra el PDF original, (2) reproducir el colapso Tipo-N, (3) varias semillas, intervalos de confianza y búsqueda de hiperparámetros para la LSTM, (4) validar la fidelidad de XAI contra intervenciones en el modelo y en el simulador, contrafactuales, interfaz Streamlit, y la presentación y el guion del video repartidos entre los tres integrantes.

**Qué funcionó**
- Convertir la lista de pendientes del README en el prompt dio un alcance claro y verificable. Cada ítem se cerró con un resultado medible, no con una afirmación.
- La verificación contra el PDF **encontró un error real** de la Fase 1 (ver abajo). Valió la pena insistir en leer el original en lugar de confiar en la reproducción de un tercero.

**Qué falló o necesitó corrección, y por qué**
1. *Error en la mortalidad de las élites (Fase 1).* La IA implementó α_E con `C_E/(κ·s·x_E)` suponiendo que κ se cancelaba, de modo que élites y plebeyos compartían el mismo término de hambruna. El paper (Ec. 6) usa `C_E/(s·x_E)`, así que las élites solo pasan hambre cuando w/w_th ≤ 1/κ. **Lo habíamos "verificado" en la Fase 1 contra un artículo que mostraba la fórmula correcta, pero la IA la reinterpretó y la simplificó mal.** Lección: al verificar, copiar la fórmula tal cual y no "simplificar" a partir de una suposición; un test unitario del caso límite (`test_elite_famine_uses_kappa`) habría detectado el error. Consecuencia: se regeneraron el dataset y todos los resultados; las cifras de la Fase 1 quedan obsoletas.
2. *Acceso al PDF.* La herramienta de lectura no pudo abrir el PDF del sitio universitario (certificado) ni los PDF binarios; se encontró otra copia con certificado válido y se extrajo el texto. No se desactivó la verificación de certificados.
3. *Tipo-N.* La hipótesis inicial (el paper usa Euler en precisión simple, lo que explicaría la irreversibilidad) se comprobó solo a medias: con Euler dt = 1 el colapso irreversible aparece desde ≈3.25×δ*, con RK4 desde ≈9.25×δ*; el paper reporta 4× reversible y 5.5× irreversible, entre ambos. No se pudo reproducir el umbral exacto sin su código.
4. *Rangos del dataset.* Tras corregir α_E la supervivencia cayó de 63 % a 15 %; hubo que re-ajustar rangos (κ ≤ 3, élites ≤ 10 %).
5. *Herramientas.* Varios comandos de shell con comillas fallaron de nuevo y se reescribieron como archivos; el paquete no estaba instalado en el venv y la primera cadena de entrenamientos falló sin avisar por eso.

**Cómo evaluamos las respuestas:** tests unitarios nuevos (mortalidad de élites, escenarios del paper, Shapley exacto en un modelo lineal, métricas de fidelidad), comparación con las figuras del paper, y chequeos cruzados (SHAP exacto vs. KernelSHAP, completitud de IG).
