# Historial de prompts de IA

La rúbrica pide los prompts relevantes, qué funcionó, qué falló y por qué, y cómo evaluamos las respuestas. Esta bitácora cubre la Fase 1. Los prompts de la etapa de aprobación del tema (anteriores a este repositorio) **no están registrados aquí**; el equipo debe agregarlos [PENDIENTE].

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
