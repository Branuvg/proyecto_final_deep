# 5. Herramientas aplicadas

## 5.1 Herramientas y relación con el curso

[[tabla: Herramientas aplicadas y sesión del curso con la que se relacionan.]]
| Herramienta | Uso en el proyecto | Sesión del curso |
|---|---|---|
| PyTorch 2.11 (CUDA 12.8) | Redes escritas como `nn.Module` propios | S1, S3, S8 |
| Regresión logística (scikit-learn) | Baseline lineal; equivale a una red de una neurona | S1 |
| MLP con dropout y AdamW | Tarea (a) y contrafactuales | S1, S8 |
| LSTM con secuencias empaquetadas | Tarea (b), alerta temprana | S3, S12 |
| Búsqueda aleatoria, varias semillas y *bootstrap* | Hiperparámetros e intervalos de confianza | S8 |
| SHAP, Integrated Gradients (Captum) | Atribuciones | S15 |
| Curvas de eliminación y prueba de cordura | Validación de las explicaciones | S15 |
| Contrafactuales por gradiente | Cambios mínimos accionables | S15 |
| LightGBM | Baseline de gradient boosting y modelo para TreeSHAP | no vista en clase |
| Streamlit | Interfaz interactiva | no vista en clase |
| NumPy, SciPy | Simulador vectorizado y muestreo Latin Hypercube | no vista en clase |
| python-docx, python-pptx | Documento y presentación generados desde código | no vista en clase |

**Herramientas no vistas en clase, en detalle.** *LightGBM* construye árboles de decisión de forma secuencial, cada uno corrigiendo el error del anterior, con crecimiento por hojas y histogramas para ser rápido; lo usamos por ser el estándar fuerte para datos tabulares y por ser compatible con TreeSHAP. *Latin Hypercube* es un muestreo estratificado que cubre cada factor de forma más uniforme que el azar simple con el mismo número de puntos. *Valores de Shapley*: reparten la diferencia entre la predicción y el baseline entre los factores, promediando su contribución marginal sobre todas las coaliciones; con 11 factores se calculan de forma exacta. *Bootstrap pareado*: remuestrea los planetas de test con reemplazo y compara dos modelos sobre la misma muestra, lo que da un intervalo de confianza para la diferencia de una métrica. *Números aleatorios comunes*: simular el planeta original y el intervenido con las mismas semillas de eventos para que su diferencia tenga poca varianza. *Streamlit* convierte un script de Python en una aplicación web con controles interactivos.

## 5.2 Historial de prompts de IA (resumen)

El historial completo, con lo que funcionó y lo que falló, está en `docs/PROMPTS.md`. Resumen:

- **Prompt 1 (primer prompt del proyecto): simulador, dataset, modelos, XAI y documento.** *Funcionó:* fijar de antemano el encuadre (surrogate, baselines obligatorios, fidelidad frente al simulador) y exigir pruebas, un escenario canónico y honestidad al reportar. *Falló:* la IA no pudo abrir el PDF original y verificó las ecuaciones contra un artículo posterior; ese artículo mostraba la fórmula correcta de la mortalidad de las élites, pero la IA la reinterpretó mal. También falló un primer umbral de colapso (30 %), detectado al graficar los escenarios.
- **Prompt 2 (pendientes de la Fase 2).** *Funcionó:* convertir la lista de pendientes en objetivos medibles. Al leer por fin el PDF original se encontró el error de la mortalidad de las élites, que invalidaba los resultados previos; se corrigió, se agregó un test y se regeneró todo. *Falló:* el umbral del colapso Tipo-N del paper solo se reprodujo de forma parcial, y varios comandos de shell y una primera cadena de entrenamientos fallaron por problemas de herramientas.

**Cómo evaluamos a la IA:** pruebas unitarias, comparación con las figuras y fórmulas del paper, comprobaciones de eficiencia de las atribuciones, comparación entre tres métodos y, sobre todo, contraste con el simulador. **Contribución de la IA:** código, borradores del documento, de la presentación y del guion. **Contribución del equipo:** el tema y su encuadre, la decisión de verificar las explicaciones con el simulador, la revisión de los resultados y de las ecuaciones contra el paper y la presentación final. *(El equipo debe ajustar esta frase si no refleja cómo trabajó.)*
