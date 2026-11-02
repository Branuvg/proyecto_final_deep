# Guion del video (≈ 14 minutos con la demo completa; ≈ 12 sin lo opcional)

**Qué debe mostrar el video (rúbrica):** el funcionamiento de la solución, su implementación y los hallazgos principales.
**Reparto:** Gabriel (0:00–3:15) · David (3:15–6:30) · Jonathan (6:30–8:45) · Gabriel, demo (8:45–14:15) · cierre conjunto (14:15–14:45).
Cada parte combina **voz + pantalla compartida**. Lo que va entre comillas es el texto sugerido (léanlo con sus palabras); lo que va en *cursiva* es lo que se ve en pantalla.

## Antes de grabar (todos)

1. Clonar el repo y entrar al entorno: `venv\Scripts\activate`.
2. Tener generado lo necesario: `python scripts/run_all.py --skip-notebooks` (con GPU tarda pocos minutos; en CPU ≈ 25 min). Alternativa para la demo: solo `python -m worldsim.data.generate` y `python -m worldsim.train.run_task_a`.
3. Abrir en pestañas: `notebooks/01…05` (ya ejecutados), `src/worldsim/simulator/handy.py`, `src/worldsim/models/nets.py`, la presentación.
4. Terminal con letra grande (≥ 16 pt), resolución 1080p, notificaciones apagadas.
5. Probar una vez `streamlit run app/streamlit_app.py` y dejar la pestaña del navegador lista.
6. Grabar cada parte por separado (OBS o la grabadora de pantalla de Windows) y unirlas; si alguien se equivoca, repite solo su escena.

---

## Parte 1 · Gabriel (0:00–3:15)

### Escena 1 (0:00–0:30) · Portada — *diapositiva 1, cámara opcional*
"Hola, somos Gabriel Bran, David Domínguez y Jonathan Díaz, de Deep Learning, sección 10. Este es nuestro proyecto final: un simulador de mundos habitables donde una red neuronal predice si una civilización sobrevive, y donde podemos comprobar si sus explicaciones son verdaderas. Yo presento el problema, el simulador y la demo; David, los datos y modelos; y Jonathan, las explicaciones."

### Escena 2 (0:30–1:30) · El problema — *diapositiva 2*
"Los simuladores científicos son caros, así que se entrenan redes como modelos sustitutos. Pero una red es una caja negra, y los métodos de explicabilidad, como SHAP, casi nunca tienen una explicación verdadera para compararse. Nuestro truco es usar un problema donde sí la hay: el propio simulador. Podemos cambiar un factor y ver qué pasa de verdad. Entonces nuestra pregunta es: ¿cambia el resultado como dice la explicación?"

### Escena 3 (1:30–3:15) · El simulador — *VS Code en `handy.py` → terminal → notebook 01*
- *Mostrar el docstring de `src/worldsim/simulator/handy.py` con las ecuaciones (3) a (6) del paper.*
"El simulador implementa HANDY, un modelo de 2014 con plebeyos, élites, naturaleza y riqueza. Aquí están las ecuaciones con la numeración del artículo original."
- *Terminal: `pytest -q` → 15 tests pasan.*
"Tiene quince pruebas unitarias: no negatividad, determinismo, el equilibrio analítico del paper, y una que agregamos al verificar contra el artículo original, porque encontramos un error nuestro en la mortalidad de las élites. Lo corregimos y regeneramos todos los resultados."
- *Abrir `notebooks/01_simulador_validacion.ipynb` y mostrar la figura de escenarios.*
"Con las ecuaciones correctas reproducimos los escenarios del paper: equilibrio, oscilaciones, ciclos de colapso, y el colapso por desigualdad y por agotamiento de la naturaleza. Lo extendimos con temperatura, agua y eventos extremos aleatorios, para tener once factores; cuatro son palancas accionables. Una salvedad: el umbral exacto del colapso irreversible igualitario depende del integrador y no lo igualamos con exactitud. Paso a David."

---

## Parte 2 · David (3:15–6:30)

### Escena 4 (3:15–4:45) · Dataset — *terminal → notebook 02*
- *Terminal: `python -m worldsim.data.generate --config configs/dataset.yaml`.* (tarda ≈ 10 s)
"Generamos 50,000 planetas con Latin Hypercube. El simulador está vectorizado, así que tarda unos diez segundos. Ajustamos los rangos para que la frontera no fuera trivial: la supervivencia queda en 49.6%."
- *Notebook 02: gráfica de clases y de tiempos de colapso; luego las seis trayectorias.*
"Aquí vemos tres planetas que sobreviven y tres que colapsan; las franjas naranja son eventos extremos. Como los eventos son aleatorios, para 500 planetas de test corrimos 32 réplicas y obtuvimos la probabilidad verdadera de supervivencia; en el 31% de ellos es intermedia, o sea que hay incertidumbre real."

### Escena 5 (4:45–6:30) · Modelos y resultados — *`nets.py` → notebook 03*
- *Mostrar `MLP` y `LSTMNet` en `src/worldsim/models/nets.py` (unos 10 s).*
"Escribimos las redes como módulos propios de PyTorch: un MLP con dropout para las condiciones iniciales y una LSTM, de la sesión tres, para la alerta temprana, entrenada con truncamiento aleatorio del tiempo observado."
- *Notebook 03: tabla de la tarea (a) y diferencias pareadas.*
"Comparamos contra regresión logística y LightGBM, con 5 semillas y búsqueda de hiperparámetros. La logística llega a 0.73 de ROC-AUC porque la frontera es no lineal; LightGBM, 0.966; y el MLP, 0.973. La ventaja del MLP es pequeña, +0.0062, pero su intervalo de confianza del 95 % va de +0.0044 a +0.0080, así que es real."
- *Notebook 03: curva de desempeño frente a k.*
"En la alerta temprana, solo evaluamos planetas que siguen vivos al final de la ventana, para no medir algo trivial. Con una ventana larga, la LSTM llega a 0.987 y LightGBM con resúmenes a 0.987; sin los parámetros, la LSTM parte del azar. La trayectoria ayuda, sobre todo con ventanas largas. Le paso a Jonathan."

---

## Parte 3 · Jonathan (6:30–8:45)

### Escena 6 (6:30–7:30) · Explicaciones — *notebook 04*
- *Mostrar el beeswarm y una cascada.*
"Calculamos tres explicaciones: SHAP exacto, porque con once factores caben las 2048 coaliciones; Integrated Gradients con Captum; y TreeSHAP sobre LightGBM. Todas usan el mismo baseline, el planeta medio de entrenamiento, y la misma salida, el logit de supervivencia. Coinciden en que la depleción, la desigualdad y la fracción de élites son lo más influyente. Pero que coincidan entre sí no prueba que sean ciertas."

### Escena 7 (7:30–8:45) · Fidelidad frente al simulador — *notebook 05*
- *Mostrar `fig_fidelidad_spearman` y la tabla.*
"Para comprobarlo, devolvemos cada factor a su valor base y medimos el efecto real en el simulador, con 64 réplicas. La correlación por planeta entre la atribución y el efecto real es de 0.67 para SHAP, 0.69 para Integrated Gradients y 0.63 para TreeSHAP en planetas cerca de la frontera. Bastante bien, pero lejos de perfecto: el signo acierta en cerca del 91%, y el factor más importante, solo en la mitad de los casos."
- *Mostrar `fig_fidelidad_eliminacion` y los números de cordura.*
"Si quitamos los factores en el orden de SHAP, la probabilidad real cambia en -0.27; en orden aleatorio, en +0.17. Y la prueba de cordura pasa: con los pesos de la red aleatorizados, la concordancia con el simulador cae de 0.65 a -0.13. Las explicaciones son útiles y dependen del modelo, pero no son una verdad. Le paso a Gabriel para la demo."

## Parte 4 · Gabriel, demo en vivo (8:45–14:15)

**Preparación:** `streamlit run app/streamlit_app.py` ya abierto en el navegador, a pantalla completa y con la barra lateral visible. Las cifras de cada escenario salen de la app con el modelo actual (simulador con 48 réplicas); pueden variar uno o dos puntos si cambian las réplicas. **Si el video se alarga, recorten lo marcado *(opcional)*.**

### Escena 8a (8:45–9:30) · Qué es la app y qué se agregó — *pantalla de inicio de Streamlit*
"Esto es lo que agregamos en esta segunda fase: una interfaz interactiva que junta las tres piezas del proyecto en una sola pantalla. A la izquierda están las once condiciones del planeta; arriba, tres pestañas: la simulación, la explicación con SHAP y el contrafactual. Cada vez que movemos una condición pasan dos cosas a la vez: el simulador corre decenas de planetas con esas condiciones, y la red neuronal da su predicción. Así vemos, en vivo, si la red y el simulador están de acuerdo, y si la explicación y el contrafactual se sostienen."
- *Señalar el selector "Escenario" y el control "Réplicas del simulador".*
"El selector carga escenarios predefinidos, y el control de réplicas decide cuántos planetas simulamos con las mismas condiciones. Como los eventos extremos son aleatorios, cada réplica es un destino posible del mismo planeta."

### Escena 8b (9:30–10:45) · Las once condiciones del planeta — *barra lateral, recorrer de arriba abajo*
"Cada deslizador es una condición del planeta. Las que llevan una llave son **accionables**: son palancas de política que una sociedad podría cambiar. Las que llevan un termómetro son **datos del entorno**: no se pueden elegir."
- 🔧 **δ · depleción de la naturaleza:** "cuánta naturaleza destruye cada persona al producir. Es la palanca más fuerte: si es alta, la sociedad agota su base de recursos."
- 🔧 **κ · desigualdad:** "cuántas veces más consumen las élites que un plebeyo. Con κ = 1 todos consumen igual; con κ = 3, una élite consume el triple."
- 🔧 **s · consumo per cápita:** "el consumo mínimo de subsistencia de cada persona; a más consumo, más presión sobre los recursos."
- 🔧 **Eficiencia hídrica:** "la fracción del agua que se ahorra; reduce la demanda y protege la producción cuando hay poca agua."
- 🌡️ **β · natalidad:** "la tasa de nacimientos de la población."
- 🌡️ **γ · regeneración de la naturaleza:** "qué tan rápido se recupera la naturaleza por sí sola."
- 🌡️ **Anomalía térmica (°C):** "cuánto más caliente es el planeta; frena la regeneración de la naturaleza y sube la mortalidad base."
- 🌡️ **Oferta de agua:** "cuánta agua hay frente a lo que demanda la población; con poca agua, la producción rinde menos."
- 🌡️ **Eventos extremos (por año):** "con qué frecuencia golpean desastres al azar."
- 🌡️ **Severidad de los eventos:** "qué fracción de la naturaleza y de la riqueza destruye cada evento."
- 🌡️ **Fracción inicial de élites:** "qué parte de la población empieza siendo élite; sin élites no hay desigualdad."
"Cuatro son palancas y siete son entorno. Esa distinción importa porque, más adelante, el contrafactual solo puede mover las palancas."

### Escena 8c (10:45–11:30) · Escenario "Equilibrio" y qué mide la simulación — *pestaña "Simulación"*
- *Elegir "Equilibrio (δ cerca de δ\*, sin élites ni eventos)". Mostrar las tres métricas y las dos gráficas.*
"Aquí tenemos un planeta sano: depleción baja, sin élites y sin eventos. La simulación mide tres cosas. Primero, **la probabilidad de sobrevivir según el simulador**: de las 48 réplicas, cuántas no colapsan en 600 años; aquí, 100 %. Segundo, **la probabilidad que predice la red neuronal**, que también da cerca del 100 %; debajo vemos la diferencia en puntos con el simulador. Tercero, **el año de colapso** de las réplicas que colapsan; aquí no hay ninguna."
"Las gráficas muestran la población total y la naturaleza a lo largo de 600 años: la línea es la mediana de las réplicas y la banda sombreada, los percentiles 10 a 90. Colapso significa que la población cae por debajo del 10 % de su pico, o del 20 % de la inicial."
- *(opcional) Elegir "Planeta medio": es el promedio del entrenamiento, sobrevive, y SHAP vale cero porque es el punto de comparación.*

### Escena 8d (11:30–12:00) · Escenario "Sobreexplotación" — *pestaña "Simulación"*
- *Elegir "Sobreexplotación (δ alto)".*
"Solo subimos la depleción. La naturaleza se agota y la población colapsa en todas las réplicas, con un año de colapso mediano cerca del 416; la red neuronal también predice 0 %. Es un caso donde el sustituto y el simulador coinciden."

### Escena 8e (12:00–12:30) · Qué es la explicación (SHAP) — *pestaña "Explicación (SHAP)"*
"SHAP responde: ¿qué empujó a la red a predecir esto? Reparte la predicción entre las once condiciones, en comparación con un planeta medio de entrenamiento. Barra verde: la condición empuja hacia sobrevivir; barra roja: hacia colapsar; y el tamaño de la barra es cuánto. Aquí la depleción es, por mucho, la barra roja más larga: la red dice que colapsa sobre todo por la depleción."
"Pero una explicación convincente no es necesariamente correcta. En el proyecto la comprobamos interviniendo en el simulador, y SHAP acierta bastante pero no siempre. Aquí podemos probarlo en vivo: si regresamos la depleción a un valor bajo, ¿el planeta se salva de verdad?"

### Escena 8f (12:30–13:15) · Qué es el contrafactual — *pestaña "Contrafactual" → "Buscar contrafactual"*
"Un contrafactual responde: ¿cuál es el cambio mínimo para que este planeta sobreviva? Busca, solo entre las cuatro palancas, el cambio más pequeño con el que la red prediga al menos 80 % de supervivencia. Después lo verificamos en el simulador, que es la verdad."
- *Pulsar "Buscar contrafactual". Mostrar las dos métricas y la tabla.*
"La red pasa de 0 % a cerca de 92 % y el simulador lo confirma: de 0 % a 100 %. El cambio propuesto es bajar la depleción, de 5 por 10 a la menos 5 a unos 1.4 por 10 a la menos 5. Con una sola palanca fue suficiente."

### Escena 8g (13:15–14:15) · Dos casos donde se ve el límite *(opcional)* — *barra lateral → escenarios*
- *Elegir "Desigualdad alta con élites" (κ = 3, 10 % de élites).* "Aquí el simulador dice 0 %, pero la red predice cerca de 35 %: el sustituto se equivoca. SHAP señala la desigualdad como lo que más empuja al colapso, y el contrafactual baja κ de 3 a unos 2.5, que el simulador confirma."
- *Elegir "Mundo caliente con eventos" (+4.5 °C, eventos frecuentes y severos) y buscar contrafactual.* "Aquí lo que más pesa es la anomalía térmica, que no es una palanca: el contrafactual solo puede tocar las cuatro condiciones accionables, y la supervivencia sube apenas de 6 % a unos 31 %. Es un caso donde cambiar políticas no basta."
"En general, el 90% de los contrafactuales que probamos funcionan en el simulador, y la probabilidad real promedio pasa de 0.07 a 0.83."

---

## Cierre conjunto (14:15–14:45) · *diapositiva 10*
- **Gabriel:** "La red sí aporta frente a los baselines, aunque por poco, y la trayectoria ayuda con ventanas largas."
- **David:** "Nuestros límites: efectos de un factor a la vez, constantes de temperatura y agua sin calibrar, y una sola familia de simulador."
- **Jonathan:** "Lo importante: una explicación convincente no es una explicación correcta; con un simulador pudimos medirlo. Todo el código, el documento y el historial de uso de IA están en el repositorio: github.com/Branuvg/proyecto_final_deep. Gracias."

## Si algo falla durante la grabación
- *Streamlit tarda en abrir:* esperar 10 s; si falla, grabar el navegador con la app ya abierta.
- *Falta `data/worlds.npz` o `outputs/`:* ejecutar `python scripts/run_all.py --skip-notebooks`.
- *Un notebook no muestra las salidas:* ejecutar `python scripts/build_notebooks.py`.
