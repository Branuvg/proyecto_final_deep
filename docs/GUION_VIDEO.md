# Guion del video (≈ 10 minutos, tres hablantes)

**Qué debe mostrar el video (rúbrica):** el funcionamiento de la solución, su implementación y los hallazgos principales.
**Reparto:** David (0:00–3:15) · Jonathan (3:15–6:30) · Gabriel (6:30–10:15) · cierre conjunto (10:15–10:45).
Cada parte combina **voz + pantalla compartida**. Lo que va entre comillas es el texto sugerido (léanlo con sus palabras); lo que va en *cursiva* es lo que se ve en pantalla.

## Antes de grabar (todos)

1. Clonar el repo y entrar al entorno: `venv\Scripts\activate`.
2. Tener generado lo necesario: `python scripts/run_all.py --skip-notebooks` (con GPU tarda pocos minutos; en CPU ≈ 25 min). Alternativa para la demo: solo `python -m worldsim.data.generate` y `python -m worldsim.train.run_task_a`.
3. Abrir en pestañas: `notebooks/01…05` (ya ejecutados), `src/worldsim/simulator/handy.py`, `src/worldsim/models/nets.py`, la presentación.
4. Terminal con letra grande (≥ 16 pt), resolución 1080p, notificaciones apagadas.
5. Probar una vez `streamlit run app/streamlit_app.py` y dejar la pestaña del navegador lista.
6. Grabar cada parte por separado (OBS o la grabadora de pantalla de Windows) y unirlas; si alguien se equivoca, repite solo su escena.

---

## Parte 1 · David (0:00–3:15)

### Escena 1 (0:00–0:30) · Portada — *diapositiva 1, cámara opcional*
"Hola, somos David Domínguez, Jonathan Díaz y Gabriel Bran, de Deep Learning, sección 10. Este es nuestro proyecto final: un simulador de mundos habitables donde una red neuronal predice si una civilización sobrevive, y donde podemos comprobar si sus explicaciones son verdaderas. Yo presento el problema y el simulador; Jonathan, los datos y modelos; y Gabriel, las explicaciones y la demo."

### Escena 2 (0:30–1:30) · El problema — *diapositiva 2*
"Los simuladores científicos son caros, así que se entrenan redes como modelos sustitutos. Pero una red es una caja negra, y los métodos de explicabilidad, como SHAP, casi nunca tienen una explicación verdadera para compararse. Nuestro truco es usar un problema donde sí la hay: el propio simulador. Podemos cambiar un factor y ver qué pasa de verdad. Entonces nuestra pregunta es: ¿cambia el resultado como dice la explicación?"

### Escena 3 (1:30–3:15) · El simulador — *VS Code en `handy.py` → terminal → notebook 01*
- *Mostrar el docstring de `src/worldsim/simulator/handy.py` con las ecuaciones (3) a (6) del paper.*
"El simulador implementa HANDY, un modelo de 2014 con plebeyos, élites, naturaleza y riqueza. Aquí están las ecuaciones con la numeración del artículo original."
- *Terminal: `pytest -q` → 15 tests pasan.*
"Tiene quince pruebas unitarias: no negatividad, determinismo, el equilibrio analítico del paper, y una que agregamos al verificar contra el artículo original, porque encontramos un error nuestro en la mortalidad de las élites. Lo corregimos y regeneramos todos los resultados."
- *Abrir `notebooks/01_simulador_validacion.ipynb` y mostrar la figura de escenarios.*
"Con las ecuaciones correctas reproducimos los escenarios del paper: equilibrio, oscilaciones, ciclos de colapso, y el colapso por desigualdad y por agotamiento de la naturaleza. Lo extendimos con temperatura, agua y eventos extremos aleatorios, para tener once factores; cuatro son palancas accionables. Una salvedad: el umbral exacto del colapso irreversible igualitario depende del integrador y no lo igualamos con exactitud. Paso a Jonathan."

---

## Parte 2 · Jonathan (3:15–6:30)

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
"En la alerta temprana, solo evaluamos planetas que siguen vivos al final de la ventana, para no medir algo trivial. Con una ventana larga, la LSTM llega a 0.987 y LightGBM con resúmenes a 0.987; sin los parámetros, la LSTM parte del azar. La trayectoria ayuda, sobre todo con ventanas largas. Le paso a Gabriel."

---

## Parte 3 · Gabriel (6:30–10:15)

### Escena 6 (6:30–7:30) · Explicaciones — *notebook 04*
- *Mostrar el beeswarm y una cascada.*
"Calculamos tres explicaciones: SHAP exacto, porque con once factores caben las 2048 coaliciones; Integrated Gradients con Captum; y TreeSHAP sobre LightGBM. Todas usan el mismo baseline, el planeta medio de entrenamiento, y la misma salida, el logit de supervivencia. Coinciden en que la depleción, la desigualdad y la fracción de élites son lo más influyente. Pero que coincidan entre sí no prueba que sean ciertas."

### Escena 7 (7:30–8:45) · Fidelidad frente al simulador — *notebook 05*
- *Mostrar `fig_fidelidad_spearman` y la tabla.*
"Para comprobarlo, devolvemos cada factor a su valor base y medimos el efecto real en el simulador, con 64 réplicas. La correlación por planeta entre la atribución y el efecto real es de 0.67 para SHAP, 0.69 para Integrated Gradients y 0.63 para TreeSHAP en planetas cerca de la frontera. Bastante bien, pero lejos de perfecto: el signo acierta en cerca del 91%, y el factor más importante, solo en la mitad de los casos."
- *Mostrar `fig_fidelidad_eliminacion` y los números de cordura.*
"Si quitamos los factores en el orden de SHAP, la probabilidad real cambia en -0.27; en orden aleatorio, en +0.17. Y la prueba de cordura pasa: con los pesos de la red aleatorizados, la concordancia con el simulador cae de 0.65 a -0.13. Las explicaciones son útiles y dependen del modelo, pero no son una verdad."

### Escena 8 (8:45–10:15) · Demo en vivo — *navegador con Streamlit*
- *Terminal: `streamlit run app/streamlit_app.py`.*
1. *Escenario "Equilibrio": mostrar la simulación (banda de réplicas) y las dos probabilidades.* "Aquí el simulador y la red coinciden en que sobrevive."
2. *Cambiar el escenario a "Sobreexplotación (δ alto)".* "Subimos la depleción: el simulador colapsa en cada réplica y la red también lo predice."
3. *Pestaña "Explicación (SHAP)".* "SHAP dice que la depleción es el factor que empuja hacia el colapso."
4. *Pestaña "Contrafactual" → "Buscar contrafactual".* "Le pedimos el cambio mínimo en las palancas accionables. Propone bajar la depleción, y el simulador lo verifica: la supervivencia pasa de 0 % a casi 100 %."
5. *Opcional: escenario "Mundo caliente con eventos" para mostrar un caso donde el contrafactual no basta.*
"En general, el 90% de los contrafactuales que probamos funcionan en el simulador, y la probabilidad real promedio pasa de 0.07 a 0.83."

---

## Cierre conjunto (10:15–10:45) · *diapositiva 10*
- **David:** "La red sí aporta frente a los baselines, aunque por poco, y la trayectoria ayuda con ventanas largas."
- **Jonathan:** "Nuestros límites: efectos de un factor a la vez, constantes de temperatura y agua sin calibrar, y una sola familia de simulador."
- **Gabriel:** "Lo importante: una explicación convincente no es una explicación correcta; con un simulador pudimos medirlo. Todo el código, el documento y el historial de uso de IA están en el repositorio: github.com/Branuvg/proyecto_final_deep. Gracias."

## Si algo falla durante la grabación
- *Streamlit tarda en abrir:* esperar 10 s; si falla, grabar el navegador con la app ya abierta.
- *Falta `data/worlds.npz` o `outputs/`:* ejecutar `python scripts/run_all.py --skip-notebooks`.
- *Un notebook no muestra las salidas:* ejecutar `python scripts/build_notebooks.py`.
