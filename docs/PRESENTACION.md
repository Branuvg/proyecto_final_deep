# Presentación: reparto entre los tres hablantes

Archivo de diapositivas: `report/Presentacion.pptx` (el texto que dice cada persona también está en las *notas del orador* de cada diapositiva).
Para entregar el PDF: abrir en PowerPoint → Archivo → Exportar → PDF (máximo 10 diapositivas: son exactamente 10).

| Hablante | Diapositivas | Palabras | Tiempo aprox. (130 pal/min) |
|---|---|---|---|
| David | 1, 2, 3, 10 | 593 | 4.6 min |
| Jonathan | 4, 5, 6 | 467 | 3.6 min |
| Gabriel | 7, 8, 9 | 528 | 4.1 min |

Total: 1588 palabras ≈ 12.2 minutos al leerlo con calma. **Si el tiempo asignado es menor, recorten las notas, no las diapositivas.**

Reglas sugeridas: quien habla cede la palabra al terminar su última diapositiva diciendo el nombre de la siguiente persona; cada quien debe poder responder preguntas de *todo* el proyecto (ver la lista al final).

## Diapositiva 1 · Portada y pregunta central — David

Buenas tardes. Somos David Domínguez, Jonathan Díaz y Gabriel Bran, y les presentamos nuestro proyecto final de Deep Learning. Empezamos con una pregunta: cuando una red neuronal nos da una explicación de su predicción, ¿podemos creerle? En casi todos los problemas reales no hay forma de comprobarlo. Nosotros construimos un problema donde sí: un simulador de civilizaciones, una red que predice si cada civilización sobrevive, y explicaciones que comprobamos contra el propio simulador. Yo, David, presento el problema y el simulador; Jonathan, los datos y los modelos; Gabriel, las explicaciones y su verificación; y cierro con las conclusiones.

## Diapositiva 2 · El problema — David

Muchos sistemas científicos —clima, epidemias, ecología— se estudian con simuladores que son lentos y caros. Por eso se entrenan redes neuronales como modelos sustitutos: aprenden a imitar al simulador y responden al instante. El problema es que una red es una caja negra. Existen métodos de explicabilidad, como SHAP o Integrated Gradients, que dicen qué factores influyeron. Pero casi nunca existe una explicación verdadera con la cual compararlas, así que no sabemos si son correctas o solo convincentes. Nuestra idea: usar un problema donde el mecanismo real sí está disponible. El simulador nos deja intervenir un factor y ver qué pasa de verdad. Entonces la pregunta del proyecto no es solo predecir, sino medir qué tan fieles son las explicaciones.

## Diapositiva 3 · El simulador — David

El simulador está basado en HANDY, un modelo publicado en 2014 en Ecological Economics. Tiene cuatro variables: plebeyos, élites, naturaleza y riqueza. La naturaleza se regenera, la población la agota para producir riqueza, y si la riqueza no alcanza para el consumo mínimo hay hambruna. Verificamos las ecuaciones contra el artículo original y al hacerlo encontramos un error nuestro: habíamos escrito mal la mortalidad de las élites. Lo corregimos, agregamos una prueba unitaria y regeneramos todo el dataset. Con las ecuaciones correctas reproducimos los escenarios del paper: equilibrio, oscilaciones, ciclos de colapso y los colapsos por desigualdad y por agotamiento de la naturaleza. Una salvedad honesta: el umbral exacto del colapso irreversible igualitario depende del integrador, y no lo igualamos con exactitud. Lo extendimos con temperatura, agua y eventos extremos aleatorios, hasta tener once factores. Cuatro son accionables, o sea palancas de política: depleción, desigualdad, consumo y eficiencia hídrica. Los demás son datos del entorno. Colapso significa que la población cae por debajo del diez por ciento de su pico.

## Diapositiva 4 · Datos y verdad de referencia — Jonathan

Gracias, David. Con el simulador generamos 50,000 planetas, muestreando las once condiciones con Latin Hypercube. Ajustamos los rangos para que la frontera no fuera trivial: la supervivencia quedó en 49.6%. La generación tarda unos diez segundos porque el simulador está vectorizado. El split es 70, 15 y 15 por planeta, y la normalización se ajusta solo con train. Como los eventos extremos son aleatorios, las mismas condiciones pueden dar resultados distintos. Por eso, para 500 planetas de test corrimos 32 réplicas con distintas semillas y obtuvimos la probabilidad verdadera de supervivencia. En el 31% de esos planetas la probabilidad es intermedia, o sea que hay incertidumbre real. Esta probabilidad verdadera nos sirve para evaluar la calibración de los modelos y, después, como verdad para las explicaciones. A la derecha ven tres planetas que sobreviven y tres que colapsan; las franjas naranja son eventos extremos.

## Diapositiva 5 · Modelos — Jonathan

Planteamos dos tareas. En la tarea a, un MLP lee las once condiciones iniciales y predice la probabilidad de supervivencia. Usa dropout, AdamW y early stopping, como vimos en la sesión ocho. En la tarea b, una LSTM, de la sesión tres, lee los primeros k por ciento de la trayectoria junto con los parámetros, como alerta temprana. La entrenamos con truncamiento aleatorio de k, así un solo modelo sirve para cualquier ventana. Para no medir una ventaja trivial, solo evaluamos planetas que siguen vivos al final de la ventana, y reportamos cuántos quedan fuera. Exigimos que la red demuestre que aporta algo: comparamos con regresión logística y con gradient boosting, LightGBM. Hicimos búsqueda de hiperparámetros, entrenamos con 5 semillas y calculamos intervalos de confianza con bootstrap pareado sobre los planetas de test.

## Diapositiva 6 · Resultados de modelos — Jonathan

Resultados de la tarea a, con media y desviación sobre 5 semillas. La regresión logística llega a ROC-AUC de 0.73: la frontera es muy no lineal. LightGBM obtiene 0.966 y el MLP 0.973. La diferencia del MLP con LightGBM es de +0.0062 de AUC, con intervalo de confianza de +0.0044 a +0.0080; como el intervalo no incluye cero, la ventaja es real, aunque pequeña. El MLP también está mejor calibrado y su error contra la probabilidad verdadera Monte Carlo es de 0.054. Para la tarea b, la gráfica muestra el desempeño al observar más trayectoria. Con k igual a cero la LSTM solo ve los parámetros. En ventanas largas, de sesenta por ciento, la LSTM alcanza 0.987, LightGBM con resúmenes 0.987 y el MLP que no ve la trayectoria 0.965. Entre la LSTM y LightGBM, empatan (diferencia +0.0003, intervalo -0.0014 a +0.0019). Sin parámetros estáticos, la LSTM parte de 0.57, o sea casi azar, y sube a 0.98: la trayectoria sola acaba informando, pero con poca ventana los parámetros son indispensables. Es importante aclarar que entre distintos k no se pueden comparar las métricas directamente, porque cambia la población evaluada.

## Diapositiva 7 · Explicaciones y verificación — Gabriel

Gracias, Jonathan. Ahora la parte central del proyecto. Calculamos tres explicaciones sobre el MLP y el boosting: SHAP, que con once factores calculamos de forma exacta, Integrated Gradients con Captum, y TreeSHAP sobre LightGBM. Algo clave: todos usan el mismo baseline, el planeta medio de entrenamiento, y el mismo espacio de salida, el logit de supervivencia. Así las comparaciones son justas. Comprobamos que SHAP exacto coincide con KernelSHAP y que cada método cumple su propiedad de completitud. Pero que las explicaciones coincidan entre sí no prueba que sean correctas. Por eso las verificamos de cuatro maneras: uno, intervenir un factor en el modelo; dos, intervenirlo en el simulador, con sesenta y cuatro réplicas y los mismos números aleatorios para que la comparación tenga poco ruido; tres, curvas de eliminación: si devolvemos al valor base los factores en el orden que dice la explicación, el resultado debe degradarse más rápido que con un orden aleatorio; y cuatro, una prueba de cordura de Adebayo y colaboradores: si destruimos el modelo, la explicación debe dejar de concordar con el simulador.

## Diapositiva 8 · Fidelidad: resultados — Gabriel

Resultados de la fidelidad. En la gráfica, las barras naranja comparan cada atribución con el efecto real en el simulador. Con planetas cercanos a la frontera de decisión, la correlación de Spearman por planeta es de 0.67 para SHAP, 0.69 para Integrated Gradients y 0.63 para TreeSHAP; con planetas Monte Carlo aleatorios, entre 0.58 y 0.64. Es decir, concuerdan bastante, pero lejos de perfecto. El signo coincide en cerca de 91% de los casos con efecto real, pero el factor más importante solo coincide en 50% a 61%. Parte del error viene del propio modelo: la línea azul indica que incluso el efecto que el MLP predice por sí mismo solo llega a 0.82 de Spearman con el simulador. En las curvas de eliminación, quitar factores en el orden de SHAP cambia la probabilidad real en -0.27 en promedio, mientras que el orden aleatorio la cambia en +0.17: las explicaciones sí apuntan a lo que importa. Y la prueba de cordura pasa: con el modelo aleatorizado la concordancia con el simulador cae de 0.65 a -0.13. La conclusión honesta: las explicaciones son útiles y dependen del modelo, pero no son una verdad; hay que verificarlas.

## Diapositiva 9 · Contrafactuales y demo — Gabriel

Una explicación útil debería permitir actuar. Por eso buscamos contrafactuales: el cambio mínimo en las palancas accionables —depleción, desigualdad, consumo y eficiencia hídrica— que lleva el planeta a sobrevivir. Los factores del entorno, como la temperatura o los eventos, no se tocan. Los aplicamos a los planetas que el modelo predice que colapsarán, y los verificamos en el simulador. El método de gradiente resuelve el 100% de los casos; en el simulador, el 90% de los contrafactuales realmente sobrevive, y la probabilidad real pasa de 0.07 a 0.83, con 2.0 palancas cambiadas en promedio. Cambiar una sola palanca resuelve solo el 87% de los casos. Ahora les mostramos la aplicación en Streamlit: movemos los factores de un planeta, vemos la simulación con sus réplicas, la explicación SHAP y pedimos un contrafactual que el simulador verifica. [DEMO EN VIVO: elegir el escenario 'Sobreexplotación (δ alto)', mostrar el colapso, abrir la pestaña de explicación y pulsar 'Buscar contrafactual'.]

## Diapositiva 10 · Conclusiones — David

Para cerrar. Primero, el modelo: la red sí aporta frente a los baselines en la tarea a, con una ventaja pequeña pero con intervalo de confianza que excluye cero; la regresión logística, en cambio, falla. En alerta temprana, la trayectoria ayuda sobre todo en ventanas largas. Segundo, las explicaciones: concuerdan con la realidad del simulador de forma moderada, mejor que el azar y que un orden aleatorio, y pasan la prueba de cordura, pero no son una verdad. Tercero, los contrafactuales verificados con el simulador funcionan en la mayoría de casos. Entre las limitaciones: los efectos de una intervención individual no capturan interacciones; las constantes de temperatura y agua son supuestos nuestros; y no reprodujimos con exactitud el umbral del colapso irreversible del paper. Sobre el uso de la IA: la usamos para escribir código y borradores, y está documentado en el repositorio, con lo que funcionó y lo que falló. El caso más importante: al verificar contra el paper original encontramos un error en la mortalidad de las élites que la IA había introducido; lo detectamos, lo corregimos con una prueba unitaria y rehicimos todos los resultados. Lección: la IA acelera, pero hay que verificar contra la fuente. Gracias; quedamos atentos a sus preguntas.

## Preguntas probables y quién las toma

| Pregunta | Responde | Respuesta corta |
|---|---|---|
| ¿Por qué un simulador y no datos reales? | David | Solo así existe una verdad de referencia para verificar las explicaciones. |
| ¿Qué error encontraron en HANDY? | David | La mortalidad de las élites usaba κ en el denominador; el paper usa s·x_E. Lo corregimos con un test y regeneramos todo. |
| ¿La red es mejor que LightGBM? | Jonathan | Por poco (ΔAUC con IC 95 % que excluye 0), pero sí; y mucho mejor que la logística. |
| ¿Por qué solo planetas vivos en la tarea (b)? | Jonathan | Para evitar la fuga trivial: un planeta ya colapsado es obvio. Se reporta cuántos quedan fuera. |
| ¿Por qué SHAP exacto? | Gabriel | Con 11 factores hay 2048 coaliciones: se calcula exacto y se comprobó contra KernelSHAP. |
| ¿Qué significa que ρ ≈ 0.65? | Gabriel | Concuerdan bastante con el efecto real pero no perfectamente; parte del error es del propio modelo. |
| ¿Los contrafactuales son confiables? | Gabriel | Se verificaron con el simulador: ≈90 % funcionan; el resto muestra dónde el sustituto se equivoca. |