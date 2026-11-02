# Notas de la presentación y preguntas probables

Este archivo se genera con `python report/build_presentation.py`; las cifras salen de `outputs/*/results.json`. Reparto y tiempos: ver `docs/PRESENTACION.md`. Las notas también están en las *notas del orador* de `report/Presentacion.pptx`. Son un guion de apoyo: cada quien debe decirlo con sus palabras.

Cada quien debe poder contestar preguntas de **todo** el proyecto; la columna «Responde» solo indica quién toma la pregunta primero.

---

# Parte 1 · Notas por diapositiva

## Diapositiva 1 · Portada y pregunta central — Gabriel

Buenas tardes. Somos Gabriel Bran, David Domínguez y Jonathan Díaz, y les presentamos nuestro proyecto final de Deep Learning. Empezamos con una pregunta: cuando una red neuronal nos da una explicación de su predicción, ¿podemos creerle? En casi todos los problemas reales no hay forma de comprobarlo. Nosotros construimos un problema donde sí: un simulador de civilizaciones, una red que predice si cada civilización sobrevive, y explicaciones que comprobamos contra el propio simulador. Yo, Gabriel, presento el problema y el simulador, y cierro con las conclusiones; David presenta los datos y los modelos; y Jonathan, las explicaciones y su verificación.

## Diapositiva 2 · El problema — Gabriel

Muchos sistemas científicos —clima, epidemias, ecología— se estudian con simuladores lentos y caros. Por eso se entrenan redes neuronales como modelos sustitutos, o surrogates: aprenden la relación entre condiciones de entrada y resultado, y responden al instante. El problema es que una red es una caja negra. Existen métodos de explicabilidad, como SHAP o Integrated Gradients, que dicen qué factores influyeron. Pero, como vimos en la sesión 15, una explicación post-hoc es un modelo del modelo: puede ser útil sin ser fiel. El ejemplo clásico es el clasificador de lobos que en realidad miraba la nieve del fondo, y la exactitud de validación no lo delata. Los criterios para juzgar una explicación son fidelidad, estabilidad y comprensibilidad, pero la fidelidad casi nunca se puede medir, porque no existe una explicación verdadera con la cual compararla. Nuestra idea: usar un problema donde el mecanismo real sí está disponible. El simulador nos deja intervenir un factor y ver qué pasa de verdad, así que podemos medir la fidelidad directamente. El proyecto tiene tres capas: el simulador, que genera la verdad; una red que lo imita, el sustituto; y las explicaciones de esa red, que verificamos contra el simulador. Todo se mide con varias semillas e intervalos de confianza, para no confundir una casualidad con un resultado.

## Diapositiva 3 · El simulador — Gabriel

El simulador está basado en HANDY, un modelo publicado en 2014 en Ecological Economics. Es un sistema de cuatro ecuaciones diferenciales: plebeyos, élites, naturaleza y riqueza. La naturaleza crece de forma logística, con tasa de regeneración gamma y capacidad lambda igual a cien, y la población la agota a una tasa delta por plebeyo. Esa extracción alimenta la riqueza, y la riqueza se consume. Si la riqueza es menor que el umbral rho por plebeyos más kappa rho por élites, el consumo se recorta y la mortalidad sube de uno por ciento a siete por ciento; kappa es cuántas veces más consume una élite que un plebeyo. Usamos los valores canónicos del paper: natalidad 0.03, rho 0.005, consumo s de 0.0005, gamma 0.01 y un punto de equilibrio delta asterisco de unos 6.67 por diez a la menos seis. Resolvemos con Runge-Kutta de orden cuatro, paso de un año, seiscientos años, guardando un punto cada seis años: ciento un puntos por planeta, vectorizado en NumPy para simular miles de planetas a la vez. Verificamos las ecuaciones contra el artículo original y al hacerlo encontramos un error nuestro: en la mortalidad de las élites usábamos kappa por s por x_E en el denominador, cuando la ecuación 6 usa s por x_E. Lo corregimos, agregamos una prueba unitaria y regeneramos todo el dataset; por eso todos los números que verán son de la versión corregida. Con las ecuaciones correctas reproducimos los escenarios del paper: equilibrio, oscilaciones, ciclos de colapso y los colapsos por desigualdad y por agotamiento de la naturaleza. Una salvedad honesta: el umbral exacto del colapso irreversible igualitario depende del integrador. Con Runge-Kutta aparece desde unas 9.25 veces delta asterisco y con Euler en precisión simple desde unas 3.25, y el paper reporta 5.5; no lo igualamos. Lo extendimos con tres mecanismos propios. Temperatura: reduce la regeneración como gamma por exponencial de menos T sobre tres, al cuadrado, y sube la mortalidad base quince por ciento por grado. Agua: limita la productividad de la naturaleza según la razón entre oferta y demanda, y la eficiencia hídrica reduce la demanda. Eventos extremos: llegan como un proceso de Poisson y cada uno destruye una fracción aleatoria de la naturaleza y de la riqueza, proporcional a su severidad. Son once factores; cuatro son accionables, o sea palancas de política: depleción, desigualdad, consumo y eficiencia hídrica. Colapso significa que la población cae por debajo del diez por ciento de su pico, o del veinte por ciento de la inicial, antes del año seiscientos. Estas trayectorias son las series multivariadas que después leerá la LSTM de la sesión tres.

## Diapositiva 4 · Datos y verdad de referencia — David

Gracias, Gabriel. Con el simulador generamos 50,000 planetas, muestreando las once condiciones con Latin Hypercube, que cubre el espacio de forma más uniforme que un muestreo aleatorio simple. Es un diseño de experimentos, no una muestra de datos reales. Delta y kappa se muestrean en escala logarítmica porque sus rangos abarcan órdenes de magnitud: delta de 5 por diez a la menos seis a 6 por diez a la menos cinco, kappa de uno a tres. El resto es uniforme: consumo s de 3.5 a 6.5 por diez a la menos cuatro, eficiencia hídrica de 0 a 0.6, temperatura de 0 a 5 grados, agua de 0.05 a 1.2, eventos de 0 a 0.12 por año con severidad de 0.1 a 0.9, y fracción inicial de élites de 0 a 10 %. Ajustamos esos rangos en varias iteraciones documentadas para que la frontera entre sobrevivir y colapsar no fuera trivial; la supervivencia quedó en 49.6%, así que las clases están casi balanceadas. Generar todo tarda unos diez segundos. El split es 70, 15 y 15 por planeta: 35,000 para entrenar, 7,500 para validar y 7,500 para probar. Para el preprocesamiento aplicamos logaritmo a delta y kappa y luego z-score, con medias y desviaciones calculadas solo con el conjunto de entrenamiento, para no filtrar información. Para la tarea de series, cada planeta es una secuencia de 101 pasos con seis canales: logaritmo de plebeyos, logaritmo de élites, naturaleza dividida entre cien, logaritmo de la riqueza, número de eventos del intervalo y el tiempo normalizado, también estandarizados con train. Esto es lo que vimos en series de tiempo, sesión 12: convertir una serie en un problema supervisado con ventanas, cuidando que no haya fuga de información del futuro. Como los eventos extremos son aleatorios, las mismas condiciones pueden dar resultados distintos. Por eso, para 500 planetas de test corrimos 32 réplicas con distintas semillas, o sea 16 mil simulaciones, y obtuvimos la probabilidad verdadera de supervivencia. En el 31% de esos planetas la probabilidad es intermedia, o sea que hay incertidumbre real. Esta probabilidad verdadera nos sirve para evaluar la calibración de los modelos y, después, como verdad para las explicaciones. A la derecha ven tres planetas que sobreviven y tres que colapsan; las franjas naranja son eventos extremos.

## Diapositiva 5 · Modelos — David

Planteamos dos tareas. En la tarea a, un MLP lee las once condiciones estandarizadas y predice la probabilidad de supervivencia. Son capas Linear, ReLU y Dropout, como en la sesión uno y la ocho; la mejor configuración de la búsqueda fue 11 a 128 a 64 a 1, con dropout 0.2 y 9 857 parámetros. La salida es un logit y la pérdida es entropía cruzada binaria con logits, que integra la sigmoide por estabilidad numérica y, como vimos en la sesión ocho, entrena probabilidades bien calibradas. Optimizamos con AdamW, que desacopla el weight decay del gradiente: tasa de aprendizaje 0.003, decaimiento 0.0001, lotes de 512, y early stopping sobre la pérdida de validación con paciencia de 15 épocas, restaurando los mejores pesos. En la tarea b, una LSTM, de la sesión tres, lee los primeros k por ciento de la trayectoria, hasta k igual a 60 %, o sea hasta 360 años, junto con los parámetros. Las compuertas de la LSTM controlan qué se recuerda de la historia. Los once factores estáticos se concatenan a cada paso de tiempo y otra vez a la entrada de la cabeza; usamos secuencias empaquetadas para que cada planeta use su último estado oculto válido, sin contar el relleno. La configuración elegida es de 2 capas de 128 unidades, dropout 0.1, 225 409 parámetros, AdamW con tasa 0.002 y weight decay 0.01. La entrenamos con truncamiento aleatorio: en cada lote k se sortea entre 0 y 70 pasos, así un solo modelo sirve para cualquier ventana, y funciona como un aumento de datos. Para no medir una ventaja trivial, solo evaluamos planetas que siguen vivos al final de la ventana, y reportamos cuántos quedan fuera. Exigimos que la red demuestre que aporta algo. Comparamos con regresión logística; con LightGBM, de mil árboles, 31 hojas y tasa 0.05; con LightGBM sobre resúmenes de la ventana, como el último valor, la pendiente y el máximo; con un MLP sobre la ventana aplanada; y con un MLP que solo ve los parámetros. Además hicimos una ablación: la LSTM sin parámetros estáticos. Los hiperparámetros salen de una búsqueda aleatoria: 12 configuraciones para el MLP, variando capas, dropout, tasa y weight decay, y 8 para la LSTM, con la semilla cero. La mejor se reentrena con 5 semillas y calculamos intervalos de confianza del 95 % con bootstrap pareado de mil remuestreos sobre los planetas de test.

## Diapositiva 6 · Resultados de modelos — David

Resultados de la tarea a, con media y desviación sobre 5 semillas. Usamos cuatro métricas. ROC-AUC mide qué tan bien ordena el modelo planetas que sobreviven y que colapsan. Brier es el error cuadrático de las probabilidades y ECE mide la calibración, o sea si cuando dice setenta por ciento, de verdad sobrevive el setenta por ciento. Y como tenemos la probabilidad verdadera Monte Carlo, calculamos el error absoluto medio contra ella. La regresión logística llega a ROC-AUC de 0.73: la frontera es muy no lineal, y un modelo lineal en los factores no la captura. LightGBM obtiene 0.966 y el MLP 0.973. La diferencia del MLP con LightGBM es de +0.0062 de AUC, con intervalo de confianza de +0.0044 a +0.0080; como el intervalo no incluye cero, la ventaja es real, aunque pequeña. El MLP también tiene mejor Brier, 0.066 contra 0.074, y su error contra la probabilidad verdadera Monte Carlo es de 0.054, contra 0.077 de LightGBM. Para la tarea b, la gráfica muestra el desempeño al observar más trayectoria; k es el porcentaje de los seiscientos años que se ve. Con k igual a cero la LSTM solo ve los parámetros. En ventanas largas, de sesenta por ciento, la LSTM alcanza 0.987, LightGBM con resúmenes 0.987 y el MLP que no ve la trayectoria 0.965. Entre la LSTM y LightGBM, empatan (diferencia +0.0003, intervalo -0.0014 a +0.0019). Con k igual a 40 por ciento, la LSTM supera al MLP estático, 0.982 contra 0.970: la historia aporta información que los parámetros iniciales no tienen. Sin parámetros estáticos, la LSTM parte de 0.57, o sea casi azar, y sube a 0.98: la trayectoria sola acaba informando, pero con poca ventana los parámetros son indispensables. Es importante aclarar que entre distintos k no se pueden comparar las métricas directamente, porque cambia la población evaluada.

## Diapositiva 7 · Explicaciones y verificación — Jonathan

Gracias, David. Ahora la parte central del proyecto. Calculamos tres explicaciones. SHAP sobre el MLP, que con once factores calculamos de forma exacta: son dos a la once, 2 048 coaliciones, y el valor de Shapley de cada factor promedia su contribución marginal sobre todos los órdenes posibles. Integrated Gradients, de la sesión 15, con Captum y mil veinticuatro pasos de integración: acumula el gradiente a lo largo del camino entre el baseline y el planeta. Y TreeSHAP sobre LightGBM, que calcula los valores de Shapley de un árbol en tiempo polinomial. Algo clave: todos usan el mismo baseline, el planeta medio de entrenamiento, que en el espacio estandarizado es el vector cero, y el mismo espacio de salida, el logit de supervivencia. Así las comparaciones son justas. Comprobamos las propiedades de cada método. SHAP exacto coincide con KernelSHAP con una diferencia máxima de 2.6e-07; cumple la eficiencia, es decir, la suma de las atribuciones es la predicción menos la del baseline, con error de 3e-06; y Integrated Gradients cumple la completitud con un error máximo de 0.03 logits. Pero que las explicaciones coincidan entre sí no prueba que sean correctas. Por eso las verificamos de cuatro maneras. Uno, intervenir un factor en el modelo: lo devolvemos a su valor base y vemos cuánto cambia el logit. Dos, intervenirlo en el simulador: corremos 64 réplicas con los mismos números aleatorios en ambos lados para reducir el ruido, y medimos el cambio real en la probabilidad de supervivencia; consideramos que hay un efecto real si cambia más de 0.10. Lo hacemos en 980 planetas cercanos a la frontera de decisión y 382 planetas Monte Carlo con algún efecto real. Comparamos con Spearman por planeta, acuerdo de signo, y si coincide el factor más importante o los tres más importantes. Tres, curvas de eliminación: si devolvemos al valor base los factores en el orden que dice la explicación, la probabilidad real debe degradarse más rápido que con un orden aleatorio. Y cuatro, una prueba de cordura de Adebayo y colaboradores: si aleatorizamos los pesos del modelo, la explicación debe dejar de concordar con el simulador.

## Diapositiva 8 · Fidelidad: resultados — Jonathan

Resultados de la fidelidad. En la gráfica, las barras naranja comparan cada atribución con el efecto real en el simulador, y las grises con el efecto en el modelo; las líneas son intervalos de confianza del 95 %. Con planetas cercanos a la frontera de decisión, la correlación de Spearman por planeta es de 0.67 para SHAP, 0.69 para Integrated Gradients y 0.63 para TreeSHAP; con planetas Monte Carlo aleatorios, entre 0.58 y 0.64. Es decir, concuerdan bastante, pero lejos de perfecto. El signo coincide en cerca de 91% de los casos con efecto real, pero el factor más importante solo coincide en 50% a 61%. Parte del error viene del propio modelo: la línea azul indica que incluso el efecto que el MLP predice por sí mismo, al intervenir un factor, solo llega a 0.82 de Spearman con el simulador. Eso significa que la explicación no puede ser más fiel que el modelo que explica, y también que el modelo sustituto se equivoca en algunos planetas. En las curvas de eliminación, quitar factores en el orden de SHAP cambia la probabilidad real en -0.27 en promedio, mientras que el orden aleatorio la cambia en +0.17: las explicaciones sí apuntan a lo que importa. Y la prueba de cordura pasa: con el modelo aleatorizado la concordancia con el simulador cae de 0.65 a -0.13. Eso descarta que la explicación sea solo un reflejo de los datos de entrada. La conclusión honesta, relacionada con la sesión 15: una explicación convincente no es necesariamente una explicación fiel. Las explicaciones son útiles y dependen del modelo, pero no son una verdad; hay que verificarlas.

## Diapositiva 9 · Contrafactuales y demo — Jonathan

Una explicación útil debería permitir actuar. Por eso buscamos contrafactuales, como en Wachter y colaboradores, de la sesión 15: el cambio mínimo en las palancas accionables —depleción, desigualdad, consumo y eficiencia hídrica— que lleva el planeta a sobrevivir. Los factores del entorno, como la temperatura o los eventos, no se tocan. Los aplicamos a los planetas que el modelo predice que colapsarán. La búsqueda es por descenso de gradiente sobre el MLP: optimizamos con Adam un desplazamiento en el espacio estandarizado, 600 pasos con tasa 0.03, solo en las cuatro palancas y recortado al rango de muestreo. La pérdida tiene dos términos: una bisagra que penaliza que el logit esté por debajo del objetivo, que es 80 % de probabilidad de supervivencia, y una penalización L1 de peso 0.5 sobre el cambio relativo al rango. Como vimos en la sesión ocho, la L1 favorece soluciones dispersas, es decir, que cambien pocos factores; además descartamos cambios menores al uno por ciento del rango. Lo importante es que no confiamos en la red: cada contrafactual se verifica en el simulador con 64 réplicas. El método de gradiente resuelve el 100% de los casos según el modelo; en el simulador, el 90% de los contrafactuales realmente sobrevive, y la probabilidad real pasa de 0.07 a 0.83, con 2.0 palancas cambiadas en promedio. Cambiar una sola palanca, con una búsqueda en rejilla de 201 valores, resuelve solo el 87% de los casos. Ahora les mostramos la aplicación en Streamlit: movemos los factores de un planeta, vemos la simulación con sus réplicas, la explicación SHAP y pedimos un contrafactual que el simulador verifica. [DEMO EN VIVO: elegir el escenario 'Sobreexplotación (δ alto)', mostrar el colapso, abrir la pestaña de explicación y pulsar 'Buscar contrafactual'.]

## Diapositiva 10 · Conclusiones — Gabriel

Gracias, Jonathan. Para cerrar. Primero, el modelo: en la tarea a, el MLP llega a ROC-AUC de 0.973, contra 0.966 de LightGBM y 0.732 de la regresión logística: la red aporta, con una ventaja pequeña pero con intervalo de confianza que excluye cero, y la logística falla porque la frontera es no lineal. En alerta temprana, la trayectoria ayuda sobre todo en ventanas largas: con k igual a 40 por ciento la LSTM llega a 0.982 contra 0.970 del MLP estático, y con 60 por ciento empata con LightGBM. Un modelo clásico bien alimentado compite con la red recurrente, y eso es parte del hallazgo. Segundo, las explicaciones: concuerdan con la realidad del simulador de forma moderada, con Spearman de 0.63 a 0.69, mejor que el azar y que un orden aleatorio, y pasan la prueba de cordura, pero no son una verdad. Es la lección de la sesión 15: desconfiar de las explicaciones y verificarlas. Tercero, los contrafactuales verificados con el simulador funcionan en el 90% de los casos. Entre las limitaciones: los efectos de una intervención individual no capturan interacciones; las constantes de temperatura y agua son supuestos nuestros; los modelos se entrenaron con 5 semillas y un solo simulador; y no reprodujimos con exactitud el umbral del colapso irreversible del paper. Gracias; quedamos atentos a sus preguntas.

---

# Parte 2 · Preguntas probables del catedrático

## A. Planteamiento y simulador

**P: ¿Por qué usar un simulador y no datos reales?** *(responde Gabriel)*

R: Porque con datos reales no existe la explicación verdadera con la cual comparar SHAP o IG. El simulador sí es el mecanismo real: podemos intervenir un factor y medir qué pasa. El costo es que los datos son sintéticos y las conclusiones valen para esta familia de simuladores, no para civilizaciones reales.

**P: Si el simulador tarda segundos, ¿para qué un modelo sustituto?** *(responde Gabriel)*

R: Aquí el simulador es rápido porque está vectorizado (50 000 planetas en unos 10 s), así que el sustituto no se justifica por velocidad sino como banco de pruebas: es la misma situación que en simuladores caros (clima, epidemias), pero con verdad disponible. Aun así, el sustituto da una predicción instantánea de la probabilidad, mientras que la verdadera requiere 32 a 64 réplicas por planeta.

**P: ¿Qué es HANDY y qué variables tiene?** *(responde Gabriel)*

R: Human and Nature Dynamics (Motesharrei, Rivas y Kalnay, 2014): cuatro ecuaciones diferenciales para plebeyos, élites, naturaleza y riqueza. La naturaleza se regenera de forma logística, la población la agota (tasa δ), la extracción genera riqueza y, si la riqueza no alcanza el umbral de consumo, sube la mortalidad (de 0.01 a 0.07). κ es cuántas veces más consume una élite.

**P: ¿Qué error encontraron y qué cambió?** *(responde Gabriel)*

R: En la mortalidad de las élites, la Fase 1 usaba κ·s·x_E en el denominador cuando la ecuación 6 del paper usa s·x_E. Lo detectamos al verificar contra el artículo, lo corregimos con una prueba unitaria y regeneramos todo. Con los rangos anteriores la supervivencia bajó de 63 % a 15 %, así que re-ajustamos los rangos (κ ≤ 3, élites ≤ 10 %, δ ≤ 6·10⁻⁵) hasta llegar a 49.6%. Todos los resultados que mostramos son de la versión corregida.

**P: ¿Por qué no reprodujeron el umbral del colapso Tipo-N del paper?** *(responde Gabriel)*

R: El paper reporta ≈4× δ* (reversible) y 5.5× (irreversible). Con RK4 el colapso irreversible aparece desde ≈9.25× δ* y con Euler en precisión simple (como el paper) desde ≈3.25×. El umbral depende del integrador y no tenemos el código original, así que lo dejamos como límite documentado. Sí reproducimos equilibrio, oscilaciones, ciclos y los colapsos Tipo-L y Tipo-N con κ = 100.

**P: ¿Cómo definen colapso y por qué ese umbral?** *(responde Gabriel)*

R: Población total < máx(10 % de su pico, 20 % de la inicial) antes del año 600. El 10 % del pico captura un colapso real; el piso del 20 % de la inicial evita contar como colapso a planetas que nunca despegaron y evita que un pico pequeño haga trivial la condición. Es una decisión nuestra, documentada en `docs/DECISIONES.md` (D-06).

**P: ¿Qué extensiones agregaron y de dónde salen sus constantes?** *(responde Gabriel)*

R: Temperatura (reduce la regeneración como γ·exp(−(T/3)²) y sube la mortalidad base 15 % por °C), agua (productividad ∝ oferta/demanda, con la eficiencia hídrica reduciendo la demanda) y eventos de Poisson que destruyen una fracción de naturaleza y riqueza. Las constantes son supuestos nuestros, sin calibración empírica: es una limitación declarada.

## B. Datos

**P: ¿Por qué Latin Hypercube?** *(responde David)*

R: Cubre el espacio de 11 factores de forma más uniforme que un muestreo aleatorio con el mismo número de planetas (McKay et al., 1979). Es un diseño de experimentos computacionales, no una muestra de población.

**P: ¿Por qué δ y κ en escala logarítmica?** *(responde David)*

R: Porque sus rangos abarcan órdenes de magnitud (δ de 5·10⁻⁶ a 6·10⁻⁵); un muestreo log-uniforme no concentra los planetas en el extremo alto. En los modelos entran como log y luego z-score.

**P: ¿Cómo evitaron la fuga de información?** *(responde David)*

R: Tres cosas: el split es por planeta (35 000 / 7 500 / 7 500); media y desviación de la normalización se calculan solo con train; y en la tarea (b) solo se evalúan planetas todavía vivos al final de la ventana (si ya colapsó, la respuesta sería trivial). Reportamos cuántos quedan fuera: 360 de 7500 en k = 40 %, por ejemplo.

**P: ¿Qué es la «probabilidad verdadera» y qué tan confiable es?** *(responde David)*

R: Para 500 planetas de test corrimos 32 réplicas con distintas semillas de eventos y tomamos la fracción que sobrevive. Con 32 réplicas el error estándar máximo es √(0.25/32) ≈ 0.09, suficiente para calibración global pero ruidoso por planeta. En el 31% de los planetas la probabilidad es intermedia: hay incertidumbre real, no solo ruido del modelo.

**P: ¿Las clases están balanceadas?** *(responde David)*

R: Sí, casi: la supervivencia es 49.6%. Lo logramos ajustando los rangos, no con remuestreo.

**P: ¿Por qué 50 000 planetas?** *(responde David)*

R: Es un compromiso: la generación tarda ≈10 s, y con 35 000 planetas para entrenar el MLP (9 857 parámetros) y la LSTM (225 409) hay datos suficientes; si hiciera falta más, se genera en segundos.

## C. Modelos y entrenamiento (sesiones 1, 3, 8 y 12)

**P: ¿Por qué BCE con logits y no MSE?** *(responde David)*

R: La BCE es la verosimilitud negativa de una variable Bernoulli: entrena probabilidades bien calibradas (ECE del MLP: 0.010). Con logits integra la sigmoide y es numéricamente estable, y los logits son además el espacio donde SHAP es aditivo.

**P: ¿Por qué AdamW, dropout y early stopping?** *(responde David)*

R: AdamW desacopla el weight decay del paso adaptativo de Adam (Loshchilov y Hutter, 2019), lo que regulariza mejor que la L2 dentro del gradiente. El dropout (0.2 en el MLP, 0.1 en la LSTM) y el early stopping con paciencia de 15 y 12 épocas sobre la BCE de validación, restaurando los mejores pesos, frenan el sobreajuste. Todo es la sesión 8.

**P: ¿Cómo eligieron los hiperparámetros sin contaminar el test?** *(responde David)*

R: Búsqueda aleatoria (12 configuraciones para el MLP y 8 para la LSTM) con la semilla 0, escogiendo por BCE de validación; el test solo se usa para reportar. La mejor configuración se reentrena con 5 semillas.

**P: ¿Por qué falla la regresión logística?** *(responde David)*

R: Su ROC-AUC es 0.732: la frontera entre sobrevivir y colapsar es muy no lineal (interacciones como δ·κ·élites) y un modelo lineal en los factores no la capta. Sirve para mostrar que el problema no es trivial.

**P: ¿La mejora del MLP sobre LightGBM es relevante?** *(responde David)*

R: Es pequeña pero real: ΔAUC +0.0062 con IC 95 % [+0.0044, +0.0080] por bootstrap pareado, que excluye 0. También mejora Brier (0.066 vs 0.074) y el error contra la probabilidad Monte Carlo (0.054 vs 0.077). Que sea pequeña es parte del hallazgo: un modelo clásico bien afinado compite con la red.

**P: ¿Por qué una LSTM y no un GRU o un Transformer?** *(responde David)*

R: Usamos la LSTM de la sesión 3 porque las secuencias son cortas (101 pasos, hasta 60 de ellos observados); no probamos GRU ni Transformer, lo mencionamos como trabajo futuro. Lo que sí hicimos fue compararla con baselines que no ven la serie o la ven resumida (LightGBM con resúmenes, MLP sobre la ventana).

**P: ¿La LSTM es mejor que los baselines?** *(responde David)*

R: Depende de k. Con k = 0 pierde (0.955 vs 0.973 del MLP estático), con k = 40 % gana (0.982 vs 0.970) y con k = 60 % empatan con LightGBM con resúmenes (0.987 vs 0.987). La trayectoria aporta información nueva sobre todo con ventanas largas.

**P: ¿Para qué concatenar los parámetros a la LSTM?** *(responde David)*

R: Los parámetros del planeta son información que la trayectoria temprana no contiene. La ablación lo muestra: sin ellos, la LSTM parte de 0.57 de AUC (azar) con k = 0 y llega a 0.98 con k = 60 %.

**P: ¿Qué es el truncamiento aleatorio de k?** *(responde David)*

R: En cada lote se sortea k entre 0 y 70 pasos y se corta la secuencia (con secuencias empaquetadas). Así un solo modelo sirve para cualquier ventana y funciona como aumento de datos; se evalúa en k = 0, 5, 10, 20, 40 y 60 %.

**P: ¿Por qué no comparar las métricas entre distintos k?** *(responde David)*

R: Porque la población evaluada cambia: al aumentar k se excluyen los planetas que ya colapsaron dentro de la ventana, y los que quedan son más difíciles. Solo se comparan modelos dentro del mismo k.

**P: ¿Qué tan estables son los resultados?** *(responde David)*

R: Todos los modelos se entrenan con 5 semillas (2026–2030). La desviación del ROC-AUC del MLP es 0.001 y la de LightGBM 0.000, mucho menor que las diferencias que interpretamos.

## D. Explicabilidad (sesión 15)

**P: ¿Por qué SHAP exacto y no KernelSHAP?** *(responde Jonathan)*

R: Con 11 factores hay 2¹¹ = 2 048 coaliciones: se pueden evaluar todas y obtener el valor de Shapley exacto, sin muestreo. Comprobamos que coincide con KernelSHAP con diferencia máxima ≈3·10⁻⁷ y que cumple la eficiencia (la suma de atribuciones es f(x) − f(baseline)).

**P: ¿Por qué el baseline es el «planeta medio»?** *(responde Jonathan)*

R: Es el punto de referencia común de SHAP, IG y TreeSHAP, y en espacio estandarizado es el vector cero. Es una decisión que condiciona los resultados: las atribuciones miden «respecto a un planeta medio», no en absoluto. Con otro baseline, los valores cambiarían.

**P: ¿Qué diferencia hay entre SHAP, IG y TreeSHAP?** *(responde Jonathan)*

R: SHAP exacto promedia la contribución marginal de cada factor sobre todos los órdenes (propiedades de eficiencia y simetría). IG integra el gradiente del logit sobre la recta del baseline al planeta (completitud, 1 024 pasos, error ≤ 0.03). TreeSHAP calcula los valores de Shapley de un modelo de árboles en tiempo polinomial. Los tres usan el mismo baseline y el mismo espacio de salida (logit) para poder compararlos.

**P: ¿Cómo miden la fidelidad de una explicación?** *(responde Jonathan)*

R: Para cada planeta, devolvemos un factor a su valor base y medimos cuánto cambia la probabilidad real de supervivencia en el simulador (64 réplicas, mismos números aleatorios, efecto real si |ΔP| > 0.10). Luego correlacionamos (Spearman por planeta) esos efectos con las atribuciones, y medimos acuerdo de signo y si coincide el factor más importante. También hacemos curvas de eliminación y una prueba de cordura.

**P: ¿Es bueno un Spearman de ≈ 0.65?** *(responde Jonathan)*

R: Es moderado: concuerdan bastante (0.63 a 0.69 en planetas de frontera), pero lejos de perfecto. Para interpretarlo hay un techo: el efecto que predice el propio MLP, sin ninguna explicación, solo alcanza 0.82 de Spearman con el simulador. Es decir, una parte de la brecha es error del modelo sustituto, no del método de explicación.

**P: ¿Por qué el factor más importante solo se acierta ≈ 50-60 %?** *(responde Jonathan)*

R: Porque la atribución (Shapley) promedia sobre coaliciones e incluye interacciones, mientras que la verdad que medimos es el efecto de cambiar un solo factor al baseline, y el MLP es solo una aproximación. El acierto del top-1 es 50% (frontera) y 61% (Monte Carlo) con SHAP; el signo sí coincide en ≈91% de los casos con efecto real.

**P: ¿Qué es la prueba de cordura y qué resultado dio?** *(responde Jonathan)*

R: De Adebayo et al. (2018): si aleatorizamos los pesos del modelo, una explicación fiel debe dejar de concordar con el simulador. Pasó: el Spearman contra el simulador cae de 0.65 (entrenado) a -0.13 (aleatorio).

**P: ¿Qué muestran las curvas de eliminación?** *(responde Jonathan)*

R: Devolvemos al valor base los factores en el orden de SHAP; la probabilidad real cambia en -0.27 en promedio, frente a +0.17 con orden aleatorio. O sea, las explicaciones sí señalan los factores que importan, mejor que el azar.

**P: ¿Por qué no usaron LIME?** *(responde Jonathan)*

R: LIME es un sustituto local lineal y su resultado depende del muestreo y del kernel. Con 11 factores, SHAP exacto es determinista y más defendible; dejamos LIME como trabajo futuro.

**P: ¿Qué limitación tiene medir un factor a la vez?** *(responde Jonathan)*

R: No captura interacciones: si dos factores solo importan juntos, el efecto individual al baseline puede no coincidir con su valor de Shapley. Un siguiente paso es medir la fidelidad con intervenciones conjuntas.

## E. Contrafactuales y aplicación

**P: ¿Qué es un contrafactual y cómo lo calculan?** *(responde Jonathan)*

R: El cambio mínimo en las palancas accionables (δ, κ, s, eficiencia hídrica) que lleva al planeta a una probabilidad de supervivencia ≥ 0.8 según el MLP. Se optimiza un desplazamiento con Adam (600 pasos, lr 0.03) minimizando una bisagra hasta el logit objetivo más 0.5·L1 del cambio relativo al rango; la L1 favorece pocas palancas (sesión 8).

**P: ¿Por qué solo cambian las palancas accionables?** *(responde Jonathan)*

R: Porque un contrafactual debe ser algo que una sociedad pueda hacer: no se puede cambiar la temperatura, los eventos extremos ni la natalidad. Verificamos que los factores no accionables nunca cambian (0 en todos los casos).

**P: ¿Cómo saben que el contrafactual funciona de verdad?** *(responde Jonathan)*

R: Cada uno se verifica en el simulador con 64 réplicas: funcionan 90% de los casos (el modelo cree que 92%). La probabilidad real media pasa de 0.07 a 0.83, cambiando 2.0 palancas en promedio. Con una sola palanca (rejilla de 201 valores) resuelve 87%.

**P: ¿Qué pasa en el 10 % donde no funciona?** *(responde Jonathan)*

R: Son planetas donde el sustituto se equivoca fuera de su zona de confianza: el modelo cree que el cambio basta y el simulador no lo confirma. Es un buen ejemplo de por qué hay que verificar con el mecanismo real; la app lo señala con una advertencia.

**P: ¿Qué muestra la demo?** *(responde Jonathan)*

R: La app de Streamlit: movemos las 11 condiciones, el simulador corre decenas de réplicas y la red da su predicción en paralelo; vemos la explicación SHAP y pedimos un contrafactual que se verifica en el simulador. En «Sobreexplotación (δ alto)» el simulador y la red coinciden en 0 %, SHAP señala la depleción y el contrafactual (bajar δ) lleva la supervivencia real de 0 % a ≈100 %.

## F. Proyecto en general

**P: ¿Qué hallazgo principal tiene el proyecto?** *(responde Gabriel)*

R: Que una explicación convincente no es necesariamente fiel: SHAP, IG y TreeSHAP concuerdan con el efecto real del simulador de forma moderada, mejor que el azar y que un orden aleatorio, pero con errores que parte vienen del modelo sustituto. Y que con un simulador sí se puede medir cuánto.

**P: ¿Qué harían distinto o qué sigue?** *(responde Gabriel)*

R: Medir fidelidad con intervenciones conjuntas (interacciones), calibrar temperatura y agua con datos o un modelo climático, probar otras arquitecturas (GRU, Transformer) y otros métodos (LIME, explicaciones basadas en conceptos), e igualar el umbral Tipo-N del paper.

**P: ¿Cómo usaron la IA y qué aprendieron?** *(responde Gabriel)*

R: La usamos para escribir código y borradores; el historial está en `docs/PROMPTS.md`, con lo que funcionó y lo que falló. El caso clave: al verificar contra el paper original encontramos un error en la mortalidad de las élites que la IA había introducido. Lección: la IA acelera, pero hay que verificar contra la fuente.

**P: ¿Cómo se reproduce todo?** *(responde Gabriel)*

R: `pip install -r requirements.txt` y `pip install -e .`, y luego `python scripts/run_all.py` (≈40 min con GPU; con CPU toma varias horas). Las semillas están fijas y las cifras del documento salen de los JSON de `outputs/`. La app corre con `streamlit run app/streamlit_app.py`.

**P: ¿Cuáles son las limitaciones?** *(responde Gabriel)*

R: Datos sintéticos de un solo simulador; constantes de temperatura y agua sin calibrar; intervenciones de un factor a la vez; 5 semillas; y un umbral Tipo-N que no igualamos con el del paper.
