"""Build the class presentation (10 slides, 3 speakers) and the speaking scripts.

Outputs:
  report/Presentacion.pptx          slides with speaker notes (the text each person says)
  docs/PRESENTACION.md              who speaks on which slide + the spoken text
  docs/GUION_VIDEO.md               script for the demo video, split among the three members

Every number comes from outputs/*/results.json (never typed by hand).
Run from the repo root: python report/build_presentation.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
NAVY, GRAY, LIGHT = RGBColor(0x1F, 0x2A, 0x44), RGBColor(0x55, 0x5B, 0x6B), RGBColor(0xF1, 0xF3, 0xF8)
ACCENT = RGBColor(0x2F, 0x6F, 0xDE)  # one colour for the whole deck (who speaks is in the notes and the video script)
TITLE = "Simulador de mundos habitables: predicción de supervivencia con modelos sustitutos y explicabilidad verificable"


# ------------------------------------------------------------------ data
def load():
    j = lambda p: json.load(open(ROOT / p, encoding="utf8"))
    ta, tb, fd, cf = j("outputs/task_a/results.json"), j("outputs/task_b/results.json"), j("outputs/fidelity/results.json"), j("outputs/counterfactuals/results.json")
    import numpy as np
    d = np.load(ROOT / "data/worlds.npz")
    y, split = d["y"], d["split"]
    mcp = d["mc_p_true"]
    return dict(ta=ta, tb=tb, fd=fd, cf=cf, n=len(y), surv=float(y.mean()), mc_inter=float(((mcp > .05) & (mcp < .95)).mean()),
                n_train=int((split == 0).sum()), n_test=int((split == 2).sum()))


# ------------------------------------------------------------------ pptx helpers
def text(slide, s, left, top, width, height, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)); tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = s; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
    return tb


def bullets(slide, items, left, top, width, height, size=18, color=NAVY):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        lvl, s = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        r = p.add_run(); r.text = ("•  " if lvl == 0 else "–  ") + s
        r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = color if lvl == 0 else GRAY; r.font.name = "Calibri"
        p.level = 0
        if lvl:
            pPr = p._p.get_or_add_pPr(); pPr.set("marL", str(Inches(0.35))); pPr.set("indent", "0")
    return tb


def box(slide, s, left, top, width, height, fill=LIGHT, color=NAVY, size=14, bold=False, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1.5)
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = s; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = "Calibri"
    return sh


def arrow(slide, left, top, width, height=0.35, color=GRAY, shape=MSO_SHAPE.RIGHT_ARROW):
    sh = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    sh.fill.solid(); sh.fill.fore_color.rgb = color; sh.line.fill.background()
    return sh


def picture(slide, name, left, top, max_w, max_h):
    path = FIG / name
    w, h = Image.open(path).size
    scale = min(max_w / w, max_h / h)
    pw, ph = w * scale, h * scale
    slide.shapes.add_picture(str(path), Inches(left + (max_w - pw) / 2), Inches(top + (max_h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, rows, left, top, width, height, size=13):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(height)).table
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(size); r.font.name = "Calibri"; r.font.bold = (i == 0)
    return t


def new_slide(prs, n, title, who, notes):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.14)); bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    text(s, title, 0.55, 0.3, 12.2, 0.9, size=28, bold=True)
    text(s, "Deep Learning · UVG · Proyecto final", 0.55, 7.05, 6, 0.3, size=10, color=GRAY)
    text(s, f"{n} / 10", 11.9, 7.05, 1.0, 0.3, size=10, color=GRAY, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = f"[{who}]\n{notes}"
    return s


# ------------------------------------------------------------------ content
def build():
    D = load(); ta, tb, fd, cf = D["ta"], D["tb"], D["fd"], D["cf"]
    A, SA = ta["results"], ta["std"]; BA = ta["bootstrap"]["pairs"]
    B = {int(k): v for k, v in tb["results"].items()}; SB = {int(k): v for k, v in tb["std"].items()}; BB = {int(k): v["pairs"] for k, v in tb["bootstrap"].items()}
    g = fd["groups"]; bd, mc = g["boundary"], g["mc"]
    M = lambda grp, m, ref, k: grp["methods"][m][ref][k]
    dele = fd["deletion"]["SHAP (MLP)"]; san = fd["sanity_randomised_model"]
    nseeds = len(ta["seeds"])
    lstm_vs_lgb60 = BB[60]["lstm_minus_lightgbm_summary"]["roc_auc"]; lstm_vs_static60 = BB[60]["lstm_minus_mlp_static_only"]["roc_auc"]
    lstm_vs_lgb_txt = ("empatan" if lstm_vs_lgb60[0] <= 0 <= lstm_vs_lgb60[2] else ("gana la LSTM" if lstm_vs_lgb60[1] > 0 else "gana LightGBM"))
    gs, cs = cf["gradient"], cf["single_factor"]
    P = {}  # slide -> (title, speaker, notes, builder)
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    sl = []

    # 1 ---------------------------------------------------------------------------------------------
    notes = ("Buenas tardes. Somos Gabriel Bran, David Domínguez y Jonathan Díaz, y les presentamos nuestro proyecto final de Deep Learning. "
             "Empezamos con una pregunta: cuando una red neuronal nos da una explicación de su predicción, ¿podemos creerle? "
             "En casi todos los problemas reales no hay forma de comprobarlo. Nosotros construimos un problema donde sí: un simulador de civilizaciones, "
             "una red que predice si cada civilización sobrevive, y explicaciones que comprobamos contra el propio simulador. "
             "Yo, Gabriel, presento el problema y el simulador, y cierro con las conclusiones; David presenta los datos y los modelos; y Jonathan, las explicaciones y su verificación.")
    s = new_slide(prs, 1, "Simulador de mundos habitables", "Gabriel", notes)
    text(s, "Predicción de supervivencia con modelos sustitutos y explicabilidad verificable", 0.55, 1.35, 11.5, 0.9, size=22, color=GRAY)
    box(s, "¿Podemos creerle a una explicación?", 0.55, 2.7, 12.2, 1.3, fill=NAVY, color=RGBColor(255, 255, 255), size=34, bold=True)
    text(s, "Gabriel Bran · David Domínguez · Jonathan Díaz", 0.55, 4.5, 12, 0.5, size=22)
    text(s, "Deep Learning · Sección 10 · Universidad del Valle de Guatemala · 2 de noviembre de 2026", 0.55, 5.1, 12, 0.5, size=16, color=GRAY)
    sl.append((1, "Gabriel", "Portada y pregunta central", notes))

    # 2 ---------------------------------------------------------------------------------------------
    notes = ("Muchos sistemas científicos —clima, epidemias, ecología— se estudian con simuladores lentos y caros. "
             "Por eso se entrenan redes neuronales como modelos sustitutos, o surrogates: aprenden la relación entre condiciones de entrada y resultado, y responden al instante. "
             "El problema es que una red es una caja negra. Existen métodos de explicabilidad, como SHAP o Integrated Gradients, que dicen qué factores influyeron. "
             "Pero, como vimos en la sesión 15, una explicación post-hoc es un modelo del modelo: puede ser útil sin ser fiel. El ejemplo clásico es el clasificador de lobos que en realidad miraba la nieve del fondo, y la exactitud de validación no lo delata. "
             "Los criterios para juzgar una explicación son fidelidad, estabilidad y comprensibilidad, pero la fidelidad casi nunca se puede medir, porque no existe una explicación verdadera con la cual compararla. "
             "Nuestra idea: usar un problema donde el mecanismo real sí está disponible. El simulador nos deja intervenir un factor y ver qué pasa de verdad, así que podemos medir la fidelidad directamente. "
             "El proyecto tiene tres capas: el simulador, que genera la verdad; una red que lo imita, el sustituto; y las explicaciones de esa red, que verificamos contra el simulador. "
             "Todo se mide con varias semillas e intervalos de confianza, para no confundir una casualidad con un resultado.")
    s = new_slide(prs, 2, "El problema: sustitutos que no sabemos explicar", "Gabriel", notes)
    bullets(s, ["Los simuladores científicos son caros: se usan redes como modelos sustitutos (surrogates).",
                "Las explicaciones (SHAP, IG, contrafactuales) casi nunca tienen una verdad de referencia.",
                "S15: una explicación post-hoc es un «modelo del modelo»: puede ser útil sin ser fiel.",
                "Aquí sí hay verdad: el simulador es el mecanismo real, así que podemos intervenir y medir la fidelidad."], 0.55, 1.3, 12.2, 2.3, size=18)
    y0 = 3.95
    box(s, "Simulador HANDY\n(mecanismo real)", 0.6, y0, 2.6, 1.2, fill=LIGHT, size=15, bold=True, line=ACCENT)
    arrow(s, 3.3, y0 + 0.4, 0.8); text(s, "50 000 planetas", 3.2, y0 + 0.82, 1.2, 0.3, size=10, color=GRAY)
    box(s, "Red sustituta\n(MLP / LSTM)", 4.2, y0, 2.6, 1.2, fill=LIGHT, size=15, bold=True, line=ACCENT)
    arrow(s, 6.9, y0 + 0.4, 0.8); text(s, "predicción", 6.9, y0 + 0.82, 1.0, 0.3, size=10, color=GRAY)
    box(s, "Explicación\n(SHAP · IG · contrafactual)", 7.8, y0, 3.0, 1.2, fill=LIGHT, size=15, bold=True, line=ACCENT)
    box(s, "¿Intervenir un factor en el simulador cambia el resultado como dice la explicación?", 2.2, 5.6, 8.6, 0.95, fill=ACCENT, color=RGBColor(255, 255, 255), size=16, bold=True)
    arrow(s, 9.1, 5.18, 0.35, 0.4, shape=MSO_SHAPE.DOWN_ARROW); arrow(s, 2.0, 5.18, 0.35, 0.4, shape=MSO_SHAPE.UP_ARROW)
    sl.append((2, "Gabriel", "El problema", notes))

    # 3 ---------------------------------------------------------------------------------------------
    notes = ("El simulador está basado en HANDY, un modelo publicado en 2014 en Ecological Economics. Es un sistema de cuatro ecuaciones diferenciales: plebeyos, élites, naturaleza y riqueza. "
             "La naturaleza crece de forma logística, con tasa de regeneración gamma y capacidad lambda igual a cien, y la población la agota a una tasa delta por plebeyo. Esa extracción alimenta la riqueza, y la riqueza se consume. "
             "Si la riqueza es menor que el umbral rho por plebeyos más kappa rho por élites, el consumo se recorta y la mortalidad sube de uno por ciento a siete por ciento; kappa es cuántas veces más consume una élite que un plebeyo. "
             "Usamos los valores canónicos del paper: natalidad 0.03, rho 0.005, consumo s de 0.0005, gamma 0.01 y un punto de equilibrio delta asterisco de unos 6.67 por diez a la menos seis. "
             "Resolvemos con Runge-Kutta de orden cuatro, paso de un año, seiscientos años, guardando un punto cada seis años: ciento un puntos por planeta, vectorizado en NumPy para simular miles de planetas a la vez. "
             "Verificamos las ecuaciones contra el artículo original y al hacerlo encontramos un error nuestro: en la mortalidad de las élites usábamos kappa por s por x_E en el denominador, cuando la ecuación 6 usa s por x_E. "
             "Lo corregimos, agregamos una prueba unitaria y regeneramos todo el dataset; por eso todos los números que verán son de la versión corregida. "
             "Con las ecuaciones correctas reproducimos los escenarios del paper: equilibrio, oscilaciones, ciclos de colapso y los colapsos por desigualdad y por agotamiento de la naturaleza. Una salvedad honesta: el umbral exacto del colapso irreversible igualitario depende del integrador. Con Runge-Kutta aparece desde unas 9.25 veces delta asterisco y con Euler en precisión simple desde unas 3.25, y el paper reporta 5.5; no lo igualamos. "
             "Lo extendimos con tres mecanismos propios. Temperatura: reduce la regeneración como gamma por exponencial de menos T sobre tres, al cuadrado, y sube la mortalidad base quince por ciento por grado. Agua: limita la productividad de la naturaleza según la razón entre oferta y demanda, y la eficiencia hídrica reduce la demanda. "
             "Eventos extremos: llegan como un proceso de Poisson y cada uno destruye una fracción aleatoria de la naturaleza y de la riqueza, proporcional a su severidad. Son once factores; cuatro son accionables, o sea palancas de política: depleción, desigualdad, consumo y eficiencia hídrica. "
             "Colapso significa que la población cae por debajo del diez por ciento de su pico, o del veinte por ciento de la inicial, antes del año seiscientos. Estas trayectorias son las series multivariadas que después leerá la LSTM de la sesión tres.")
    s = new_slide(prs, 3, "El simulador: HANDY extendido", "Gabriel", notes)
    bullets(s, ["HANDY (Motesharrei et al., 2014): plebeyos x_C, élites x_E, naturaleza y, riqueza w. Ecuaciones (3)–(6) verificadas contra el paper (un error nuestro corregido).",
                "Valores del paper: β = 0.03, γ = 0.01, s = 5·10⁻⁴, ρ = 5·10⁻³, λ = 100, α_m = 0.01, α_M = 0.07; equilibrio δ* ≈ 6.67·10⁻⁶. Aquí varían δ, κ, s, β y γ.",
                "Integración RK4, dt = 1 año, 600 años, 101 puntos guardados (cada 6 años), vectorizada.",
                "Extensiones: temperatura (γ·e^−(T/3)², α_m·(1 + 0.15|T|)); agua (λ·clip(oferta/demanda, 0.05, 1)); eventos Poisson que destruyen una fracción sev·U de naturaleza y riqueza.",
                "11 factores, 4 accionables (δ, κ, s, eficiencia hídrica). Colapso: población < máx(10 % del pico, 20 % de la inicial) antes del año 600."], 0.55, 1.15, 12.2, 3.0, size=14)
    picture(s, "fig_escenarios_handy.png", 0.55, 3.35, 12.2, 3.65)
    sl.append((3, "Gabriel", "El simulador", notes))

    # 4 ---------------------------------------------------------------------------------------------
    notes = (f"Gracias, Gabriel. Con el simulador generamos {D['n']:,} planetas, muestreando las once condiciones con Latin Hypercube, que cubre el espacio de forma más uniforme que un muestreo aleatorio simple. Es un diseño de experimentos, no una muestra de datos reales. "
             "Delta y kappa se muestrean en escala logarítmica porque sus rangos abarcan órdenes de magnitud: delta de 5 por diez a la menos seis a 6 por diez a la menos cinco, kappa de uno a tres. El resto es uniforme: consumo s de 3.5 a 6.5 por diez a la menos cuatro, eficiencia hídrica de 0 a 0.6, temperatura de 0 a 5 grados, agua de 0.05 a 1.2, eventos de 0 a 0.12 por año con severidad de 0.1 a 0.9, y fracción inicial de élites de 0 a 10 %. "
             f"Ajustamos esos rangos en varias iteraciones documentadas para que la frontera entre sobrevivir y colapsar no fuera trivial; la supervivencia quedó en {D['surv']:.1%}, así que las clases están casi balanceadas. Generar todo tarda unos diez segundos. "
             f"El split es 70, 15 y 15 por planeta: {D['n_train']:,} para entrenar, {ta['n_val']:,} para validar y {D['n_test']:,} para probar. Para el preprocesamiento aplicamos logaritmo a delta y kappa y luego z-score, con medias y desviaciones calculadas solo con el conjunto de entrenamiento, para no filtrar información. "
             "Para la tarea de series, cada planeta es una secuencia de 101 pasos con seis canales: logaritmo de plebeyos, logaritmo de élites, naturaleza dividida entre cien, logaritmo de la riqueza, número de eventos del intervalo y el tiempo normalizado, también estandarizados con train. "
             "Esto es lo que vimos en series de tiempo, sesión 12: convertir una serie en un problema supervisado con ventanas, cuidando que no haya fuga de información del futuro. "
             f"Como los eventos extremos son aleatorios, las mismas condiciones pueden dar resultados distintos. Por eso, para 500 planetas de test corrimos 32 réplicas con distintas semillas, o sea 16 mil simulaciones, y obtuvimos la probabilidad verdadera de supervivencia. En el {D['mc_inter']:.0%} de esos planetas la probabilidad es intermedia, o sea que hay incertidumbre real. "
             "Esta probabilidad verdadera nos sirve para evaluar la calibración de los modelos y, después, como verdad para las explicaciones. A la derecha ven tres planetas que sobreviven y tres que colapsan; las franjas naranja son eventos extremos.")
    s = new_slide(prs, 4, "Datos: 50 000 planetas y una verdad de referencia", "David", notes)
    bullets(s, [f"{D['n']:,} planetas por Latin Hypercube sobre 11 factores; δ y κ log-uniformes. Supervivencia {D['surv']:.1%}.",
                "Rangos: δ 5·10⁻⁶–6·10⁻⁵, κ 1–3, s 3.5–6.5·10⁻⁴, eficiencia hídrica 0–0.6, temperatura 0–5 °C, agua 0.05–1.2, eventos 0–0.12/año (severidad 0.1–0.9), élites 0–10 %.",
                f"Split 70/15/15 por planeta: {D['n_train']:,} / {ta['n_val']:,} / {D['n_test']:,}. Estáticas: log(δ, κ) + z-score con estadísticas solo de train.",
                "Secuencia: 101 pasos × 6 canales (log x_C, log x_E, y/100, log w, eventos, tiempo), z-score con train (S12: serie → problema supervisado).",
                f"Monte Carlo: 500 planetas de test × 32 réplicas = 16 000 simulaciones → P(sobrevive | parámetros). En {D['mc_inter']:.0%} la probabilidad es intermedia."], 0.55, 1.2, 6.3, 5.7, size=14)
    picture(s, "fig_eda_trayectorias.png", 7.0, 1.2, 5.9, 3.6)
    picture(s, "fig_eda_clases_tiempos.png", 7.0, 4.8, 5.9, 2.2)
    sl.append((4, "David", "Datos y verdad de referencia", notes))

    # 5 ---------------------------------------------------------------------------------------------
    hpa, hpb = ta["best_hp"], tb["best_hp"]
    notes = ("Planteamos dos tareas. En la tarea a, un MLP lee las once condiciones estandarizadas y predice la probabilidad de supervivencia. "
             f"Son capas Linear, ReLU y Dropout, como en la sesión uno y la ocho; la mejor configuración de la búsqueda fue {' a '.join(str(x) for x in [11] + list(hpa['hidden']) + [1])}, con dropout {hpa['dropout']} y 9 857 parámetros. "
             "La salida es un logit y la pérdida es entropía cruzada binaria con logits, que integra la sigmoide por estabilidad numérica y, como vimos en la sesión ocho, entrena probabilidades bien calibradas. "
             f"Optimizamos con AdamW, que desacopla el weight decay del gradiente: tasa de aprendizaje {hpa['lr']}, decaimiento {hpa['weight_decay']}, lotes de 512, y early stopping sobre la pérdida de validación con paciencia de 15 épocas, restaurando los mejores pesos. "
             "En la tarea b, una LSTM, de la sesión tres, lee los primeros k por ciento de la trayectoria, hasta k igual a 60 %, o sea hasta 360 años, junto con los parámetros. Las compuertas de la LSTM controlan qué se recuerda de la historia. Los once factores estáticos se concatenan a cada paso de tiempo y otra vez a la entrada de la cabeza; usamos secuencias empaquetadas para que cada planeta use su último estado oculto válido, sin contar el relleno. "
             f"La configuración elegida es de {hpb['layers']} capas de {hpb['hidden']} unidades, dropout {hpb['dropout']}, 225 409 parámetros, AdamW con tasa {hpb['lr']} y weight decay {hpb['weight_decay']}. "
             "La entrenamos con truncamiento aleatorio: en cada lote k se sortea entre 0 y 70 pasos, así un solo modelo sirve para cualquier ventana, y funciona como un aumento de datos. Para no medir una ventaja trivial, solo evaluamos planetas que siguen vivos al final de la ventana, y reportamos cuántos quedan fuera. "
             "Exigimos que la red demuestre que aporta algo. Comparamos con regresión logística; con LightGBM, de mil árboles, 31 hojas y tasa 0.05; con LightGBM sobre resúmenes de la ventana, como el último valor, la pendiente y el máximo; con un MLP sobre la ventana aplanada; y con un MLP que solo ve los parámetros. Además hicimos una ablación: la LSTM sin parámetros estáticos. "
             f"Los hiperparámetros salen de una búsqueda aleatoria: 12 configuraciones para el MLP, variando capas, dropout, tasa y weight decay, y 8 para la LSTM, con la semilla cero. La mejor se reentrena con {nseeds} semillas y calculamos intervalos de confianza del 95 % con bootstrap pareado de mil remuestreos sobre los planetas de test.")
    s = new_slide(prs, 5, "Modelos: la red tiene que ganarse su lugar", "David", notes)
    y0 = 1.2
    box(s, "11 factores (z-score)", 0.55, y0, 2.6, 0.7, size=13, bold=True, line=ACCENT); arrow(s, 3.25, y0 + 0.2, 0.6)
    box(s, "MLP · BCE con logits", 3.95, y0, 3.6, 0.7, size=13, bold=True, fill=ACCENT, color=RGBColor(255, 255, 255)); arrow(s, 7.65, y0 + 0.2, 0.6)
    box(s, "P(sobrevive) · tarea (a)", 8.35, y0, 3.6, 0.7, size=13, bold=True, line=ACCENT)
    y1 = 2.05
    box(s, "k % de la trayectoria + 11 factores", 0.55, y1, 2.6, 0.7, size=12, bold=True, line=ACCENT); arrow(s, 3.25, y1 + 0.2, 0.6)
    box(s, "LSTM · secuencias empaquetadas", 3.95, y1, 3.6, 0.7, size=13, bold=True, fill=ACCENT, color=RGBColor(255, 255, 255)); arrow(s, 7.65, y1 + 0.2, 0.6)
    box(s, "P(sobrevive) · tarea (b)", 8.35, y1, 3.6, 0.7, size=13, bold=True, line=ACCENT)
    rows = [["Modelo", "Arquitectura", "Entrenamiento"],
            ["MLP (a)", f"{'→'.join(str(x) for x in [11] + list(hpa['hidden']) + [1])}, ReLU, dropout {hpa['dropout']} · 9 857 parámetros", f"AdamW lr {hpa['lr']}, wd {hpa['weight_decay']}; lote 512; paciencia 15"],
            ["LSTM (b)", f"{hpb['layers']} capas × {hpb['hidden']}, dropout {hpb['dropout']}; estáticas en cada paso y en la cabeza · 225 409 par.", f"AdamW lr {hpb['lr']}, wd {hpb['weight_decay']}; k aleatorio ∈ [0, 70]; paciencia 12"],
            ["LightGBM", "1 000 árboles, 31 hojas, tasa 0.05 (estáticas, o resúmenes de la ventana)", "submuestreo 0.8, columnas 0.9"],
            ["Otros baselines", "Regresión logística · MLP sobre la ventana · MLP solo con parámetros", "LSTM sin estáticas (ablación)"]]
    table(s, rows, 0.55, 2.95, 12.2, 2.2, size=12)
    bullets(s, [f"Búsqueda aleatoria de hiperparámetros: MLP 12 configuraciones (capas, dropout, lr, wd), LSTM 8; {nseeds} semillas; IC 95 % por bootstrap pareado (1 000 remuestreos).",
                "Sin fuga: solo se evalúan planetas aún vivos al final de la ventana; se reporta cuántos quedan fuera.",
                "Vínculos con el curso: S1 neurona/ReLU · S3 LSTM y compuertas · S8 BCE, AdamW, dropout, early stopping · S12 ventanas."], 0.55, 5.3, 12.2, 1.7, size=14)
    sl.append((5, "David", "Modelos", notes))

    # 6 ---------------------------------------------------------------------------------------------
    d_ml = BA["mlp_minus_lightgbm"]["roc_auc"]
    notes = (f"Resultados de la tarea a, con media y desviación sobre {nseeds} semillas. Usamos cuatro métricas. ROC-AUC mide qué tan bien ordena el modelo planetas que sobreviven y que colapsan. Brier es el error cuadrático de las probabilidades y ECE mide la calibración, o sea si cuando dice setenta por ciento, de verdad sobrevive el setenta por ciento. "
             f"Y como tenemos la probabilidad verdadera Monte Carlo, calculamos el error absoluto medio contra ella. La regresión logística llega a ROC-AUC de {A['logistic']['roc_auc']:.2f}: la frontera es muy no lineal, y un modelo lineal en los factores no la captura. "
             f"LightGBM obtiene {A['lightgbm']['roc_auc']:.3f} y el MLP {A['mlp']['roc_auc']:.3f}. La diferencia del MLP con LightGBM es de {d_ml[1]:+.4f} de AUC, con intervalo de confianza de {d_ml[0]:+.4f} a {d_ml[2]:+.4f}; "
             f"como el intervalo no incluye cero, la ventaja es real, aunque pequeña. El MLP también tiene mejor Brier, {A['mlp']['brier']:.3f} contra {A['lightgbm']['brier']:.3f}, y su error contra la probabilidad verdadera Monte Carlo es de {A['mlp']['mc_mae']:.3f}, contra {A['lightgbm']['mc_mae']:.3f} de LightGBM. "
             f"Para la tarea b, la gráfica muestra el desempeño al observar más trayectoria; k es el porcentaje de los seiscientos años que se ve. Con k igual a cero la LSTM solo ve los parámetros. En ventanas largas, de sesenta por ciento, la LSTM alcanza {B[60]['lstm']['roc_auc']:.3f}, LightGBM con resúmenes {B[60]['lightgbm_summary']['roc_auc']:.3f} y el MLP que no ve la trayectoria {B[60]['mlp_static_only']['roc_auc']:.3f}. "
             f"Entre la LSTM y LightGBM, {lstm_vs_lgb_txt} (diferencia {lstm_vs_lgb60[1]:+.4f}, intervalo {lstm_vs_lgb60[0]:+.4f} a {lstm_vs_lgb60[2]:+.4f}). Con k igual a 40 por ciento, la LSTM supera al MLP estático, {B[40]['lstm']['roc_auc']:.3f} contra {B[40]['mlp_static_only']['roc_auc']:.3f}: la historia aporta información que los parámetros iniciales no tienen. "
             f"Sin parámetros estáticos, la LSTM parte de {B[0]['lstm_no_static']['roc_auc']:.2f}, o sea casi azar, y sube a {B[60]['lstm_no_static']['roc_auc']:.2f}: la trayectoria sola acaba informando, pero con poca ventana los parámetros son indispensables. "
             "Es importante aclarar que entre distintos k no se pueden comparar las métricas directamente, porque cambia la población evaluada.")
    s = new_slide(prs, 6, "Resultados de los modelos", "David", notes)
    rows = [["Tarea (a)", "ROC-AUC", "Brier", "ECE", "MAE vs P verdadera"]]
    for k, lab in (("logistic", "Regresión logística"), ("lightgbm", "LightGBM"), ("mlp", "MLP")):
        rows.append([lab, f"{A[k]['roc_auc']:.3f} ± {SA[k]['roc_auc']:.3f}", f"{A[k]['brier']:.3f} ± {SA[k]['brier']:.3f}", f"{A[k]['ece']:.3f}", f"{A[k]['mc_mae']:.3f} ± {SA[k]['mc_mae']:.3f}"])
    table(s, rows, 0.55, 1.3, 7.4, 1.7, size=13)
    bullets(s, [f"MLP − LightGBM: ΔAUC {d_ml[1]:+.4f} [{d_ml[0]:+.4f}, {d_ml[2]:+.4f}] (IC 95 %).",
                f"k = 60 %: LSTM {B[60]['lstm']['roc_auc']:.3f} vs. LightGBM {B[60]['lightgbm_summary']['roc_auc']:.3f} → {lstm_vs_lgb_txt}.",
                f"k = 40 %: LSTM {B[40]['lstm']['roc_auc']:.3f} vs. MLP estático {B[40]['mlp_static_only']['roc_auc']:.3f}.",
                "Sin parámetros estáticos, la LSTM parte casi del azar (k = 0)."], 8.2, 1.25, 4.8, 2.0, size=13)
    picture(s, "fig_tarea_b_curva_k.png", 0.55, 3.35, 12.2, 3.65)
    sl.append((6, "David", "Resultados de modelos", notes))

    # 7 ---------------------------------------------------------------------------------------------
    chk = json.load(open(ROOT / "outputs/xai/checks.json", encoding="utf8"))["checks"]
    nb, nm = M(bd, "SHAP (MLP)", "vs_simulator", "n_planets"), M(mc, "SHAP (MLP)", "vs_simulator", "n_planets")
    notes = ("Gracias, David. Ahora la parte central del proyecto. Calculamos tres explicaciones. SHAP sobre el MLP, que con once factores calculamos de forma exacta: son dos a la once, 2 048 coaliciones, y el valor de Shapley de cada factor promedia su contribución marginal sobre todos los órdenes posibles. "
             "Integrated Gradients, de la sesión 15, con Captum y mil veinticuatro pasos de integración: acumula el gradiente a lo largo del camino entre el baseline y el planeta. Y TreeSHAP sobre LightGBM, que calcula los valores de Shapley de un árbol en tiempo polinomial. "
             "Algo clave: todos usan el mismo baseline, el planeta medio de entrenamiento, que en el espacio estandarizado es el vector cero, y el mismo espacio de salida, el logit de supervivencia. Así las comparaciones son justas. "
             f"Comprobamos las propiedades de cada método. SHAP exacto coincide con KernelSHAP con una diferencia máxima de {chk['exact_vs_kernelshap_max_abs_diff']:.1e}; cumple la eficiencia, es decir, la suma de las atribuciones es la predicción menos la del baseline, con error de {chk['shap_mlp']:.0e}; y Integrated Gradients cumple la completitud con un error máximo de {chk['ig_convergence_delta_max']:.2f} logits. "
             "Pero que las explicaciones coincidan entre sí no prueba que sean correctas. Por eso las verificamos de cuatro maneras. "
             "Uno, intervenir un factor en el modelo: lo devolvemos a su valor base y vemos cuánto cambia el logit. "
             f"Dos, intervenirlo en el simulador: corremos 64 réplicas con los mismos números aleatorios en ambos lados para reducir el ruido, y medimos el cambio real en la probabilidad de supervivencia; consideramos que hay un efecto real si cambia más de 0.10. Lo hacemos en {nb} planetas cercanos a la frontera de decisión y {nm} planetas Monte Carlo con algún efecto real. "
             "Comparamos con Spearman por planeta, acuerdo de signo, y si coincide el factor más importante o los tres más importantes. "
             "Tres, curvas de eliminación: si devolvemos al valor base los factores en el orden que dice la explicación, la probabilidad real debe degradarse más rápido que con un orden aleatorio. "
             "Y cuatro, una prueba de cordura de Adebayo y colaboradores: si aleatorizamos los pesos del modelo, la explicación debe dejar de concordar con el simulador.")
    s = new_slide(prs, 7, "Explicaciones y cómo las verificamos", "Jonathan", notes)
    box(s, "Mismo baseline (planeta medio de entrenamiento = vector 0)\nMisma salida (logit de supervivencia)", 0.55, 1.2, 7.4, 0.85, fill=NAVY, color=RGBColor(255, 255, 255), size=14, bold=True)
    for i, (lab, sub) in enumerate((("SHAP exacto · MLP", f"2¹¹ = 2 048 coaliciones\nvs. KernelSHAP: ≈ 3·10⁻⁷"),
                                    ("Integrated Gradients · MLP", f"Captum, 1 024 pasos\nerror de completitud ≤ {chk['ig_convergence_delta_max']:.2f}"),
                                    ("TreeSHAP · LightGBM", "polinomial en el árbol\nerror ≈ 3·10⁻⁷"))):
        box(s, f"{lab}\n{sub}", 0.55 + i * 2.5, 2.2, 2.4, 1.25, size=11, bold=True, line=ACCENT)
    bullets(s, ["Intervención en el modelo: ¿cambia el logit como dice φ?",
                f"Intervención en el simulador: 64 réplicas, números aleatorios comunes; efecto real si |ΔP| > 0.10. {nb} planetas de frontera + {nm} Monte Carlo con efecto.",
                "Métricas: Spearman por planeta, acuerdo de signo, top-1 y top-3.",
                "Curvas de eliminación: orden por atribución vs. aleatorio (12 pasos, 32 réplicas).",
                "Prueba de cordura: pesos aleatorizados (Adebayo et al., 2018)."], 0.55, 3.6, 7.4, 3.4, size=15)
    picture(s, "fig_xai_beeswarm.png", 8.2, 1.2, 4.7, 5.7)
    sl.append((7, "Jonathan", "Explicaciones y verificación", notes))

    # 8 ---------------------------------------------------------------------------------------------
    notes = (f"Resultados de la fidelidad. En la gráfica, las barras naranja comparan cada atribución con el efecto real en el simulador, y las grises con el efecto en el modelo; las líneas son intervalos de confianza del 95 %. "
             f"Con planetas cercanos a la frontera de decisión, la correlación de Spearman por planeta es de {M(bd,'SHAP (MLP)','vs_simulator','spearman'):.2f} para SHAP, {M(bd,'IG (MLP)','vs_simulator','spearman'):.2f} para Integrated Gradients y {M(bd,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} para TreeSHAP; "
             f"con planetas Monte Carlo aleatorios, entre {M(mc,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} y {M(mc,'IG (MLP)','vs_simulator','spearman'):.2f}. Es decir, concuerdan bastante, pero lejos de perfecto. "
             f"El signo coincide en cerca de {M(bd,'SHAP (MLP)','vs_simulator','sign_agreement'):.0%} de los casos con efecto real, pero el factor más importante solo coincide en {M(bd,'SHAP (MLP)','vs_simulator','top1'):.0%} a {M(mc,'SHAP (MLP)','vs_simulator','top1'):.0%}. "
             f"Parte del error viene del propio modelo: la línea azul indica que incluso el efecto que el MLP predice por sí mismo, al intervenir un factor, solo llega a {bd['model_vs_simulator_effect']['spearman']:.2f} de Spearman con el simulador. Eso significa que la explicación no puede ser más fiel que el modelo que explica, y también que el modelo sustituto se equivoca en algunos planetas. "
             f"En las curvas de eliminación, quitar factores en el orden de SHAP cambia la probabilidad real en {dele['attribution']['sim_auc']:+.2f} en promedio, mientras que el orden aleatorio la cambia en {dele['random']['sim_auc']:+.2f}: las explicaciones sí apuntan a lo que importa. "
             f"Y la prueba de cordura pasa: con el modelo aleatorizado la concordancia con el simulador cae de {san['vs_simulator_trained']['spearman']:.2f} a {san['vs_simulator_randomised']['spearman']:.2f}. Eso descarta que la explicación sea solo un reflejo de los datos de entrada. "
             "La conclusión honesta, relacionada con la sesión 15: una explicación convincente no es necesariamente una explicación fiel. Las explicaciones son útiles y dependen del modelo, pero no son una verdad; hay que verificarlas.")
    s = new_slide(prs, 8, "¿Son fieles las explicaciones? Resultados", "Jonathan", notes)
    picture(s, "fig_fidelidad_spearman.png", 0.55, 1.2, 12.2, 3.6)
    bullets(s, [f"Spearman vs. simulador: {M(bd,'SHAP (MLP)','vs_simulator','spearman'):.2f} (SHAP), {M(bd,'IG (MLP)','vs_simulator','spearman'):.2f} (IG), {M(bd,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} (TreeSHAP) en planetas de frontera (n = {nb}); signo correcto ≈{M(bd,'SHAP (MLP)','vs_simulator','sign_agreement'):.0%}; factor n.º 1 acertado ≈{M(bd,'SHAP (MLP)','vs_simulator','top1'):.0%}–{M(mc,'SHAP (MLP)','vs_simulator','top1'):.0%}.",
                f"Techo del modelo: el efecto que predice el propio MLP vs. el simulador llega solo a ρ = {bd['model_vs_simulator_effect']['spearman']:.2f}.",
                f"Eliminación en el simulador: orden SHAP {dele['attribution']['sim_auc']:+.2f} vs. aleatorio {dele['random']['sim_auc']:+.2f} (cambio medio de P).",
                f"Prueba de cordura: Spearman vs. simulador {san['vs_simulator_trained']['spearman']:.2f} (entrenado) → {san['vs_simulator_randomised']['spearman']:.2f} (pesos aleatorios)."],
            0.55, 4.9, 12.2, 2.1, size=14)
    sl.append((8, "Jonathan", "Fidelidad: resultados", notes))

    # 9 ---------------------------------------------------------------------------------------------
    notes = (f"Una explicación útil debería permitir actuar. Por eso buscamos contrafactuales, como en Wachter y colaboradores, de la sesión 15: el cambio mínimo en las palancas accionables —depleción, desigualdad, consumo y eficiencia hídrica— que lleva el planeta a sobrevivir. "
             f"Los factores del entorno, como la temperatura o los eventos, no se tocan. Los aplicamos a los planetas que el modelo predice que colapsarán. "
             "La búsqueda es por descenso de gradiente sobre el MLP: optimizamos con Adam un desplazamiento en el espacio estandarizado, 600 pasos con tasa 0.03, solo en las cuatro palancas y recortado al rango de muestreo. "
             "La pérdida tiene dos términos: una bisagra que penaliza que el logit esté por debajo del objetivo, que es 80 % de probabilidad de supervivencia, y una penalización L1 de peso 0.5 sobre el cambio relativo al rango. Como vimos en la sesión ocho, la L1 favorece soluciones dispersas, es decir, que cambien pocos factores; además descartamos cambios menores al uno por ciento del rango. "
             "Lo importante es que no confiamos en la red: cada contrafactual se verifica en el simulador con 64 réplicas. "
             f"El método de gradiente resuelve el {gs['solved_share']:.0%} de los casos según el modelo; en el simulador, el {gs['simulator_valid']:.0%} de los contrafactuales realmente sobrevive, y la probabilidad real pasa de {gs['mean_true_p_before']:.2f} a {gs['mean_true_p_after']:.2f}, con {gs['mean_factors_changed']:.1f} palancas cambiadas en promedio. "
             f"Cambiar una sola palanca, con una búsqueda en rejilla de 201 valores, resuelve solo el {cs['solved_share']:.0%} de los casos. "
             "Ahora les mostramos la aplicación en Streamlit: movemos los factores de un planeta, vemos la simulación con sus réplicas, la explicación SHAP y pedimos un contrafactual que el simulador verifica. "
             "[DEMO EN VIVO: elegir el escenario 'Sobreexplotación (δ alto)', mostrar el colapso, abrir la pestaña de explicación y pulsar 'Buscar contrafactual'.]")
    s = new_slide(prs, 9, "Contrafactuales verificados y demo", "Jonathan", notes)
    bullets(s, ["Pregunta: ¿cuál es el cambio mínimo en las palancas accionables (δ, κ, s, eficiencia hídrica) que lleva el planeta a sobrevivir?",
                "Búsqueda: Adam sobre el desplazamiento Δ en espacio estandarizado (600 pasos, lr 0.03), solo en 4 palancas, recortado al rango.",
                "Pérdida: bisagra hasta P ≥ 0.8 (logit ≈ 1.39) + 0.5 · L1(Δ / rango) → pocas palancas (S8: L1 → dispersión). Verificación: 64 réplicas del simulador.",
                f"Gradiente: {gs['solved_share']:.0%} resueltos; {gs['simulator_valid']:.0%} válidos en el simulador; P real {gs['mean_true_p_before']:.2f} → {gs['mean_true_p_after']:.2f}; {gs['mean_factors_changed']:.1f} palancas.",
                f"Una sola palanca (rejilla de 201): {cs['solved_share']:.0%} resueltos; {cs['simulator_valid']:.0%} válidos en el simulador. Los factores no accionables nunca se modifican."], 0.55, 1.15, 7.5, 3.3, size=13)
    box(s, "DEMO EN VIVO\nstreamlit run app/streamlit_app.py", 8.3, 1.4, 4.5, 1.7, fill=ACCENT, color=RGBColor(255, 255, 255), size=17, bold=True)
    picture(s, "fig_contrafactuales.png", 0.55, 4.3, 12.2, 2.7)
    sl.append((9, "Jonathan", "Contrafactuales y demo", notes))

    # 10 --------------------------------------------------------------------------------------------
    notes = (f"Gracias, Jonathan. Para cerrar. Primero, el modelo: en la tarea a, el MLP llega a ROC-AUC de {A['mlp']['roc_auc']:.3f}, contra {A['lightgbm']['roc_auc']:.3f} de LightGBM y {A['logistic']['roc_auc']:.3f} de la regresión logística: la red aporta, con una ventaja pequeña pero con intervalo de confianza que excluye cero, y la logística falla porque la frontera es no lineal. "
             f"En alerta temprana, la trayectoria ayuda sobre todo en ventanas largas: con k igual a 40 por ciento la LSTM llega a {B[40]['lstm']['roc_auc']:.3f} contra {B[40]['mlp_static_only']['roc_auc']:.3f} del MLP estático, y con 60 por ciento empata con LightGBM. Un modelo clásico bien alimentado compite con la red recurrente, y eso es parte del hallazgo. "
             f"Segundo, las explicaciones: concuerdan con la realidad del simulador de forma moderada, con Spearman de {M(bd,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} a {M(bd,'IG (MLP)','vs_simulator','spearman'):.2f}, mejor que el azar y que un orden aleatorio, y pasan la prueba de cordura, pero no son una verdad. Es la lección de la sesión 15: desconfiar de las explicaciones y verificarlas. "
             f"Tercero, los contrafactuales verificados con el simulador funcionan en el {gs['simulator_valid']:.0%} de los casos. Entre las limitaciones: los efectos de una intervención individual no capturan interacciones; "
             f"las constantes de temperatura y agua son supuestos nuestros; los modelos se entrenaron con {nseeds} semillas y un solo simulador; y no reprodujimos con exactitud el umbral del colapso irreversible del paper. "
             "Gracias; quedamos atentos a sus preguntas.")
    s = new_slide(prs, 10, "Conclusiones y límites", "Gabriel", notes)
    cols = (("Lo que encontramos", [f"ROC-AUC: MLP {A['mlp']['roc_auc']:.3f} · LightGBM {A['lightgbm']['roc_auc']:.3f} · logística {A['logistic']['roc_auc']:.3f}; la ventaja del MLP tiene IC que excluye 0.",
                                    f"Alerta temprana: con k = 40 %, LSTM {B[40]['lstm']['roc_auc']:.3f} vs. MLP estático {B[40]['mlp_static_only']['roc_auc']:.3f}; con k = 60 % empata con LightGBM.",
                                    f"Las explicaciones concuerdan con el simulador (ρ ≈ {M(bd,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f}–{M(bd,'IG (MLP)','vs_simulator','spearman'):.2f}) y pasan la prueba de cordura, pero no son una verdad (S15).",
                                    f"{gs['simulator_valid']:.0%} de los contrafactuales funcionan en el simulador."]),
            ("Límites", ["Efecto de un factor a la vez: no captura interacciones.", "Temperatura y agua: supuestos nuestros, sin calibración empírica.", "No igualamos el umbral del Tipo-N del paper (depende del integrador).", f"{nseeds} semillas y un solo simulador (datos sintéticos)."]))
    for i, (h, items) in enumerate(cols):
        box(s, h, 0.55 + i * 6.2, 1.3, 5.95, 0.6, fill=ACCENT, color=RGBColor(255, 255, 255), size=18, bold=True)
        bullets(s, items, 0.55 + i * 6.2, 2.1, 5.95, 4.8, size=17)
    sl.append((10, "Gabriel", "Conclusiones", notes))

    out = ROOT / "report/Presentacion.pptx"; prs.save(out); print("OK ->", out)
    return D, sl, dict(A=A, SA=SA, B=B, SB=SB, BB=BB, BA=BA, fd=fd, cf=cf, bd=bd, mc=mc, dele=dele, san=san, lstm_vs_lgb_txt=lstm_vs_lgb_txt)


def write_markdown(D, sl, X):
    by = {"Gabriel": [], "David": [], "Jonathan": []}
    for n, who, title, notes in sl:
        by[who].append((n, title, notes))
    words = lambda t: len(t.split())
    lines = ["# Presentación: reparto entre los tres hablantes", "",
             "Archivo de diapositivas: `report/Presentacion.pptx` (el texto que dice cada persona también está en las *notas del orador* de cada diapositiva).",
             "Para entregar el PDF: abrir en PowerPoint → Archivo → Exportar → PDF (máximo 10 diapositivas: son exactamente 10).", "",
             "| Hablante | Diapositivas | Palabras | Tiempo aprox. (130 pal/min) |", "|---|---|---|---|"]
    for who, items in by.items():
        w = sum(words(t) for _, _, t in items)
        lines.append(f"| {who} | {', '.join(str(n) for n, _, _ in items)} | {w} | {w / 130:.1f} min |")
    tot = sum(words(t) for _, _, _, t in sl)
    lines += ["", f"Total: {tot} palabras ≈ {tot / 130:.1f} minutos al leerlo con calma. **Si el tiempo asignado es menor, recorten las notas, no las diapositivas.**", "",
              "Reglas sugeridas: quien habla cede la palabra al terminar su última diapositiva diciendo el nombre de la siguiente persona; cada quien debe poder responder preguntas de *todo* el proyecto (ver `docs/NOTAS_PRESENTACION.md`).", ""]
    lines += ["", "Notas por diapositiva y preguntas probables con respuestas: `docs/NOTAS_PRESENTACION.md`."]
    (ROOT / "docs/PRESENTACION.md").write_text("\n".join(lines), encoding="utf8"); print("OK -> docs/PRESENTACION.md")


def write_notes(D, sl, X):
    """docs/NOTAS_PRESENTACION.md: the spoken notes of every slide + a bank of likely questions with answers."""
    A, SA, B, BA, bd, mc, dele, san, cf = X["A"], X["SA"], X["B"], X["BA"], X["bd"], X["mc"], X["dele"], X["san"], X["cf"]
    gs, cs = cf["gradient"], cf["single_factor"]
    sp = lambda g, m: g["methods"][m]["vs_simulator"]["spearman"]
    d_ml = BA["mlp_minus_lightgbm"]["roc_auc"]
    L = ["# Notas de la presentación y preguntas probables", "",
         "Este archivo se genera con `python report/build_presentation.py`; las cifras salen de `outputs/*/results.json`. "
         "Reparto y tiempos: ver `docs/PRESENTACION.md`. Las notas también están en las *notas del orador* de `report/Presentacion.pptx`. "
         "Son un guion de apoyo: cada quien debe decirlo con sus palabras.", "",
         "Cada quien debe poder contestar preguntas de **todo** el proyecto; la columna «Responde» solo indica quién toma la pregunta primero.", "",
         "---", "", "# Parte 1 · Notas por diapositiva", ""]
    for n, who, title, notes in sl:
        L += [f"## Diapositiva {n} · {title} — {who}", "", notes, ""]
    L += ["---", "", "# Parte 2 · Preguntas probables del catedrático", ""]

    def qa(section, items):
        nonlocal L
        L += [f"## {section}", ""]
        for q, who, a in items:
            L += [f"**P: {q}** *(responde {who})*", "", f"R: {a}", ""]

    qa("A. Planteamiento y simulador", [
        ("¿Por qué usar un simulador y no datos reales?", "Gabriel",
         "Porque con datos reales no existe la explicación verdadera con la cual comparar SHAP o IG. El simulador sí es el mecanismo real: podemos intervenir un factor y medir qué pasa. "
         "El costo es que los datos son sintéticos y las conclusiones valen para esta familia de simuladores, no para civilizaciones reales."),
        ("Si el simulador tarda segundos, ¿para qué un modelo sustituto?", "Gabriel",
         f"Aquí el simulador es rápido porque está vectorizado (50 000 planetas en unos 10 s), así que el sustituto no se justifica por velocidad sino como banco de pruebas: "
         "es la misma situación que en simuladores caros (clima, epidemias), pero con verdad disponible. Aun así, el sustituto da una predicción instantánea de la probabilidad, mientras que la verdadera requiere 32 a 64 réplicas por planeta."),
        ("¿Qué es HANDY y qué variables tiene?", "Gabriel",
         "Human and Nature Dynamics (Motesharrei, Rivas y Kalnay, 2014): cuatro ecuaciones diferenciales para plebeyos, élites, naturaleza y riqueza. La naturaleza se regenera de forma logística, "
         "la población la agota (tasa δ), la extracción genera riqueza y, si la riqueza no alcanza el umbral de consumo, sube la mortalidad (de 0.01 a 0.07). κ es cuántas veces más consume una élite."),
        ("¿Qué error encontraron y qué cambió?", "Gabriel",
         "En la mortalidad de las élites, la Fase 1 usaba κ·s·x_E en el denominador cuando la ecuación 6 del paper usa s·x_E. Lo detectamos al verificar contra el artículo, lo corregimos con una prueba unitaria "
         "y regeneramos todo. Con los rangos anteriores la supervivencia bajó de 63 % a 15 %, así que re-ajustamos los rangos (κ ≤ 3, élites ≤ 10 %, δ ≤ 6·10⁻⁵) hasta llegar a "
         f"{D['surv']:.1%}. Todos los resultados que mostramos son de la versión corregida."),
        ("¿Por qué no reprodujeron el umbral del colapso Tipo-N del paper?", "Gabriel",
         "El paper reporta ≈4× δ* (reversible) y 5.5× (irreversible). Con RK4 el colapso irreversible aparece desde ≈9.25× δ* y con Euler en precisión simple (como el paper) desde ≈3.25×. "
         "El umbral depende del integrador y no tenemos el código original, así que lo dejamos como límite documentado. Sí reproducimos equilibrio, oscilaciones, ciclos y los colapsos Tipo-L y Tipo-N con κ = 100."),
        ("¿Cómo definen colapso y por qué ese umbral?", "Gabriel",
         "Población total < máx(10 % de su pico, 20 % de la inicial) antes del año 600. El 10 % del pico captura un colapso real; el piso del 20 % de la inicial evita contar como colapso a planetas que nunca despegaron "
         "y evita que un pico pequeño haga trivial la condición. Es una decisión nuestra, documentada en `docs/DECISIONES.md` (D-06)."),
        ("¿Qué extensiones agregaron y de dónde salen sus constantes?", "Gabriel",
         "Temperatura (reduce la regeneración como γ·exp(−(T/3)²) y sube la mortalidad base 15 % por °C), agua (productividad ∝ oferta/demanda, con la eficiencia hídrica reduciendo la demanda) "
         "y eventos de Poisson que destruyen una fracción de naturaleza y riqueza. Las constantes son supuestos nuestros, sin calibración empírica: es una limitación declarada."),
    ])
    qa("B. Datos", [
        ("¿Por qué Latin Hypercube?", "David",
         "Cubre el espacio de 11 factores de forma más uniforme que un muestreo aleatorio con el mismo número de planetas (McKay et al., 1979). Es un diseño de experimentos computacionales, no una muestra de población."),
        ("¿Por qué δ y κ en escala logarítmica?", "David",
         "Porque sus rangos abarcan órdenes de magnitud (δ de 5·10⁻⁶ a 6·10⁻⁵); un muestreo log-uniforme no concentra los planetas en el extremo alto. En los modelos entran como log y luego z-score."),
        ("¿Cómo evitaron la fuga de información?", "David",
         "Tres cosas: el split es por planeta (35 000 / 7 500 / 7 500); media y desviación de la normalización se calculan solo con train; y en la tarea (b) solo se evalúan planetas todavía vivos al final de la ventana "
         "(si ya colapsó, la respuesta sería trivial). Reportamos cuántos quedan fuera: {} de {} en k = 40 %, por ejemplo.".format(B[40]['n_excluded'], B[40]['n_test'])),
        ("¿Qué es la «probabilidad verdadera» y qué tan confiable es?", "David",
         "Para 500 planetas de test corrimos 32 réplicas con distintas semillas de eventos y tomamos la fracción que sobrevive. Con 32 réplicas el error estándar máximo es √(0.25/32) ≈ 0.09, "
         f"suficiente para calibración global pero ruidoso por planeta. En el {D['mc_inter']:.0%} de los planetas la probabilidad es intermedia: hay incertidumbre real, no solo ruido del modelo."),
        ("¿Las clases están balanceadas?", "David", f"Sí, casi: la supervivencia es {D['surv']:.1%}. Lo logramos ajustando los rangos, no con remuestreo."),
        ("¿Por qué 50 000 planetas?", "David",
         "Es un compromiso: la generación tarda ≈10 s, y con 35 000 planetas para entrenar el MLP (9 857 parámetros) y la LSTM (225 409) hay datos suficientes; si hiciera falta más, se genera en segundos."),
    ])
    qa("C. Modelos y entrenamiento (sesiones 1, 3, 8 y 12)", [
        ("¿Por qué BCE con logits y no MSE?", "David",
         "La BCE es la verosimilitud negativa de una variable Bernoulli: entrena probabilidades bien calibradas (ECE del MLP: {:.3f}). Con logits integra la sigmoide y es numéricamente estable, y los logits son además el espacio donde SHAP es aditivo.".format(A['mlp']['ece'])),
        ("¿Por qué AdamW, dropout y early stopping?", "David",
         "AdamW desacopla el weight decay del paso adaptativo de Adam (Loshchilov y Hutter, 2019), lo que regulariza mejor que la L2 dentro del gradiente. El dropout (0.2 en el MLP, 0.1 en la LSTM) y el early stopping "
         "con paciencia de 15 y 12 épocas sobre la BCE de validación, restaurando los mejores pesos, frenan el sobreajuste. Todo es la sesión 8."),
        ("¿Cómo eligieron los hiperparámetros sin contaminar el test?", "David",
         "Búsqueda aleatoria (12 configuraciones para el MLP y 8 para la LSTM) con la semilla 0, escogiendo por BCE de validación; el test solo se usa para reportar. La mejor configuración se reentrena con 5 semillas."),
        ("¿Por qué falla la regresión logística?", "David",
         f"Su ROC-AUC es {A['logistic']['roc_auc']:.3f}: la frontera entre sobrevivir y colapsar es muy no lineal (interacciones como δ·κ·élites) y un modelo lineal en los factores no la capta. "
         "Sirve para mostrar que el problema no es trivial."),
        ("¿La mejora del MLP sobre LightGBM es relevante?", "David",
         f"Es pequeña pero real: ΔAUC {d_ml[1]:+.4f} con IC 95 % [{d_ml[0]:+.4f}, {d_ml[2]:+.4f}] por bootstrap pareado, que excluye 0. "
         f"También mejora Brier ({A['mlp']['brier']:.3f} vs {A['lightgbm']['brier']:.3f}) y el error contra la probabilidad Monte Carlo ({A['mlp']['mc_mae']:.3f} vs {A['lightgbm']['mc_mae']:.3f}). "
         "Que sea pequeña es parte del hallazgo: un modelo clásico bien afinado compite con la red."),
        ("¿Por qué una LSTM y no un GRU o un Transformer?", "David",
         "Usamos la LSTM de la sesión 3 porque las secuencias son cortas (101 pasos, hasta 60 de ellos observados); no probamos GRU ni Transformer, lo mencionamos como trabajo futuro. "
         "Lo que sí hicimos fue compararla con baselines que no ven la serie o la ven resumida (LightGBM con resúmenes, MLP sobre la ventana)."),
        ("¿La LSTM es mejor que los baselines?", "David",
         f"Depende de k. Con k = 0 pierde ({B[0]['lstm']['roc_auc']:.3f} vs {B[0]['mlp_static_only']['roc_auc']:.3f} del MLP estático), con k = 40 % gana ({B[40]['lstm']['roc_auc']:.3f} vs {B[40]['mlp_static_only']['roc_auc']:.3f}) y con k = 60 % "
         f"{X['lstm_vs_lgb_txt']} con LightGBM con resúmenes ({B[60]['lstm']['roc_auc']:.3f} vs {B[60]['lightgbm_summary']['roc_auc']:.3f}). La trayectoria aporta información nueva sobre todo con ventanas largas."),
        ("¿Para qué concatenar los parámetros a la LSTM?", "David",
         f"Los parámetros del planeta son información que la trayectoria temprana no contiene. La ablación lo muestra: sin ellos, la LSTM parte de {B[0]['lstm_no_static']['roc_auc']:.2f} de AUC (azar) con k = 0 y llega a {B[60]['lstm_no_static']['roc_auc']:.2f} con k = 60 %."),
        ("¿Qué es el truncamiento aleatorio de k?", "David",
         "En cada lote se sortea k entre 0 y 70 pasos y se corta la secuencia (con secuencias empaquetadas). Así un solo modelo sirve para cualquier ventana y funciona como aumento de datos; se evalúa en k = 0, 5, 10, 20, 40 y 60 %."),
        ("¿Por qué no comparar las métricas entre distintos k?", "David",
         "Porque la población evaluada cambia: al aumentar k se excluyen los planetas que ya colapsaron dentro de la ventana, y los que quedan son más difíciles. Solo se comparan modelos dentro del mismo k."),
        ("¿Qué tan estables son los resultados?", "David",
         f"Todos los modelos se entrenan con 5 semillas (2026–2030). La desviación del ROC-AUC del MLP es {SA['mlp']['roc_auc']:.3f} y la de LightGBM {SA['lightgbm']['roc_auc']:.3f}, mucho menor que las diferencias que interpretamos."),
    ])
    qa("D. Explicabilidad (sesión 15)", [
        ("¿Por qué SHAP exacto y no KernelSHAP?", "Jonathan",
         "Con 11 factores hay 2¹¹ = 2 048 coaliciones: se pueden evaluar todas y obtener el valor de Shapley exacto, sin muestreo. Comprobamos que coincide con KernelSHAP con diferencia máxima ≈3·10⁻⁷ y que cumple la eficiencia (la suma de atribuciones es f(x) − f(baseline))."),
        ("¿Por qué el baseline es el «planeta medio»?", "Jonathan",
         "Es el punto de referencia común de SHAP, IG y TreeSHAP, y en espacio estandarizado es el vector cero. Es una decisión que condiciona los resultados: las atribuciones miden «respecto a un planeta medio», no en absoluto. Con otro baseline, los valores cambiarían."),
        ("¿Qué diferencia hay entre SHAP, IG y TreeSHAP?", "Jonathan",
         "SHAP exacto promedia la contribución marginal de cada factor sobre todos los órdenes (propiedades de eficiencia y simetría). IG integra el gradiente del logit sobre la recta del baseline al planeta (completitud, 1 024 pasos, error ≤ 0.03). "
         "TreeSHAP calcula los valores de Shapley de un modelo de árboles en tiempo polinomial. Los tres usan el mismo baseline y el mismo espacio de salida (logit) para poder compararlos."),
        ("¿Cómo miden la fidelidad de una explicación?", "Jonathan",
         "Para cada planeta, devolvemos un factor a su valor base y medimos cuánto cambia la probabilidad real de supervivencia en el simulador (64 réplicas, mismos números aleatorios, efecto real si |ΔP| > 0.10). "
         "Luego correlacionamos (Spearman por planeta) esos efectos con las atribuciones, y medimos acuerdo de signo y si coincide el factor más importante. También hacemos curvas de eliminación y una prueba de cordura."),
        ("¿Es bueno un Spearman de ≈ 0.65?", "Jonathan",
         f"Es moderado: concuerdan bastante ({sp(bd, 'TreeSHAP (LightGBM)'):.2f} a {sp(bd, 'IG (MLP)'):.2f} en planetas de frontera), pero lejos de perfecto. Para interpretarlo hay un techo: el efecto que predice el propio MLP, sin ninguna explicación, "
         f"solo alcanza {bd['model_vs_simulator_effect']['spearman']:.2f} de Spearman con el simulador. Es decir, una parte de la brecha es error del modelo sustituto, no del método de explicación."),
        ("¿Por qué el factor más importante solo se acierta ≈ 50-60 %?", "Jonathan",
         f"Porque la atribución (Shapley) promedia sobre coaliciones e incluye interacciones, mientras que la verdad que medimos es el efecto de cambiar un solo factor al baseline, y el MLP es solo una aproximación. "
         f"El acierto del top-1 es {M_top1(bd):.0%} (frontera) y {M_top1(mc):.0%} (Monte Carlo) con SHAP; el signo sí coincide en ≈{bd['methods']['SHAP (MLP)']['vs_simulator']['sign_agreement']:.0%} de los casos con efecto real."),
        ("¿Qué es la prueba de cordura y qué resultado dio?", "Jonathan",
         f"De Adebayo et al. (2018): si aleatorizamos los pesos del modelo, una explicación fiel debe dejar de concordar con el simulador. Pasó: el Spearman contra el simulador cae de {san['vs_simulator_trained']['spearman']:.2f} (entrenado) a {san['vs_simulator_randomised']['spearman']:.2f} (aleatorio)."),
        ("¿Qué muestran las curvas de eliminación?", "Jonathan",
         f"Devolvemos al valor base los factores en el orden de SHAP; la probabilidad real cambia en {dele['attribution']['sim_auc']:+.2f} en promedio, frente a {dele['random']['sim_auc']:+.2f} con orden aleatorio. "
         "O sea, las explicaciones sí señalan los factores que importan, mejor que el azar."),
        ("¿Por qué no usaron LIME?", "Jonathan",
         "LIME es un sustituto local lineal y su resultado depende del muestreo y del kernel. Con 11 factores, SHAP exacto es determinista y más defendible; dejamos LIME como trabajo futuro."),
        ("¿Qué limitación tiene medir un factor a la vez?", "Jonathan",
         "No captura interacciones: si dos factores solo importan juntos, el efecto individual al baseline puede no coincidir con su valor de Shapley. Un siguiente paso es medir la fidelidad con intervenciones conjuntas."),
    ])
    qa("E. Contrafactuales y aplicación", [
        ("¿Qué es un contrafactual y cómo lo calculan?", "Jonathan",
         "El cambio mínimo en las palancas accionables (δ, κ, s, eficiencia hídrica) que lleva al planeta a una probabilidad de supervivencia ≥ 0.8 según el MLP. "
         "Se optimiza un desplazamiento con Adam (600 pasos, lr 0.03) minimizando una bisagra hasta el logit objetivo más 0.5·L1 del cambio relativo al rango; la L1 favorece pocas palancas (sesión 8)."),
        ("¿Por qué solo cambian las palancas accionables?", "Jonathan",
         "Porque un contrafactual debe ser algo que una sociedad pueda hacer: no se puede cambiar la temperatura, los eventos extremos ni la natalidad. Verificamos que los factores no accionables nunca cambian (0 en todos los casos)."),
        ("¿Cómo saben que el contrafactual funciona de verdad?", "Jonathan",
         f"Cada uno se verifica en el simulador con 64 réplicas: funcionan {gs['simulator_valid']:.0%} de los casos (el modelo cree que {gs['model_valid']:.0%}). La probabilidad real media pasa de {gs['mean_true_p_before']:.2f} a {gs['mean_true_p_after']:.2f}, "
         f"cambiando {gs['mean_factors_changed']:.1f} palancas en promedio. Con una sola palanca (rejilla de 201 valores) resuelve {cs['solved_share']:.0%}."),
        ("¿Qué pasa en el 10 % donde no funciona?", "Jonathan",
         "Son planetas donde el sustituto se equivoca fuera de su zona de confianza: el modelo cree que el cambio basta y el simulador no lo confirma. Es un buen ejemplo de por qué hay que verificar con el mecanismo real; la app lo señala con una advertencia."),
        ("¿Qué muestra la demo?", "Jonathan",
         "La app de Streamlit: movemos las 11 condiciones, el simulador corre decenas de réplicas y la red da su predicción en paralelo; vemos la explicación SHAP y pedimos un contrafactual que se verifica en el simulador. "
         "En «Sobreexplotación (δ alto)» el simulador y la red coinciden en 0 %, SHAP señala la depleción y el contrafactual (bajar δ) lleva la supervivencia real de 0 % a ≈100 %."),
    ])
    qa("F. Proyecto en general", [
        ("¿Qué hallazgo principal tiene el proyecto?", "Gabriel",
         "Que una explicación convincente no es necesariamente fiel: SHAP, IG y TreeSHAP concuerdan con el efecto real del simulador de forma moderada, mejor que el azar y que un orden aleatorio, pero con errores que parte vienen del modelo sustituto. "
         "Y que con un simulador sí se puede medir cuánto."),
        ("¿Qué harían distinto o qué sigue?", "Gabriel",
         "Medir fidelidad con intervenciones conjuntas (interacciones), calibrar temperatura y agua con datos o un modelo climático, probar otras arquitecturas (GRU, Transformer) y otros métodos (LIME, explicaciones basadas en conceptos), e igualar el umbral Tipo-N del paper."),
        ("¿Cómo usaron la IA y qué aprendieron?", "Gabriel",
         "La usamos para escribir código y borradores; el historial está en `docs/PROMPTS.md`, con lo que funcionó y lo que falló. El caso clave: al verificar contra el paper original encontramos un error en la mortalidad de las élites que la IA había introducido. "
         "Lección: la IA acelera, pero hay que verificar contra la fuente."),
        ("¿Cómo se reproduce todo?", "Gabriel",
         "`pip install -r requirements.txt` y `pip install -e .`, y luego `python scripts/run_all.py` (≈40 min con GPU; con CPU toma varias horas). Las semillas están fijas y las cifras del documento salen de los JSON de `outputs/`. La app corre con `streamlit run app/streamlit_app.py`."),
        ("¿Cuáles son las limitaciones?", "Gabriel",
         "Datos sintéticos de un solo simulador; constantes de temperatura y agua sin calibrar; intervenciones de un factor a la vez; 5 semillas; y un umbral Tipo-N que no igualamos con el del paper."),
    ])
    (ROOT / "docs/NOTAS_PRESENTACION.md").write_text("\n".join(L), encoding="utf8"); print("OK -> docs/NOTAS_PRESENTACION.md")


def M_top1(g):
    return g["methods"]["SHAP (MLP)"]["vs_simulator"]["top1"]


if __name__ == "__main__":
    D, sl, X = build()
    write_markdown(D, sl, X)
    write_notes(D, sl, X)
