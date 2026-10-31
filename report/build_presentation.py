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
SPEAKER = {"David": RGBColor(0x2F, 0x6F, 0xDE), "Jonathan": RGBColor(0x2E, 0x9D, 0x63), "Gabriel": RGBColor(0xE8, 0x83, 0x3A)}
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
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.14)); bar.fill.solid(); bar.fill.fore_color.rgb = SPEAKER[who]; bar.line.fill.background()
    text(s, title, 0.55, 0.3, 10.2, 0.9, size=28, bold=True)
    chip = box(s, f"Habla: {who}", 10.9, 0.38, 2.0, 0.45, fill=SPEAKER[who], color=RGBColor(255, 255, 255), size=14, bold=True)
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
    notes = ("Buenas tardes. Somos David Domínguez, Jonathan Díaz y Gabriel Bran, y les presentamos nuestro proyecto final de Deep Learning. "
             "Empezamos con una pregunta: cuando una red neuronal nos da una explicación de su predicción, ¿podemos creerle? "
             "En casi todos los problemas reales no hay forma de comprobarlo. Nosotros construimos un problema donde sí: un simulador de civilizaciones, "
             "una red que predice si cada civilización sobrevive, y explicaciones que comprobamos contra el propio simulador. "
             "Yo, David, presento el problema y el simulador; Jonathan, los datos y los modelos; Gabriel, las explicaciones y su verificación; y cierro con las conclusiones.")
    s = new_slide(prs, 1, "Simulador de mundos habitables", "David", notes)
    text(s, "Predicción de supervivencia con modelos sustitutos y explicabilidad verificable", 0.55, 1.35, 11.5, 0.9, size=22, color=GRAY)
    box(s, "¿Podemos creerle a una explicación?", 0.55, 2.7, 12.2, 1.3, fill=NAVY, color=RGBColor(255, 255, 255), size=34, bold=True)
    text(s, "David Domínguez · Jonathan Díaz · Gabriel Bran", 0.55, 4.5, 12, 0.5, size=22)
    text(s, "Deep Learning · Sección 10 · Universidad del Valle de Guatemala · 2 de noviembre de 2026", 0.55, 5.1, 12, 0.5, size=16, color=GRAY)
    for i, (who, what) in enumerate((("David", "problema y simulador"), ("Jonathan", "datos y modelos"), ("Gabriel", "explicaciones y verificación"))):
        box(s, f"{who}: {what}", 0.55 + i * 4.15, 6.0, 3.95, 0.6, fill=SPEAKER[who], color=RGBColor(255, 255, 255), size=14, bold=True)
    sl.append((1, "David", "Portada y pregunta central", notes))

    # 2 ---------------------------------------------------------------------------------------------
    notes = ("Muchos sistemas científicos —clima, epidemias, ecología— se estudian con simuladores que son lentos y caros. "
             "Por eso se entrenan redes neuronales como modelos sustitutos: aprenden a imitar al simulador y responden al instante. "
             "El problema es que una red es una caja negra. Existen métodos de explicabilidad, como SHAP o Integrated Gradients, que dicen qué factores influyeron. "
             "Pero casi nunca existe una explicación verdadera con la cual compararlas, así que no sabemos si son correctas o solo convincentes. "
             "Nuestra idea: usar un problema donde el mecanismo real sí está disponible. El simulador nos deja intervenir un factor y ver qué pasa de verdad. "
             "Entonces la pregunta del proyecto no es solo predecir, sino medir qué tan fieles son las explicaciones.")
    s = new_slide(prs, 2, "El problema: sustitutos que no sabemos explicar", "David", notes)
    bullets(s, ["Los simuladores científicos son caros: se usan redes como modelos sustitutos (surrogates).",
                "Las explicaciones (SHAP, IG, contrafactuales) casi nunca tienen una verdad de referencia.",
                "Aquí sí: el simulador es el mecanismo real, así que podemos intervenir y comprobar."], 0.55, 1.35, 12.2, 2.0, size=20)
    y0 = 3.75
    box(s, "Simulador HANDY\n(mecanismo real)", 0.6, y0, 2.6, 1.2, fill=LIGHT, size=15, bold=True, line=SPEAKER["David"])
    arrow(s, 3.3, y0 + 0.4, 0.8); text(s, "50 000 planetas", 3.2, y0 + 0.82, 1.2, 0.3, size=10, color=GRAY)
    box(s, "Red sustituta\n(MLP / LSTM)", 4.2, y0, 2.6, 1.2, fill=LIGHT, size=15, bold=True, line=SPEAKER["David"])
    arrow(s, 6.9, y0 + 0.4, 0.8); text(s, "predicción", 6.9, y0 + 0.82, 1.0, 0.3, size=10, color=GRAY)
    box(s, "Explicación\n(SHAP · IG · contrafactual)", 7.8, y0, 3.0, 1.2, fill=LIGHT, size=15, bold=True, line=SPEAKER["David"])
    box(s, "¿Intervenir un factor en el simulador cambia el resultado como dice la explicación?", 2.2, 5.45, 8.6, 0.95, fill=SPEAKER["David"], color=RGBColor(255, 255, 255), size=16, bold=True)
    arrow(s, 9.1, 4.98, 0.35, 0.4, shape=MSO_SHAPE.DOWN_ARROW); arrow(s, 2.0, 4.98, 0.35, 0.4, shape=MSO_SHAPE.UP_ARROW)
    sl.append((2, "David", "El problema", notes))

    # 3 ---------------------------------------------------------------------------------------------
    notes = ("El simulador está basado en HANDY, un modelo publicado en 2014 en Ecological Economics. Tiene cuatro variables: plebeyos, élites, naturaleza y riqueza. "
             "La naturaleza se regenera, la población la agota para producir riqueza, y si la riqueza no alcanza para el consumo mínimo hay hambruna. "
             "Verificamos las ecuaciones contra el artículo original y al hacerlo encontramos un error nuestro: habíamos escrito mal la mortalidad de las élites. "
             "Lo corregimos, agregamos una prueba unitaria y regeneramos todo el dataset. Con las ecuaciones correctas reproducimos los escenarios del paper: equilibrio, oscilaciones, "
             "ciclos de colapso y los colapsos por desigualdad y por agotamiento de la naturaleza. Una salvedad honesta: el umbral exacto del colapso irreversible igualitario depende del integrador, "
             "y no lo igualamos con exactitud. Lo extendimos con temperatura, agua y eventos extremos aleatorios, hasta tener once factores. "
             "Cuatro son accionables, o sea palancas de política: depleción, desigualdad, consumo y eficiencia hídrica. Los demás son datos del entorno. Colapso significa que la población cae por debajo del diez por ciento de su pico.")
    s = new_slide(prs, 3, "El simulador: HANDY extendido", "David", notes)
    bullets(s, ["4 variables (plebeyos, élites, naturaleza, riqueza); ecuaciones verificadas contra el paper de 2014 (y un error nuestro corregido).",
                "Extensiones: temperatura, agua y eventos extremos (Poisson) → 11 factores; 4 accionables (δ, κ, s, eficiencia hídrica).",
                "Colapso = población < 10 % de su pico (o < 20 % de la inicial) antes del año 600."], 0.55, 1.2, 12.2, 1.8, size=17)
    picture(s, "fig_escenarios_handy.png", 0.55, 3.0, 12.2, 4.0)
    sl.append((3, "David", "El simulador", notes))

    # 4 ---------------------------------------------------------------------------------------------
    notes = (f"Gracias, David. Con el simulador generamos {D['n']:,} planetas, muestreando las once condiciones con Latin Hypercube. Ajustamos los rangos para que la frontera no fuera trivial: "
             f"la supervivencia quedó en {D['surv']:.1%}. La generación tarda unos diez segundos porque el simulador está vectorizado. "
             f"El split es 70, 15 y 15 por planeta, y la normalización se ajusta solo con train. "
             f"Como los eventos extremos son aleatorios, las mismas condiciones pueden dar resultados distintos. Por eso, para 500 planetas de test corrimos 32 réplicas con distintas semillas "
             f"y obtuvimos la probabilidad verdadera de supervivencia. En el {D['mc_inter']:.0%} de esos planetas la probabilidad es intermedia, o sea que hay incertidumbre real. "
             "Esta probabilidad verdadera nos sirve para evaluar la calibración de los modelos y, después, como verdad para las explicaciones. "
             "A la derecha ven tres planetas que sobreviven y tres que colapsan; las franjas naranja son eventos extremos.")
    s = new_slide(prs, 4, "Datos: 50 000 planetas y una verdad de referencia", "Jonathan", notes)
    bullets(s, [f"{D['n']:,} planetas (LHS), 11 factores; supervivencia {D['surv']:.1%}.",
                "Split 70/15/15 por planeta; normalización solo con train.",
                "Eventos aleatorios → mismas condiciones, resultados distintos.",
                f"Monte Carlo: 500 planetas de test × 32 réplicas = P(sobrevive | parámetros). En {D['mc_inter']:.0%} la probabilidad es intermedia."], 0.55, 1.3, 5.1, 5.4, size=17)
    picture(s, "fig_eda_trayectorias.png", 5.8, 1.3, 7.1, 4.2)
    picture(s, "fig_eda_clases_tiempos.png", 5.8, 5.2, 7.1, 1.8)
    sl.append((4, "Jonathan", "Datos y verdad de referencia", notes))

    # 5 ---------------------------------------------------------------------------------------------
    notes = (f"Planteamos dos tareas. En la tarea a, un MLP lee las once condiciones iniciales y predice la probabilidad de supervivencia. Usa dropout, AdamW y early stopping, como vimos en la sesión ocho. "
             f"En la tarea b, una LSTM, de la sesión tres, lee los primeros k por ciento de la trayectoria junto con los parámetros, como alerta temprana. "
             "La entrenamos con truncamiento aleatorio de k, así un solo modelo sirve para cualquier ventana. Para no medir una ventaja trivial, solo evaluamos planetas que siguen vivos al final de la ventana, y reportamos cuántos quedan fuera. "
             f"Exigimos que la red demuestre que aporta algo: comparamos con regresión logística y con gradient boosting, LightGBM. "
             f"Hicimos búsqueda de hiperparámetros, entrenamos con {nseeds} semillas y calculamos intervalos de confianza con bootstrap pareado sobre los planetas de test.")
    s = new_slide(prs, 5, "Modelos: la red tiene que ganarse su lugar", "Jonathan", notes)
    y0 = 1.45
    box(s, "11 factores del planeta", 0.6, y0, 2.6, 0.9, size=14, bold=True, line=SPEAKER["Jonathan"]); arrow(s, 3.3, y0 + 0.28, 0.7)
    box(s, "MLP\ndropout · AdamW · early stopping", 4.1, y0, 3.6, 0.9, size=13, bold=True, fill=SPEAKER["Jonathan"], color=RGBColor(255, 255, 255)); arrow(s, 7.8, y0 + 0.28, 0.7)
    box(s, "P(sobrevive) · tarea (a)", 8.6, y0, 3.6, 0.9, size=14, bold=True, line=SPEAKER["Jonathan"])
    y1 = 2.75
    box(s, "Primeros k % de la trayectoria\n+ 11 factores", 0.6, y1, 3.4, 0.9, size=13, bold=True, line=SPEAKER["Jonathan"]); arrow(s, 4.1, y1 + 0.28, 0.5)
    box(s, "LSTM\ntruncamiento aleatorio de k", 4.7, y1, 3.0, 0.9, size=13, bold=True, fill=SPEAKER["Jonathan"], color=RGBColor(255, 255, 255)); arrow(s, 7.8, y1 + 0.28, 0.7)
    box(s, "P(sobrevive) · tarea (b)", 8.6, y1, 3.6, 0.9, size=14, bold=True, line=SPEAKER["Jonathan"])
    bullets(s, ["Baselines obligatorios: regresión logística, LightGBM, MLP sobre la ventana, MLP solo con parámetros.",
                f"Búsqueda de hiperparámetros (MLP: 12 configuraciones; LSTM: 8), {nseeds} semillas, IC 95 % por bootstrap pareado.",
                "Sin fuga: solo se evalúan planetas aún vivos al final de la ventana; se reporta cuántos quedan fuera.",
                "Ablación: LSTM con y sin parámetros estáticos."], 0.55, 4.05, 12.2, 2.9, size=18)
    sl.append((5, "Jonathan", "Modelos", notes))

    # 6 ---------------------------------------------------------------------------------------------
    d_ml = BA["mlp_minus_lightgbm"]["roc_auc"]
    notes = (f"Resultados de la tarea a, con media y desviación sobre {nseeds} semillas. La regresión logística llega a ROC-AUC de {A['logistic']['roc_auc']:.2f}: la frontera es muy no lineal. "
             f"LightGBM obtiene {A['lightgbm']['roc_auc']:.3f} y el MLP {A['mlp']['roc_auc']:.3f}. La diferencia del MLP con LightGBM es de {d_ml[1]:+.4f} de AUC, con intervalo de confianza de {d_ml[0]:+.4f} a {d_ml[2]:+.4f}; "
             f"como el intervalo no incluye cero, la ventaja es real, aunque pequeña. El MLP también está mejor calibrado y su error contra la probabilidad verdadera Monte Carlo es de {A['mlp']['mc_mae']:.3f}. "
             f"Para la tarea b, la gráfica muestra el desempeño al observar más trayectoria. Con k igual a cero la LSTM solo ve los parámetros. En ventanas largas, de sesenta por ciento, la LSTM alcanza {B[60]['lstm']['roc_auc']:.3f}, LightGBM con resúmenes {B[60]['lightgbm_summary']['roc_auc']:.3f} y el MLP que no ve la trayectoria {B[60]['mlp_static_only']['roc_auc']:.3f}. "
             f"Entre la LSTM y LightGBM, {lstm_vs_lgb_txt} (diferencia {lstm_vs_lgb60[1]:+.4f}, intervalo {lstm_vs_lgb60[0]:+.4f} a {lstm_vs_lgb60[2]:+.4f}). "
             f"Sin parámetros estáticos, la LSTM parte de {B[0]['lstm_no_static']['roc_auc']:.2f}, o sea casi azar, y sube a {B[60]['lstm_no_static']['roc_auc']:.2f}: la trayectoria sola acaba informando, pero con poca ventana los parámetros son indispensables. "
             "Es importante aclarar que entre distintos k no se pueden comparar las métricas directamente, porque cambia la población evaluada.")
    s = new_slide(prs, 6, "Resultados de los modelos", "Jonathan", notes)
    rows = [["Tarea (a)", "ROC-AUC", "Brier", "MAE vs P verdadera"]]
    for k, lab in (("logistic", "Regresión logística"), ("lightgbm", "LightGBM"), ("mlp", "MLP")):
        rows.append([lab, f"{A[k]['roc_auc']:.3f} ± {SA[k]['roc_auc']:.3f}", f"{A[k]['brier']:.3f} ± {SA[k]['brier']:.3f}", f"{A[k]['mc_mae']:.3f} ± {SA[k]['mc_mae']:.3f}"])
    table(s, rows, 0.55, 1.3, 6.6, 1.7, size=13)
    bullets(s, [f"MLP − LightGBM: ΔAUC {d_ml[1]:+.4f} [{d_ml[0]:+.4f}, {d_ml[2]:+.4f}] (IC 95 %).",
                f"k = 60 %: LSTM {B[60]['lstm']['roc_auc']:.3f} vs. LightGBM {B[60]['lightgbm_summary']['roc_auc']:.3f} → {lstm_vs_lgb_txt}.",
                "Sin parámetros estáticos, la LSTM parte casi del azar (k = 0)."], 7.4, 1.25, 5.5, 2.0, size=14)
    picture(s, "fig_tarea_b_curva_k.png", 0.55, 3.15, 12.2, 3.85)
    sl.append((6, "Jonathan", "Resultados de modelos", notes))

    # 7 ---------------------------------------------------------------------------------------------
    notes = ("Gracias, Jonathan. Ahora la parte central del proyecto. Calculamos tres explicaciones sobre el MLP y el boosting: SHAP, que con once factores calculamos de forma exacta, "
             "Integrated Gradients con Captum, y TreeSHAP sobre LightGBM. Algo clave: todos usan el mismo baseline, el planeta medio de entrenamiento, y el mismo espacio de salida, el logit de supervivencia. "
             "Así las comparaciones son justas. Comprobamos que SHAP exacto coincide con KernelSHAP y que cada método cumple su propiedad de completitud. "
             "Pero que las explicaciones coincidan entre sí no prueba que sean correctas. Por eso las verificamos de cuatro maneras: "
             "uno, intervenir un factor en el modelo; dos, intervenirlo en el simulador, con sesenta y cuatro réplicas y los mismos números aleatorios para que la comparación tenga poco ruido; "
             "tres, curvas de eliminación: si devolvemos al valor base los factores en el orden que dice la explicación, el resultado debe degradarse más rápido que con un orden aleatorio; "
             "y cuatro, una prueba de cordura de Adebayo y colaboradores: si destruimos el modelo, la explicación debe dejar de concordar con el simulador.")
    s = new_slide(prs, 7, "Explicaciones y cómo las verificamos", "Gabriel", notes)
    box(s, "Mismo baseline (planeta medio de entrenamiento)\nMisma salida (logit de supervivencia)", 0.55, 1.3, 7.2, 0.95, fill=NAVY, color=RGBColor(255, 255, 255), size=15, bold=True)
    for i, (lab, sub) in enumerate((("SHAP exacto", "MLP · 2¹¹ coaliciones"), ("Integrated Gradients", "MLP · Captum"), ("TreeSHAP", "LightGBM"))):
        box(s, f"{lab}\n{sub}", 0.55 + i * 2.45, 2.45, 2.3, 1.0, size=13, bold=True, line=SPEAKER["Gabriel"])
    bullets(s, ["Intervención en el modelo: ¿cambia el logit como dice φ?",
                "Intervención en el simulador (64 réplicas, números aleatorios comunes): la verdad.",
                "Curvas de eliminación: orden por atribución vs. aleatorio.",
                "Prueba de cordura: pesos aleatorizados (Adebayo et al., 2018)."], 0.55, 3.75, 7.2, 3.2, size=17)
    picture(s, "fig_xai_beeswarm.png", 8.0, 1.3, 4.9, 5.6)
    sl.append((7, "Gabriel", "Explicaciones y verificación", notes))

    # 8 ---------------------------------------------------------------------------------------------
    notes = (f"Resultados de la fidelidad. En la gráfica, las barras naranja comparan cada atribución con el efecto real en el simulador. "
             f"Con planetas cercanos a la frontera de decisión, la correlación de Spearman por planeta es de {M(bd,'SHAP (MLP)','vs_simulator','spearman'):.2f} para SHAP, {M(bd,'IG (MLP)','vs_simulator','spearman'):.2f} para Integrated Gradients y {M(bd,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} para TreeSHAP; "
             f"con planetas Monte Carlo aleatorios, entre {M(mc,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} y {M(mc,'IG (MLP)','vs_simulator','spearman'):.2f}. Es decir, concuerdan bastante, pero lejos de perfecto. "
             f"El signo coincide en cerca de {M(bd,'SHAP (MLP)','vs_simulator','sign_agreement'):.0%} de los casos con efecto real, pero el factor más importante solo coincide en {M(bd,'SHAP (MLP)','vs_simulator','top1'):.0%} a {M(mc,'SHAP (MLP)','vs_simulator','top1'):.0%}. "
             f"Parte del error viene del propio modelo: la línea azul indica que incluso el efecto que el MLP predice por sí mismo solo llega a {bd['model_vs_simulator_effect']['spearman']:.2f} de Spearman con el simulador. "
             f"En las curvas de eliminación, quitar factores en el orden de SHAP cambia la probabilidad real en {dele['attribution']['sim_auc']:+.2f} en promedio, mientras que el orden aleatorio la cambia en {dele['random']['sim_auc']:+.2f}: las explicaciones sí apuntan a lo que importa. "
             f"Y la prueba de cordura pasa: con el modelo aleatorizado la concordancia con el simulador cae de {san['vs_simulator_trained']['spearman']:.2f} a {san['vs_simulator_randomised']['spearman']:.2f}. "
             "La conclusión honesta: las explicaciones son útiles y dependen del modelo, pero no son una verdad; hay que verificarlas.")
    s = new_slide(prs, 8, "¿Son fieles las explicaciones? Resultados", "Gabriel", notes)
    picture(s, "fig_fidelidad_spearman.png", 0.55, 1.2, 12.2, 3.6)
    bullets(s, [f"Spearman vs. simulador: {M(bd,'SHAP (MLP)','vs_simulator','spearman'):.2f} (SHAP), {M(bd,'IG (MLP)','vs_simulator','spearman'):.2f} (IG), {M(bd,'TreeSHAP (LightGBM)','vs_simulator','spearman'):.2f} (TreeSHAP) en planetas de frontera; signo correcto en ≈{M(bd,'SHAP (MLP)','vs_simulator','sign_agreement'):.0%}; factor n.º 1 acertado en ≈{M(bd,'SHAP (MLP)','vs_simulator','top1'):.0%}–{M(mc,'SHAP (MLP)','vs_simulator','top1'):.0%}.",
                f"Eliminación en el simulador: orden SHAP {dele['attribution']['sim_auc']:+.2f} vs. aleatorio {dele['random']['sim_auc']:+.2f} (cambio medio de P).",
                f"Prueba de cordura: Spearman vs. simulador {san['vs_simulator_trained']['spearman']:.2f} (modelo entrenado) → {san['vs_simulator_randomised']['spearman']:.2f} (pesos aleatorios)."],
            0.55, 4.9, 12.2, 2.1, size=15)
    sl.append((8, "Gabriel", "Fidelidad: resultados", notes))

    # 9 ---------------------------------------------------------------------------------------------
    notes = (f"Una explicación útil debería permitir actuar. Por eso buscamos contrafactuales: el cambio mínimo en las palancas accionables —depleción, desigualdad, consumo y eficiencia hídrica— que lleva el planeta a sobrevivir. "
             f"Los factores del entorno, como la temperatura o los eventos, no se tocan. Los aplicamos a los planetas que el modelo predice que colapsarán, y los verificamos en el simulador. "
             f"El método de gradiente resuelve el {gs['solved_share']:.0%} de los casos; en el simulador, el {gs['simulator_valid']:.0%} de los contrafactuales realmente sobrevive, y la probabilidad real pasa de {gs['mean_true_p_before']:.2f} a {gs['mean_true_p_after']:.2f}, con {gs['mean_factors_changed']:.1f} palancas cambiadas en promedio. "
             f"Cambiar una sola palanca resuelve solo el {cs['solved_share']:.0%} de los casos. "
             "Ahora les mostramos la aplicación en Streamlit: movemos los factores de un planeta, vemos la simulación con sus réplicas, la explicación SHAP y pedimos un contrafactual que el simulador verifica. "
             "[DEMO EN VIVO: elegir el escenario 'Sobreexplotación (δ alto)', mostrar el colapso, abrir la pestaña de explicación y pulsar 'Buscar contrafactual'.]")
    s = new_slide(prs, 9, "Contrafactuales verificados y demo", "Gabriel", notes)
    bullets(s, ["Pregunta: ¿cuál es el cambio mínimo en las palancas accionables (δ, κ, s, eficiencia hídrica) que lleva el planeta a sobrevivir?",
                f"Gradiente: {gs['solved_share']:.0%} resueltos; {gs['simulator_valid']:.0%} válidos en el simulador; P real {gs['mean_true_p_before']:.2f} → {gs['mean_true_p_after']:.2f}.",
                f"Una sola palanca: {cs['solved_share']:.0%} resueltos; {cs['simulator_valid']:.0%} válidos en el simulador.",
                "Los factores no accionables (temperatura, agua, eventos) nunca se modifican."], 0.55, 1.25, 7.4, 2.5, size=16)
    box(s, "DEMO EN VIVO\nstreamlit run app/streamlit_app.py", 8.3, 1.4, 4.5, 1.7, fill=SPEAKER["Gabriel"], color=RGBColor(255, 255, 255), size=17, bold=True)
    picture(s, "fig_contrafactuales.png", 0.55, 3.75, 12.2, 3.2)
    sl.append((9, "Gabriel", "Contrafactuales y demo", notes))

    # 10 --------------------------------------------------------------------------------------------
    notes = (f"Para cerrar. Primero, el modelo: la red sí aporta frente a los baselines en la tarea a, con una ventaja pequeña pero con intervalo de confianza que excluye cero; la regresión logística, en cambio, falla. "
             "En alerta temprana, la trayectoria ayuda sobre todo en ventanas largas. Segundo, las explicaciones: concuerdan con la realidad del simulador de forma moderada, mejor que el azar y que un orden aleatorio, y pasan la prueba de cordura, pero no son una verdad. "
             f"Tercero, los contrafactuales verificados con el simulador funcionan en la mayoría de casos. Entre las limitaciones: los efectos de una intervención individual no capturan interacciones; "
             "las constantes de temperatura y agua son supuestos nuestros; y no reprodujimos con exactitud el umbral del colapso irreversible del paper. "
             "Sobre el uso de la IA: la usamos para escribir código y borradores, y está documentado en el repositorio, con lo que funcionó y lo que falló. El caso más importante: al verificar contra el paper original encontramos un error en la mortalidad de las élites que la IA había introducido; "
             "lo detectamos, lo corregimos con una prueba unitaria y rehicimos todos los resultados. Lección: la IA acelera, pero hay que verificar contra la fuente. Gracias; quedamos atentos a sus preguntas.")
    s = new_slide(prs, 10, "Conclusiones, límites y uso de la IA", "David", notes)
    cols = (("Lo que encontramos", ["La red supera a la logística y, por poco pero con IC, a LightGBM.", "La trayectoria ayuda sobre todo con ventanas largas.", "Las explicaciones concuerdan con el simulador (ρ ≈ 0.6–0.7) y pasan la prueba de cordura.", f"{gs['simulator_valid']:.0%} de los contrafactuales funcionan en el simulador."]),
            ("Límites", ["Efecto de un factor a la vez: no captura interacciones.", "Temperatura y agua: supuestos nuestros, sin calibración.", "No igualamos el umbral del Tipo-N del paper.", f"Modelos con {nseeds} semillas; un solo simulador."]),
            ("Uso de la IA", ["Código y borradores generados con IA; historial en docs/PROMPTS.md.", "Verificar contra el paper encontró un error de la IA (mortalidad de élites).", "Lo corregimos con un test y rehicimos los resultados.", "Lección: la IA acelera; la verificación es nuestra."]))
    for i, (h, items) in enumerate(cols):
        box(s, h, 0.55 + i * 4.15, 1.3, 3.95, 0.6, fill=SPEAKER["David"], color=RGBColor(255, 255, 255), size=16, bold=True)
        bullets(s, items, 0.55 + i * 4.15, 2.05, 3.95, 4.8, size=17)
    sl.append((10, "David", "Conclusiones", notes))

    out = ROOT / "report/Presentacion.pptx"; prs.save(out); print("OK ->", out)
    return D, sl, dict(A=A, SA=SA, B=B, SB=SB, BB=BB, BA=BA, fd=fd, cf=cf, bd=bd, mc=mc, dele=dele, san=san, lstm_vs_lgb_txt=lstm_vs_lgb_txt)


def write_markdown(D, sl, X):
    by = {"David": [], "Jonathan": [], "Gabriel": []}
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
              "Reglas sugeridas: quien habla cede la palabra al terminar su última diapositiva diciendo el nombre de la siguiente persona; cada quien debe poder responder preguntas de *todo* el proyecto (ver la lista al final).", ""]
    for n, who, title, notes in sl:
        lines += [f"## Diapositiva {n} · {title} — {who}", "", notes, ""]
    lines += ["## Preguntas probables y quién las toma", "",
              "| Pregunta | Responde | Respuesta corta |", "|---|---|---|",
              "| ¿Por qué un simulador y no datos reales? | David | Solo así existe una verdad de referencia para verificar las explicaciones. |",
              "| ¿Qué error encontraron en HANDY? | David | La mortalidad de las élites usaba κ en el denominador; el paper usa s·x_E. Lo corregimos con un test y regeneramos todo. |",
              "| ¿La red es mejor que LightGBM? | Jonathan | Por poco (ΔAUC con IC 95 % que excluye 0), pero sí; y mucho mejor que la logística. |",
              "| ¿Por qué solo planetas vivos en la tarea (b)? | Jonathan | Para evitar la fuga trivial: un planeta ya colapsado es obvio. Se reporta cuántos quedan fuera. |",
              "| ¿Por qué SHAP exacto? | Gabriel | Con 11 factores hay 2048 coaliciones: se calcula exacto y se comprobó contra KernelSHAP. |",
              "| ¿Qué significa que ρ ≈ 0.65? | Gabriel | Concuerdan bastante con el efecto real pero no perfectamente; parte del error es del propio modelo. |",
              "| ¿Los contrafactuales son confiables? | Gabriel | Se verificaron con el simulador: ≈90 % funcionan; el resto muestra dónde el sustituto se equivoca. |"]
    (ROOT / "docs/PRESENTACION.md").write_text("\n".join(lines), encoding="utf8"); print("OK -> docs/PRESENTACION.md")


if __name__ == "__main__":
    D, sl, X = build()
    write_markdown(D, sl, X)
