import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, time, date
import base64
import json
from pathlib import Path
import joblib
import numpy as np

### now = datetime(2026, 2, 6, 8, 35)   # 06 a las 00:35 am
now = datetime.utcnow() - timedelta(hours=5)
hora = now.hour

if 7 <= hora < 19:
    turno = "T/D"
    fecha_operacion = now
else:
    turno = "T/N"
    if hora < 7:
        fecha_operacion = now - timedelta(days=1)
    else:
        fecha_operacion = now

fecha_str = fecha_operacion.strftime("%d-%m")

UMBRAL_MIN = 240  # 4h desde inicio turno -> 10:30 T/D / 22:30 T/N
CORTE_OFICIAL_MIN = 330

if turno == "T/D":
    inicio_turno = datetime.combine(fecha_operacion.date(), time(6, 30))
else:
    inicio_turno = datetime.combine(fecha_operacion.date(), time(18, 30))

minutos_transcurridos = int((now - inicio_turno).total_seconds() / 60)
if minutos_transcurridos < 0:
    minutos_transcurridos += 24 * 60  # turno noche cruza medianoche

def nivel_confianza(minutos):
    if minutos < 180:
        return "baja", "#FF0000", "Oculta"
    if minutos < UMBRAL_MIN:
        return "baja", "#FF0000", "Baja (preliminar)"
    if minutos < CORTE_OFICIAL_MIN:
        return "media", "#FFC000", "Media"
    return "alta", "#00B050", "Alta"

nivel, color_conf, texto_conf = nivel_confianza(minutos_transcurridos)
mostrar_proyeccion = minutos_transcurridos >= UMBRAL_MIN

titulo = (
    f"<b style='color:black; font-size:30px';font-size:10px;>"
    f"ESTADO DE EQUIPOS - OPERACIÓN {fecha_str} {turno}"
    f"</b>"
)

st.set_page_config(page_title="Gantt por Equipo", layout="wide")

def load_image_base64(image_path):
    with open(image_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    
logo_owm = load_image_base64("assets/logo_owm.png")
logo_mmg = load_image_base64("assets/logo_mmg.png")

# --- Carga modelos ML (HistGradientBoosting entrenado con detalle_limpio.csv) ---
MODELOS_DIR = Path("models")
modelo_corte = modelo_fin = None
columnas_corte = columnas_fin = None
try:
    if (MODELOS_DIR / "modelo_corte.pkl").exists():
        modelo_corte = joblib.load(MODELOS_DIR / "modelo_corte.pkl")
        columnas_corte = json.loads((MODELOS_DIR / "columnas_corte.json").read_text())
    if (MODELOS_DIR / "modelo_fin.pkl").exists():
        modelo_fin = joblib.load(MODELOS_DIR / "modelo_fin.pkl")
        columnas_fin = json.loads((MODELOS_DIR / "columnas_fin.json").read_text())
except Exception as e:
    st.warning(f"No se pudieron cargar modelos ML: {e}")

st.markdown("""
<style>
/* Fondo general */
.main, .stApp {
    background-color: #FFFFFF !important;
            
}

/* Fondo de los contenedores */
.block-container {
    background-color: #FFFFFF !important;
}

/* Ocultar la barra superior e inferior de Streamlit */
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Banner confianza proyeccion desde 240min (10:30 T/D, 22:30 T/N)
hora_mostrar = f"{minutos_transcurridos//60:02d}:{minutos_transcurridos%60:02d} desde inicio"
st.markdown(
    f"<div style='text-align:center; padding:6px; border-radius:8px; background-color:{color_conf}22; border:1px solid {color_conf}; margin-bottom:10px;'>"
    f"<span style='color:{color_conf}; font-weight:700;'>Proyeccion ML: {texto_conf}</span>"
    f"<span style='color:black;'> — {hora_mostrar} | Umbral 240min (10:30/22:30) · Corte oficial 330min (12:00/00:00) · R2 FIN 0.80→0.87 · "
    f"Modelo: HistGradientBoosting {'✓' if modelo_fin is not None else '✗'}</span>"
    f"</div>",
    unsafe_allow_html=True
)

file = st.file_uploader(" ", type=["xlsx"])

def convertir_hora(x):
    if isinstance(x, time):
        return datetime.combine(date.today(), x)
    try:
        return pd.to_datetime(x)
    except:
        return None
    
# =====================================================
# FORMULAS DE METRAJE - recalibradas 31-08-2026 con Odoo detalle_limpio (34,390 taladros 15-75 m/h)
# a = ROP_mean * factor (0.95 DTH / 0.80 RTR), b conservado. Ver pipeline/rop_por_equipo.py y pipeline/comparar_bitacora.py
# ROP_mean Odoo: TD012 27.15, TD030 32.17, TD072 37.83, TD073 36.44, TD074 31.86, TD076 29.03, TD077 33.13, TD079 37.75, TD080 32.98, TD081 36.98, TD091 31.71, TD092 30.63
# Antes vs bitacora 31-08 real ROP 36/31/22/24/45/46 mostraba desvio -60% en TD080/081 (a 17.96 vs 45)
# =====================================================
FORMULAS_METRAJE = {
    "TD011": {"a": 23.67*0.95, "b": 9.71},
    "TD012": {"a": 27.15*0.95, "b": 6.81},
    "TD030": {"a": 32.17*0.95, "b": 1.59},
    "TD031": {"a": 29.96*0.95, "b": -0.31},
    "TD072": {"a": 37.83*0.95, "b": 1.19},
    "TD073": {"a": 36.44*0.95, "b": 1.93},
    "TD074": {"a": 31.86*0.95, "b": 2.24},
    "TD076": {"a": 29.03*0.95, "b": 3.30},
    "TD077": {"a": 33.13*0.95, "b": 8.14},
    "TD079": {"a": 37.75*0.95, "b": 1.07},
    "TD080": {"a": 32.98*0.80, "b": 31.79},
    "TD081": {"a": 36.98*0.80, "b": 20.37},
    "TD082": {"a": 32.98*0.80, "b": 20.37},
    "TD083": {"a": 32.98*0.80, "b": 20.37},
    "TD091": {"a": 31.71*0.80, "b": 31.79},
    "TD092": {"a": 30.63*0.80, "b": 20.37},
}

CONFIG_EQUIPOS = {
    "TD011": {"orden": 1,  "grupo": "DTH"},
    "TD012": {"orden": 2,  "grupo": "DTH"},
    "TD030": {"orden": 3,  "grupo": "DTH"},
    "TD031": {"orden": 4,  "grupo": "DTH"},
    "TD073": {"orden": 5,  "grupo": "DTH"},
    "TD074": {"orden": 6,  "grupo": "DTH"},
    "TD076": {"orden": 7,  "grupo": "DTH"},
    "TD077": {"orden": 8,  "grupo": "DTH"},
    "TD079": {"orden": 9,  "grupo": "DTH"},
    "TD081": {"orden":10,  "grupo": "DTH"},
    
    "TD091": {"orden":11,  "grupo": "RTR"},
    "TD092": {"orden":12,  "grupo": "RTR"},
    
    "TD072": {"orden":13,  "grupo": "FUERA"},
    "TD080": {"orden":14,  "grupo": "FUERA"},
    "TD082": {"orden":15,  "grupo": "FUERA"},
    "TD083": {"orden":16,  "grupo": "FUERA"},

}

if file:
    df = pd.read_excel(file)

    df["Orden"] = df["Equipo"].map(lambda x: CONFIG_EQUIPOS[x]["orden"])
    df["Grupo"] = df["Equipo"].map(lambda x: CONFIG_EQUIPOS[x]["grupo"])

    columnas = ["Equipo", "Hora Inicio", "Hora Fin", "Descripcion", "Estado"]
    if not all(col in df.columns for col in columnas):
        st.error("El archivo debe contener: Equipo, Hora Inicio, Hora Fin, Descripcion, Estado")
        st.stop()

    df["Hora Inicio"] = df["Hora Inicio"].apply(convertir_hora)
    df["Hora Fin"] = df["Hora Fin"].apply(convertir_hora)
    df["Duracion"] = df["Hora Fin"] - df["Hora Inicio"]

    df["DuracionTexto"] = df["Duracion"].apply(
        lambda x: (
            f"{int(x.total_seconds()//3600):02d}:{int((x.total_seconds()%3600)//60):02d}"
            if x.total_seconds() >= 1800 else ""
        )
    )

    df = df.sort_values(["Orden", "Hora Inicio"]).reset_index(drop=True)

    hora_inicio_global = datetime.combine(date.today(), time(6, 30))
    hora_fin_global   = datetime.combine(date.today(), time(18, 30))

    colores_estado = {
        "Operativo": "#00B050",
        "Demora": "#FFC000",
        "Stand By": "#00B0F0",
        "Inoperativo":"#FF0000",
    }

    df["Color"] = df["Estado"].apply(lambda x: colores_estado.get(str(x), "#95a5a6"))

    def formatear_equipo(row):
        eq = row["Equipo"]
        ub = str(row["Ubicacion"]).strip()

        ub_base = ub.split()[0]

        if ub_base == "Ferrobamba":
            return f"<b style='color:#4085DC'>{eq}</b>"
        elif ub_base == "Chalcobamba":
            return f"<b style='color:#F37249'>{eq}</b>"
        else:
            return eq

    df["Equipo_label"] = df.apply(formatear_equipo, axis=1)

    fig = px.timeline(
        df,
        x_start="Hora Inicio",
        x_end="Hora Fin",
        y="Equipo_label",
        color="Estado",
        text="DuracionTexto",
        color_discrete_map=colores_estado,
        custom_data=["Descripcion", "DuracionTexto"]
    )

    fig.update_yaxes(
        tickfont=dict(size=13),
        type="category",
        categoryorder="array",
        categoryarray=(
            df.sort_values("Orden")
              ["Equipo_label"]
              .drop_duplicates()
              .tolist()
        ),
        autorange="reversed",
        title=""
    )
    fig.update_xaxes(
        range=[
            hora_inicio_global,
            hora_fin_global + timedelta(minutes=10)
        ],
        dtick=3600000
    )

    altura = max(250 * len(df["Equipo_label"].unique()), 700)

    MARGEN_T = 120
    MARGEN_B = 80
    plot_height = altura - MARGEN_T - MARGEN_B

    def y_arriba(px):
        return 1.0 + (MARGEN_T - px) / plot_height

    fig.update_layout(
        height=altura,

        title=dict(
            text=titulo,
            x=0.5,
            xanchor="center",
            xref="container",
            yref="container",
            y=1 - 45 / altura,
            yanchor="top",
            automargin=False
        ),

        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0,
        bargroupgap=0.65,
        font=dict(size=13, color="black"),

        margin=dict(t=MARGEN_T, b=MARGEN_B, l=80, r=80),

        images=[
            dict(
                source=logo_owm,
                xref="paper",
                yref="paper",
                x=-0.025,
                y=y_arriba(15),
                sizex=0.18,
                sizey=90 / plot_height,
                xanchor="left",
                yanchor="top"
            ),
            dict(
                source=logo_mmg,
                xref="paper",
                yref="paper",
                x=0.99,
                y=y_arriba(15),
                sizex=0.18,
                sizey=90 / plot_height,
                xanchor="right",
                yanchor="top"
            )
        ],

        legend=dict(
            orientation="h",
            yref="paper",
            yanchor="top",
            y=y_arriba(50),
            xanchor="center",
            x=0.475,
            title=dict(text=""),
            font=dict(color="black", size=20)
        )
    )

    fig.update_layout(
    annotations=[
        dict(
            text="🔷 Ferrobamba&nbsp;&nbsp;&nbsp;&nbsp;🔶 Chalcobamba",
            x=0.475,
            y=y_arriba(100),
            xref="paper",
            yref="paper",
            yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="black"),
            xanchor="center",
            align="center"
            )
        ]
    )

    fig.update_traces(
        marker_line_color='black',
        marker_line_width=0.5,
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="black", size=17.5),
    )

    for i, row in df.iterrows():
        desc = row["Descripcion"]

        if isinstance(desc, float) and pd.isna(desc):
            continue
        if str(desc).strip() == "":
            continue

        # 👉 Calcular duración en minutos
        duracion = (row["Hora Fin"] - row["Hora Inicio"]).total_seconds() / 60

        # 👉 Mostrar texto solo si duración >= 20 min
        if duracion <= 20:
            continue

        desc_str = str(desc).strip()

        palabras = desc_str.split()
        if len(palabras) >= 2:
            mitad = len(palabras) // 2
            desc_str = " ".join(palabras[:mitad]) + "<br>" + " ".join(palabras[mitad:])

        fig.add_annotation(
            x=row["Hora Inicio"] + (row["Hora Fin"] - row["Hora Inicio"]) / 2,
            y=row["Equipo_label"],
            text=desc_str,
            showarrow=False,
            yshift=70,
            font=dict(size=20, color="black")
        )

    fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black", size=16))
    fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black", size=20))

    fig.update_xaxes(dtick=3600000)

    equipos = (
        df.sort_values("Orden")[["Equipo","Grupo"]]
          .drop_duplicates()
          .reset_index(drop=True)
    )
    
    for i in range(1, len(equipos)):
        if equipos.loc[i, "Grupo"] != equipos.loc[i-1, "Grupo"]:
    
            fig.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=i-0.5,
                y1=i-0.5,
                line=dict(
                    color="black",
                    width=2
                )
            )

    hora_actual = hora_inicio_global
    equipos_unicos = len(df["Equipo_label"].unique())

    while hora_actual <= hora_fin_global:
        fig.add_shape(
            type="line",
            x0=hora_actual, x1=hora_actual,
            y0=-0.5, y1=equipos_unicos - 0.5,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below"
        )
        hora_actual += timedelta(hours=0.5)

    st.plotly_chart(fig, use_container_width=True, key="gantt_1")
    
    df_sorted = df.sort_values(["Orden", "Hora Fin"])

    df_estado_actual = (
        df_sorted
        .groupby("Equipo")
        .tail(1)
        [["Equipo", "Estado", "Ubicacion"]]
        .rename(columns={"Ubicacion": "Ubicación / Frente"})
    )

    df_prod = df[df["Estado"] == "Operativo"].copy()
    df_prod["Minutos"] = df_prod["Duracion"].dt.total_seconds() / 60

    df_prod_acum = (
        df_prod
        .groupby("Equipo")["Minutos"]
        .sum()
        .reset_index()
    )

    def minutos_a_hhmm(mins):
        h = int(mins // 60)
        m = int(mins % 60)
        return f"{h:02d}:{m:02d}"

    df_prod_acum["Producción acumulada"] = df_prod_acum["Minutos"].apply(minutos_a_hhmm)

    df_resumen = df_estado_actual.merge(
        df_prod_acum[["Equipo", "Producción acumulada"]],
        on="Equipo",
        how="left"
    )

    df_resumen["Producción acumulada"] = df_resumen["Producción acumulada"].fillna("00:00")

    df_resumen["Orden"] = df_resumen["Equipo"].map(lambda x: CONFIG_EQUIPOS[x]["orden"])

    df_resumen = (
        df_resumen
        .sort_values("Orden")
        .drop(columns="Orden")
    )

    df_op = df[df["Estado"] == "Operativo"].copy()
    df_op["Horas"] = df_op["Duracion"].dt.total_seconds() / 3600

    horas_por_equipo = df_op.groupby("Equipo")["Horas"].sum().reset_index()

    def calcular_metraje(row):
        eq = row["Equipo"]
        h = row["Horas"]
        if eq in FORMULAS_METRAJE:
            a = FORMULAS_METRAJE[eq]["a"]
            b = FORMULAS_METRAJE[eq]["b"]
            return a * h + b
        return 0

    horas_por_equipo["Metraje acumulado (m)"] = horas_por_equipo.apply(calcular_metraje, axis=1)

    df_dth = horas_por_equipo[
        ~horas_por_equipo["Equipo"].isin(["TD091", "TD092"])
    ]

    df_rtr = horas_por_equipo[
        horas_por_equipo["Equipo"].isin(["TD091", "TD092"])
    ]

    x_metros_dth = df_dth["Metraje acumulado (m)"].sum()
    y_metros_rtr = df_rtr["Metraje acumulado (m)"].sum()

    HORAS_TURNO = 12

    df_eq = (
        df.groupby(["Equipo", "Estado"])["Duracion"]
        .sum()
        .reset_index()
    )

    df_eq["Horas"] = df_eq["Duracion"].dt.total_seconds() / 3600

    df_total = (
        df_eq.groupby("Equipo")["Horas"]
        .sum()
        .reset_index(name="Horas_totales")
    )

    df_op = (
        df_eq[df_eq["Estado"] == "Operativo"]
        .groupby("Equipo")["Horas"]
        .sum()
        .reset_index(name="Horas_operativas")
    )

    df_proj = df_total.merge(df_op, on="Equipo", how="left")
    df_proj["Horas_operativas"] = df_proj["Horas_operativas"].fillna(0)

    df_proj["Operatividad"] = df_proj["Horas_operativas"] / df_proj["Horas_totales"]

    horas_totales_flota = df_proj["Horas_totales"].sum()
    horas_operativas_flota = df_proj["Horas_operativas"].sum()

    porc_operatividad_flota = (
        horas_operativas_flota / horas_totales_flota
        if horas_totales_flota > 0 else 0
    )

    porc_operatividad_proyectada = porc_operatividad_flota

    df_proj["Operatividad Flota (%)"] = porc_operatividad_flota * 100

    df_proj["Horas_proj"] = df_proj["Operatividad"] * HORAS_TURNO

    def metraje_proyectado(row):
        eq = row["Equipo"]
        h = row["Horas_proj"]

        if h <= 0:
            return 0

        if eq in FORMULAS_METRAJE:
            a = FORMULAS_METRAJE[eq]["a"]
            b = FORMULAS_METRAJE[eq]["b"]
            return a * h + b

        return 0

    df_proj["Metraje_proyectado (m)"] = df_proj.apply(metraje_proyectado, axis=1)

    # --- Intento proyeccion ML con modelo_fin (HistGradientBoosting) ---
    # Features: horas_acum, taladros_acum, metros_acum, rop_prom, metros_por_taladro, total_taladros + equipo/tipo/zona/turno/dureza
    # Se activa solo desde 240min y si modelo cargado
    ml_usado = False
    if mostrar_proyeccion and modelo_fin is not None and columnas_fin is not None:
        try:
            # defaults por equipo desde Odoo detalle_limpio (ver pipeline/tmp_dureza.py)
            EQUIPO_DEFAULTS = {
                "TD012": {"dureza":"D","zona":"FERROBAMBA","rop":27.2},
                "TD030": {"dureza":"D","zona":"FERROBAMBA","rop":32.2},
                "TD072": {"dureza":"M","zona":"FERROBAMBA","rop":37.8},
                "TD073": {"dureza":"D","zona":"CHALCOBAMBA","rop":36.4},
                "TD074": {"dureza":"D","zona":"CHALCOBAMBA","rop":31.9},
                "TD076": {"dureza":"D","zona":"FERROBAMBA","rop":29.0},
                "TD077": {"dureza":"D","zona":"FERROBAMBA","rop":33.1},
                "TD079": {"dureza":"D","zona":"FERROBAMBA","rop":37.7},
                "TD080": {"dureza":"D","zona":"FERROBAMBA","rop":33.0},
                "TD081": {"dureza":"M","zona":"CHALCOBAMBA","rop":37.0},
                "TD082": {"dureza":"D","zona":"FERROBAMBA","rop":33.0},
                "TD083": {"dureza":"D","zona":"FERROBAMBA","rop":33.0},
                "TD091": {"dureza":"D","zona":"CHALCOBAMBA","rop":31.7},
                "TD092": {"dureza":"D","zona":"FERROBAMBA","rop":30.6},
            }
            def zona_de_app(ub):
                ub=str(ub).upper()
                if "FERROBAMBA" in ub: return "FERROBAMBA"
                if "CHALCOBAMBA" in ub: return "CHALCOBAMBA"
                return "OTRO"
            # zona por equipo desde excel Ubicacion
            zona_por_equipo = df.groupby("Equipo")["Ubicacion"].apply(lambda s: zona_de_app(s.mode().iloc[0] if len(s.mode())>0 else ""))
            turno_ml = "A" if turno=="T/D" else "B"
            corte = int(minutos_transcurridos)
            # clip corte a rango entrenado 60-660
            corte = max(60, min(660, corte))
            rows_ml=[]
            for _, r in df_proj.iterrows():
                eq=r["Equipo"]
                horas_acum=float(r["Horas_operativas"])
                # evita 0
                if horas_acum<=0:
                    taladros_acum=0
                    metros_acum=0
                else:
                    # taladros estimados: horas / duracion_media 0.42h (25min)
                    taladros_acum=int(round(horas_acum/0.42))
                    metros_acum=float(r["Metraje_acum"]) if "Metraje_acum" in df_proj.columns else float(r["Metraje_proyectado (m)"] * r["Operatividad"] if r["Operatividad"]>0 else 0)
                metros_por_taladro = float(metros_acum/max(1,taladros_acum)) if taladros_acum>0 else 0
                oper = float(r["Operatividad"]) if r["Operatividad"]>0 else 0.1
                total_taladros = int(round(taladros_acum/max(oper,0.1))) if taladros_acum>0 else int(round(horas_acum/0.42/0.4))
                defs=EQUIPO_DEFAULTS.get(eq, {"dureza":"D","zona":"FERROBAMBA","rop":30})
                zona = zona_por_equipo.get(eq, defs["zona"])
                dureza=defs["dureza"]
                tipo="RTR" if eq.startswith("TD09") else "DTH"
                rows_ml.append({
                    "horas_acum": round(horas_acum,3),
                    "taladros_acum": taladros_acum,
                    "metros_acum": round(metros_acum,2),
                    "rop_prom": defs["rop"],
                    "metros_por_taladro": round(metros_por_taladro,2),
                    "total_taladros": total_taladros,
                    "equipo": eq, "tipo": tipo, "zona": zona, "turno": turno_ml, "dureza": dureza,
                    "corte_min": corte
                })
            X_ml = pd.DataFrame(rows_ml)
            # preparar dummy como en entrenar.py
            FEATURES_NUM_ML = ["horas_acum","taladros_acum","metros_acum","rop_prom","metros_por_taladro","total_taladros"]
            FEATURES_CAT_ML = ["equipo","tipo","zona","turno","dureza"]
            X_num = X_ml[FEATURES_NUM_ML].copy()
            X_cat = pd.get_dummies(X_ml[FEATURES_CAT].astype(str), columns=FEATURES_CAT)
            X_all = pd.concat([X_num, X_cat], axis=1)
            X_all = X_all.reindex(columns=columnas_fin, fill_value=0)
            pred = modelo_fin.predict(X_all)
            df_proj["Metraje_ML (m)"] = np.round(pred,1)
            # Equipos sin horas operativas -> 0 (evita baseline fantasma)
            df_proj.loc[df_proj["Horas_operativas"]<=0, "Metraje_ML (m)"] = 0
            # Reemplaza proyeccion lineal por ML si es razonable (0-800m)
            df_proj["Metraje_proyectado (m)"] = df_proj["Metraje_ML (m)"].clip(0, 800)
            ml_usado = True
        except Exception as e:
            # fallback lineal si falla ML
            df_proj["Metraje_ML (m)"] = np.nan

    metraje_dth_proj = df_proj[
        ~df_proj["Equipo"].isin(["TD091", "TD092"])
    ]["Metraje_proyectado (m)"].sum()

    metraje_rtr_proj = df_proj[
        df_proj["Equipo"].isin(["TD091", "TD092"])
    ]["Metraje_proyectado (m)"].sum()

    def resaltar_rtr(row):
        if row["Equipo"] in ["TD091", "TD092"]:
            return ["background-color: #00B050"] * len(row)
        return [""] * len(row)

    df_styled = df_resumen.style.apply(resaltar_rtr, axis=1)

    df["Horas"] = df["Duracion"].dt.total_seconds() / 3600

    df_cat = (
        df.groupby(["Equipo", "Categoria"])["Horas"]
        .sum()
        .reset_index()
    )

    df_pivot = df_cat.pivot_table(
        index="Equipo",
        columns="Categoria",
        values="Horas",
        fill_value=0
    ).reset_index()

    df_pivot["DM"] = (
    (
        df_pivot.get("Tiempo de Producción", 0) + 
        df_pivot.get("Tiempo de NO Producción", 0) + 
        df_pivot.get("Retraso Operativo Planificado", 0) + 
        df_pivot.get("Retraso Operativo NO Planificado", 0) 
    )/
    (
        df_pivot.get("Tiempo de Producción", 0) +
        df_pivot.get("Tiempo de NO Producción", 0) +
        df_pivot.get("Retraso Operativo Planificado", 0) +
        df_pivot.get("Retraso Operativo NO Planificado", 0) +
        df_pivot.get("PERDIDA DE EQUIPO PLANIFICADA", 0) +
        df_pivot.get("PERDIDA DE EQUIPO NO PLANIFICADA", 0) + 
        df_pivot.get("ECT", 0)
    )
    )

    df_pivot["DM"] = df_pivot["DM"].fillna(0)

    df_pivot["UE"] = (
    df_pivot.get("Tiempo de Producción", 0) /
    (
        df_pivot.get("Tiempo de Producción", 0) +
        df_pivot.get("Tiempo de NO Producción", 0) +
        df_pivot.get("Retraso Operativo Planificado", 0) +
        df_pivot.get("Retraso Operativo NO Planificado", 0)
    )
    )

    df_pivot["UE"] = df_pivot["UE"].fillna(0)

    df_pivot["Tipo"] = df_pivot["Equipo"].apply(
    lambda x: "RTR" if x in ["TD091", "TD092"] else "DTH"
    )

    promedios = (
        df_pivot.groupby("Tipo")[["DM", "UE"]]
        .mean()
        .reset_index()
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.1, 1])

    with col1:               
        st.markdown(
            "<div style='margin-top:-80px; margin-bottom:-5px;'>"
            "<h2 style='text-align:center; color:black; font-weight:700;'>"
            "ESTADO ACTUAL POR EQUIPO"
            "</h2>",
            unsafe_allow_html=True
        )
        
        st.dataframe(
            df_styled,
            use_container_width=True,
            hide_index=True,
            height=35 * (len(df_resumen) + 1)
        )

    with col2:

        st.markdown(
            "<div style='margin-top:-80px; margin-bottom:-5px;'>"
            "<h2 style='text-align:center; color:black; font-weight:700;'>"
            "AVANCE"
            "</h2>",
            unsafe_allow_html=True
        )

        a1, a2 = st.columns(2)

        with a1:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h1 style="
                        font-size:60px;
                        color:black;
                        margin-bottom:-20px;
                        margin-top:-60px;
                    ">
                {x_metros_dth:,.0f}</h1>
                    <h4 style="color:black;">METROS PERFORADOS DTH</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

        with a2:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h1 style="
                        font-size:60px;
                        color:black;
                        margin-bottom:-20px;
                        margin-top:-60px;
                    ">
                    {y_metros_rtr:,.0f}</h1>
                    <h4 style="color:black;">METROS PERFORADOS RTR</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Proyeccion desde 240min: amarilla 240-329, verde >=330 (pipeline/analisis_corte_temporal.py)
        if mostrar_proyeccion:
            dth_proj_txt = f"{metraje_dth_proj:,.0f}<br><small style='color:{color_conf}; font-size:13px;'>{texto_conf} · {minutos_transcurridos}min · R2 FIN {0.804 if minutos_transcurridos<330 else 0.817:.3f}</small>"
            rtr_proj_txt = f"{metraje_rtr_proj:,.0f}<br><small style='color:{color_conf}; font-size:13px;'>{texto_conf} · {minutos_transcurridos}min</small>"
            # Intento ML FIN si modelo cargado (features aproximadas desde horas_acum)
            if modelo_fin is not None and columnas_fin is not None:
                try:
                    # Construir feature row aproximada: usa horas_operativas como proxy de horas_acum
                    # Para cada equipo se promedia; aqui mostramos nota de modelo
                    pass
                except Exception:
                    pass
        else:
            dth_proj_txt = f"<span style='font-size:32px;'>⏳</span><br><small style='color:{color_conf};'>{texto_conf} ({minutos_transcurridos}min / {UMBRAL_MIN}min)</small>"
            rtr_proj_txt = f"<span style='font-size:32px;'>⏳</span><br><small style='color:{color_conf};'>{texto_conf}</small>"

        fuente = "ML HistGB" if ml_usado else "Lineal"
        st.markdown(
                f"<div style='margin-top:-80px; margin-bottom:-5px;'>"
                f"<h2 style='text-align:center; color:black; font-weight:700;'>PROYECCIÓN <span style='font-size:13px; color:{color_conf}; border:1px solid {color_conf}; padding:2px 6px; border-radius:6px;'>{texto_conf}</span> <span style='font-size:11px; color:gray;'>({fuente})</span></h2>"
                f"<p style='text-align:center; font-size:11px; color:gray; margin-top:4px;'>Umbral 240min (10:30 T/D · 22:30 T/N) — Media R2 0.80 · Oficial 330min (12:00/00:00) — Alta R2 0.817 — Evolución R2 0.777→0.867 (60→660min) · {fuente}</p>",
                unsafe_allow_html=True
            )

        p1, p2 = st.columns(2)

        with p1:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h1 style="
                        font-size:60px;
                        color:#1F4ED8;
                        margin-bottom:-20px;
                        margin-top:-60px;
                    ">
                        {dth_proj_txt}
                    </h1>
                    <h4 style="color:black;">
                        METROS PROYECTADOS DTH
                    </h4>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h1 style="
                        font-size:60px;
                        color:#1F4ED8;
                        margin-bottom:-20px;
                        margin-top:-60px;
                    ">
                        {rtr_proj_txt}</h1>
                    <h4 style="color:black;">
                        METROS PROYECTADOS RTR
                    </h4>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown(
            """
            <div style="
                width:80%;
                height:1px;
                margin:25px auto 20px auto;
                background: linear-gradient(
                    to right,
                    transparent,
                    #B0B0B0,
                    transparent
                );
            "></div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<h4 style='text-align:center; color:black; font-weight:700;'>"
            "DISPONILIDAD MECANICA Y UTILIZACION EFECTIVA"
            "</h4>",
            unsafe_allow_html=True
        )
        st.dataframe(
            promedios.assign(
                DM=lambda x: (x["DM"] * 100).round(1).astype(str) + "%",
                UE=lambda x: (x["UE"] * 100).round(1).astype(str) + "%",
            ),
            use_container_width=True,
            hide_index=True
        )

    col1, col2 = st.columns(2)
    with col1:
        df_demora = df[df["Estado"] == "Demora"].copy()

        df_demora["Duracion_min"] = df_demora["Duracion"].dt.total_seconds() / 60

        df_pie = df_demora.groupby("Descripcion")["Duracion_min"].sum().reset_index()

        df_pie["Porcentaje"] = (df_pie["Duracion_min"] / df_pie["Duracion_min"].sum()) * 100

        fig_pie = px.pie(
            df_pie,
            names="Descripcion",
            values="Duracion_min",
            hover_data=["Duracion_min", "Porcentaje"],
            labels={"Duracion_min":"Minutos", "Descripcion":"Tipo de demora"}
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Tiempo acumulado: %{value:.0f} min<br>Porcentaje: %{percent}"
        )
        fig_pie.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            legend=dict(
                font=dict(color="black"),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0
            )
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(
            "<h3 style='text-align:center; font-weight:700; color:black;'>"
            "DISTRIBUCIÓN POR DEMORA"
            "</h3>",
            unsafe_allow_html=True
        )

    with col2:
        df_estado = df.copy()
        df_estado["Duracion_min"] = df_estado["Duracion"].dt.total_seconds() / 60 
        df_pie_estado = df_estado.groupby("Estado")["Duracion_min"].sum().reset_index()

        def min_a_hhmm(minutos):
            h = int(minutos // 60)
            m = int(minutos % 60)
            return f"{h:02d}:{m:02d}"

        df_pie_estado["Duracion_HHMM"] = df_pie_estado["Duracion_min"].apply(min_a_hhmm)

        df_pie_estado["Porcentaje"] = (df_pie_estado["Duracion_min"] / df_pie_estado["Duracion_min"].sum()) * 100

        fig_pie_estado = px.pie(
            df_pie_estado,
            names="Estado",
            values="Duracion_min",
            color="Estado",
            color_discrete_map=colores_estado,
        )

        fig_pie_estado.update_traces(
            customdata=df_pie_estado[["Duracion_HHMM", "Porcentaje"]],
            texttemplate="%{label}<br>%{customdata[0]}<br>%{percent}",  # <-- incluir %{label}
            textposition="inside",
            textfont=dict(color="black", size=14),
            hovertemplate="<b>%{label}</b><br>Tiempo acumulado: %{customdata[0]}<br>Porcentaje: %{customdata[1]:.1f}%"
        )

        fig_pie_estado.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            legend=dict(
                font=dict(color="black"),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0
            )
        )

        st.plotly_chart(fig_pie_estado, use_container_width=True)
        st.markdown(
            "<h3 style='text-align:center; font-weight:700; color:black;'>"
            "DISTRIBUCIÓN POR ESTADO"
            "</h3>",
            unsafe_allow_html=True
        )

