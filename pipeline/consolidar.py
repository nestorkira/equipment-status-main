import pandas as pd
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
DETALLE = BASE / "data" / "detalle_diario.csv"
HISTORICO = BASE / "data" / "historico_detalle.csv"
REQUIRED_COLS = {"id", "contract", "date", "shift", "area", "equipo",
                 "hour_from", "hour_to", "rop", "hardness", "high"}

try:
    if not DETALLE.exists():
        print(f"ERROR: no existe {DETALLE}. Ejecuta extraer_odoo.py primero.")
        sys.exit(1)

    df_new = pd.read_csv(DETALLE)
    cols_faltantes = REQUIRED_COLS - set(df_new.columns)
    if cols_faltantes:
        print(f"ERROR: columnas faltantes en detalle: {cols_faltantes}")
        sys.exit(1)

    if HISTORICO.exists():
        df_hist = pd.read_csv(HISTORICO)
        df = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df = df_new.copy()

    df = df.drop_duplicates(subset="id", keep="last")
    df = df.sort_values(["date", "shift", "equipo", "hour_from"]).reset_index(drop=True)
    df.to_csv(HISTORICO, index=False)

    duracion = df["hour_to"] - df["hour_from"]
    df_tmp = df.assign(duracion=duracion.where(duracion >= 0, duracion + 24))
    resumen = (
        df_tmp.groupby(["date", "area", "equipo", "shift"], as_index=False)
        .agg(
            taladros=("id", "count"),
            horas_trabajadas=("duracion", "sum"),
            metros_fin_turno=("high", "sum"),
            rop_promedio=("rop", "mean"),
            dureza_promedio=("hardness", "mean"),
        )
        .rename(
            columns={
                "date": "Fecha", "equipo": "Maquina", "shift": "Turno",
                "horas_trabajadas": "Horas Trabajadas", "taladros": "Taladros",
                "metros_fin_turno": "Metros",
            }
        )
    )
    resumen["Horas Trabajadas"] = resumen["Horas Trabajadas"].round(2)
    resumen["Metros"] = resumen["Metros"].round(2)
    resumen.to_excel(BASE / "data" / "resumen_diario.xlsx", index=False)
    print(f"Historico: {len(df)} registros | Resumen: {len(resumen)} filas equipo-turno")

except Exception as e:
    print("ERROR FATAL en consolidar.py:")
    traceback = __import__("traceback"); traceback.print_exc()
    sys.exit(1)