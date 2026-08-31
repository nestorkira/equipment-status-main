import numpy as np
import pandas as pd
import traceback
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DETALLE = BASE / "data" / "detalle_limpio.csv"
SNAPSHOTS = BASE / "data" / "snapshots.csv"

INICIO_TURNO = {"A": 6.5, "B": 18.5}
DURACION_TURNO_MIN = 720
CORTE_OFICIAL_MIN = 330
CORTES_MIN = [60, 120, 180, 240, 300, 330, 360, 420, 480, 540, 600, 660]

RTR_PREFIX = ("TD09",)
EQUIPOS_RTR_MANUAL = set()

def es_rtr(equipo):
    equipo = str(equipo).upper().strip()
    if equipo in EQUIPOS_RTR_MANUAL:
        return True
    return any(equipo.startswith(p) for p in RTR_PREFIX)

def zona_de(area):
    area = str(area).upper()
    if "FERROBAMBA" in area:
        return "FERROBAMBA"
    if "CHALCOBAMBA" in area:
        return "CHALCOBAMBA"
    return "OTRO"

def minutos_desde_inicio(hora, inicio):
    return (hora - inicio) % 24 * 60

def solapamiento(ini_min, fin_min, corte):
    ini = max(0.0, ini_min)
    fin = min(float(corte), fin_min)
    return max(0.0, fin - ini)

try:
    if not DETALLE.exists():
        print(f"ERROR: no existe {DETALLE}. Ejecuta limpiar_datos.py primero.")
        sys.exit(1)

    df = pd.read_csv(DETALLE, low_memory=False)
    REQUIRED = {"date", "shift", "equipo", "area", "hour_from", "hour_to", "high", "rop"}
    if not REQUIRED.issubset(set(df.columns)):
        print(f"ERROR: columnas faltantes: {REQUIRED - set(df.columns)}")
        sys.exit(1)

    df["zona"] = df["area"].apply(zona_de)
    df["tipo"] = df["equipo"].apply(lambda x: "RTR" if es_rtr(x) else "DTH")

    equipos_info = df.groupby("equipo")["tipo"].first().to_dict()
    print(f"Equipos detectados ({len(equipos_info)}):")
    for eq, tp in sorted(equipos_info.items()):
        n = len(df[df["equipo"] == eq])
        print(f"  {eq}: {tp} ({n} registros)")

    if "hardness" in df.columns:
        df["hardness"] = df["hardness"].fillna("N/A")
        df["hardness"] = df["hardness"].replace("False", "N/A")
        df["hardness"] = df["hardness"].replace(False, "N/A")
    else:
        df["hardness"] = "N/A"

    if "contract" not in df.columns:
        df["contract"] = "N/A"

    df["hour_from"] = pd.to_numeric(df["hour_from"], errors="coerce")
    df["hour_to"] = pd.to_numeric(df["hour_to"], errors="coerce")
    df["rop"] = pd.to_numeric(df["rop"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")

    duracion = df["hour_to"] - df["hour_from"]
    df["duracion_min"] = duracion.where(duracion >= 0, duracion + 24) * 60

    snapshots = []
    for (fecha, turno, equipo), g in df.groupby(["date", "shift", "equipo"]):
        inicio = INICIO_TURNO[turno]
        zona = g["zona"].iloc[0]
        tipo = g["tipo"].iloc[0]
        hardness_mode = g["hardness"].mode()
        dureza = hardness_mode.iloc[0] if len(hardness_mode) > 0 else "N/A"
        contract = g["contract"].mode().iloc[0] if "contract" in g.columns and len(g["contract"].mode()) > 0 else "N/A"

        intervalos = []
        for _, r in g.iterrows():
            if pd.isna(r["hour_from"]) or pd.isna(r["hour_to"]) or pd.isna(r["high"]) or pd.isna(r["rop"]):
                continue
            ini_rel = minutos_desde_inicio(r["hour_from"], inicio)
            fin_rel = ini_rel + r["duracion_min"]
            if fin_rel > ini_rel:
                intervalos.append((ini_rel, fin_rel, float(r["high"]), float(r["rop"])))

        if not intervalos:
            continue

        total_metros = sum(iv[2] for iv in intervalos)
        total_taladros = len(intervalos)

        for corte in CORTES_MIN:
            horas_acum = 0.0
            metros_acum = 0.0
            taladros_acum = 0
            pesos_rop = 0.0
            rop_pond = 0.0
            for ini_rel, fin_rel, high, rop in intervalos:
                ov = solapamiento(ini_rel, fin_rel, corte)
                if ov <= 0:
                    continue
                fraccion = ov / (fin_rel - ini_rel)
                horas_acum += ov / 60.0
                metros_acum += high * fraccion
                if ini_rel < corte:
                    taladros_acum += 1
                    pesos_rop += ov
                    rop_pond += rop * ov

            rop_prom = round(rop_pond / pesos_rop, 2) if pesos_rop > 0 else 0.0
            metros_por_taladro = round(metros_acum / taladros_acum, 2) if taladros_acum > 0 else 0.0

            acum_al_corte = round(
                sum(
                    iv[2] * (solapamiento(iv[0], iv[1], CORTE_OFICIAL_MIN) / max(iv[1] - iv[0], 1e-9))
                    for iv in intervalos
                ), 2) if corte <= CORTE_OFICIAL_MIN else np.nan

            snapshots.append({
                "fecha": fecha, "turno": turno, "equipo": equipo, "tipo": tipo,
                "zona": zona, "contract": contract,
                "corte_min": corte,
                "hora_corte": round((inicio + corte / 60.0) % 24, 2),
                "horas_acum": round(horas_acum, 3),
                "taladros_acum": taladros_acum, "metros_acum": round(metros_acum, 2),
                "rop_prom": rop_prom,
                "metros_por_taladro": metros_por_taladro,
                "dureza": dureza,
                "metros_al_corte": acum_al_corte,
                "metros_fin_turno": round(total_metros, 2),
                "total_taladros": total_taladros,
            })

    out = pd.DataFrame(snapshots)
    out.to_csv(SNAPSHOTS, index=False)
    print(f"\nSnapshots: {len(out)} filas -> data/snapshots.csv")
    print(f"Columnas: {list(out.columns)}")
except Exception as e:
    print("ERROR FATAL en generar_snapshots.py:")
    traceback.print_exc()
    sys.exit(1)
