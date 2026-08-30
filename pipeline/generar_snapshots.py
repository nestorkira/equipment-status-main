import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
HISTORICO = BASE / "data" / "historico_detalle.csv"
SNAPSHOTS = BASE / "data" / "snapshots.csv"

INICIO_TURNO = {"A": 6.5, "B": 18.5}
DURACION_TURNO_MIN = 720
CORTE_OFICIAL_MIN = 330
CORTES_MIN = [60, 120, 180, 240, 300, 330, 360, 420, 480, 540, 600, 660]

EQUIPOS_RTR = {"TD091", "TD092"}


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


def main():
    df = pd.read_csv(HISTORICO)
    df["zona"] = df["area"].apply(zona_de)
    df["tipo"] = df["equipo"].apply(lambda x: "RTR" if x in EQUIPOS_RTR else "DTH")

    duracion = df["hour_to"] - df["hour_from"]
    df["duracion_min"] = duracion.where(duracion >= 0, duracion + 24) * 60

    snapshots = []
    for (fecha, turno, equipo), g in df.groupby(["date", "shift", "equipo"]):
        inicio = INICIO_TURNO[turno]
        zona = g["zona"].iloc[0]
        tipo = g["tipo"].iloc[0]
        dureza_prom = g["hardness"].mean()

        intervalos = []
        for _, r in g.iterrows():
            ini_rel = minutos_desde_inicio(r["hour_from"], inicio)
            fin_rel = ini_rel + r["duracion_min"]
            intervalos.append((ini_rel, fin_rel, float(r["high"]), float(r["rop"])))

        total_metros = sum(iv[2] for iv in intervalos)

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
                fraccion = ov / (fin_rel - ini_rel) if fin_rel > ini_rel else 1.0
                horas_acum += ov / 60.0
                metros_acum += high * fraccion
                if ini_rel < corte:
                    taladros_acum += 1
                    pesos_rop += ov
                    rop_pond += rop * ov

            snapshots.append(
                {
                    "fecha": fecha,
                    "turno": turno,
                    "equipo": equipo,
                    "tipo": tipo,
                    "zona": zona,
                    "corte_min": corte,
                    "hora_corte": round((inicio + corte / 60.0) % 24, 2),
                    "horas_acum": round(horas_acum, 3),
                    "taladros_acum": taladros_acum,
                    "metros_acum": round(metros_acum, 2),
                    "rop_prom": round(rop_pond / pesos_rop, 2) if pesos_rop > 0 else 0.0,
                    "dureza_prom": round(dureza_prom, 2),
                    "horas_restantes": round((DURACION_TURNO_MIN - corte) / 60.0, 2),
                    "metros_al_corte": round(
                        sum(
                            iv[2]
                            * (
                                solapamiento(iv[0], iv[1], CORTE_OFICIAL_MIN)
                                / max(iv[1] - iv[0], 1e-9)
                            )
                            for iv in intervalos
                        ),
                        2,
                    )
                    if corte <= CORTE_OFICIAL_MIN
                    else np.nan,
                    "metros_fin_turno": round(total_metros, 2),
                }
            )

    out = pd.DataFrame(snapshots)
    out.to_csv(SNAPSHOTS, index=False)
    print(f"Snapshots: {len(out)} filas -> data/snapshots.csv")
    print(out.head())


if __name__ == "__main__":
    main()
