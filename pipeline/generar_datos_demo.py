import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

EQUIPOS = {
    "TD030": {"tipo": "DTH", "zona": "CHALCOBAMBA", "ritmo": 30},
    "TD072": {"tipo": "DTH", "zona": "FERROBAMBA", "ritmo": 33},
    "TD073": {"tipo": "DTH", "zona": "CHALCOBAMBA", "ritmo": 29},
    "TD074": {"tipo": "DTH", "zona": "CHALCOBAMBA", "ritmo": 31},
    "TD076": {"tipo": "DTH", "zona": "FERROBAMBA", "ritmo": 28},
    "TD077": {"tipo": "DTH", "zona": "FERROBAMBA", "ritmo": 27},
    "TD079": {"tipo": "DTH", "zona": "FERROBAMBA", "ritmo": 32},
    "TD091": {"tipo": "RTR", "zona": "CHALCOBAMBA", "ritmo": 25},
    "TD092": {"tipo": "RTR", "zona": "CHALCOBAMBA", "ritmo": 24},
}

TURNOS = {"A": 6.5, "B": 18.5}
BLOQUES = ["B101", "B206", "B215", "B323"]
CONTRATOS = [
    "Mina Bambas Rotativas - Ferrobamba",
    "Mina Bambas Rotativas - Chalcobamba",
]

fechas = pd.date_range("2026-01-05", periods=120, freq="D")
registros = []
rid = 1

for fecha in fechas:
    for eq, cfg in EQUIPOS.items():
        if np.random.rand() < 0.12:
            continue
        for turno, inicio in TURNOS.items():
            if np.random.rand() < 0.15:
                continue
            n_taladros = np.random.poisson(14) + 1
            ritmo_eq = cfg["ritmo"] * np.random.uniform(0.85, 1.15)
            dureza_dia = np.random.uniform(4, 9)
            hora_cursor = inicio + np.random.uniform(0.0, 1.5)
            for _ in range(n_taladros):
                dur = np.random.uniform(0.35, 1.4)
                hora_fin = hora_cursor + dur
                fin_rel = (hora_fin - inicio) % 24
                if fin_rel > 12:
                    break
                rop = ritmo_eq * np.random.uniform(0.9, 1.1)
                metros = max(rop * dur * np.random.uniform(0.85, 1.1), 1.0)
                registros.append({
                    "id": rid,
                    "contract": np.random.choice(CONTRATOS),
                    "date": fecha.strftime("%Y-%m-%d"),
                    "shift": turno,
                    "area": cfg["zona"],
                    "block": np.random.choice(BLOQUES),
                    "equipo": eq,
                    "bank": f"BK-{np.random.randint(100, 999)}",
                    "project": "PRODUCCION",
                    "drill_type": cfg["tipo"],
                    "drill_code": f"{cfg['tipo']}-{np.random.randint(10, 99)}",
                    "hour_from": round(hora_cursor % 24, 2),
                    "hour_to": round(hora_fin % 24, 2),
                    "rop": round(rop, 2),
                    "hardness": round(dureza_dia, 1),
                    "high": round(metros, 1),
                })
                rid += 1
                hora_cursor += dur + np.random.uniform(0.02, 0.5)

df = pd.DataFrame(registros)
out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)
df.to_csv(out / "detalle_diario.csv", index=False)
print(f"Demo generada: {len(df)} registros -> data/detalle_diario.csv")
print(df.head())
