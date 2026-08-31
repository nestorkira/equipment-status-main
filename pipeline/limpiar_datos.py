import pandas as pd
import numpy as np
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

print("=" * 70)
print("LIMPIEZA DE DETALLE DIARIO")
print("=" * 70)

df = pd.read_csv(DATA / "detalle_diario.csv", low_memory=False)
print(f"Registros originales: {len(df)}")

# 1. ROP 15-75
df["rop"] = pd.to_numeric(df["rop"], errors="coerce")
antes = len(df)
df = df[(df["rop"] >= 15) & (df["rop"] <= 75)]
print(f"ROP 15-75: {antes} -> {len(df)} (-{antes - len(df)})")

# 2. High (metros) razonable
df["high"] = pd.to_numeric(df["high"], errors="coerce")
antes = len(df)
df = df[(df["high"] >= 0.5) & (df["high"] <= 25)]
print(f"High 0.5-25: {antes} -> {len(df)} (-{antes - len(df)})")

# 3. Drill_code valido
antes = len(df)
df = df[df["drill_code"].notna() & (df["drill_code"] != False) & (df["drill_code"] != "")]
print(f"Drill_code: {antes} -> {len(df)} (-{antes - len(df)})")

# 4. Equipos con >=50 registros
eq_counts = df["equipo"].value_counts()
equipos_validos = eq_counts[eq_counts >= 50].index.tolist()
antes = len(df)
df = df[df["equipo"].isin(equipos_validos)]
print(f"Equipos >=50: {antes} -> {len(df)} (-{antes - len(df)})")
print(f"  Equipos: {sorted(equipos_validos)}")

# 5. Duracion turno (calcular y filtrar)
df["hour_from"] = pd.to_numeric(df["hour_from"], errors="coerce")
df["hour_to"] = pd.to_numeric(df["hour_to"], errors="coerce")
df["duracion"] = df["hour_to"] - df["hour_from"]
df.loc[df["duracion"] < 0, "duracion"] = df["duracion"] + 24
antes = len(df)
df = df[(df["duracion"] >= 0.1) & (df["duracion"] <= 13)]
print(f"Duracion 0.1-13h: {antes} -> {len(df)} (-{antes - len(df)})")

# Guardar
df.to_csv(DATA / "detalle_limpio.csv", index=False)
print(f"\nGuardado: data/detalle_limpio.csv ({len(df)} registros)")

# Resumen
print("\n" + "=" * 70)
print("RESUMEN POST-LIMPIEZA")
print("=" * 70)
print(f"Registros: {len(df)}")
print(f"Fechas: {df['date'].min()} a {df['date'].max()}")
print(f"Dias: {df['date'].nunique()}")
print(f"Equipos: {df['equipo'].nunique()}")
print(f"Turnos: {df['shift'].value_counts().to_dict()}")
print(f"Hardness: {df['hardness'].value_counts().to_dict()}")

print("\nEstadisticas:")
for col in ["rop", "high", "duracion"]:
    s = df[col]
    print(f"  {col:12s}: min={s.min():.2f}  media={s.mean():.2f}  max={s.max():.2f}")

print("\nPor equipo:")
for eq in sorted(df["equipo"].unique()):
    n = len(df[df["equipo"] == eq])
    tipo = "RTR" if eq.startswith("TD09") else "DTH"
    rop_s = df[df["equipo"] == eq]["rop"]
    print(f"  {eq:12s}: {n:>5} registros | ROP {rop_s.mean():.1f} ({rop_s.min():.1f}-{rop_s.max():.1f}) | {tipo}")
