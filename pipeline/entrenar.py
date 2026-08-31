import json
import traceback
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BASE = Path(__file__).resolve().parents[1]
SNAPSHOTS = BASE / "data" / "snapshots.csv"
MODELOS_DIR = BASE / "models"
METRICAS_PATH = BASE / "metrics" / "metricas.csv"
CORTE_OFICIAL_MIN = 330
FRACCION_TEST_DIAS = 0.2

FEATURES_NUM = [
    "horas_acum", "taladros_acum", "metros_acum",
    "rop_prom", "metros_por_taladro", "total_taladros",
]
FEATURES_CAT = ["equipo", "tipo", "zona", "turno", "dureza"]

MODELOS = {
    "RegresionLineal": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42
    ),
    "RedNeuronal": lambda: make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42),
    ),
}

def preparar_X(df):
    X_num = df[FEATURES_NUM].copy()
    X_cat = pd.get_dummies(df[FEATURES_CAT], columns=FEATURES_CAT)
    X = pd.concat([X_num, X_cat], axis=1)
    return X

def evaluar(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    rmse = round(root_mean_squared_error(y_true, y_pred), 3)
    mask = y_true > 0
    if mask.sum() == 0:
        mape = None
    else:
        mape = round(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100, 2)
    r2 = round(r2_score(y_true, y_pred), 4)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def split_temporal(df):
    fechas = sorted(df["fecha"].unique())
    n_test = max(int(len(fechas) * FRACCION_TEST_DIAS), 1)
    n_test = min(n_test, max(1, len(fechas) - 1))
    test_fechas = set(fechas[-n_test:])
    return df[~df["fecha"].isin(test_fechas)], df[df["fecha"].isin(test_fechas)]

def entrenar_objetivo(nombre, df, target):
    if df.empty:
        print(f"ADVERTENCIA: datos vacios para '{nombre}'. Se omite.")
        return None
    df = df.dropna(subset=[target])
    if df.empty:
        print(f"ADVERTENCIA: sin filas validas para '{nombre}'. Se omite.")
        return None
    train, test = split_temporal(df)
    if train.empty or test.empty:
        print(f"ADVERTENCIA: split insuficiente para '{nombre}' (train={len(train)}, test={len(test)}). Se omite.")
        return None
    X_train = preparar_X(train)
    X_test = preparar_X(test)
    all_cols = X_train.columns.union(X_test.columns)
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)
    y_train, y_test = train[target].values, test[target].values

    print(f"\n[{nombre}] Train: {len(train)} filas | Test: {len(test)} filas | Features: {len(all_cols)}")

    resultados = {}
    mejor = None
    for nombre_modelo, fabrica in MODELOS.items():
        try:
            modelo = fabrica()
            modelo.fit(X_train, y_train)
            m = evaluar(y_test, modelo.predict(X_test))
            resultados[nombre_modelo] = m
            print(f"  {nombre_modelo:20s}: R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.1f}%")
            if mejor is None or m["RMSE"] < resultados[mejor]["RMSE"]:
                mejor = nombre_modelo
        except Exception as e:
            print(f"  {nombre_modelo:20s}: FALLO - {e}")

    if mejor is None:
        print(f"ERROR: ningun modelo entreno bien para '{nombre}'.")
        return None

    modelo_final = MODELOS[mejor]()
    X_full = preparar_X(df)
    all_cols_full = X_full.columns
    modelo_final.fit(X_full, df[target].values)
    ruta = MODELOS_DIR / f"modelo_{nombre}.pkl"
    joblib.dump(modelo_final, ruta)
    with open(MODELOS_DIR / f"columnas_{nombre}.json", "w") as f:
        json.dump(list(all_cols_full), f)
    print(f"\n  >>> Modelo '{nombre}' guardado: {ruta} via {mejor}")
    return {
        "modelo": nombre, "target": target, "mejor_algoritmo": mejor,
        "metricas_test": json.dumps(resultados, ensure_ascii=False),
        "n_train": len(train), "n_test": len(test),
    }

def main():
    MODELOS_DIR.mkdir(exist_ok=True)
    (BASE / "metrics").mkdir(exist_ok=True)
    if not SNAPSHOTS.exists():
        print(f"ERROR: no existe {SNAPSHOTS}. Ejecuta generar_snapshots.py primero.")
        sys.exit(1)
    df = pd.read_csv(SNAPSHOTS)
    print(f"Snapshots cargados: {len(df)} filas")
    print(f"Fechas: {df['fecha'].min()} a {df['fecha'].max()}")
    print(f"Equipos: {sorted(df['equipo'].unique())}")

    df_corte = df[df["corte_min"] <= CORTE_OFICIAL_MIN].dropna(subset=["metros_al_corte"])
    df_fin = df.dropna(subset=["metros_fin_turno"])

    filas = []
    r = entrenar_objetivo("corte", df_corte, "metros_al_corte")
    if r: filas.append(r)
    r = entrenar_objetivo("fin", df_fin, "metros_fin_turno")
    if r: filas.append(r)

    if not filas:
        print("ERROR: no se entreno ningun modelo.")
        sys.exit(1)

    hoy = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    salida = pd.DataFrame(filas)
    salida.insert(0, "fecha_entrenamiento", hoy)
    if METRICAS_PATH.exists():
        try:
            historico = pd.read_csv(METRICAS_PATH)
            salida = pd.concat([historico, salida], ignore_index=True)
        except Exception:
            pass
    salida.to_csv(METRICAS_PATH, index=False)
    print("\nMetricas guardadas en:", METRICAS_PATH)
    print(salida.to_string())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR FATAL en entrenar.py:")
        traceback.print_exc()
        sys.exit(1)
