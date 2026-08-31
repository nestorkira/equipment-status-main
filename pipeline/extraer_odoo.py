import xmlrpc.client
import pandas as pd
import traceback
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import time

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

try:
    url = os.getenv("ODOO_URL")
    db = os.getenv("ODOO_DB")
    username = os.getenv("ODOO_USER")
    password = os.getenv("ODOO_PASSWORD")

    if not all([url, db, username, password]):
        print("ERROR: faltan credenciales ODOO en entorno (secrets ODOO_URL, ODOO_DB, ODOO_USER, ODOO_PASSWORD)")
        sys.exit(1)

    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, username, password, {})
    if not uid:
        print("ERROR: autenticacion fallida en Odoo")
        sys.exit(1)
    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")
    print("UID:", uid)

    fecha_fin = datetime.now(timezone.utc).replace(tzinfo=None)
    fecha_ini = fecha_fin - timedelta(days=14)
    fecha_ini_str = fecha_ini.strftime("%Y-%m-%d")
    fecha_fin_str = fecha_fin.strftime("%Y-%m-%d")
    print(f"Rango solicitado: {fecha_ini_str} -> {fecha_fin_str}")

    DIALES_RANGO_AMPLIO = [
        ("2025-01-01", "2025-03-31"),
        ("2025-04-01", "2025-06-30"),
        ("2025-07-01", "2025-08-31"),
        ("2025-09-01", fecha_fin_str),
    ]
    print("Verificando rangos disponibles en Odoo...")
    for di, df_r in DIALES_RANGO_AMPLIO:
        dom_check = [
            ("date", ">=", di),
            ("date", "<=", df_r),
            ("contract", "in", [
                "Mina Bambas Rotativas - Ferrobamba",
                "Mina Bambas Rotativas - Chalcobamba",
                "Mina Bambas",
            ]),
        ]
        cnt = models.execute_kw(db, uid, password, MODELO_ODOO, "search_count", [dom_check])
        print(f"  {di} -> {df_r}: {cnt} registros")

    domain = [
        ("date", ">=", fecha_ini_str),
        ("date", "<=", fecha_fin_str),
        (
            "contract",
            "in",
            [
                "Mina Bambas Rotativas - Ferrobamba",
                "Mina Bambas Rotativas - Chalcobamba",
                "Mina Bambas",
            ],
        ),
    ]

    MODELO_ODOO = "project.steel.report"

    FIELDS_LIST = [
        "id", "contract", "date", "shift", "area", "block", "equipo",
        "bank", "project", "drill_type", "drill_code",
        "hour_from", "hour_to", "rop", "hardness", "high",
    ]

    def safe_read_batch(model, ids_batch):
        try:
            return models.execute_kw(
                db, uid, password, model, "read", [ids_batch],
                {"fields": FIELDS_LIST},
            )
        except Exception as e:
            print("Error en lote grande. Dividiendo...", e)
            if len(ids_batch) == 1:
                print(f"Registro problematico ID {ids_batch[0]} omitido")
                return []
            mid = len(ids_batch) // 2
            left = safe_read_batch(model, ids_batch[:mid])
            right = safe_read_batch(model, ids_batch[mid:])
            return left + right

    count = models.execute_kw(db, uid, password, MODELO_ODOO, "search_count", [domain])
    print("Total registros:", count)

    inicio = time.time()
    ids = models.execute_kw(db, uid, password, MODELO_ODOO, "search", [domain])

    registros = []
    batch_size = 1000
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        print(f"Leyendo {i} - {i + len(batch_ids)}")
        data = safe_read_batch(MODELO_ODOO, batch_ids)
        registros.extend(data)

    print("Tiempo total:", round(time.time() - inicio, 2), "segundos")

    df = pd.DataFrame(registros)
    if df.empty:
        print("ERROR: no se recuperaron registros de Odoo")
        sys.exit(1)
    print("Columnas obtenidas:", list(df.columns))

    for col in ["hour_from", "hour_to", "rop", "hardness", "high"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        fechas = sorted(df["date"].dropna().unique())
        print(f"Fechas unicas en Odoo ({len(fechas)}): {fechas[:5]} ... {fechas[-5:]}" if len(fechas) > 10 else f"Fechas unicas: {fechas}")

    df = df[df["drill_code"].notna()]
    df = df[df["drill_code"] != False]
    df = df[df["drill_code"] != ""]

    ruta_detalle = DATA_DIR / "detalle_diario.csv"
    df.to_csv(ruta_detalle, index=False)
    print(f"Detalle guardado en: {ruta_detalle} | filas: {len(df)}")

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
    )
    resumen["Horas Trabajadas"] = resumen["horas_trabajadas"].round(2)
    resumen["Metros"] = resumen["metros_fin_turno"].round(2)
    ruta_resumen = DATA_DIR / "resumen_diario.xlsx"
    resumen.to_excel(ruta_resumen, index=False)
    print(f"Resumen guardado en: {ruta_resumen} | filas: {len(resumen)}")

except Exception as e:
    print("ERROR FATAL en extraer_odoo.py:")
    traceback.print_exc()
    sys.exit(1)
