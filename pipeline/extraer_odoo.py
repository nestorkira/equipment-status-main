import xmlrpc.client
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os
import time

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

url = os.getenv("ODOO_URL")
db = os.getenv("ODOO_DB")
username = os.getenv("ODOO_USER")
password = os.getenv("ODOO_PASSWORD")

common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
uid = common.authenticate(db, username, password, {})
models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")

print("UID:", uid)

fecha_fin = datetime.today()
fecha_ini = fecha_fin - timedelta(days=14)

fecha_ini_str = fecha_ini.strftime("%Y-%m-%d")
fecha_fin_str = fecha_fin.strftime("%Y-%m-%d")

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
    "id",
    "contract",
    "date",
    "shift",
    "area",
    "block",
    "equipo",
    "bank",
    "project",
    "drill_type",
    "drill_code",
    "hour_from",
    "hour_to",
    "rop",
    "hardness",
    "high",
]


def safe_read_batch(model, ids_batch):
    try:
        return models.execute_kw(
            db,
            uid,
            password,
            model,
            "read",
            [ids_batch],
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


inicio = time.time()

ids = models.execute_kw(db, uid, password, MODELO_ODOO, "search", [domain])
print("Total registros:", len(ids))

registros = []
batch_size = 1000

for i in range(0, len(ids), batch_size):
    batch_ids = ids[i : i + batch_size]
    print(f"Leyendo {i} - {i + len(batch_ids)}")
    data = safe_read_batch(MODELO_ODOO, batch_ids)
    registros.extend(data)

print("Tiempo total:", round(time.time() - inicio, 2), "segundos")

df = pd.DataFrame(registros)

df = df[df["drill_code"].notna()]
df = df[df["drill_code"] != False]
df = df[df["drill_code"] != ""]

ruta_detalle = DATA_DIR / "detalle_diario.csv"
df.to_csv(ruta_detalle, index=False)
print("Detalle guardado en:", ruta_detalle, "| filas:", len(df))
