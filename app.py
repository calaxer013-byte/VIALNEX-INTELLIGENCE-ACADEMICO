
from flask import Flask, render_template, request, jsonify, Response
from pathlib import Path
from datetime import date
import sqlite3
import csv
import io
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# 1. CONFIGURACION GENERAL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "pavement.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.json.sort_keys = False

FEATURES = [
    "antiguedad",
    "trafico",
    "grietas",
    "losas_danadas",
    "escalonamiento",
    "baches"
]

FEATURE_LABELS = [
    "Antigüedad",
    "Tráfico",
    "Grietas",
    "Losas dañadas",
    "Escalonamiento",
    "Baches"
]

# ==========================================================
# 2. CONEXION A SQLITE
# ==========================================================

def db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


# ==========================================================
# 3. CLASIFICACION DEL PAVIMENTO
# ==========================================================

def condition(pci):

    if pci >= 85:
        return "Bueno", "good"

    elif pci >= 70:
        return "Satisfactorio", "good"

    elif pci >= 55:
        return "Regular", "warning"

    elif pci >= 40:
        return "Malo", "danger"

    return "Muy malo", "danger"


def recommendation(pci):

    if pci >= 85:
        return (
            "El pavimento presenta condiciones favorables. "
            "Se recomienda conservación preventiva e inspecciones periódicas."
        )

    elif pci >= 70:
        return (
            "Se recomienda mantenimiento preventivo para conservar "
            "las condiciones actuales de servicio."
        )

    elif pci >= 55:
        return (
            "Se recomienda programar intervenciones de mantenimiento "
            "y realizar seguimiento de los deterioros."
        )

    elif pci >= 40:
        return (
            "Se recomienda priorizar una evaluación técnica detallada "
            "y planificar trabajos de rehabilitación."
        )

    return (
        "Se recomienda una evaluación técnica prioritaria para determinar "
        "la necesidad de rehabilitación mayor o reconstrucción."
    )


# ==========================================================
# 4. CREACION DE BASE DE DATOS
# ==========================================================

def init_db():

    with db() as con:

        con.execute("""
            CREATE TABLE IF NOT EXISTS inspections (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                sector TEXT NOT NULL,
                date TEXT NOT NULL,

                antiguedad REAL NOT NULL,
                trafico REAL NOT NULL,
                grietas REAL NOT NULL,

                baches REAL NOT NULL,
                losas_danadas REAL NOT NULL,
                escalonamiento REAL NOT NULL,

                pci REAL NOT NULL,

                observaciones TEXT DEFAULT ''
            )
        """)

        count = con.execute(
            "SELECT COUNT(*) FROM inspections"
        ).fetchone()[0]

        # Datos demostrativos para iniciar el sistema.
        if count == 0:

            rng = np.random.default_rng(2026)

            sectors = [
                "Jr. Dos de Mayo",
                "Av. Alameda",
                "Jr. Huallayco",
                "Av. Universitaria",
                "Jr. Constitución",
                "Av. Esteban Pavletich",
                "Jr. Abtao",
                "Av. Circunvalación"
            ]

            for i in range(100):

                age = float(rng.uniform(1, 24))
                traffic = float(rng.uniform(150, 2600))
                cracks = float(rng.uniform(0.5, 42))
                slabs = float(rng.uniform(0.5, 38))
                faulting = float(rng.uniform(0, 18))
                potholes = float(rng.uniform(0, 18))

                pci = (
                    100
                    - 0.72 * age
                    - 0.007 * traffic
                    - 0.72 * cracks
                    - 0.82 * slabs
                    - 1.10 * faulting
                    - 1.55 * potholes
                    + rng.normal(0, 3.2)
                )

                pci = float(np.clip(pci, 5, 99))

                con.execute("""
                    INSERT INTO inspections (
                        sector,
                        date,
                        antiguedad,
                        trafico,
                        grietas,
                        baches,
                        losas_danadas,
                        escalonamiento,
                        pci,
                        observaciones
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sectors[i % len(sectors)],
                    f"2026-{(i % 10) + 1:02d}-{(i % 27) + 1:02d}",
                    age,
                    traffic,
                    cracks,
                    potholes,
                    slabs,
                    faulting,
                    pci,
                    "Registro demostrativo de capacitación"
                ))

            con.commit()


# ==========================================================
# 5. ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING
# ==========================================================

def load_model():

    with db() as con:

        rows = con.execute("""
            SELECT
                antiguedad,
                trafico,
                grietas,
                losas_danadas,
                escalonamiento,
                baches,
                pci
            FROM inspections
        """).fetchall()

    total = len(rows)

    if total < 8:
        return None, None, total

    # Convertir los registros en una matriz numérica.
    data = np.array(
        [list(row) for row in rows],
        dtype=float
    )

    # CORRECCION PRINCIPAL:
    # Las primeras 6 columnas son variables independientes.
    # La columna 7 es el PCI que queremos predecir.

    X = data[:, :6]
    y = data[:, 6]

    # Separación de entrenamiento y evaluación.
    test_size = 0.25

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    # Entrenar modelo de evaluación.
    evaluation_model = LinearRegression()

    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    # Métricas de evaluación.
    metrics = {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)), 2
        ),

        "rmse": round(
            float(np.sqrt(mean_squared_error(y_test, predictions))), 2
        ),

        "r2": round(
            float(r2_score(y_test, predictions)), 3
        ) if len(y_test) >= 2 else None,

        "samples": total,
        "train": len(X_train),
        "test": len(X_test)
    }

    # Modelo definitivo entrenado con todos los registros.
    final_model = LinearRegression()

    final_model.fit(X, y)

    return final_model, metrics, total


# ==========================================================
# 6. VALIDACION DE VARIABLES
# ==========================================================

def validate_features(data):

    values = {}

    for feature in FEATURES:

        value = float(data[feature])

        if not np.isfinite(value) or value < 0:
            raise ValueError()

        values[feature] = value

    # Validaciones según la naturaleza de las variables.
    if values["antiguedad"] > 150:
        raise ValueError()

    if values["grietas"] > 100:
        raise ValueError()

    if values["losas_danadas"] > 100:
        raise ValueError()

    return values


# ==========================================================
# 7. PAGINA PRINCIPAL
# ==========================================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================================
# 8. DASHBOARD
# ==========================================================

@app.get("/api/dashboard")
def dashboard():

    with db() as con:

        rows = con.execute("""
            SELECT *
            FROM inspections
            ORDER BY id DESC
        """).fetchall()

    items = [dict(row) for row in rows]

    pcis = [item["pci"] for item in items]

    average = round(float(np.mean(pcis)), 1) if pcis else 0

    distribution = {
        "Bueno": 0,
        "Satisfactorio": 0,
        "Regular": 0,
        "Malo": 0,
        "Muy malo": 0
    }

    for pci in pcis:

        label, _ = condition(pci)

        distribution[label] += 1

    model, metrics, total = load_model()

    recent = items[:8]

    trend = list(reversed(items[:12]))

    return jsonify({

        "total": len(items),

        "average": average,

        "min": round(min(pcis), 1) if pcis else 0,

        "max": round(max(pcis), 1) if pcis else 0,

        "distribution": distribution,

        "recent": recent,

        "trend": trend,

        "metrics": metrics,

        "model_ready": model is not None

    })


# ==========================================================
# 9. REGISTRAR INSPECCIONES
# ==========================================================

@app.post("/api/inspections")
def create_inspection():

    data = request.get_json(silent=True) or {}

    required = [
        "sector",
        "date",
        *FEATURES,
        "pci"
    ]

    if any(
        key not in data or data[key] in (None, "")
        for key in required
    ):

        return jsonify({
            "error": "Completa todos los campos obligatorios."
        }), 400

    try:

        values = validate_features(data)

        pci = float(data["pci"])

        if not np.isfinite(pci) or not 0 <= pci <= 100:
            raise ValueError()

        sector = str(data["sector"]).strip()[:100]

        if not sector:
            raise ValueError()

        inspection_date = date.fromisoformat(
            str(data["date"])[:10]
        ).isoformat()

        observations = str(
            data.get("observaciones", "")
        )[:500]

        with db() as con:

            con.execute("""
                INSERT INTO inspections (
                    sector,
                    date,
                    antiguedad,
                    trafico,
                    grietas,
                    baches,
                    losas_danadas,
                    escalonamiento,
                    pci,
                    observaciones
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sector,
                inspection_date,
                values["antiguedad"],
                values["trafico"],
                values["grietas"],
                values["baches"],
                values["losas_danadas"],
                values["escalonamiento"],
                pci,
                observations
            ))

            con.commit()

        return jsonify({
            "message": "Inspección registrada correctamente."
        }), 201

    except (ValueError, TypeError, OverflowError):

        return jsonify({
            "error": (
                "Verifica los datos. El PCI debe estar entre 0 y 100, "
                "los porcentajes no pueden superar 100 y los demás "
                "valores deben ser números válidos no negativos."
            )
        }), 400


# ==========================================================
# 10. PREDICCION DEL PCI
# ==========================================================

@app.post("/api/predict")
def predict():

    data = request.get_json(silent=True) or {}

    try:

        values = validate_features(data)

    except (KeyError, ValueError, TypeError, OverflowError):

        return jsonify({
            "error": "Ingresa todas las variables con valores numéricos válidos."
        }), 400

    model, metrics, total = load_model()

    if model is None:

        return jsonify({
            "error": (
                "Se necesitan al menos 8 inspecciones "
                "para entrenar el modelo."
            )
        }), 400

    # Mantener exactamente el orden de FEATURES.
    X_new = np.array([
        [values[feature] for feature in FEATURES]
    ], dtype=float)

    raw_prediction = float(model.predict(X_new)[0])

    pci = round(
        float(np.clip(raw_prediction, 0, 100)),
        1
    )

    label, css = condition(pci)

    action = recommendation(pci)

    return jsonify({

        "pci": pci,

        "condition": label,

        "level": css,

        "recommendation": action,

        "metrics": metrics,

        "training_samples": total,

        "prediction_type": "Regresión lineal",

        "features": FEATURES

    })


# ==========================================================
# 11. EXPORTAR INSPECCIONES A CSV
# ==========================================================

@app.get("/api/export")
def export_csv():

    with db() as con:

        rows = con.execute("""
            SELECT *
            FROM inspections
            ORDER BY date DESC, id DESC
        """).fetchall()

    output = io.StringIO()

    writer = csv.writer(output)

    # Los encabezados coinciden con el orden de los valores.
    writer.writerow([
        "ID",
        "Sector",
        "Fecha",
        "Antiguedad_anios",
        "Trafico_vehiculos_dia",
        "Grietas_porcentaje",
        "Losas_danadas_porcentaje",
        "Escalonamiento_mm",
        "Baches_unidades",
        "PCI",
        "Observaciones"
    ])

    for row in rows:

        writer.writerow([
            row["id"],
            row["sector"],
            row["date"],
            row["antiguedad"],
            row["trafico"],
            row["grietas"],
            row["losas_danadas"],
            row["escalonamiento"],
            row["baches"],
            row["pci"],
            row["observaciones"]
        ])

    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
            "attachment; filename=vialnex_inspecciones.csv"
        }
    )


# ==========================================================
# 12. CONSULTAR UNA INSPECCION
# ==========================================================

@app.get("/api/inspections/<int:inspection_id>")
def get_inspection(inspection_id):

    with db() as con:

        row = con.execute("""
            SELECT *
            FROM inspections
            WHERE id = ?
        """, (inspection_id,)).fetchone()

    if not row:

        return jsonify({
            "error": "Registro no encontrado."
        }), 404

    return jsonify(dict(row))


# ==========================================================
# 13. ELIMINAR UNA INSPECCION
# ==========================================================

@app.delete("/api/inspections/<int:inspection_id>")
def delete_inspection(inspection_id):

    with db() as con:

        cursor = con.execute("""
            DELETE FROM inspections
            WHERE id = ?
        """, (inspection_id,))

        con.commit()

    if cursor.rowcount == 0:

        return jsonify({
            "error": "Registro no encontrado."
        }), 404

    return jsonify({
        "message": "Inspección eliminada correctamente."
    })




# ==========================================================
# 13. ACTUALIZAR INSPECCIONES (CRUD)
# ==========================================================

@app.put("/api/inspections/<int:inspection_id>")
def update_inspection(inspection_id):
    data = request.get_json(silent=True) or {}
    required = ["sector", "date", *FEATURES, "pci"]

    if any(key not in data or data[key] in (None, "") for key in required):
        return jsonify({"error": "Completa todos los campos obligatorios."}), 400

    try:
        values = validate_features(data)
        pci = float(data["pci"])
        if not np.isfinite(pci) or not 0 <= pci <= 100:
            raise ValueError("PCI fuera de rango")

        sector = str(data["sector"]).strip()[:100]
        if not sector:
            raise ValueError("Sector vacío")

        inspection_date = date.fromisoformat(str(data["date"])[:10]).isoformat()
        observations = str(data.get("observaciones", ""))[:500]

        with db() as con:
            cursor = con.execute("""
                UPDATE inspections
                SET sector=?, date=?, antiguedad=?, trafico=?, grietas=?,
                    baches=?, losas_danadas=?, escalonamiento=?, pci=?,
                    observaciones=?
                WHERE id=?
            """, (
                sector, inspection_date, values["antiguedad"],
                values["trafico"], values["grietas"], values["baches"],
                values["losas_danadas"], values["escalonamiento"],
                pci, observations, inspection_id
            ))
            con.commit()

        if cursor.rowcount == 0:
            return jsonify({"error": "Registro no encontrado."}), 404

        return jsonify({"message": "Inspección actualizada correctamente."})

    except (ValueError, TypeError, OverflowError):
        return jsonify({
            "error": "Verifica los datos: PCI de 0 a 100, porcentajes hasta 100 y valores numéricos no negativos."
        }), 400


# ==========================================================
# 14. COMPARACIÓN DE MODELOS DE MACHINE LEARNING
# ==========================================================

@app.get("/api/model-comparison")
def model_comparison():
    with db() as con:
        rows = con.execute("""
            SELECT antiguedad, trafico, grietas, losas_danadas,
                   escalonamiento, baches, pci
            FROM inspections
        """).fetchall()

    if len(rows) < 8:
        return jsonify({
            "ready": False,
            "message": "Se necesitan al menos 8 inspecciones para comparar modelos.",
            "samples": len(rows),
            "models": []
        })

    data = np.asarray([list(row) for row in rows], dtype=float)
    X, y = data[:, :6], data[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    candidates = [
        ("Regresión lineal", LinearRegression()),
        ("Random Forest", RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ]

    results = []
    for name, estimator in candidates:
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        results.append({
            "name": name,
            "mae": round(float(mean_absolute_error(y_test, pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
            "r2": round(float(r2_score(y_test, pred)), 3) if len(y_test) >= 2 else None
        })

    # El menor MAE se presenta como mejor resultado de esta partición,
    # sin asumir que será el mejor con datos nuevos.
    best = min(results, key=lambda item: item["mae"])
    return jsonify({
        "ready": True,
        "samples": len(rows),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "selection_metric": "MAE",
        "best_on_holdout": best["name"],
        "models": results,
        "note": "Comparación exploratoria con una única partición train/test. Validar con más datos y validación cruzada antes de usar para decisiones de mantenimiento."
    })


# ==========================================================
# 15. ESTADO DEL SERVIDOR
# ==========================================================

@app.get("/api/health")
def health():
    try:
        with db() as con:
            con.execute("SELECT 1").fetchone()
        return jsonify({"status": "ok", "database": "connected"})
    except sqlite3.Error:
        return jsonify({"status": "error", "database": "unavailable"}), 503


# ==========================================================
# 16. INICIAR VIALNEX INTELLIGENCE
# ==========================================================

init_db()


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       VIALNEX INTELLIGENCE")
    print("       El futuro de las vías comienza con datos.")
    print("=" * 55)
    print("\nServidor: http://127.0.0.1:5000\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("VIALNEX_DEBUG", "0") == "1"
    )
