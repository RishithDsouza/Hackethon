from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
CORS(app)

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "aadhaar_service_data.csv")

# ================= LOAD DATA =================
def load_data():
    global df
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])

load_data()

# ================= HELPERS =================
def filtered_data(state=None, district=None):
    data = df.copy()
    if state:
        data = data[data["state"] == state]
    if district:
        data = data[data["district"] == district]
    return data

# ================= ROUTES =================

@app.route("/")
def home():
    return jsonify({"status": "Aadhaar Intelligence System API Running"})

# ---------- SUMMARY ----------
@app.route("/summary")
def summary():
    data = filtered_data(request.args.get("state"), request.args.get("district"))
    return jsonify({
        "total_enrolments": int(data["new_enrolments"].sum()),
        "total_updates": int(data["update_requests"].sum()),
        "total_failures": int(data["failures"].sum()),
        "records": len(data)
    })

# ---------- KPIs ----------
@app.route("/kpis")
def kpis():
    data = filtered_data(request.args.get("state"), request.args.get("district"))

    enrol = data["new_enrolments"].sum()
    updates = data["update_requests"].sum()
    failures = data["failures"].sum()

    load_ratio = (updates / (data["operators"] * data["service_hours"])).mean()
    failure_rate = (failures / max(enrol + updates, 1)) * 100

    return jsonify({
        "avg_daily_enrolments": round(enrol / max(data["date"].nunique(), 1), 2),
        "failure_rate": round(failure_rate, 2),
        "avg_service_load": round(load_ratio, 2)
    })

# ---------- BAR DATA ----------
@app.route("/bar-data")
def bar_data():
    state = request.args.get("state")
    if state:
        grp = df[df["state"] == state].groupby("district")["new_enrolments"].sum()
    else:
        grp = df.groupby("state")["new_enrolments"].sum()

    return jsonify(grp.sort_values(ascending=False).to_dict())

# ---------- HEATMAP DATA ----------
@app.route("/heatmap-data")
def heatmap_data():
    return jsonify(df.groupby("state")["new_enrolments"].sum().to_dict())

# ---------- TIME SERIES ----------
@app.route("/timeseries")
def timeseries():
    data = filtered_data(request.args.get("state"), request.args.get("district"))
    ts = data.groupby("date")["new_enrolments"].sum().sort_index()
    return jsonify({"dates": ts.index.astype(str).tolist(), "values": ts.values.tolist()})

# ---------- SERVICE LOAD ----------
@app.route("/service-load")
def service_load():
    data = filtered_data(request.args.get("state"), request.args.get("district"))
    load = data.groupby("date").apply(
        lambda x: x["update_requests"].sum() / max(x["operators"].sum(), 1)
    )
    return jsonify({"dates": load.index.astype(str).tolist(), "values": load.values.tolist()})

# ---------- DISTRIBUTION ----------
@app.route("/distribution")
def distribution():
    data = filtered_data(request.args.get("state"), request.args.get("district"))
    return jsonify({
        "enrolments": int(data["new_enrolments"].sum()),
        "updates": int(data["update_requests"].sum())
    })

# ---------- FORECAST (WITH CONFIDENCE BAND) ----------
@app.route("/forecast")
def forecast():
    data = filtered_data(request.args.get("state"), request.args.get("district"))
    ts = data.groupby("date")["new_enrolments"].sum().sort_index()

    if len(ts) < 5:
        return jsonify({"dates": [], "forecast": [], "upper": [], "lower": []})

    X = np.arange(len(ts)).reshape(-1, 1)
    y = ts.values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(ts), len(ts) + 30).reshape(-1, 1)
    forecast = model.predict(future_X)

    return jsonify({
        "dates": pd.date_range(ts.index[-1], periods=31, freq="D")[1:].astype(str).tolist(),
        "forecast": forecast.round().astype(int).tolist(),
        "upper": (forecast * 1.15).round().astype(int).tolist(),
        "lower": (forecast * 0.85).round().astype(int).tolist()
    })

# ---------- TIME SERIES ANOMALIES ----------
@app.route("/timeseries-anomalies")
def timeseries_anomalies():
    data = filtered_data(request.args.get("state"), request.args.get("district")).copy()
    data["load_ratio"] = data["update_requests"] / (data["operators"] * data["service_hours"])

    daily = data.groupby("date")["load_ratio"].mean().reset_index()
    if len(daily) < 10:
        return jsonify({"dates": [], "values": []})

    model = IsolationForest(contamination=0.1, random_state=42)
    daily["anomaly"] = model.fit_predict(daily[["load_ratio"]])

    anomalies = daily[daily["anomaly"] == -1]
    return jsonify({
        "dates": anomalies["date"].astype(str).tolist(),
        "values": anomalies["load_ratio"].round(2).tolist()
    })

# ---------- INSIGHTS ----------
@app.route("/insights")
def insights():
    data = filtered_data(request.args.get("state"), request.args.get("district"))
    enrol, updates, failures = (
        data["new_enrolments"].sum(),
        data["update_requests"].sum(),
        data["failures"].sum()
    )

    return jsonify([
        "Update demand exceeds enrolments." if updates > enrol else "Enrolments dominate demand.",
        "Failure rate is high." if failures / max(enrol + updates, 1) > 0.05 else "Failure rate acceptable.",
        "Operational load is high." if (updates / (data["operators"] * data["service_hours"])).mean() > 1.2 else "Operational load balanced."
    ])

# ---------- DISTRICTS ----------
@app.route("/districts")
def districts():
    state = request.args.get("state")
    if not state:
        return jsonify([])
    return jsonify(sorted(df[df["state"] == state]["district"].dropna().unique()))

if __name__ == "__main__":
    app.run(debug=True)
