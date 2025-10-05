# app.py - FastAPI ML inference service
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os, numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "models/eco_model.pkl")
app = FastAPI(title="EcoRoute ML Service")

# Vehicle-specific assumptions (can be tuned with real-world data)
VEHICLE_FACTORS = {
    "car": {"weight": 1200, "co2_factor": 1.0},     # baseline
    "van": {"weight": 2500, "co2_factor": 1.4},     # heavier → higher CO₂
    "bike": {"weight": 200, "co2_factor": 0.2},     # very low CO₂
    "ev": {"weight": 1800, "co2_factor": 0.0},      # no tailpipe CO₂
}

class Features(BaseModel):
    distance_km: float
    elevation_gain_m: float = 0.0
    avg_speed_kph: float = 50.0
    turns: int = 0
    humps: int = 0
    weight_kg: float = 1000.0
    traffic_index: float = 1.0
    route_type: str = "fast"
    vehicle: str = "car"   # NEW: include vehicle type

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model missing at {MODEL_PATH}. Run train_model.py to create it.")
    model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(feat: Features):
    # Encode route type
    route_type_val = 0 if feat.route_type == "fast" else 1

    # Vehicle adjustments
    v = VEHICLE_FACTORS.get(feat.vehicle, VEHICLE_FACTORS["car"])
    base_weight = v["weight"]
    co2_factor = v["co2_factor"]

    X = np.array([[ 
        feat.distance_km,
        feat.elevation_gain_m,
        feat.avg_speed_kph,
        feat.turns,
        feat.humps,
        base_weight,               # override weight by vehicle
        feat.traffic_index,
        route_type_val
    ]])

    # Model predicts baseline [fuel_l, co2_kg]
    pred = model.predict(X)[0]
    fuel, co2 = float(pred[0]), float(pred[1])

    # Adjust CO₂ depending on vehicle
    co2 *= co2_factor

    # Special case: EV → report energy consumption in kWh instead of liters
    if feat.vehicle == "ev":
        # Convert fuel_l baseline into energy (assume 0.2 kWh/km)
        energy_kwh = feat.distance_km * 0.2
        return {"fuel_l": 0.0, "energy_kwh": energy_kwh, "co2_kg": co2}

    return {"fuel_l": fuel, "co2_kg": co2}
