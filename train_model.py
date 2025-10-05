# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib, os

# Example synthetic dataset generator (replace with real data if you have it)
def generate_data(n=2000):
    np.random.seed(42)
    data = []
    for i in range(n):
        distance_km = np.random.uniform(5, 1000)
        elevation_gain_m = np.random.uniform(0, 500)
        avg_speed_kph = np.random.uniform(20, 100)
        turns = np.random.randint(5, 200)
        humps = np.random.randint(0, 50)
        weight_kg = np.random.uniform(800, 3000)
        traffic_index = np.random.uniform(0.5, 2.0)
        route_type = np.random.choice(["fast", "eco"])  # NEW FEATURE

        # Baseline fuel/CO₂
        fuel = (distance_km / 12) + (weight_kg / 1000) * 0.5
        co2 = fuel * 2.31

        # Adjust depending on route_type
        if route_type == "eco":
            fuel *= 0.9 + np.random.uniform(-0.05, 0.05)
            co2 *= 0.9 + np.random.uniform(-0.05, 0.05)
        else:  # fast
            fuel *= 1.05 + np.random.uniform(-0.05, 0.05)
            co2 *= 1.05 + np.random.uniform(-0.05, 0.05)

        data.append([distance_km, elevation_gain_m, avg_speed_kph,
                     turns, humps, weight_kg, traffic_index,
                     0 if route_type=="fast" else 1,  # encode categorical
                     fuel, co2])
    cols = ["distance_km","elevation_gain_m","avg_speed_kph","turns","humps",
            "weight_kg","traffic_index","route_type","fuel_l","co2_kg"]
    return pd.DataFrame(data, columns=cols)

# Generate synthetic dataset
df = generate_data(5000)

X = df.drop(["fuel_l","co2_kg"], axis=1)
y = df[["fuel_l","co2_kg"]]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train ML model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model R^2:", model.score(X_test, y_test))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/eco_model.pkl")
print("✅ Model saved to models/eco_model.pkl")
