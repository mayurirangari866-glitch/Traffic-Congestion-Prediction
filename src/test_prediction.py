import pandas as pd
import joblib

from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "models" / "traffic_volume_model.pkl"

model = joblib.load(model_path)

# Sample traffic conditions
sample_data = pd.DataFrame([{
    "holiday": "None",
    "temp": 288.5,
    "rain_1h": 0.0,
    "snow_1h": 0.0,
    "clouds_all": 40,
    "weather_main": "Clouds",
    "weather_description": "scattered clouds",
    "hour": 8,
    "day": 23,
    "month": 11,
    "day_of_week": 0,
    "is_weekend": 0
}])

prediction = model.predict(sample_data)

print("\nTRAFFIC VOLUME PREDICTION")
print("-------------------------")
print("Predicted Traffic Volume:", round(prediction[0]), "vehicles")

if prediction[0] < 2000:
    level = "Low"
elif prediction[0] < 4000:
    level = "Medium"
else:
    level = "High"

print("Traffic Congestion Level:", level)