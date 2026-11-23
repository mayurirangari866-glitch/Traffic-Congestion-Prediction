
from pathlib import Path
import joblib
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model_files = [
    "traffic_volume_model.pkl",
    "traffic_arima.pkl",
    "traffic_prophet.pkl",
    "lstm_scaler.pkl"
]

print("=" * 45)
print("FINAL MODEL FILE VERIFICATION")
print("=" * 45)

for filename in model_files:
    path = MODEL_DIR / filename

    if not path.exists():
        print(filename, "MISSING")
        continue

    try:
        joblib.load(path)
        print(filename, "LOAD SUCCESS")
    except Exception as error:
        print(filename, "LOAD FAILED:", error)

lstm_path = MODEL_DIR / "traffic_lstm.keras"

if lstm_path.exists():
    try:
        tf.keras.models.load_model(lstm_path)
        print("traffic_lstm.keras LOAD SUCCESS")
    except Exception as error:
        print("traffic_lstm.keras LOAD FAILED:", error)
else:
    print("traffic_lstm.keras MISSING")

print("=" * 45)
