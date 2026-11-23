
import pandas as pd
from pathlib import Path

# Load live traffic data
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "outputs" / "live_traffic_data.csv"

df = pd.read_csv(file_path)

if df.empty:
    print("No live traffic data available.")
    raise SystemExit

# Calculate speed reduction
df["speed_reduction_percent"] = (
    (df["free_flow_speed_kmph"] - df["current_speed_kmph"])
    / df["free_flow_speed_kmph"].replace(0, float("nan"))
) * 100

def detect_congestion(row):
    if row["road_closure"]:
        return "Road Closed"

    reduction = row["speed_reduction_percent"]

    if pd.isna(reduction):
        return "Insufficient Data"
    elif reduction <= 10:
        return "Low Congestion"
    elif reduction <= 30:
        return "Moderate Congestion"
    elif reduction <= 50:
        return "High Congestion"
    else:
        return "Severe Congestion"

df["congestion_level"] = df.apply(detect_congestion, axis=1)

# Display latest reading
latest = df.iloc[-1]

print("=" * 45)
print("LIVE TRAFFIC CONGESTION DETECTION")
print("=" * 45)
print("Location:", latest["latitude"], latest["longitude"])
print("Current Speed:", latest["current_speed_kmph"], "km/h")
print("Free-flow Speed:", latest["free_flow_speed_kmph"], "km/h")
print("Speed Reduction:", round(latest["speed_reduction_percent"], 2), "%")
print("Congestion Level:", latest["congestion_level"])
print("=" * 45)
