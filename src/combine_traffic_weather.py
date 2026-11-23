
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

traffic_file = OUTPUT_DIR / "live_traffic_multi_location.csv"
weather_file = OUTPUT_DIR / "live_weather_data.csv"

traffic = pd.read_csv(traffic_file)
weather = pd.read_csv(weather_file)

# Convert timestamps to UTC
traffic["timestamp_utc"] = pd.to_datetime(
    traffic["timestamp_utc"], utc=True, errors="coerce"
)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"], utc=True, errors="coerce"
)

# Remove invalid timestamps
traffic = traffic.dropna(subset=["timestamp_utc"])
weather = weather.dropna(subset=["timestamp"])

# Sort before matching
traffic = traffic.sort_values("timestamp_utc")
weather = weather.sort_values("timestamp")

# Match nearest weather observation
combined = pd.merge_asof(
    traffic,
    weather,
    left_on="timestamp_utc",
    right_on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min"),
    suffixes=("_traffic", "_weather")
)

output_file = OUTPUT_DIR / "combined_live_data.csv"
combined.to_csv(output_file, index=False)

print("=" * 50)
print("MULTI-LOCATION TRAFFIC AND WEATHER DATA")
print("=" * 50)

print("Traffic records:", len(traffic))
print("Weather records:", len(weather))
print("Combined records:", len(combined))
print("Matched weather records:", combined["temperature_c"].notna().sum())
print("Unmatched records:", combined["temperature_c"].isna().sum())

print("\nRecords by location:")
print(combined["location_name"].value_counts())

print("\nCombined Data Preview:")
print(combined[[
    "timestamp_utc",
    "location_name",
    "current_speed_kmph",
    "free_flow_speed_kmph",
    "temperature_c",
    "humidity_percent",
    "rain_mm"
]].to_string(index=False))

print("\nOutput:", output_file)
print("=" * 50)
