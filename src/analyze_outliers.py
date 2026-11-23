
import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_FOLDER
    / "dataset"
    / "Metro_Interstate_Traffic_Volume.csv"
)

df = pd.read_csv(DATA_PATH)

print("EXTREME RAINFALL VERIFICATION")
print("=" * 50)

result = df[df["rain_1h"] > 100]

print("\nRecords with rainfall above 100 mm:")
print(result[
    ["date_time", "rain_1h", "traffic_volume"]
].to_string(index=False))

print("\nNumber of records:", len(result))

print("\nVerification completed.")

