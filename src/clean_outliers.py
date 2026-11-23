
import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_FOLDER / "outputs" / "processed_traffic.csv"
OUTPUT_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"

df = pd.read_csv(INPUT_PATH)

print("RAINFALL OUTLIER TREATMENT")
print("=" * 50)

median_rainfall = df["rain_1h"].median()

suspicious = df["rain_1h"] > 100

print("Suspicious records:", suspicious.sum())
print("Median rainfall:", median_rainfall)

df.loc[suspicious, "rain_1h"] = median_rainfall

print("Corrected records:", suspicious.sum())
print("New maximum rainfall:", df["rain_1h"].max())

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print(OUTPUT_PATH)
