
import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
OUTPUT_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"

df = pd.read_csv(DATA_PATH, keep_default_na=False)

print("HOLIDAY MISSING VALUE CORRECTION")
print("=" * 50)

print("Missing values before:", df.isnull().sum().sum())

df["holiday"] = df["holiday"].replace("", "None")

print("Missing values after:", df.isnull().sum().sum())

print("\nHoliday distribution:")
print(df["holiday"].value_counts().head(10))

df.to_csv(OUTPUT_PATH, index=False)

print("\nCorrected dataset saved successfully!")
