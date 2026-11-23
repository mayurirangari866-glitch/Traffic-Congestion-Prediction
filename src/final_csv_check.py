
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "outputs" / "cleaned_traffic.csv"

df = pd.read_csv(file_path, keep_default_na=False)

print("=" * 45)
print("HISTORICAL DATA QUALITY FINAL CHECK")
print("=" * 45)

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Missing values:", int(df.isna().sum().sum()))
print("Holiday None entries:", (df["holiday"] == "None").sum())

print("\nDuplicate records:", df.duplicated().sum())

print("\nDate range:")
print(df["date_time"].min(), "to", df["date_time"].max())

print("\nVerification complete.")
print("=" * 45)
