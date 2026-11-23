
import pandas as pd
from pathlib import Path

# Locate the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "dataset" / "Metro_Interstate_Traffic_Volume.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display basic information
print("TRAFFIC CONGESTION PREDICTION PROJECT")
print("-------------------------------------")

print("Dataset loaded successfully!")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

print("\nTRAFFIC VOLUME SUMMARY:")
print("Minimum:", df["traffic_volume"].min())
print("Maximum:", df["traffic_volume"].max())
print("Average:", df["traffic_volume"].mean())

# FEATURE ENGINEERING

print("\nFEATURE ENGINEERING STARTED")

# Convert date_time into datetime format
df["date_time"] = pd.to_datetime(df["date_time"])

# Extract time-based features
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.day
df["month"] = df["date_time"].dt.month
df["day_of_week"] = df["date_time"].dt.dayofweek

# Weekend feature
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Handle missing holiday values
df["holiday"] = df["holiday"].fillna("None")

# Remove original datetime column

# Remove duplicate records
df = df.drop_duplicates()
df = df.sort_values("date_time").reset_index(drop=True)
# Save processed dataset
output_path = project_folder / "outputs" / "processed_traffic.csv"
df.to_csv(output_path, index=False)

print("Feature engineering completed!")
print("New dataset shape:", df.shape)
print("Updated columns:")
print(df.columns.tolist())
print("\nProcessed dataset saved at:")
print(output_path)
print("\nDATASET VALIDATION")

print("Duplicate rows:", df.duplicated().sum())

print("\nRainfall values:")
print(df["rain_1h"].describe())

print("\nDate range:")
print(df["date_time"].min())
print(df["date_time"].max())

print("\nTraffic volume distribution:")
print(df["traffic_volume"].describe())