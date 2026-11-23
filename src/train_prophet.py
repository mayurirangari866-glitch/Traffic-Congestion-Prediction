
import pandas as pd
import joblib

from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
MODEL_PATH = PROJECT_FOLDER / "models" / "traffic_prophet.pkl"
RESULT_PATH = PROJECT_FOLDER / "outputs" / "prophet_results.csv"

# Load historical data
df = pd.read_csv(DATA_PATH, keep_default_na=False)

df["date_time"] = pd.to_datetime(df["date_time"])

# Prepare daily average traffic
daily_traffic = (
    df.groupby(df["date_time"].dt.date)["traffic_volume"]
    .mean()
    .reset_index()
)

daily_traffic.columns = ["ds", "y"]
daily_traffic["ds"] = pd.to_datetime(daily_traffic["ds"])

# Chronological split
split_index = int(len(daily_traffic) * 0.8)

train = daily_traffic.iloc[:split_index]
test = daily_traffic.iloc[split_index:]

print("PROPHET MODEL TRAINING")
print("=" * 50)

print("Training records:", len(train))
print("Testing records:", len(test))

# Train Prophet
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

print("\nTraining Prophet...")
model.fit(train)

# Forecast test period
future = test[["ds"]].copy()
forecast = model.predict(future)

predictions = forecast["yhat"].clip(lower=0)

# Evaluation
mae = mean_absolute_error(test["y"], predictions)
rmse = mean_squared_error(test["y"], predictions) ** 0.5
r2 = r2_score(test["y"], predictions)

print("\nPROPHET EVALUATION RESULTS")
print("-" * 40)
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model and results
joblib.dump(model, MODEL_PATH)

results = pd.DataFrame({
    "Date": test["ds"].values,
    "Actual": test["y"].values,
    "Predicted": predictions.values
})

results.to_csv(RESULT_PATH, index=False)

print("\nProphet model saved:", MODEL_PATH)
print("Forecast results saved:", RESULT_PATH)
print("\nProphet training completed!")
