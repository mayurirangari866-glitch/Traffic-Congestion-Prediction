
import pandas as pd
import joblib

from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
MODEL_PATH = PROJECT_FOLDER / "models" / "traffic_arima.pkl"
RESULT_PATH = PROJECT_FOLDER / "outputs" / "arima_results.csv"

# Load dataset
df = pd.read_csv(DATA_PATH, keep_default_na=False)

# Prepare chronological traffic time series
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time")

# Convert hourly observations into daily average traffic
daily_traffic = (
    df.set_index("date_time")["traffic_volume"]
    .resample("D")
    .mean()
    .interpolate()
)

# Chronological 80:20 split
split_index = int(len(daily_traffic) * 0.8)

train = daily_traffic.iloc[:split_index]
test = daily_traffic.iloc[split_index:]

print("ARIMA MODEL TRAINING")
print("=" * 50)

print("Training records:", len(train))
print("Testing records:", len(test))

# Train ARIMA
print("\nTraining ARIMA(2,1,2)...")

model = ARIMA(train, order=(2, 1, 2))
fitted_model = model.fit()

# Forecast test period
predictions = fitted_model.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test, predictions)
rmse = mean_squared_error(test, predictions) ** 0.5
r2 = r2_score(test, predictions)

print("\nARIMA EVALUATION RESULTS")
print("-" * 40)
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model and results
joblib.dump(fitted_model, MODEL_PATH)

results = pd.DataFrame({
    "Actual": test.values,
    "Predicted": predictions.values
})

results.to_csv(RESULT_PATH, index=False)

print("\nARIMA model saved:", MODEL_PATH)
print("Forecast results saved:", RESULT_PATH)
print("\nARIMA training completed!")
