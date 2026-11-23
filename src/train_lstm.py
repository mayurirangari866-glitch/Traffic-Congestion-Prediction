
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
MODEL_PATH = PROJECT_FOLDER / "models" / "traffic_lstm.keras"
SCALER_PATH = PROJECT_FOLDER / "models" / "lstm_scaler.pkl"
RESULT_PATH = PROJECT_FOLDER / "outputs" / "lstm_results.csv"

# Load and sort historical data
df = pd.read_csv(DATA_PATH, keep_default_na=False)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time")

# Use hourly traffic volume
traffic = df.set_index("date_time")["traffic_volume"]
traffic = traffic.resample("h").mean().interpolate()

values = traffic.values.reshape(-1, 1)

# Chronological train-test split
split_index = int(len(values) * 0.8)
train_values = values[:split_index]
test_values = values[split_index:]

# Scale using training data only
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_values)
test_scaled = scaler.transform(test_values)

# Create sequences of previous 24 hours
LOOKBACK = 24

def create_sequences(data, lookback):
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])

    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, LOOKBACK)

# Include previous training observations for the first test sequence
combined_test = np.vstack([train_scaled[-LOOKBACK:], test_scaled])
X_test, y_test = create_sequences(combined_test, LOOKBACK)

X_train = X_train.reshape(X_train.shape[0], LOOKBACK, 1)
X_test = X_test.reshape(X_test.shape[0], LOOKBACK, 1)

print("LSTM TRAFFIC PREDICTION MODEL")
print("=" * 50)
print("Training sequences:", len(X_train))
print("Testing sequences:", len(X_test))
print("Lookback window:", LOOKBACK, "hours")

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("\nTraining LSTM...")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# Predict and convert back to vehicle counts
predicted_scaled = model.predict(X_test, verbose=0)

predictions = scaler.inverse_transform(predicted_scaled).flatten()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

predictions = np.maximum(predictions, 0)

# Evaluation
mae = mean_absolute_error(actual, predictions)
rmse = mean_squared_error(actual, predictions) ** 0.5
r2 = r2_score(actual, predictions)

print("\nLSTM EVALUATION RESULTS")
print("-" * 40)
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model, scaler, and predictions
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

results = pd.DataFrame({
    "Actual": actual,
    "Predicted": predictions
})

results.to_csv(RESULT_PATH, index=False)

print("\nLSTM model saved:", MODEL_PATH)
print("Scaler saved:", SCALER_PATH)
print("Results saved:", RESULT_PATH)
print("\nLSTM training completed!")
