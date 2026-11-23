
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
MODEL_PATH = PROJECT_FOLDER / "models" / "traffic_svr.pkl"
RESULT_PATH = PROJECT_FOLDER / "outputs" / "svr_results.csv"

# Load data
df = pd.read_csv(DATA_PATH, keep_default_na=False)
df["date_time"] = pd.to_datetime(df["date_time"])

# Sort chronologically
df = df.sort_values("date_time")

# Select features and target
X = df.drop(columns=["traffic_volume", "date_time"])
y = df["traffic_volume"]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

# Chronological train-test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("SUPPORT VECTOR REGRESSION TRAINING")
print("=" * 50)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ("numerical", StandardScaler(), numerical_columns)
])

# SVR pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", SVR(kernel="rbf", C=100, epsilon=0.1))
])

print("\nTraining SVR...")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test).clip(min=0)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nSVR EVALUATION RESULTS")
print("-" * 40)
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save model
joblib.dump(model, MODEL_PATH)

# Save predictions
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

results.to_csv(RESULT_PATH, index=False)

print("\nSVR model saved:", MODEL_PATH)
print("Prediction results saved:", RESULT_PATH)
print("\nSVR training completed!")
