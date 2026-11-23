
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
RESULT_PATH = PROJECT_FOLDER / "outputs" / "model_comparison.csv"
CHART_PATH = PROJECT_FOLDER / "outputs" / "model_comparison.png"

# Load data
df = pd.read_csv(DATA_PATH, keep_default_na=False)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time")

# Features and target
X = df.drop(columns=["traffic_volume", "date_time"])
y = df["traffic_volume"]

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

# Chronological 80:20 split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ("numerical", "passthrough", numerical_columns)
])

# Models
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

print("HOURLY MACHINE LEARNING MODEL COMPARISON")
print("=" * 55)

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 4))

# Save comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

results_df.to_csv(RESULT_PATH, index=False)

print("\nFINAL MODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))

# Plot comparison
plt.figure(figsize=(10, 6))

plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Hourly Traffic Prediction - RMSE Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(CHART_PATH, dpi=300)
plt.close()

print("\nComparison saved:", RESULT_PATH)
print("Chart saved:", CHART_PATH)
print("\nModel comparison completed!")
