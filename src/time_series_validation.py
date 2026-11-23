
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
RESULT_PATH = PROJECT_FOLDER / "outputs" / "time_series_validation.csv"

# Load historical data
df = pd.read_csv(DATA_PATH, keep_default_na=False)
df["date_time"] = pd.to_datetime(df["date_time"])
df = df.sort_values("date_time").reset_index(drop=True)

# Features and target
X = df.drop(columns=["traffic_volume", "date_time"])
y = df["traffic_volume"]

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

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

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

results = []

print("TIME SERIES CROSS-VALIDATION")
print("=" * 55)
print("Number of folds:", tscv.get_n_splits())

for name, algorithm in models.items():

    print(f"\nValidating {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring=scoring,
        n_jobs=1
    )

    mae_scores = -scores["test_MAE"]
    rmse_scores = -scores["test_RMSE"]
    r2_scores = scores["test_R2"]

    results.append({
        "Model": name,
        "Mean MAE": mae_scores.mean(),
        "Mean RMSE": rmse_scores.mean(),
        "Mean R2": r2_scores.mean(),
        "MAE Std": mae_scores.std(),
        "RMSE Std": rmse_scores.std()
    })

    print("Mean MAE:", round(mae_scores.mean(), 2))
    print("Mean RMSE:", round(rmse_scores.mean(), 2))
    print("Mean R2:", round(r2_scores.mean(), 4))

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Mean RMSE")

results_df.to_csv(RESULT_PATH, index=False)

print("\nFINAL CROSS-VALIDATION RESULTS")
print("=" * 55)
print(results_df.to_string(index=False))

print("\nResults saved:", RESULT_PATH)
print("\nTime series validation completed!")
