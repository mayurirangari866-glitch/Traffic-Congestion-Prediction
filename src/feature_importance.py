import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

data_path = project_folder / "outputs" / "processed_traffic.csv"
model_path = project_folder / "models" / "traffic_volume_model.pkl"

df = pd.read_csv(data_path)

X = df.drop(columns=["traffic_volume", "date_time"])

model = joblib.load(model_path)

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
importances = model.named_steps["regressor"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
top_features = importance_df.head(10).sort_values("Importance")

plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.title("Top 10 Traffic Prediction Features")
plt.tight_layout()

output_path = project_folder / "outputs" / "feature_importance.png"
plt.savefig(output_path)

print("\nGraph saved at:", output_path)

plt.show()