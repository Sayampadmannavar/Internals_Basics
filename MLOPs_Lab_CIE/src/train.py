import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# FIX PATH (important)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "training_data.csv")

df = pd.read_csv(data_path)

X = df.drop("completion_days", axis=1)
y = df["completion_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("edutrack-completion-days")

def evaluate(model):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    return mae, rmse, r2, mape

results = []
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mae, rmse, r2, mape = evaluate(model)

        mlflow.log_param("model_name", name)
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })
        mlflow.set_tag("domain", "edtech")

        mlflow.sklearn.log_model(model, "model")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "edutrack-completion-days",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

results_path = os.path.join(BASE_DIR, "results")
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, "step1_s1.json"), "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 1 DONE")