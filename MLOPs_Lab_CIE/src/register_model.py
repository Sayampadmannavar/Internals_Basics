import mlflow
import json

model_name = "edutrack-completion-days-predictor"

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("edutrack-completion-days")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"]
)

if len(runs) == 0:
    raise Exception("❌ No runs found. Run train.py first")

best_run = runs[0]

model_uri = f"runs:/{best_run.info.run_id}/model"

result = mlflow.register_model(model_uri, model_name)

output = {
    "registered_model_name": model_name,
    "version": result.version,
    "run_id": best_run.info.run_id,
    "source_metric": "rmse",
    "source_metric_value": best_run.data.metrics["rmse"]
}

with open("../results/step2_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 2 DONE")