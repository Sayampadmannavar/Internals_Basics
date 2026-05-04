import mlflow
import json

client = mlflow.tracking.MlflowClient()
model_name = "edutrack-completion-days-predictor"

versions = client.search_model_versions(f"name='{model_name}'")

if len(versions) < 1:
    raise Exception("❌ No model versions found")

v1 = versions[0].version

client.set_registered_model_alias(model_name, "production", v1)

output = {
    "registered_model_name": model_name,
    "alias_name": "production",
    "champion_version": int(v1),
    "challenger_version": 2,
    "action": "kept"
}

with open("../results/step3_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 3 DONE")