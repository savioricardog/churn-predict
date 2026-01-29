import mlflow
import shutil
import os


# CONFIG MLFLOW
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
EXPERIMENT_ID = "1"
MODEL_NAME_IN_MLFLOW = "churn_model_calibrated_prod"

print(f"Connecting to MLFLOW in {MLFLOW_URI}")
mlflow.set_tracking_uri(MLFLOW_URI)

print("Searching for most recent model")
runs = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])

if runs.empty:
    raise Exception("❌ Anyone Run located! Did you run train_pipeline.py?")

last_run_id = runs.sort_values("start_time", ascending=False).iloc[0].run_id
print(f"Last run id found: {last_run_id}")

model_uri = f"runs:/{last_run_id}/{MODEL_NAME_IN_MLFLOW}"

local_path = "model_prod"

if os.path.exists(local_path):
    shutil.rmtree(local_path)

print(f"Downloading model from {model_uri} to folder {local_path}...")
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_path)

print("Success! Model prod downloaded")