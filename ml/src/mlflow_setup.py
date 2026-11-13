
from pathlib import Path

import mlflow


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_DIR = PROJECT_ROOT / "ml"

TRACKING_DB = ML_DIR / "mlflow.db"


# --------------------------------------------------
# 2. Configure SQLite tracking backend
# --------------------------------------------------

tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"

mlflow.set_tracking_uri(tracking_uri)


# --------------------------------------------------
# 3. Create or select experiment
# --------------------------------------------------

EXPERIMENT_NAME = "IntelliChurn-Model-Experiments"

experiment = mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 4. Display configuration
# --------------------------------------------------

print("\nMLflow configuration completed!")

print(f"Experiment name: {EXPERIMENT_NAME}")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Tracking database: {TRACKING_DB}")

print("\nSQLite tracking backend is ready.")
