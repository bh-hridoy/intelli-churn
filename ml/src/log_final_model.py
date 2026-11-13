
from pathlib import Path
import json
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"

TRAIN_PATH = ML_DIR / "data" / "processed" / "splits" / "train.csv"
TEST_PATH = ML_DIR / "data" / "processed" / "splits" / "test.csv"

TRACKING_DB = ML_DIR / "mlflow.db"

TARGET = "Churn Value"
THRESHOLD = 0.30


# --------------------------------------------------
# 2. Configure MLflow
# --------------------------------------------------

tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("IntelliChurn-Model-Experiments")


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# --------------------------------------------------
# 4. Build final model pipeline
# --------------------------------------------------

print("\nBuilding final Random Forest pipeline...")

pipeline = Pipeline(
    steps=[
        ("preprocessing", build_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# --------------------------------------------------
# 5. Train and log final model
# --------------------------------------------------

with mlflow.start_run(
    run_name="Final Random Forest Model"
):

    print("\nTraining final model...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Generate test probabilities and predictions
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Calculate final test metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
    }

    cm = confusion_matrix(y_test, predictions)

    tn, fp, fn, tp = cm.ravel()

    # Log parameters
    mlflow.log_params({
        "model_name": "Random Forest",
        "model_status": "Final Selected Model",
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "decision_threshold": THRESHOLD,
        "threshold_selection_method": "Maximum validation F1",
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "input_features": len(X_train.columns),
    })

    # Log metrics
    mlflow.log_metrics(metrics)

    mlflow.log_metrics({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    })

    # Add identifying tags
    mlflow.set_tags({
        "project": "IntelliChurn",
        "model_role": "final_model",
        "algorithm": "Random Forest",
        "dataset": "IBM Telco Customer Churn",
    })

    # --------------------------------------------------
    # 6. Log trained model
    # --------------------------------------------------

       # --------------------------------------------------
    # 6. Log trained model
    # --------------------------------------------------

    print("\nLogging trained model to MLflow...")

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="random_forest_pipeline",
        input_example=X_train.head(5),
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
        ],
    )

    # --------------------------------------------------
    # 7. Save metadata artifact
    # --------------------------------------------------

    metadata = {
        "project": "IntelliChurn",
        "model_name": "Random Forest",
        "model_type": "RandomForestClassifier",
        "model_status": "Final Selected Model",
        "decision_threshold": THRESHOLD,
        "threshold_selection_method": "Maximum validation F1",
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "input_features": list(X_train.columns),
        "parameters": {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }

    with tempfile.TemporaryDirectory() as temp_dir:

        metadata_path = Path(temp_dir) / "model_metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="model_metadata",
        )

    print("\nFinal model logged successfully!")

    print("\nFinal model metrics:")

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print(f"\nConfusion matrix:\n{cm}")

    print(f"\nMLflow model URI: {model_info.model_uri}")


print("\nStage 14.5 completed successfully!")
