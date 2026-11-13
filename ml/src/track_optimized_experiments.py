
from pathlib import Path
import tempfile
import json

import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
# 1. Project paths and MLflow configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"

TRAIN_PATH = ML_DIR / "data" / "processed" / "splits" / "train.csv"
TEST_PATH = ML_DIR / "data" / "processed" / "splits" / "test.csv"

TRACKING_DB = ML_DIR / "mlflow.db"

tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("IntelliChurn-Model-Experiments")


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

TARGET = "Churn Value"
RANDOM_STATE = 42

# These thresholds were selected using validation data.
optimized_thresholds = {
    "Logistic Regression": 0.45,
    "Random Forest": 0.30,
    "HistGradientBoosting": 0.20,
}


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
# 4. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    ),
}


# --------------------------------------------------
# 5. Run optimized experiments
# --------------------------------------------------

experiment_results = []

for model_name, estimator in models.items():

    threshold = optimized_thresholds[model_name]

    print(f"\nStarting optimized experiment: {model_name}")
    print(f"Decision threshold: {threshold}")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(
        run_name=f"Optimized - {model_name}"
    ):

        # ------------------------------------------
        # Log parameters
        # ------------------------------------------

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("experiment_type", "threshold_optimized")
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_param("threshold_selection_method", "Maximum validation F1")
        mlflow.log_param("training_records", len(X_train))
        mlflow.log_param("testing_records", len(X_test))
        mlflow.log_param("number_of_features", len(X_train.columns))
        mlflow.log_param("random_state", RANDOM_STATE)

        # ------------------------------------------
        # Train model
        # ------------------------------------------

        print("Training model...")

        pipeline.fit(X_train, y_train)

        # ------------------------------------------
        # Generate predictions
        # ------------------------------------------

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        # ------------------------------------------
        # Calculate metrics
        # ------------------------------------------

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

        mlflow.log_metrics(metrics)

        # ------------------------------------------
        # Confusion matrix
        # ------------------------------------------

        cm = confusion_matrix(y_test, predictions)

        tn, fp, fn, tp = cm.ravel()

        mlflow.log_metrics({
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

        # ------------------------------------------
        # Save evaluation artifacts
        # ------------------------------------------

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = Path(temp_dir)

            confusion_path = temp_path / "confusion_matrix.csv"

            pd.DataFrame(
                cm,
                index=["Actual_No_Churn", "Actual_Churn"],
                columns=["Predicted_No_Churn", "Predicted_Churn"],
            ).to_csv(confusion_path)

            mlflow.log_artifact(
                str(confusion_path),
                artifact_path="evaluation",
            )

            report_path = temp_path / "evaluation_report.json"

            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "model": model_name,
                        "decision_threshold": threshold,
                        "threshold_selection_method": "Maximum validation F1",
                        "metrics": metrics,
                        "confusion_matrix": cm.tolist(),
                    },
                    file,
                    indent=4,
                )

            mlflow.log_artifact(
                str(report_path),
                artifact_path="evaluation",
            )

        # ------------------------------------------
        # Store results
        # ------------------------------------------

        experiment_results.append({
            "Model": model_name,
            "Threshold": threshold,
            **metrics,
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
        })

        print("Experiment completed.")
        print(f"Metrics: {metrics}")
        print(f"Confusion matrix:\n{cm}")


# --------------------------------------------------
# 6. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame(experiment_results)

results_path = ML_DIR / "results" / "mlflow_optimized_comparison.csv"

results_df.to_csv(results_path, index=False)

print("\nAll optimized experiments completed!")

print("\nOptimized experiment comparison:")
print(results_df.to_string(index=False))

print(f"\nComparison saved: {results_path}")
