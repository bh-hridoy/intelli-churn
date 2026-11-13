
from pathlib import Path
import json
import tempfile

import mlflow
import mlflow.sklearn
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

TRAIN_PATH = (
    ML_DIR / "data" / "processed" / "splits" / "train.csv"
)

TEST_PATH = (
    ML_DIR / "data" / "processed" / "splits" / "test.csv"
)

TRACKING_DB = ML_DIR / "mlflow.db"

tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

EXPERIMENT_NAME = "IntelliChurn-Model-Experiments"

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

TARGET = "Churn Value"
THRESHOLD = 0.50


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
    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "parameters": {
            "max_iter": 1000,
            "random_state": 42,
        },
    },

    "Random Forest": {
        "estimator": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "parameters": {
            "n_estimators": 300,
            "random_state": 42,
            "n_jobs": -1,
        },
    },

    "HistGradientBoosting": {
        "estimator": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=42,
        ),
        "parameters": {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "random_state": 42,
        },
    },
}


# --------------------------------------------------
# 5. Run experiments
# --------------------------------------------------

experiment_results = []

for model_name, configuration in models.items():

    print(f"\nStarting experiment: {model_name}")

    estimator = configuration["estimator"]

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", estimator),
        ]
    )

    with mlflow.start_run(run_name=model_name):

        # Log experiment parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("decision_threshold", THRESHOLD)
        mlflow.log_param("training_records", len(X_train))
        mlflow.log_param("testing_records", len(X_test))
        mlflow.log_param("number_of_features", len(X_train.columns))
        mlflow.log_param("preprocessing", "ColumnTransformer")
        mlflow.log_param("numerical_scaling", "StandardScaler")
        mlflow.log_param("categorical_encoding", "OneHotEncoder")

        mlflow.log_params(configuration["parameters"])

        # Train model
        print("Training model...")

        pipeline.fit(X_train, y_train)

        # Generate predictions
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        # Calculate metrics
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

        # Log metrics
        mlflow.log_metrics(metrics)

        # Calculate and save confusion matrix
        cm = confusion_matrix(y_test, predictions)

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

            report_path = temp_path / "metrics.json"

            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "model": model_name,
                        "threshold": THRESHOLD,
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

        # Store results for comparison
        experiment_results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

        print("Experiment completed.")
        print(f"Metrics: {metrics}")
        print(f"Confusion matrix:\n{cm}")


# --------------------------------------------------
# 6. Save experiment comparison
# --------------------------------------------------

results_df = pd.DataFrame(experiment_results)

results_path = ML_DIR / "results" / "mlflow_model_comparison.csv"

results_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(results_path, index=False)

print("\nAll experiments completed!")

print("\nExperiment comparison:")
print(results_df.to_string(index=False))

print(f"\nComparison saved: {results_path}")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
