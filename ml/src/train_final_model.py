
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET = "Churn Value"

THRESHOLD = 0.30
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load training data
# --------------------------------------------------

print("Loading training dataset...")

train_df = pd.read_csv(TRAIN_PATH)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

print(f"Training records: {len(X_train)}")
print(f"Features: {len(X_train.columns)}")


# --------------------------------------------------
# 3. Build final pipeline
# --------------------------------------------------

print("\nBuilding final Random Forest pipeline...")

final_pipeline = Pipeline(
    steps=[
        ("preprocessing", build_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)


# --------------------------------------------------
# 4. Train on the complete training dataset
# --------------------------------------------------

print("\nTraining final model...")

final_pipeline.fit(X_train, y_train)

print("Final model training completed.")


# --------------------------------------------------
# 5. Save model pipeline
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(final_pipeline, MODEL_PATH)

print(f"\nSaved model pipeline: {MODEL_PATH}")


# --------------------------------------------------
# 6. Save decision threshold
# --------------------------------------------------

threshold_data = {
    "threshold": THRESHOLD,
    "selection_method": "Maximum validation F1-score",
    "model": "Random Forest",
}

with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
    json.dump(threshold_data, file, indent=4)

print(f"Saved decision threshold: {THRESHOLD_PATH}")


# --------------------------------------------------
# 7. Save model metadata
# --------------------------------------------------

metadata = {
    "model_name": "Random Forest",
    "model_type": "RandomForestClassifier",
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "training_records": len(X_train),
    "number_of_input_features": len(X_train.columns),
    "input_features": list(X_train.columns),
    "target": TARGET,
    "decision_threshold": THRESHOLD,
    "test_metrics": {
        "accuracy": 0.757984,
        "precision": 0.531309,
        "recall": 0.748663,
        "f1_score": 0.621532,
        "confusion_matrix": [
            [788, 247],
            [94, 280],
        ],
    },
    "dataset": "IBM Telco Customer Churn",
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print(f"Saved model metadata: {METADATA_PATH}")

print("\nFinal model artifacts created successfully!")
