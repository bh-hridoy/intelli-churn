
from pathlib import Path
import json

import joblib
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TEST_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)


# --------------------------------------------------
# 2. Load saved artifacts
# --------------------------------------------------

print("Loading saved model...")

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = threshold_config["threshold"]

print("Model loaded successfully.")
print(f"Model: {metadata['model_name']}")
print(f"Threshold: {threshold}")


# --------------------------------------------------
# 3. Load test dataset
# --------------------------------------------------

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["Churn Value"])
y_test = test_df["Churn Value"]

print(f"Test records: {len(X_test)}")


# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

print("\nGenerating predictions...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)

print("Predictions generated successfully.")


# --------------------------------------------------
# 5. Display sample predictions
# --------------------------------------------------

sample_results = pd.DataFrame(
    {
        "Actual": y_test.iloc[:10].values,
        "Churn Probability": probabilities[:10],
        "Predicted": predictions[:10],
    }
)

sample_results["Predicted Risk"] = sample_results[
    "Churn Probability"
].apply(
    lambda probability:
        "High" if probability >= threshold else "Low"
)

print("\nSample Predictions:")
print(sample_results.to_string(index=False))


# --------------------------------------------------
# 6. Verify prediction output
# --------------------------------------------------

assert len(probabilities) == len(X_test)
assert len(predictions) == len(X_test)
assert probabilities.min() >= 0
assert probabilities.max() <= 1

print("\nPrediction verification passed.")

print(f"Minimum probability: {probabilities.min():.4f}")
print(f"Maximum probability: {probabilities.max():.4f}")

print("\nSaved model testing completed successfully!")
