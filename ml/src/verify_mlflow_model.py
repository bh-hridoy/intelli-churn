
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"

TEST_PATH = (
    ML_DIR / "data" / "processed" / "splits" / "test.csv"
)

TRACKING_DB = ML_DIR / "mlflow.db"

# Configure MLflow
tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"
mlflow.set_tracking_uri(tracking_uri)

MODEL_URI = "models:/m-1d670de66d5f4a18a9845cb4d48d5ff0"

# Load test data
test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["Churn Value"])

# Load model from MLflow
print("Loading model from MLflow...")

model = mlflow.sklearn.load_model(MODEL_URI)

print("Model loaded successfully!")

# Generate predictions
probabilities = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.30
predictions = (probabilities >= THRESHOLD).astype(int)

# Display sample results
results = pd.DataFrame({
    "Actual": test_df["Churn Value"].head(10),
    "Churn Probability": probabilities[:10],
    "Predicted": predictions[:10],
})

print("\nSample predictions:")
print(results.to_string(index=False))

print("\nPrediction summary:")
print(f"Total predictions: {len(predictions)}")
print(f"Minimum probability: {probabilities.min():.4f}")
print(f"Maximum probability: {probabilities.max():.4f}")
print(f"Predicted churners: {predictions.sum()}")

print("\nMLflow model verification completed!")
