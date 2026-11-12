
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "splits" / "train.csv"
TEST_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "splits" / "test.csv"

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

THRESHOLD = 0.30
RANDOM_STATE = 42


# --------------------------------------------------
# 2. Load training and testing data
# --------------------------------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "Churn Value"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# --------------------------------------------------
# 3. Build and train selected model
# --------------------------------------------------

print("\nTraining Random Forest...")

model = Pipeline(
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

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 4. Predict churn probabilities
# --------------------------------------------------

print("\nPredicting test customers...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= THRESHOLD).astype(int)


# --------------------------------------------------
# 5. Identify error types
# --------------------------------------------------

results = X_test.copy().reset_index(drop=True)

results["Actual"] = y_test.reset_index(drop=True)
results["Churn Probability"] = probabilities
results["Predicted"] = predictions

results["Error Type"] = "Correct"

results.loc[
    (results["Actual"] == 0) & (results["Predicted"] == 1),
    "Error Type",
] = "False Positive"

results.loc[
    (results["Actual"] == 1) & (results["Predicted"] == 0),
    "Error Type",
] = "False Negative"

results.loc[
    (results["Actual"] == 1) & (results["Predicted"] == 1),
    "Error Type",
] = "True Positive"

results.loc[
    (results["Actual"] == 0) & (results["Predicted"] == 0),
    "Error Type",
] = "True Negative"


# --------------------------------------------------
# 6. Print overall error summary
# --------------------------------------------------

cm = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("\nError Summary:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 7. Save misclassified customers
# --------------------------------------------------

misclassified = results[
    results["Error Type"].isin(["False Positive", "False Negative"])
].copy()

misclassified.insert(
    0,
    "Test Row",
    misclassified.index + 1,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

misclassified.to_csv(
    RESULTS_DIR / "misclassified_customers.csv",
    index=False,
)

print(f"\nMisclassified customers: {len(misclassified)}")

print(
    "Saved:",
    RESULTS_DIR / "misclassified_customers.csv",
)


# --------------------------------------------------
# 8. Analyze errors by customer groups
# --------------------------------------------------

def summarize_group_errors(df, group_column):
    summary = (
        df.groupby(group_column, dropna=False)
        .agg(
            Customers=("Actual", "size"),
            Actual_Churners=("Actual", "sum"),
            Predicted_Churners=("Predicted", "sum"),
            False_Positives=(
                "Error Type",
                lambda values: (values == "False Positive").sum(),
            ),
            False_Negatives=(
                "Error Type",
                lambda values: (values == "False Negative").sum(),
            ),
        )
        .reset_index()
    )

    summary["False Positive Rate"] = (
        summary["False_Positives"]
        / (summary["Customers"] - summary["Actual_Churners"]).replace(0, pd.NA)
    )

    summary["False Negative Rate"] = (
        summary["False_Negatives"]
        / summary["Actual_Churners"].replace(0, pd.NA)
    )

    return summary


group_columns = [
    "Contract",
    "Internet Service",
    "Payment Method",
]

for column in group_columns:
    summary = summarize_group_errors(results, column)

    output_path = RESULTS_DIR / f"error_analysis_by_{column.lower().replace(' ', '_')}.csv"

    summary.to_csv(output_path, index=False)

    print(f"\nError analysis by {column}:")
    print(summary.to_string(index=False))
    print(f"Saved: {output_path}")


# --------------------------------------------------
# 9. Analyze errors by tenure groups
# --------------------------------------------------

results["Tenure Group"] = pd.cut(
    results["Tenure Months"],
    bins=[-1, 12, 24, 48, 60, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-60 months",
        "61-72 months",
    ],
)

tenure_summary = summarize_group_errors(results, "Tenure Group")

tenure_summary.to_csv(
    RESULTS_DIR / "error_analysis_by_tenure.csv",
    index=False,
)

print("\nError analysis by Tenure Group:")
print(tenure_summary.to_string(index=False))


# --------------------------------------------------
# 10. Confusion matrix visualization
# --------------------------------------------------

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"],
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("Random Forest Error Analysis (Threshold = 0.30)")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "error_analysis_confusion_matrix.png",
    dpi=300,
)

plt.close()


# --------------------------------------------------
# 11. False positive and false negative visualization
# --------------------------------------------------

error_counts = misclassified["Error Type"].value_counts()

fig, ax = plt.subplots(figsize=(7, 5))

error_counts.plot(
    kind="bar",
    ax=ax,
)

ax.set_title("False Positives vs False Negatives")
ax.set_xlabel("Error Type")
ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "false_positive_vs_false_negative.png",
    dpi=300,
)

plt.close()

print("\nError analysis completed successfully!")
