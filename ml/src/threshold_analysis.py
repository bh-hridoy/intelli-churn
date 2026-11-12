
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

TRAIN_PATH = DATA_DIR / "splits" / "train.csv"
TEST_PATH = DATA_DIR / "splits" / "test.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn Value"
RANDOM_STATE = 42

THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.05), 2)


# --------------------------------------------------
# 2. FEATURE CONFIGURATION
# --------------------------------------------------

NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 3. PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# --------------------------------------------------
# 4. MODEL CONFIGURATION
# --------------------------------------------------

def create_models():

    return {
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


def create_pipeline(model):

    return Pipeline([
        ("preprocessing", create_preprocessor()),
        ("model", model),
    ])


# --------------------------------------------------
# 5. LOAD THE EXISTING TRAIN/TEST SPLIT
# --------------------------------------------------

print("Loading existing dataset splits...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X = train_df[FEATURES]
y = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"Training records: {len(X)}")
print(f"Testing records: {len(X_test)}")


# --------------------------------------------------
# 6. CREATE VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Model training records: {len(X_train)}")
print(f"Validation records: {len(X_val)}")

print("\nOriginal test set remains untouched during threshold selection.")


# --------------------------------------------------
# 7. THRESHOLD ANALYSIS ON VALIDATION DATA
# --------------------------------------------------

validation_results = []
selected_thresholds = {}

models = create_models()

for model_name, model in models.items():

    print(f"\nAnalyzing {model_name}...")

    pipeline = create_pipeline(model)

    # Fit only on the smaller training subset
    pipeline.fit(X_train, y_train)

    # Predict churn probabilities on validation data
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    model_results = []

    for threshold in THRESHOLDS:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )

        recall = recall_score(
            y_val, predictions, zero_division=0
        )

        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        accuracy = accuracy_score(y_val, predictions)

        result = {
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
        }

        validation_results.append(result)
        model_results.append(result)

    model_df = pd.DataFrame(model_results)

    # Select threshold with the highest validation F1-score.
    # In case of a tie, prefer the higher threshold.
    best_row = model_df.sort_values(
        ["F1_Score", "Threshold"],
        ascending=[False, False],
    ).iloc[0]

    selected_thresholds[model_name] = float(best_row["Threshold"])

    print(f"Selected threshold: {best_row['Threshold']:.2f}")
    print(f"Validation precision: {best_row['Precision']:.4f}")
    print(f"Validation recall: {best_row['Recall']:.4f}")
    print(f"Validation F1-score: {best_row['F1_Score']:.4f}")


# --------------------------------------------------
# 8. SAVE VALIDATION RESULTS
# --------------------------------------------------

validation_df = pd.DataFrame(validation_results)

validation_output = RESULTS_DIR / "validation_threshold_analysis.csv"

validation_df.to_csv(
    validation_output,
    index=False,
    float_format="%.6f",
)

selected_df = pd.DataFrame([
    {
        "Model": model_name,
        "Selected_Threshold": threshold,
    }
    for model_name, threshold in selected_thresholds.items()
])

selected_output = RESULTS_DIR / "selected_thresholds.csv"

selected_df.to_csv(
    selected_output,
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 9. FINAL EVALUATION ON UNTOUCHED TEST SET
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST SET EVALUATION")
print("=" * 65)

test_results = []
confusion_matrices = {}

for model_name, model in create_models().items():

    print(f"\nEvaluating {model_name}...")

    # Retrain using the complete original training set
    pipeline = create_pipeline(model)
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    threshold = selected_thresholds[model_name]

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    test_results.append(result)
    confusion_matrices[model_name] = np.array([
        [tn, fp],
        [fn, tp],
    ])

    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1-score: {result['F1_Score']:.4f}")
    print(f"Confusion matrix: [[{tn}, {fp}], [{fn}, {tp}]]")


test_results_df = pd.DataFrame(test_results)

test_output = RESULTS_DIR / "selected_threshold_test_results.csv"

test_results_df.to_csv(
    test_output,
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 10. VISUALIZE VALIDATION THRESHOLD METRICS
# --------------------------------------------------

metrics = ["Precision", "Recall", "F1_Score"]

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 5),
)

for ax, metric in zip(axes, metrics):

    for model_name in models:

        model_data = validation_df[
            validation_df["Model"] == model_name
        ]

        ax.plot(
            model_data["Threshold"],
            model_data[metric],
            marker="o",
            label=model_name,
        )

        ax.axvline(
            selected_thresholds[model_name],
            linestyle="--",
            alpha=0.5,
        )

    ax.set_title(f"{metric} vs Threshold")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

axes[0].legend()

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "threshold_metrics.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 11. VISUALIZE FINAL TEST CONFUSION MATRICES
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5),
)

for ax, (model_name, matrix) in zip(
    axes,
    confusion_matrices.items(),
):

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(
        f"{model_name}\nThreshold: "
        f"{selected_thresholds[model_name]:.2f}"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "selected_threshold_confusion_matrices.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("THRESHOLD ANALYSIS SUMMARY")
print("=" * 65)

print("\nSelected thresholds:")
print(selected_df.to_string(index=False))

print("\nFinal test results:")
print(
    test_results_df[
        [
            "Model",
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
        ]
    ].to_string(index=False)
)

print("\nGenerated files:")
print("- results/validation_threshold_analysis.csv")
print("- results/selected_thresholds.csv")
print("- results/selected_threshold_test_results.csv")
print("- results/plots/threshold_metrics.png")
print("- results/plots/selected_threshold_confusion_matrices.png")

print("\nTHRESHOLD ANALYSIS COMPLETED SUCCESSFULLY.")
