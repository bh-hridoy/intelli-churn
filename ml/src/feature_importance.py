
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT / "ml" / "data" / "processed"
    / "splits" / "train.csv"
)

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn Value"
RANDOM_STATE = 42

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
# 2. LOAD TRAINING DATA AND CREATE VALIDATION SET
# --------------------------------------------------

print("Loading training data...")

df = pd.read_csv(TRAIN_PATH)

X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training records: {len(X_train)}")
print(f"Validation records: {len(X_val)}")


# --------------------------------------------------
# 3. BUILD AND TRAIN RANDOM FOREST PIPELINE
# --------------------------------------------------

print("\nTraining Random Forest...")

model = Pipeline([
    ("preprocessing", build_preprocessor()),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)

print("Model training completed.")

probabilities = model.predict_proba(X_val)[:, 1]

print(
    f"Validation PR-AUC: "
    f"{average_precision_score(y_val, probabilities):.4f}"
)


# --------------------------------------------------
# 4. RANDOM FOREST BUILT-IN FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCalculating built-in feature importance...")

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["model"]

encoded_feature_names = preprocessor.get_feature_names_out()
encoded_importances = classifier.feature_importances_

native_importance = []

for feature in FEATURES:

    if feature in NUMERICAL_FEATURES:
        prefix = f"numerical__{feature}"
    else:
        prefix = f"categorical__{feature}_"

    indices = [
        i
        for i, name in enumerate(encoded_feature_names)
        if (
            name == prefix
            if feature in NUMERICAL_FEATURES
            else name.startswith(prefix)
        )
    ]

    importance = sum(encoded_importances[i] for i in indices)

    native_importance.append({
        "Feature": feature,
        "Importance": importance,
    })

native_df = pd.DataFrame(native_importance)

native_df = native_df.sort_values(
    "Importance",
    ascending=False,
)


# --------------------------------------------------
# 5. PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCalculating permutation importance...")
print("This may take a little time.")

permutation_result = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="average_precision",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

permutation_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance_Mean": permutation_result.importances_mean,
    "Importance_Std": permutation_result.importances_std,
})

permutation_df = permutation_df.sort_values(
    "Importance_Mean",
    ascending=False,
)


# --------------------------------------------------
# 6. SAVE FEATURE IMPORTANCE RESULTS
# --------------------------------------------------

native_output = RESULTS_DIR / "random_forest_feature_importance.csv"

permutation_output = (
    RESULTS_DIR / "random_forest_permutation_importance.csv"
)

native_df.to_csv(
    native_output,
    index=False,
    float_format="%.6f",
)

permutation_df.to_csv(
    permutation_output,
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 7. VISUALIZE BUILT-IN FEATURE IMPORTANCE
# --------------------------------------------------

top_native = native_df.head(15).sort_values(
    "Importance",
    ascending=True,
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_native["Feature"],
    top_native["Importance"],
)

plt.title("Random Forest - Feature Importance")
plt.xlabel("Aggregated Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "random_forest_feature_importance.png",
    dpi=300,
)

plt.close()


# --------------------------------------------------
# 8. VISUALIZE PERMUTATION IMPORTANCE
# --------------------------------------------------

top_permutation = permutation_df.head(15).sort_values(
    "Importance_Mean",
    ascending=True,
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_permutation["Feature"],
    top_permutation["Importance_Mean"],
    xerr=top_permutation["Importance_Std"],
    capsize=3,
)

plt.axvline(0, linestyle="--", linewidth=1)

plt.title("Random Forest - Permutation Importance")
plt.xlabel("Decrease in Validation Average Precision")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "random_forest_permutation_importance.png",
    dpi=300,
)

plt.close()


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print("\nTop 10 built-in feature importance:")
print(native_df.head(10).to_string(index=False))

print("\nTop 10 permutation feature importance:")
print(permutation_df.head(10).to_string(index=False))

print("\nGenerated files:")
print("- results/random_forest_feature_importance.csv")
print("- results/random_forest_permutation_importance.csv")
print("- results/plots/random_forest_feature_importance.png")
print("- results/plots/random_forest_permutation_importance.png")

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED.")
