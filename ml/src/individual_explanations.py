
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

CHURN_THRESHOLD = 0.30


# --------------------------------------------------
# 2. LOAD DATA AND CREATE VALIDATION SET
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
# 3. TRAIN RANDOM FOREST
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


# --------------------------------------------------
# 4. CALCULATE VALIDATION PREDICTIONS
# --------------------------------------------------

print("\nCalculating validation probabilities...")

probabilities = model.predict_proba(X_val)[:, 1]

print(
    f"Minimum probability: {probabilities.min():.4f}"
)
print(
    f"Maximum probability: {probabilities.max():.4f}"
)


# --------------------------------------------------
# 5. SELECT THREE REPRESENTATIVE CUSTOMERS
# --------------------------------------------------

# Select examples near the 10th, 50th and 90th
# percentiles of predicted churn probability.

percentiles = {
    "Low Risk Example": 0.10,
    "Medium Risk Example": 0.50,
    "High Risk Example": 0.90,
}

selected_customers = {}

for label, percentile in percentiles.items():

    target_probability = np.quantile(
        probabilities,
        percentile,
    )

    selected_position = int(
        np.argmin(
            np.abs(probabilities - target_probability)
        )
    )

    selected_customers[label] = {
        "position": selected_position,
        "probability": probabilities[selected_position],
    }


# --------------------------------------------------
# 6. CREATE A REFERENCE CUSTOMER PROFILE
# --------------------------------------------------

# Numerical features: training-set median
# Categorical features: training-set most frequent value

reference = {}

for feature in NUMERICAL_FEATURES:
    reference[feature] = X_train[feature].median()

for feature in CATEGORICAL_FEATURES:
    reference[feature] = X_train[feature].mode().iloc[0]

reference_df = pd.DataFrame(
    [reference],
    columns=FEATURES,
)


# --------------------------------------------------
# 7. INDIVIDUAL FEATURE SENSITIVITY ANALYSIS
# --------------------------------------------------

print("\nCalculating individual explanations...")

explanation_records = []
customer_records = []

for customer_label, customer_info in selected_customers.items():

    position = customer_info["position"]

    customer = X_val.iloc[[position]].copy()

    original_probability = float(
        model.predict_proba(customer)[0, 1]
    )

    original_prediction = (
        "High Churn Risk"
        if original_probability >= CHURN_THRESHOLD
        else "Low Churn Risk"
    )

    customer_record = {
        "Case": customer_label,
        "Churn_Probability": original_probability,
        "Predicted_Risk": original_prediction,
    }

    customer_record.update(customer.iloc[0].to_dict())
    customer_records.append(customer_record)

    print(f"\n{customer_label}")
    print(f"Churn probability: {original_probability:.4f}")
    print(f"Predicted risk: {original_prediction}")

    for feature in FEATURES:

        modified_customer = customer.copy()

        # Replace one feature with its reference value
        modified_customer.loc[:, feature] = (
            reference_df.iloc[0][feature]
        )

        modified_probability = float(
            model.predict_proba(modified_customer)[0, 1]
        )

        probability_change = (
            original_probability - modified_probability
        )

        if probability_change > 0:
            interpretation = "Raises predicted risk"
        elif probability_change < 0:
            interpretation = "Lowers predicted risk"
        else:
            interpretation = "No change"

        explanation_records.append({
            "Case": customer_label,
            "Original_Probability": original_probability,
            "Feature": feature,
            "Original_Value": customer.iloc[0][feature],
            "Reference_Value": reference_df.iloc[0][feature],
            "Modified_Probability": modified_probability,
            "Probability_Change": probability_change,
            "Interpretation": interpretation,
        })


# --------------------------------------------------
# 8. SAVE CUSTOMER AND EXPLANATION RESULTS
# --------------------------------------------------

customers_df = pd.DataFrame(customer_records)

explanations_df = pd.DataFrame(explanation_records)

customers_output = RESULTS_DIR / "explained_customers.csv"

explanations_output = (
    RESULTS_DIR / "individual_feature_explanations.csv"
)

customers_df.to_csv(
    customers_output,
    index=False,
    float_format="%.6f",
)

explanations_df.to_csv(
    explanations_output,
    index=False,
    float_format="%.6f",
)


# --------------------------------------------------
# 9. GENERATE INDIVIDUAL EXPLANATION CHARTS
# --------------------------------------------------

for customer_label in selected_customers:

    customer_explanations = explanations_df[
        explanations_df["Case"] == customer_label
    ].copy()

    # Display the 8 features with the largest
    # absolute changes in predicted probability.

    customer_explanations["Absolute_Change"] = (
        customer_explanations["Probability_Change"].abs()
    )

    top_features = customer_explanations.nlargest(
        8,
        "Absolute_Change",
    ).sort_values("Probability_Change")

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["Feature"],
        top_features["Probability_Change"],
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    probability = selected_customers[customer_label]["probability"]

    plt.title(
        f"{customer_label}\n"
        f"Original Churn Probability: {probability:.2%}"
    )

    plt.xlabel(
        "Original Probability - Modified Probability"
    )

    plt.ylabel("Feature")

    plt.tight_layout()

    safe_name = (
        customer_label.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        PLOTS_DIR / f"{safe_name}_explanation.png",
        dpi=300,
    )

    plt.close()


# --------------------------------------------------
# 10. DISPLAY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("INDIVIDUAL CUSTOMER EXPLANATION SUMMARY")
print("=" * 65)

print("\nSelected customers:")

print(
    customers_df[
        ["Case", "Churn_Probability", "Predicted_Risk"]
    ].to_string(index=False)
)

for customer_label in selected_customers:

    print(f"\n{customer_label} - Top influential features:")

    customer_explanations = explanations_df[
        explanations_df["Case"] == customer_label
    ].copy()

    customer_explanations["Absolute_Change"] = (
        customer_explanations["Probability_Change"].abs()
    )

    print(
        customer_explanations.nlargest(
            5,
            "Absolute_Change",
        )[
            [
                "Feature",
                "Original_Value",
                "Reference_Value",
                "Probability_Change",
                "Interpretation",
            ]
        ].to_string(index=False)
    )


print("\nGenerated files:")
print("- results/explained_customers.csv")
print("- results/individual_feature_explanations.csv")
print("- results/plots/low_risk_example_explanation.png")
print("- results/plots/medium_risk_example_explanation.png")
print("- results/plots/high_risk_example_explanation.png")

print("\nINDIVIDUAL EXPLANATION ANALYSIS COMPLETED.")
