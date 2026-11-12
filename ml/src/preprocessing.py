
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. FEATURE CONFIGURATION
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


# --------------------------------------------------
# 2. PREPROCESSING PIPELINES
# --------------------------------------------------

def build_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# --------------------------------------------------
# 3. TEST PREPROCESSING
# --------------------------------------------------

def main():

    project_root = Path(__file__).resolve().parents[2]

    train_path = (
        project_root
        / "ml"
        / "data"
        / "processed"
        / "splits"
        / "train.csv"
    )

    test_path = (
        project_root
        / "ml"
        / "data"
        / "processed"
        / "splits"
        / "test.csv"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn Value"])
    y_train = train_df["Churn Value"]

    X_test = test_df.drop(columns=["Churn Value"])
    y_test = test_df["Churn Value"]

    preprocessor = build_preprocessor()

    # Fit only on training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform test data without fitting
    X_test_transformed = preprocessor.transform(X_test)

    print("=" * 60)
    print("INTELLICHURN - PREPROCESSING PIPELINE")
    print("=" * 60)

    print("\nOriginal training shape:", X_train.shape)
    print("Original testing shape:", X_test.shape)

    print(
        "\nTransformed training shape:",
        X_train_transformed.shape
    )

    print(
        "Transformed testing shape:",
        X_test_transformed.shape
    )

    print("\nTraining target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)

    print("\nNumerical features:", len(NUMERICAL_FEATURES))
    print("Categorical features:", len(CATEGORICAL_FEATURES))

    print("\nPreprocessing pipeline:")
    print(preprocessor)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
