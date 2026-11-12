
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco_customer_churn.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
)

OUTPUT_PATH = OUTPUT_DIR / "churn_modeling_data.csv"


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

TARGET = "Churn Value"

SELECTED_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + [TARGET]
)


# --------------------------------------------------
# 3. DATA PREPARATION
# --------------------------------------------------

def prepare_dataset():

    print("=" * 60)
    print("INTELLICHURN - DATA PREPARATION")
    print("=" * 60)

    # Load original dataset
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_excel(RAW_DATA_PATH)

    print(f"\nOriginal dataset shape: {df.shape}")

    # Validate required columns
    missing_columns = [
        column
        for column in SELECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 4. CLEAN TOTAL CHARGES
    # --------------------------------------------------

    print("\n1. Cleaning Total Charges...")

    total_charges_text = (
        df["Total Charges"]
        .astype("string")
        .str.strip()
    )

    blank_mask = (
        total_charges_text.isna()
        | total_charges_text.eq("")
    )

    total_charges_numeric = pd.to_numeric(
        total_charges_text.mask(blank_mask),
        errors="coerce"
    )

    invalid_mask = (
        total_charges_numeric.isna()
        & ~blank_mask
    )

    if invalid_mask.any():
        raise ValueError(
            "Unexpected non-numeric Total Charges values found."
        )

    # Blank Total Charges are valid only for zero-tenure customers
    invalid_blank_rows = blank_mask & (
        df["Tenure Months"] != 0
    )

    if invalid_blank_rows.any():
        raise ValueError(
            "Found blank Total Charges for customers "
            "with non-zero tenure."
        )

    total_charges_numeric = total_charges_numeric.mask(
        blank_mask,
        0
    )

    df["Total Charges"] = total_charges_numeric

    print(f"Blank Total Charges handled: {blank_mask.sum()}")

    # --------------------------------------------------
    # 5. SELECT FEATURES AND TARGET
    # --------------------------------------------------

    print("\n2. Selecting model features...")

    df = df[SELECTED_COLUMNS].copy()

    print(f"Selected columns: {len(SELECTED_COLUMNS)}")
    print(f"Input features: {len(SELECTED_COLUMNS) - 1}")

    # --------------------------------------------------
    # 6. VALIDATE DATA TYPES
    # --------------------------------------------------

    print("\n3. Validating numerical features...")

    for column in NUMERICAL_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise"
    ).astype("int64")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    # --------------------------------------------------
    # 7. VALIDATE TARGET CONSISTENCY
    # --------------------------------------------------

    print("\n4. Checking target consistency...")

    original_target = pd.read_excel(
        RAW_DATA_PATH,
        usecols=["Churn Label", "Churn Value"]
    )

    expected_target = original_target["Churn Label"].map(
        {"No": 0, "Yes": 1}
    )

    if expected_target.isna().any():
        raise ValueError("Unexpected Churn Label values found.")

    if not (
        expected_target.to_numpy()
        == df[TARGET].to_numpy()
    ).all():
        raise ValueError(
            "Churn Label and Churn Value do not match."
        )

    print("Churn Label and Churn Value match.")

    # --------------------------------------------------
    # 8. CHECK MISSING VALUES
    # --------------------------------------------------

    print("\n5. Checking missing values...")

    missing_values = df.isnull().sum()

    if missing_values.any():
        raise ValueError(
            f"Unexpected missing values:\n{missing_values[missing_values > 0]}"
        )

    print("No missing values in selected columns.")

    # --------------------------------------------------
    # 9. SAVE PROCESSED DATASET
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n6. Final dataset summary")
    print(f"Shape: {df.shape}")
    print(f"Numerical features: {len(NUMERICAL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Target: {TARGET}")

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())

    print("\nSaved dataset:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    prepare_dataset()
