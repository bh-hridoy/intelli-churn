
from pathlib import Path

import pandas as pd


# Locate the project root and dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco_customer_churn.xlsx"
)


def inspect_dataset():
    print("=" * 60)
    print("INTELLICHURN - DATASET INSPECTION")
    print("=" * 60)

    # Check whether the dataset exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Load the Excel dataset
    df = pd.read_excel(DATASET_PATH)

    # 1. Dataset dimensions
    print("\n1. DATASET DIMENSIONS")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. Column names
    print("\n2. COLUMN NAMES")
    for column in df.columns:
        print(f"- {column}")

    # 3. Data types
    print("\n3. DATA TYPES")
    print(df.dtypes)

    # 4. Missing values
    print("\n4. MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    # 5. Duplicate records
    print("\n5. DUPLICATE RECORDS")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # 6. Churn distribution
    print("\n6. CHURN DISTRIBUTION")

    if "Churn Label" in df.columns:
        print(df["Churn Label"].value_counts())
        print("\nPercentage:")
        print(
            df["Churn Label"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    # 7. Sample records
    print("\n7. FIRST FIVE RECORDS")
    print(df.head().to_string())

    # 8. Unique customer IDs
    if "CustomerID" in df.columns:
        print("\n8. UNIQUE CUSTOMER IDS")
        print(df["CustomerID"].nunique())

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
