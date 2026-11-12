
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco_customer_churn.xlsx"
)


def analyze_data_quality():
    df = pd.read_excel(DATASET_PATH)

    print("=" * 65)
    print("INTELLICHURN - DATA QUALITY ANALYSIS")
    print("=" * 65)

    # 1. Check blank and whitespace-only values
    print("\n1. BLANK / WHITESPACE VALUES")

    blank_counts = {}

    for column in df.select_dtypes(include=["object", "str"]).columns:
        blank_count = (
            df[column]
            .astype("string")
            .str.strip()
            .eq("")
            .sum()
        )

        if blank_count > 0:
            blank_counts[column] = blank_count

    if blank_counts:
        print(pd.Series(blank_counts))
    else:
        print("No blank strings found.")

    # 2. Constant columns
    print("\n2. CONSTANT COLUMNS")

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) == 1
    ]

    print(constant_columns)

    # 3. Unique values in categorical columns
    print("\n3. CATEGORICAL COLUMN DETAILS")

    categorical_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}")
        print(f"Unique values: {df[column].nunique()}")
        print(df[column].value_counts().head(10).to_string())

    # 4. Total Charges inspection
    print("\n4. TOTAL CHARGES INSPECTION")

    total_charges = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    invalid_count = total_charges.isna().sum()

    print(f"Invalid or blank values: {invalid_count}")
    print("Sample affected rows:")

    print(
        df.loc[
            total_charges.isna(),
            ["CustomerID", "Tenure Months", "Total Charges"]
        ].head(10).to_string(index=False)
    )

    # 5. Numeric column summary
    print("\n5. NUMERIC COLUMN SUMMARY")

    print(df.describe().T.to_string())

    # 6. Churn-related columns
    print("\n6. CHURN-RELATED COLUMN CHECK")

    for column in [
        "Churn Label",
        "Churn Value",
        "Churn Score",
        "CLTV",
        "Churn Reason",
    ]:
        if column in df.columns:
            print(f"\n{column}")
            print(f"Unique values: {df[column].nunique(dropna=False)}")
            print(df[column].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    analyze_data_quality()
