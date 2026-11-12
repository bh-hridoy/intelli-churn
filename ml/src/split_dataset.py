
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "churn_modeling_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "splits"
)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

TARGET = "Churn Value"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# 3. TRAIN-TEST SPLITTING
# --------------------------------------------------

def split_dataset():

    print("=" * 60)
    print("INTELLICHURN - TRAIN TEST SPLITTING")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"\nTotal records: {len(df)}")
    print(f"Input features: {X.shape[1]}")

    # Split while preserving the target distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Verify that no records overlap
    assert set(X_train.index).isdisjoint(
        set(X_test.index)
    )

    # Reattach target for saving
    train_data = X_train.copy()
    train_data[TARGET] = y_train

    test_data = X_test.copy()
    test_data[TARGET] = y_test

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw train-test splits
    train_data.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    test_data.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. RESULTS
    # --------------------------------------------------

    print("\nTRAINING SET")
    print(f"Records: {len(train_data)}")
    print("\nTarget distribution:")
    print(y_train.value_counts(normalize=True).mul(100).round(2))

    print("\nTEST SET")
    print(f"Records: {len(test_data)}")
    print("\nTarget distribution:")
    print(y_test.value_counts(normalize=True).mul(100).round(2))

    print("\nSAVED FILES")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "test.csv")

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLITTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    split_dataset()
