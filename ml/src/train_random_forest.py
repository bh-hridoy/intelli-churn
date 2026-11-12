
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def main():
    # 1. Load datasets
    project_root = Path(__file__).resolve().parents[2]

    train_path = (
        project_root / "ml" / "data" / "processed" / "splits" / "train.csv"
    )
    test_path = (
        project_root / "ml" / "data" / "processed" / "splits" / "test.csv"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn Value"])
    y_train = train_df["Churn Value"]

    X_test = test_df.drop(columns=["Churn Value"])
    y_test = test_df["Churn Value"]

    # 2. Build pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 3. Train
    print("=" * 60)
    print("INTELLICHURN - RANDOM FOREST")
    print("=" * 60)

    print("\nTraining model...")
    model_pipeline.fit(X_train, y_train)
    print("Training completed successfully.")

    # 4. Predict
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # 5. Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\nMODEL PERFORMANCE")
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    # 6. Classification report
    print("\nCLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    # 7. Confusion matrix
    print("CONFUSION MATRIX")
    print("-" * 60)

    matrix = confusion_matrix(y_test, y_pred)

    print(matrix)
    print("\nRows = Actual | Columns = Predicted")
    print("                Predicted No    Predicted Yes")
    print(f"Actual No       {matrix[0][0]:<15}{matrix[0][1]}")
    print(f"Actual Yes      {matrix[1][0]:<15}{matrix[1][1]}")

    print("\n" + "=" * 60)
    print("RANDOM FOREST EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
