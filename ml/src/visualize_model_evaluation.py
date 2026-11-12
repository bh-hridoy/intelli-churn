
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def main():
    # --------------------------------------------------
    # 1. Paths and data loading
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    train_path = (
        project_root / "ml" / "data" / "processed" / "splits" / "train.csv"
    )
    test_path = (
        project_root / "ml" / "data" / "processed" / "splits" / "test.csv"
    )

    results_dir = project_root / "ml" / "results"
    plots_dir = results_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Churn Value"])
    y_train = train_df["Churn Value"]

    X_test = test_df.drop(columns=["Churn Value"])
    y_test = test_df["Churn Value"]

    # --------------------------------------------------
    # 2. Define the three models
    # --------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    model_results = []
    predictions = {}

    # --------------------------------------------------
    # 3. Train and evaluate every model
    # --------------------------------------------------

    for model_name, estimator in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, y_proba),
            "PR_AUC": average_precision_score(y_test, y_proba),
        }

        model_results.append(metrics)

        predictions[model_name] = {
            "predicted": y_pred,
            "probabilities": y_proba,
        }

        print(f"{model_name} evaluation completed.")

    results_df = pd.DataFrame(model_results)

    # Save exact metrics from this evaluation run.
    results_df.to_csv(
        results_dir / "model_comparison.csv",
        index=False,
        float_format="%.6f",
    )

    print("\nAll models evaluated successfully.")

    # --------------------------------------------------
    # 4. Model performance comparison chart
    # --------------------------------------------------

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC",
    ]

    chart_data = results_df.set_index("Model")[metric_columns]

    ax = chart_data.plot(
        kind="bar",
        figsize=(13, 6),
        width=0.8,
    )

    ax.set_title("IntelliChurn - Model Performance Comparison")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Evaluation Metrics", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        plots_dir / "model_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 5. ROC curve comparison
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    for model_name, result in predictions.items():
        fpr, tpr, _ = roc_curve(
            y_test,
            result["probabilities"],
        )

        auc = roc_auc_score(
            y_test,
            result["probabilities"],
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        plots_dir / "roc_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 6. Precision-Recall curve comparison
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    for model_name, result in predictions.items():
        precision_values, recall_values, _ = (
            precision_recall_curve(
                y_test,
                result["probabilities"],
            )
        )

        pr_auc = average_precision_score(
            y_test,
            result["probabilities"],
        )

        plt.plot(
            recall_values,
            precision_values,
            label=f"{model_name} (AP = {pr_auc:.3f})",
        )

    plt.axhline(
        y_test.mean(),
        linestyle="--",
        label="Positive class baseline",
    )

    plt.title("Precision-Recall Curve Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        plots_dir / "precision_recall_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 7. Confusion matrices
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (model_name, result) in zip(
        axes,
        predictions.items(),
    ):
        matrix = confusion_matrix(
            y_test,
            result["predicted"],
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["No Churn", "Churn"],
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            values_format="d",
        )

        ax.set_title(model_name)

    fig.suptitle("Confusion Matrix Comparison", fontsize=15)

    plt.tight_layout()
    plt.savefig(
        plots_dir / "confusion_matrices.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------
    # 8. Final output
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("INTELLICHURN - EVALUATION SUMMARY")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nGenerated files:")

    for file_path in sorted(plots_dir.glob("*.png")):
        print(file_path.name)

    print("\nUpdated: results/model_comparison.csv")
    print("\nMODEL EVALUATION AND VISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()
