
import pandas as pd

from app.core.config import settings


NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

CATEGORICAL_FEATURES = {
    "Gender": {"Male", "Female"},
    "Senior Citizen": {"Yes", "No"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "Phone Service": {"Yes", "No"},
    "Multiple Lines": {"Yes", "No", "No phone service"},
    "Internet Service": {"DSL", "Fiber optic", "No"},
    "Online Security": {"Yes", "No", "No internet service"},
    "Online Backup": {"Yes", "No", "No internet service"},
    "Device Protection": {"Yes", "No", "No internet service"},
    "Tech Support": {"Yes", "No", "No internet service"},
    "Streaming TV": {"Yes", "No", "No internet service"},
    "Streaming Movies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "Paperless Billing": {"Yes", "No"},
    "Payment Method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

REQUIRED_FEATURES = (
    NUMERICAL_FEATURES + list(CATEGORICAL_FEATURES.keys())
)


def predict_batch_churn(data: pd.DataFrame, model):
    """Validate a customer DataFrame and generate batch predictions."""

    if data.empty:
        raise ValueError("The uploaded CSV contains no customer records.")

    # Preserve the original data for the output.
    result = data.copy()

    # Check required feature columns.
    missing_columns = [
        column
        for column in REQUIRED_FEATURES
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Allow CustomerID as an additional identifier.
    allowed_columns = set(REQUIRED_FEATURES) | {"CustomerID"}

    unexpected_columns = [
        column for column in result.columns
        if column not in allowed_columns
    ]

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns: {unexpected_columns}"
        )

    # Validate numerical features.
    for column in NUMERICAL_FEATURES:
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")

        invalid_values = original.notna() & converted.isna()

        if invalid_values.any():
            raise ValueError(
                f"Column '{column}' contains non-numeric values."
            )

        result[column] = converted

    if (result["Tenure Months"].dropna() < 0).any():
        raise ValueError("Tenure Months cannot be negative.")

    if (result[["Monthly Charges", "Total Charges"]].dropna() < 0).any().any():
        raise ValueError("Charges cannot be negative.")

    if (result["Tenure Months"].dropna() > 72).any():
        raise ValueError("Tenure Months cannot exceed 72.")

    # Validate categorical features.
    for column, allowed_values in CATEGORICAL_FEATURES.items():
        invalid_values = (
            result[column].notna()
            & ~result[column].isin(allowed_values)
        )

        if invalid_values.any():
            values = result.loc[invalid_values, column].unique().tolist()

            raise ValueError(
                f"Invalid values in '{column}': {values}"
            )

    # Pass only the 19 model features to the pipeline.
    model_input = result[REQUIRED_FEATURES]

    # Generate predictions.
    probabilities = model.predict_proba(model_input)[:, 1]

    threshold = settings.model_threshold
    predictions = probabilities >= threshold

    # Add prediction results.
    result["Churn Probability"] = probabilities.round(4)
    result["Predicted Churn"] = predictions.astype(int)

    result["Risk Level"] = [
        "Low" if probability < threshold
        else "Medium" if probability < 0.60
        else "High"
        for probability in probabilities
    ]

    return result
