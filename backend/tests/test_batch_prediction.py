
import pandas as pd
import pytest
import numpy as np

from app.services.batch_prediction_service import (
    REQUIRED_FEATURES,
    predict_batch_churn,
)


class MockModel:
    """A lightweight model for testing without loading the real ML model."""

    def predict_proba(self, data):
        return np.array([
            [0.20, 0.80],
            [0.75, 0.25],
            [0.40, 0.60],
        ][:len(data)])


def create_valid_data():
    """Create three valid customer records for testing."""

    return pd.DataFrame([
        {
            "Tenure Months": 2,
            "Monthly Charges": 85.5,
            "Total Charges": 171.0,
            "Gender": "Male",
            "Senior Citizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
        },
        {
            "Tenure Months": 60,
            "Monthly Charges": 65.0,
            "Total Charges": 3900.0,
            "Gender": "Female",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "Yes",
            "Device Protection": "Yes",
            "Tech Support": "Yes",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": "Two year",
            "Paperless Billing": "No",
            "Payment Method": "Bank transfer (automatic)",
        },
        {
            "Tenure Months": 12,
            "Monthly Charges": 95.0,
            "Total Charges": 1140.0,
            "Gender": "Male",
            "Senior Citizen": "Yes",
            "Partner": "No",
            "Dependents": "No",
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "Yes",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
        },
    ], columns=REQUIRED_FEATURES)


def test_successful_batch_prediction():
    data = create_valid_data()
    model = MockModel()

    result = predict_batch_churn(data, model)

    assert len(result) == 3
    assert "Churn Probability" in result.columns
    assert "Predicted Churn" in result.columns
    assert "Risk Level" in result.columns

    assert result["Predicted Churn"].tolist() == [1, 0, 1]
    assert result["Risk Level"].tolist() == ["High", "Low", "High"]


def test_missing_required_column():
    data = create_valid_data().drop(columns=["Contract"])

    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch_churn(data, MockModel())


def test_invalid_categorical_value():
    data = create_valid_data()
    data.loc[0, "Gender"] = "Unknown"

    with pytest.raises(ValueError, match="Invalid values"):
        predict_batch_churn(data, MockModel())


def test_negative_charges():
    data = create_valid_data()
    data.loc[0, "Monthly Charges"] = -50

    with pytest.raises(ValueError, match="Charges cannot be negative"):
        predict_batch_churn(data, MockModel())


def test_empty_dataframe():
    data = pd.DataFrame(columns=REQUIRED_FEATURES)

    with pytest.raises(ValueError, match="no customer records"):
        predict_batch_churn(data, MockModel())


def test_unexpected_column():
    data = create_valid_data()
    data["Unexpected Feature"] = "value"

    with pytest.raises(ValueError, match="Unexpected columns"):
        predict_batch_churn(data, MockModel())
