
import pandas as pd

from app.schemas.prediction import CustomerPredictionRequest
from app.core.config import settings


# API field names -> exact ML training column names
FEATURE_MAPPING = {
    "tenure_months": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
}


def predict_customer_churn(
    customer: CustomerPredictionRequest,
    model,
    threshold: float,
):
    """
    Generate a churn prediction for one customer.
    """

    # Convert validated request into a dictionary
    customer_data = customer.model_dump()

    # Rename fields to match the ML training dataset
    model_input = {
        FEATURE_MAPPING[key]: value
        for key, value in customer_data.items()
    }

    # Create a single-row DataFrame
    input_df = pd.DataFrame([model_input])

    # Generate churn probability
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    # Apply selected decision threshold
    predicted_churn = probability >= threshold

    # Determine presentation risk level
    if probability < threshold:
        risk_level = "Low"
    elif probability < 0.60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "churn_probability": round(probability, 4),
        "predicted_churn": predicted_churn,
        "risk_level": risk_level,
        "decision_threshold": threshold,
        "message": (
            "Customer is likely to churn."
            if predicted_churn
            else "Customer is unlikely to churn."
        ),
    }
