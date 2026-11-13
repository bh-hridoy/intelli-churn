
from typing import Literal

from pydantic import BaseModel, Field


YesNo = Literal["Yes", "No"]


class CustomerPredictionRequest(BaseModel):
    # Numerical features
    tenure_months: int = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)

    # Customer information
    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo

    # Phone services
    phone_service: YesNo
    multiple_lines: Literal[
        "Yes", "No", "No phone service"
    ]

    # Internet services
    internet_service: Literal[
        "DSL", "Fiber optic", "No"
    ]
    online_security: Literal[
        "Yes", "No", "No internet service"
    ]
    online_backup: Literal[
        "Yes", "No", "No internet service"
    ]
    device_protection: Literal[
        "Yes", "No", "No internet service"
    ]
    tech_support: Literal[
        "Yes", "No", "No internet service"
    ]
    streaming_tv: Literal[
        "Yes", "No", "No internet service"
    ]
    streaming_movies: Literal[
        "Yes", "No", "No internet service"
    ]

    # Contract and billing
    contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class CustomerPredictionResponse(BaseModel):
    churn_probability: float
    predicted_churn: bool
    risk_level: Literal["Low", "Medium", "High"]
    decision_threshold: float
    message: str
