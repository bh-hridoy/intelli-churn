
from fastapi import APIRouter, Request

from app.schemas.prediction import (
    CustomerPredictionRequest,
    CustomerPredictionResponse,
)

from app.services.prediction_service import (
    predict_customer_churn,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["Predictions"],
)


@router.post(
    "/predict",
    response_model=CustomerPredictionResponse,
)
def predict(
    customer: CustomerPredictionRequest,
    request: Request,
):
    model = request.app.state.model
    threshold = request.app.state.model_threshold

    result = predict_customer_churn(
        customer=customer,
        model=model,
        threshold=threshold,
    )

    return result
