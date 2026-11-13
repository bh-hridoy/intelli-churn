

import io
import pandas as pd

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app.services.batch_prediction_service import predict_batch_churn

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




@router.post("/predict/batch")
def predict_batch(
    request: Request,
    file: UploadFile = File(...),
):
    # Validate file type.
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        # Read uploaded CSV.
        data = pd.read_csv(file.file)

        # Retrieve loaded model.
        model = request.app.state.model

        # Generate batch predictions.
        results = predict_batch_churn(
            data=data,
            model=model,
        )

        # Convert results to CSV.
        output = io.StringIO()
        results.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="intelli_churn_predictions.csv"'
                )
            },
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV file is empty.",
        )

    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="Could not parse the CSV file.",
        )

    finally:
        file.file.close()
