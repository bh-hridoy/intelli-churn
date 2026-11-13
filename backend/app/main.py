
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.services.model_service import load_model

from app.api.prediction import router as prediction_router





@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading IntelliChurn model...")

    app.state.model = load_model()
    app.state.model_threshold = settings.model_threshold

    print("IntelliChurn model loaded successfully!")

    yield

    print("Shutting down IntelliChurn API...")


app = FastAPI(
    title=settings.app_name,
    description="Customer Churn Prediction and Analytics Platform",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(prediction_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to IntelliChurn API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "IntelliChurn Backend",
    }


@app.get("/health/model")
def model_health_check():
    return {
        "status": "loaded",
        "model": "Random Forest",
        "threshold": app.state.model_threshold,
    }
