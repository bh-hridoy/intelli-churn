
import mlflow
import mlflow.sklearn

from app.core.config import settings


def load_model():
    """Load the trained model from MLflow."""

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model = mlflow.sklearn.load_model(
        settings.mlflow_model_uri
    )

    return model
