
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: intelli-churn/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

BACKEND_DIR = PROJECT_ROOT / "backend"
ML_DIR = PROJECT_ROOT / "ml"

TRACKING_DB = ML_DIR / "mlflow.db"


class Settings(BaseSettings):
    app_name: str = "IntelliChurn API"
    model_threshold: float = 0.30

    mlflow_model_uri: str

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def mlflow_tracking_uri(self) -> str:
        return f"sqlite:///{TRACKING_DB.resolve().as_posix()}"


settings = Settings()
