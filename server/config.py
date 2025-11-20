from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_variant: str = "ml"
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # ---- Load Local Pickle Model for Week 4 ----
    model_uri_ml: str = "file:/app/models/gbm_volatility.pkl"
    model_uri_baseline: str = "file:/app/models/gbm_volatility.pkl"
    model_uri_student1: str = "file:/app/models/gbm_volatility.pkl"
    model_uri_student2: str = "file:/app/models/gbm_volatility.pkl"

    class Config:
        env_prefix = ""  # allow plain env var names

settings = Settings()

def get_settings():
    return settings
