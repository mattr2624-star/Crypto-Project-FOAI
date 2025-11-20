from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- Routing / variant selection ----
    model_variant: str = "ml"  # default; can be overridden with MODEL_VARIANT env

    # ---- MLflow (for later remote loading) ----
    mlflow_tracking_uri: str = "http://mlflow:5000"

    # ---- Local model URIs (Week 4/5 local pickle) ----
    # Using local joblib pickle wrapped as a dict:
    # { "model": sklearn_estimator, "feature_cols": [...], "high_vol_threshold": float }
    model_uri_ml: str = "file:/app/models/gbm_volatility.pkl"
    model_uri_baseline: str = "file:/app/models/gbm_volatility.pkl"
    model_uri_student1: str = "file:/app/models/gbm_volatility.pkl"
    model_uri_student2: str = "file:/app/models/gbm_volatility.pkl"

    # ---- Metrics ----
    # Prometheus metric namespace; override via METRICS_NAMESPACE env if desired
    metrics_namespace: str = "crypto_api"

    class Config:
        env_prefix = ""  # allow plain env var names like MODEL_VARIANT, METRICS_NAMESPACE
        # Avoid pydantic complaining about "model_*" attributes
        protected_namespaces = ("settings_",)


settings = Settings()


def get_settings() -> Settings:
    return settings
