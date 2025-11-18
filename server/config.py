from functools import lru_cache
from typing import Literal

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # Service
    service_name: str = Field("crypto-vol-api", description="Service name")
    environment: str = Field("dev", description="Environment (dev|staging|prod)")

    # Kafka
    # IMPORTANT: Use the host-visible listener (127.0.0.1:29092)
    kafka_bootstrap_servers: str = Field(
        "127.0.0.1:29092",
        description="Kafka bootstrap servers (host:port)",
    )
    kafka_ticks_topic: str = Field(
        "btc_ticks",
        description="Kafka topic for BTC-USD ticks",
    )
    kafka_consumer_group: str = Field(
        "server-consumer",
        description="Kafka consumer group id for API service",
    )

    # Model / MLflow
    model_variant: Literal["ml", "baseline", "student1", "student2"] = Field(
        "ml",
        description="Which model variant to serve",
    )
    model_uri_ml: str = Field(
        "models:/crypto-vol-ml/latest",
        description="MLflow model URI for main model",
    )
    model_uri_baseline: str = Field(
        "models:/crypto-vol-baseline/latest",
        description="MLflow model URI for baseline model",
    )
    model_uri_student1: str = Field(
        "models:/crypto-vol-student1/latest",
        description="MLflow model URI for student 1 model",
    )
    model_uri_student2: str = Field(
        "models:/crypto-vol-student2/latest",
        description="MLflow model URI for student 2 model",
    )
    mlflow_tracking_uri: str = Field(
        "http://mlflow:5000",
        description="MLflow tracking server URI",
    )

    # Prometheus / metrics
    metrics_namespace: str = Field(
        "crypto_vol",
        description="Prometheus metrics namespace",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
