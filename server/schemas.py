from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Tick(BaseModel):
    ts: datetime = Field(..., description="UTC timestamp of the tick")
    price: float = Field(..., description="BTC-USD price at time ts")
    volume: Optional[float] = Field(
        None,
        description="Trade volume in this tick (if available)",
    )


class PredictRequest(BaseModel):
    ts: datetime = Field(..., description="UTC timestamp of the observation")
    price: float = Field(..., description="BTC-USD price at time ts")
    volume: Optional[float] = Field(
        None,
        description="Trade volume in this tick (if available)",
    )


class PredictResponse(BaseModel):
    volatility_score: float = Field(..., description="Predicted volatility score")
    model_name: str = Field(..., description="Name of the underlying model")
    model_version: str = Field(..., description="Model version or run id")
    model_variant: str = Field(..., description="Variant identifier (ml|baseline|student1|student2)")
    latency_ms: float = Field(..., description="Approximate inference latency in ms")
