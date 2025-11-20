from pydantic import BaseModel
from typing import List, Dict


class PredictPayload(BaseModel):
    rows: List[Dict]


class PredictResponse(BaseModel):
    volatility_score: float
    model_name: str
    model_version: str
    model_variant: str
    latency_ms: float
