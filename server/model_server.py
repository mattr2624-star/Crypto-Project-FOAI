import time
from fastapi import APIRouter, Request, HTTPException

from .schemas import PredictPayload, PredictResponse
from .model_loader import predict_row
from .config import get_settings
from .instrumentation import record_prediction

router = APIRouter()
settings = get_settings()


@router.get("/version")
def version(request: Request):
    """
    Simple version endpoint. Exposes the default model variant,
    and which variant was explicitly requested via ?variant=.
    """
    requested = request.query_params.get("variant", settings.model_variant)
    return {
        "default_model_variant": settings.model_variant,
        "requested_variant": requested,
    }


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload, request: Request):
    """
    Main prediction endpoint.
    - Accepts a list with one row of features.
    - Uses MODEL_VARIANT env by default, but can be overridden via ?variant=.
    - Returns a single volatility_score plus metadata.
    """
    start = time.time()
    variant = request.query_params.get("variant", settings.model_variant)

    try:
        row = payload.rows[0]
        score = predict_row(row, variant)
        latency_seconds = time.time() - start

        # Record success metrics
        record_prediction(
            path=request.url.path,
            method=request.method,
            status_code=200,
            latency_seconds=latency_seconds,
            model_variant=variant,
            ok=True,
        )

        return PredictResponse(
            volatility_score=score,
            model_name=f"crypto-vol-{variant}",
            model_version="local-week4",
            model_variant=variant,
            latency_ms=round(latency_seconds * 1000, 3),
        )
    except Exception as e:
        latency_seconds = time.time() - start

        # Record failure metrics
        record_prediction(
            path=request.url.path,
            method=request.method,
            status_code=400,
            latency_seconds=latency_seconds,
            model_variant=variant,
            ok=False,
        )

        raise HTTPException(status_code=400, detail=str(e))
