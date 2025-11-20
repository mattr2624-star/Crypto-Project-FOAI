from fastapi import APIRouter, Request, HTTPException
from .schemas import PredictPayload, PredictResponse
from .model_loader import load_model, predict_row
from .config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/version")
def version(request: Request):
    requested = request.query_params.get("variant", settings.model_variant)
    return {
        "default_model_variant": settings.model_variant,
        "requested_variant": requested,
    }


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictPayload, request: Request):
    try:
        variant = request.query_params.get("variant", settings.model_variant)
        row = payload.rows[0]
        score = predict_row(row, variant)
        return PredictResponse(
            volatility_score=score,
            model_name=f"crypto-vol-{variant}",
            model_version="latest",
            model_variant=variant,
            latency_ms=0,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
