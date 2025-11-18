import logging
from fastapi import FastAPI

from .config import get_settings
from .model_loader import load_model

logger = logging.getLogger("crypto-vol-api")

app = FastAPI(
    title="Crypto Volatility Real-Time Service",
    version="0.1.0"
)

settings = get_settings()
model = None


# -------------------------------------------------------
# STARTUP — Load model ONLY
# -------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global model

    print("🚀 STARTUP EVENT (model load only)")
    logger.info("API startup: model load only")

    try:
        model = load_model(settings.model_variant)
        logger.info(
            "Model loaded: name=%s version=%s variant=%s",
            model.name, model.version, model.variant
        )
        print("✔️ Model loaded")
    except Exception as e:
        print("❌ Model load failed:", e)
        logger.exception("Model load failed")


# -------------------------------------------------------
# SHUTDOWN
# -------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 SHUTDOWN EVENT")
    logger.info("API shutdown")


# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": settings.service_name}


@app.get("/version")
async def version():
    return {
        "model_name": getattr(model, "name", "none"),
        "model_version": getattr(model, "version", "none"),
        "model_variant": getattr(model, "variant", "none"),
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(payload: dict):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        return model.predict(payload)
    except Exception as e:
        return {"error": str(e), "received": payload}
