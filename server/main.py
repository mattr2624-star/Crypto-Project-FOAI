from fastapi import FastAPI

from .model_server import router as model_router
from .instrumentation import setup_instrumentation

app = FastAPI(
    title="Crypto Volatility API",
    version="1.0",
    description="Predict short-horizon volatility spikes for BTC/USD.",
)

# Mount the model router (/predict, /version)
app.include_router(model_router)


@app.get("/health")
def health():
    """
    Lightweight health check for CI and ops.
    """
    return {"status": "ok"}


# Attach Prometheus /metrics endpoint and HTTP metrics
setup_instrumentation(app)
