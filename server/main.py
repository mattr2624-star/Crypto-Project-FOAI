from fastapi import FastAPI
from .model_server import router as model_router

app = FastAPI(
    title="Crypto Volatility API",
    version="1.0",
    description="Predict short-horizon volatility spikes",
)

# Mount router (version + predict)
app.include_router(model_router)

@app.get("/health")
def health():
    return {"status": "ok"}
