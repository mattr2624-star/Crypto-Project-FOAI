# server/model_server.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
import joblib
import os
from kafka import KafkaProducer
import json
from datetime import datetime
import numpy as np
from time import perf_counter

from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/gbm_volatility.pkl")
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "model-events")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "gbm_live_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gbm_volatility")
GIT_SHA = os.environ.get("GIT_SHA", "unknown")

app = FastAPI(title="Crypto Volatility Model Server")

# -------------------------------------------------------------------------
# PROMETHEUS METRICS
# -------------------------------------------------------------------------
model_requests_total = Counter(
    "model_requests_total", "Total number of prediction requests"
)
model_errors_total = Counter(
    "model_errors_total", "Total number of failed prediction requests"
)
model_retrains_total = Counter(
    "model_retrains_total", "Total number of retrain events"
)
model_request_latency_seconds = Histogram(
    "model_request_latency_seconds", "Prediction request latency in seconds"
)

# -------------------------------------------------------------------------
# LOAD MODEL ARTIFACT
# -------------------------------------------------------------------------
artifact = None
model = None
feature_cols = None
high_vol_threshold = None

if os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
        high_vol_threshold = artifact.get("high_vol_threshold", None)
        print(f"✅ Loaded model from {MODEL_PATH}")
        print(f"   feature_cols={feature_cols}")
        print(f"   high_vol_threshold={high_vol_threshold}")
    except Exception as e:
        print(f"❌ Failed to load model artifact from {MODEL_PATH}: {e}")
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}")

# -------------------------------------------------------------------------
# KAFKA PRODUCER (lazy init so startup doesn’t crash if Kafka is not ready)
# -------------------------------------------------------------------------
_producer = None


def get_producer():
    global _producer
    if _producer is None:
        try:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"✅ KafkaProducer connected to {KAFKA_BROKER}")
        except Exception as e:
            print(f"⚠️ Could not create KafkaProducer: {e}")
            _producer = None
    return _producer


# -------------------------------------------------------------------------
# HEALTH ENDPOINT
# -------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Simple health check for orchestration & monitoring.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "kafka_broker": KAFKA_BROKER,
        "kafka_connected": get_producer() is not None,
        "version": MODEL_VERSION,
    }


# -------------------------------------------------------------------------
# VERSION ENDPOINT
# -------------------------------------------------------------------------
@app.get("/version")
def version():
    """
    Return model metadata for debugging & auditing.
    """
    return {
        "model": MODEL_NAME,
        "model_variant": "ml",  # Week 6 will introduce baseline vs ml
        "version": MODEL_VERSION,
        "git_sha": GIT_SHA,
        "feature_cols": feature_cols,
    }


# -------------------------------------------------------------------------
# PROMETHEUS METRICS ENDPOINT
# -------------------------------------------------------------------------
@app.get("/metrics")
def metrics():
    """
    Expose Prometheus metrics.
    """
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


# -------------------------------------------------------------------------
# PREDICT ENDPOINT (Assignment contract)
# -------------------------------------------------------------------------
@app.post("/predict")
def predict(payload: dict):
    """
    POST /predict
    Request:
    {
      "rows": [
        { "midprice": 100000.0, "spread": 1.2, "trade_intensity": 15, "volatility_30s": 0.0003 }
      ]
    }

    Response:
    {
      "scores": [0.74],
      "model_variant": "ml",
      "version": "gbm_live_v1",
      "ts": "2025-11-02T14:33:00Z"
    }
    """
    start = perf_counter()
    model_requests_total.inc()

    if model is None or feature_cols is None:
        model_errors_total.inc()
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded on server", "model_path": MODEL_PATH},
        )

    rows = payload.get("rows")
    if not rows or not isinstance(rows, list):
        model_errors_total.inc()
        return JSONResponse(
            status_code=400,
            content={"error": "Payload must contain non-empty 'rows' list"},
        )

    # Build feature matrix in the correct column order
    X = []
    try:
        for row in rows:
            x_vec = [row[col] for col in feature_cols]
            X.append(x_vec)
    except KeyError as e:
        model_errors_total.inc()
        return JSONResponse(
            status_code=400,
            content={"error": f"Missing feature in request: {str(e)}"},
        )

    X = np.asarray(X)

    try:
        scores = model.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        model_errors_total.inc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Model prediction failed: {str(e)}"},
        )
    finally:
        elapsed = perf_counter() - start
        model_request_latency_seconds.observe(elapsed)

    return {
        "scores": scores,
        "model_variant": "ml",
        "version": MODEL_VERSION,
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


# -------------------------------------------------------------------------
# RETRAIN EVENT EMITTER (optional, already useful for later weeks)
# -------------------------------------------------------------------------
@app.post("/retrain")
def retrain(data: dict):
    """
    Emit a 'model_retrained' event to Kafka.
    """
    model_retrains_total.inc()
    event = {
        "event": "model_retrained",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_path": MODEL_PATH,
        "metrics": data.get("metrics", {}),
        "version": MODEL_VERSION,
    }

    producer = get_producer()
    if producer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Kafka producer not available", "broker": KAFKA_BROKER},
        )

    try:
        producer.send(KAFKA_TOPIC, event)
        producer.flush()
    except Exception as e:
        model_errors_total.inc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to send retrain event: {str(e)}"},
        )

    return {"status": "retrained_event_emitted", "topic": KAFKA_TOPIC}
