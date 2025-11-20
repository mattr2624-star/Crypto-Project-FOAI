from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings

_settings = get_settings()

# Latency in seconds (Prometheus best practice), with path/method/status labels
REQUEST_LATENCY = Histogram(
    name=f"{_settings.metrics_namespace}_request_latency_seconds",
    documentation="Request latency in seconds",
    labelnames=("path", "method", "status_code"),
)

# Count of predictions served, tagged by model_variant and status
PREDICTION_COUNTER = Counter(
    name=f"{_settings.metrics_namespace}_predictions_total",
    documentation="Number of predictions served",
    labelnames=("model_variant", "status"),
)


def setup_instrumentation(app):
    """
    Attach Prometheus instrumentation to the FastAPI app.
    - Standard HTTP metrics via Instrumentator at /metrics
    """
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def record_prediction(path: str, method: str, status_code: int, latency_seconds: float, model_variant: str, ok: bool):
    """
    Custom helper to record prediction-level metrics from /predict handler.
    """
    REQUEST_LATENCY.labels(
        path=path,
        method=method,
        status_code=str(status_code),
    ).observe(latency_seconds)

    PREDICTION_COUNTER.labels(
        model_variant=model_variant,
        status="success" if ok else "error",
    ).inc()
