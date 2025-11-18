from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings

_settings = get_settings()

REQUEST_LATENCY = Histogram(
    name=f"{_settings.metrics_namespace}_request_latency_ms",
    documentation="Request latency in milliseconds",
    labelnames=("path", "method", "status_code"),
)

PREDICTION_COUNTER = Counter(
    name=f"{_settings.metrics_namespace}_predictions_total",
    documentation="Number of predictions served",
    labelnames=("model_variant", "status"),
)


def setup_instrumentation(app):
    """
    Attach Prometheus instrumentation to the FastAPI app.
    - Standard HTTP metrics via Instrumentator
    - Custom metrics defined above
    """
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
