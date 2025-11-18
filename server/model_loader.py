import mlflow.pyfunc
from functools import lru_cache
from .config import get_settings

class LoadedModel:
    def __init__(self, name: str, version: str, variant: str, model):
        self.name = name
        self.version = version
        self.variant = variant
        self.model = model

    def predict(self, payload: dict):
        """Run prediction on a single JSON dict."""
        prediction = self.model.predict([payload])[0]

        return {
            "volatility_score": float(prediction),
            "model_name": self.name,
            "model_version": self.version,
            "model_variant": self.variant,
        }


@lru_cache
def load_model(variant: str) -> LoadedModel:
    """Load an MLflow model once and cache it."""
    settings = get_settings()

    if variant == "ml":
        uri = settings.model_uri_ml
    elif variant == "baseline":
        uri = settings.model_uri_baseline
    elif variant == "student1":
        uri = settings.model_uri_student1
    elif variant == "student2":
        uri = settings.model_uri_student2
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.pyfunc.load_model(uri)

    return LoadedModel(
        name=f"dummy-volatility-{variant}",
        version="0.1.0",
        variant=variant,
        model=model
    )
