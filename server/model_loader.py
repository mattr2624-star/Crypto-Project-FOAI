import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from .config import get_settings


class LoadedModel:
    def __init__(self, name, version, variant, model, feature_cols, threshold=None):
        self.name = name
        self.version = version
        self.variant = variant
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold  # For later drift or spike classification use

    def _align_features(self, features: dict) -> pd.DataFrame:
        df = pd.DataFrame([features])

        # Add missing columns
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Keep order, remove infinities / NaNs
        df = df[self.feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
        return df

    def predict(self, row: dict) -> float:
        aligned = self._align_features(row)
        return float(self.model.predict_proba(aligned)[0][1])


@lru_cache
def load_model(variant: str) -> LoadedModel:
    settings = get_settings()

    uri_map = {
        "ml": settings.model_uri_ml,
        "baseline": settings.model_uri_baseline,
        "student1": settings.model_uri_student1,
        "student2": settings.model_uri_student2,
    }

    if variant not in uri_map:
        raise ValueError(f"Unknown model variant: {variant}")

    path = uri_map[variant].replace("file:", "").replace("file://", "")
    raw = joblib.load(path)

    if not isinstance(raw, dict):
        raise RuntimeError("Expected a dict-wrapped model with keys: model, feature_cols, high_vol_threshold")

    model = raw.get("model", None)
    feature_cols = raw.get("feature_cols", None)
    threshold = raw.get("high_vol_threshold", None)

    if model is None or not hasattr(model, "predict_proba"):
        raise RuntimeError("Model missing or does not support predict_proba.")

    if feature_cols is None or not isinstance(feature_cols, (list, tuple)):
        raise RuntimeError("Missing feature_cols array in saved model.")

    return LoadedModel(
        name=f"crypto-vol-{variant}",
        version="local-week4",
        variant=variant,
        model=model,
        feature_cols=list(feature_cols),
        threshold=threshold,
    )


def predict_row(row: dict, variant: str) -> float:
    return load_model(variant).predict(row)
