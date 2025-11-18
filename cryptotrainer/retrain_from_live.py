import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Paths
LIVE_FILE = r"C:\cp\data\btcusd_ticks_10min.csv"   # from Coinbase collector
MODEL_PATH = r"C:\cp\models\gbm_volatility.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of your featurizer.compute_features, adapted for offline training.
    Expects columns: price, best_bid, best_ask, last_size, time
    """
    for col in ["price", "best_bid", "best_ask", "last_size"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").ffill()

    df["midprice"] = (df["best_bid"] + df["best_ask"]) / 2
    df["mid_return"] = df["midprice"].pct_change().fillna(0)
    df["spread"] = df["best_ask"] - df["best_bid"]

    if "last_size" in df.columns:
        df["trade_intensity"] = (
            pd.to_numeric(df["last_size"], errors="coerce")
            .fillna(0)
            .rolling(window=5, min_periods=1)
            .sum()
        )
    else:
        df["trade_intensity"] = 0

    df["volatility_30s"] = (
        df["mid_return"].rolling(window=30, min_periods=5).std().fillna(0)
    )

    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    return df


def load_live_ticks(path: str) -> pd.DataFrame:
    """
    Load the Coinbase live tick CSV and adapt it to the schema expected
    by compute_features (best_bid, best_ask, last_size, time, price).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Live tick file not found: {path}")

    df = pd.read_csv(path)

    # Expect columns from collect_live_ticks.py: timestamp, price, volume
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError("CSV must contain at least 'timestamp' and 'price' columns.")

    # Map to featurizer schema
    df["time"] = df["timestamp"]
    df["last_size"] = df.get("volume", 0.0)

    # We don't have real order book, so approximate a tiny spread around price
    df["best_bid"] = df["price"] * (1 - 0.0005)  # -5 bps
    df["best_ask"] = df["price"] * (1 + 0.0005)  # +5 bps

    # Optional: featurizer filters on type == "ticker", so add it
    df["type"] = "ticker"

    df = df.sort_values("time")
    return df


def build_dataset(features: pd.DataFrame):
    """
    Build feature matrix X and binary target y:

    y_t = 1 if future volatility_30s (30 steps ahead) > 90th percentile, else 0.
    """
    df = features.copy()

    # Future volatility_30s over the next window (shift backwards)
    df["future_vol_30s"] = df["volatility_30s"].shift(-30)

    df = df.dropna(subset=["future_vol_30s"])

    # Threshold = 90th percentile of future vol in this dataset
    tau = df["future_vol_30s"].quantile(0.9)
    df["y_high_vol"] = (df["future_vol_30s"] > tau).astype(int)

    feature_cols = [
        "midprice",
        "mid_return",
        "spread",
        "trade_intensity",
        "volatility_30s",
    ]

    X = df[feature_cols].values
    y = df["y_high_vol"].values

    return X, y, feature_cols, tau


def train_model(X, y):
    """
    Train a GradientBoostingClassifier to predict probability of high volatility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("ROC AUC (test):", auc)
    print(classification_report(y_test, (y_proba > 0.5).astype(int)))

    return clf, auc


def main():
    print("📥 Loading live ticks from:", LIVE_FILE)
    df_raw = load_live_ticks(LIVE_FILE)

    print("✅ Loaded", len(df_raw), "raw ticks")
    features = compute_features(df_raw)
    print("✅ Computed features:", features.columns.tolist())

    X, y, feature_cols, tau = build_dataset(features)
    print(f"✅ Built dataset with {X.shape[0]} rows, high-vol threshold tau={tau:.6f}")

    clf, auc = train_model(X, y)

    artifact = {
        "model": clf,
        "feature_cols": feature_cols,
        "high_vol_threshold": float(tau),
        "trained_at": datetime.utcnow().isoformat(),
        "auc_test": float(auc),
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"💾 Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
