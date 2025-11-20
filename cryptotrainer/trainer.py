import os
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

from dotenv import load_dotenv
from cryptotrainer.utils import preprocess_data

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
load_dotenv()

DATA_PATH = "/app/data/processed"
MODEL_PATH = os.environ.get("MODEL_OUTPUT_PATH", "/app/models/gbm_volatility.pkl")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

MODEL_NAME = "crypto-vol-ml"  # 🔥 centralized name


# ----------------------------------------------------------------------
# LOAD LATEST FEATURE PARQUET FILE
# ----------------------------------------------------------------------
def load_latest_features():
    print("📂 Scanning:", DATA_PATH)
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".parquet")]

    if not files:
        raise FileNotFoundError("❌ No parquet features found in /app/data/processed")

    files.sort()
    latest = files[-1]

    print(f"📦 Loading latest features: {latest}")
    df = pd.read_parquet(os.path.join(DATA_PATH, latest))
    return df


# ----------------------------------------------------------------------
# TRAINING PIPELINE
# ----------------------------------------------------------------------
def train_model():

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("crypto_volatility")

    print("🚀 Training pipeline started.")

    df = load_latest_features()
    df = preprocess_data(df)

    # ------------------------------
    # Features used for GBM model
    # ------------------------------
    FEATURE_COLS = ["midprice", "spread", "trade_intensity", "volatility_30s"]
    X = df[FEATURE_COLS].fillna(0)

    # ------------------------------
    # Target: high volatility next window
    # ------------------------------
    vol = df["volatility_30s"].fillna(0)
    tau = np.percentile(vol, 90)
    y = (vol > tau).astype(int)

    print(f"📊 High-volatility threshold τ = {tau}")

    # ----------------------------------------------------------------------
    # Hyperparameter searches
    # ----------------------------------------------------------------------
    base_gbm = GradientBoostingClassifier(random_state=42)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.005, 0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 3, 5]
    }

    print("\n🔥 Stage 1: RandomizedSearchCV")
    random_search = RandomizedSearchCV(
        estimator=base_gbm,
        param_distributions=param_dist,
        n_iter=40,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    random_search.fit(X, y)

    best_stage1_model = random_search.best_estimator_
    print("🔥 Stage 1 best params:", random_search.best_params_)

    # ----------------------------------------------------------------------
    # Fine tuning
    # ----------------------------------------------------------------------
    print("\n🎯 Stage 2: GridSearchCV (fine tuning)")
    grid_params = {
        "n_estimators": [
            random_search.best_params_["n_estimators"] - 50,
            random_search.best_params_["n_estimators"],
            random_search.best_params_["n_estimators"] + 50
        ],
        "learning_rate": [random_search.best_params_["learning_rate"]],
        "max_depth": [
            random_search.best_params_["max_depth"] - 1,
            random_search.best_params_["max_depth"],
            random_search.best_params_["max_depth"] + 1,
        ],
        "subsample": [random_search.best_params_["subsample"]],
        "min_samples_split": [random_search.best_params_["min_samples_split"]],
        "min_samples_leaf": [random_search.best_params_["min_samples_leaf"]],
    }

    grid_search = GridSearchCV(
        estimator=best_stage1_model,
        param_grid=grid_params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X, y)

    final_model = grid_search.best_estimator_
    print("🏆 Final best params:", grid_search.best_params_)

    # ----------------------------------------------------------------------
    # MLflow Logging + Registry
    # ----------------------------------------------------------------------
    with mlflow.start_run() as run:

        mlflow.log_params({"tau_threshold": float(tau), **grid_search.best_params_})
        mlflow.log_metric("train_accuracy", final_model.score(X, y))

        # Inference signature
        signature = infer_signature(X, final_model.predict(X))

        print("\n💾 Logging model to MLflow Registry:", MODEL_NAME)

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature
        )

        # Save artifact to disk as backup
        artifact = {
            "model": final_model,
            "feature_cols": FEATURE_COLS,
            "high_vol_threshold": float(tau),
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)

    print("\n🎉 Training Completed + Model Registered!")
    return final_model


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
