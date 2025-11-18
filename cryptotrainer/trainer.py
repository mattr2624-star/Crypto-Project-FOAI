import os
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import mlflow
from dotenv import load_dotenv
from cryptotrainer.utils import preprocess_data

load_dotenv()

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = "/app/data/processed"
MODEL_PATH = os.environ.get("MODEL_OUTPUT_PATH", "/app/models/gbm_volatility.pkl")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


# ----------------------------------------------------------------------
# LOAD LATEST FEATURE PARQUET FILE
# ----------------------------------------------------------------------
def load_latest_features():
    print("Scanning:", DATA_PATH)
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".parquet")]

    if not files:
        raise FileNotFoundError("❌ No parquet features found in /app/data/processed")

    files.sort()
    latest = files[-1]

    print(f"📦 Loading latest features: {latest}")
    df = pd.read_parquet(os.path.join(DATA_PATH, latest))
    return df


# ----------------------------------------------------------------------
# MAIN TRAINING PIPELINE
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
    # Target construction (high volatility next window)
    # ------------------------------
    vol = df["volatility_30s"].fillna(0)
    tau = np.percentile(vol, 90)
    y = (vol > tau).astype(int)

    print(f"📊 High-volatility threshold τ = {tau}")

    # ------------------------------------------------------------------
    # 🔥 STAGE 1 — RANDOMIZED SEARCH (large search space)
    # ------------------------------------------------------------------
    print("\n===============================")
    print("🔥 Stage 1: RandomizedSearchCV")
    print("===============================\n")

    base_gbm = GradientBoostingClassifier(random_state=42)

    param_dist = {
        "n_estimators": [100, 200, 300, 500, 800],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 10, 20, 40],
        "min_samples_leaf": [1, 3, 5, 10]
    }

    random_search = RandomizedSearchCV(
        estimator=base_gbm,
        param_distributions=param_dist,
        n_iter=60,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print("🔍 Running wide hyperparameter search...")
    random_search.fit(X, y)

    stage1_params = random_search.best_params_
    print("\n✅ Stage 1 best parameters:", stage1_params)
    print("📈 Stage 1 best CV ROC-AUC:", random_search.best_score_)

    best_stage1_model = random_search.best_estimator_


    # ------------------------------------------------------------------
    # 🔥 STAGE 2 — GRID SEARCH (fine tuning around Stage 1 best params)
    # ------------------------------------------------------------------
    print("\n===============================")
    print("🔥 Stage 2: GridSearchCV (fine tuning)")
    print("===============================\n")

    # Helper to keep values positive and valid
    def around(val, choices):
        out = []
        for c in choices:
            x = val * c
            if x > 0:
                out.append(x)
        return sorted(set(out))

    grid_params = {
        "n_estimators": [stage1_params["n_estimators"] - 100,
                         stage1_params["n_estimators"],
                         stage1_params["n_estimators"] + 100],

        "learning_rate": around(stage1_params["learning_rate"], [0.5, 1, 1.5]),

        "max_depth": [
            max(1, stage1_params["max_depth"] - 1),
            stage1_params["max_depth"],
            stage1_params["max_depth"] + 1
        ],

        "subsample": [stage1_params["subsample"]],

        "min_samples_split": [
            stage1_params["min_samples_split"],
            max(2, stage1_params["min_samples_split"] // 2),
            stage1_params["min_samples_split"] + 5
        ],

        "min_samples_leaf": [
            stage1_params["min_samples_leaf"],
            max(1, stage1_params["min_samples_leaf"] // 2),
            stage1_params["min_samples_leaf"] + 2
        ]
    }

    grid_search = GridSearchCV(
        estimator=best_stage1_model,
        param_grid=grid_params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    print("🎯 Running second-stage fine-tuning...")
    grid_search.fit(X, y)

    final_params = grid_search.best_params_
    print("\n🎉 Final best GBM parameters:", final_params)
    print("🏆 Final best CV ROC-AUC:", grid_search.best_score_)

    final_model = grid_search.best_estimator_


    # ----------------------------------------------------------------------
    # MLflow Logging + Saving Artifact
    # ----------------------------------------------------------------------
    with mlflow.start_run():

        mlflow.log_params({
            "tau_threshold": float(tau),
            **final_params
        })

        # Train accuracy for reference
        train_acc = final_model.score(X, y)
        mlflow.log_metric("train_accuracy", train_acc)

        # -------------------------------
        # Save full artifact for server
        # -------------------------------
        artifact = {
            "model": final_model,
            "feature_cols": FEATURE_COLS,
            "high_vol_threshold": float(tau),
        }

        print("\n💾 Saving final optimized model →", MODEL_PATH)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)

        mlflow.log_artifact(MODEL_PATH)

    print("\n✅ Training complete.")
    print("📦 Artifact keys:", artifact.keys())

    return artifact


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        train_model()
        print("🎯 Finished end-to-end model training pipeline.")
    except Exception as e:
        print("❌ Training failed:", e)
        raise
