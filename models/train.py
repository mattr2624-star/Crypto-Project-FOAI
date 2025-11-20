import os
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib

DEFAULT_FEATURES_PATH = "data/processed/features_20251113_204606.parquet"
EXPERIMENT_NAME = "crypto_volatility"


# ============================================================
#                     LABEL CREATION
# ============================================================

def create_labels(
    df: pd.DataFrame,
    horizon: int = 60,
    tau_quantile: float = 0.98
) -> Tuple[pd.DataFrame, float]:
    """
    Create volatility labels:
    - Future rolling std of mid_return over next `horizon` rows.
    - Label 1 if >= tau (quantile threshold).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "mid_return" not in df.columns:
        raise ValueError("mid_return column is missing in features parquet.")

    future_vol = (
        df["mid_return"]
        .rolling(window=horizon, min_periods=horizon)
        .std()
        .shift(-horizon + 1)
    )

    df["future_vol"] = future_vol
    df = df.dropna(subset=["future_vol"]).reset_index(drop=True)

    tau = df["future_vol"].quantile(tau_quantile)
    df["label"] = (df["future_vol"] >= tau).astype(int)

    return df, float(tau)


# ============================================================
#                     TIME-BASED SPLIT
# ============================================================

def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split for train → val → test.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)

    train_df = df.iloc[: int(0.70 * n)]
    val_df = df.iloc[int(0.70 * n): int(0.85 * n)]
    test_df = df.iloc[int(0.85 * n):]

    return train_df, val_df, test_df


# ============================================================
#                     FEATURE MATRIX
# ============================================================

def get_feature_matrix(df: pd.DataFrame, feature_cols: List[str]):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    return X, y


# ============================================================
#                     BASELINE MODEL
# ============================================================

def baseline_zscore_volatility(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Baseline detector using z-scored volatility_30s.
    """
    from scipy.special import expit

    if "volatility_30s" not in train_df.columns:
        raise ValueError("volatility_30s missing (featurizer did not generate it).")

    mu = train_df["volatility_30s"].mean()
    sigma = train_df["volatility_30s"].std() + 1e-8

    def make_scores(df):
        z = (df["volatility_30s"] - mu) / sigma
        return expit(z)

    y_test = test_df["label"].values
    scores_test = make_scores(test_df)

    pr_auc = average_precision_score(y_test, scores_test)

    threshold = expit(2.0)  # heuristic
    preds = (scores_test >= threshold).astype(int)
    f1 = f1_score(y_test, preds)

    return pr_auc, f1


# ============================================================
#                     LOGISTIC REGRESSION MODEL
# ============================================================

def train_logreg_model(train_df, val_df, test_df, feature_cols):
    X_train, y_train = get_feature_matrix(train_df, feature_cols)
    X_val, y_val = get_feature_matrix(val_df, feature_cols)
    X_test, y_test = get_feature_matrix(test_df, feature_cols)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, n_jobs=-1))
        ]
    )

    pipe.fit(X_train, y_train)

    val_scores = pipe.predict_proba(X_val)[:, 1]
    test_scores = pipe.predict_proba(X_test)[:, 1]

    val_pr_auc = average_precision_score(y_val, val_scores)
    test_pr_auc = average_precision_score(y_test, test_scores)

    val_f1 = f1_score(y_val, (val_scores >= 0.5).astype(int))
    test_f1 = f1_score(y_test, (test_scores >= 0.5).astype(int))

    metrics = {
        "val_pr_auc": float(val_pr_auc),
        "test_pr_auc": float(test_pr_auc),
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
    }

    return pipe, metrics


# ============================================================
#                     MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output_model", type=str, default="models/model_latest.pkl")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--tau_quantile", type=float, default=0.98)
    args = parser.parse_args()

    print(f"Loading features from: {args.features}")
    df = pd.read_parquet(args.features)

    # Fix timestamp column if needed
    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])
        else:
            raise ValueError("No `timestamp` or `time` column found.")

    # -----------------------------
    #  Label creation
    # -----------------------------
    df_labeled, tau = create_labels(df, horizon=args.horizon, tau_quantile=args.tau_quantile)
    print(f"Label threshold tau = {tau:.6f}")
    print(f"Positive rate = {df_labeled['label'].mean():.4f}")

    # -----------------------------
    #  Time split
    # -----------------------------
    train_df, val_df, test_df = time_split(df_labeled)

    print(f"Train size = {len(train_df)}, Val = {len(val_df)}, Test = {len(test_df)}")
    print(f"Train positives = {train_df['label'].sum()}, "
          f"Val positives = {val_df['label'].sum()}, "
          f"Test positives = {test_df['label'].sum()}")

    # Feature columns
    feature_cols = [
        c for c in [
            "mid_return", "spread", "trade_intensity", "volatility_30s",
            "last_size", "best_bid", "best_ask", "price"
        ] if c in df_labeled.columns
    ]
    print("Using features:", feature_cols)

    # -----------------------------
    #  MLflow setup
    # -----------------------------
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -----------------------------
    #  Baseline
    # -----------------------------
    with mlflow.start_run(run_name="baseline_zscore"):
        pr_auc, f1 = baseline_zscore_volatility(train_df, test_df)
        mlflow.log_metric("baseline_test_pr_auc", pr_auc)
        mlflow.log_metric("baseline_test_f1", f1)
        print(f"[Baseline] PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

    # -----------------------------
    #  Logistic regression
    # -----------------------------
    with mlflow.start_run(run_name="logreg_model"):
        mlflow.log_param("horizon_rows", args.horizon)
        mlflow.log_param("tau_quantile", args.tau_quantile)
        mlflow.log_param("features", ",".join(feature_cols))

        model, metrics = train_logreg_model(train_df, val_df, test_df, feature_cols)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        print("[LogReg] Metrics:", metrics)

        # Log full model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save local .pkl
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        joblib.dump(model, args.output_model)

        print(f"Saved local model to: {args.output_model}")


if __name__ == "__main__":
    main()
