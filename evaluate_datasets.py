import os
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from dotenv import load_dotenv
load_dotenv()

from cryptotrainer.utils import preprocess_data
import mlflow


# ---------------------------------------------------------
# REQUIRED FEATURE SET (same as training)
# ---------------------------------------------------------
REQUIRED_FEATURES = ["midprice", "spread", "trade_intensity", "volatility_30s"]


# =========================================================
#    FEATURE ENGINEERING FOR EACH DATASET TYPE
# =========================================================

def build_features_from_ticks_mine(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Custom feature builder for YOUR dataset:
    columns: ['timestamp', 'price', 'volume'].
    """
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # 1-second resample
    per_sec = df.resample("1s").agg({
        "price": "last",
        "volume": ["sum", "count"]
    })
    per_sec.columns = ["price_last", "volume_sum", "trade_count"]

    per_sec["midprice"] = per_sec["price_last"].ffill()
    per_sec["spread"] = per_sec["price_last"].rolling(5).std().fillna(0)
    per_sec["trade_intensity"] = per_sec["trade_count"].fillna(0)

    per_sec["log_price"] = np.log(per_sec["midprice"].replace(0, np.nan))
    returns = per_sec["log_price"].diff()
    per_sec["volatility_30s"] = returns.rolling(30, min_periods=5).std()

    feat_df = per_sec[REQUIRED_FEATURES].dropna().reset_index(drop=True)
    return feat_df


def build_features_student1(df):
    """
    Convert Student 1's price-only dataset into the 4 ML features.
    """
    print("🔧 Building features for Student 1 (price-only dataset)...")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')

    midprice = df['price'].resample("1s").last().ffill()
    spread = pd.Series(0, index=midprice.index)

    df['count'] = 1
    trade_intensity = df['count'].resample("1s").sum().fillna(0)

    ret = midprice.pct_change()
    volatility_30s = ret.rolling(30).std().fillna(0)

    feat = pd.DataFrame({
        "midprice": midprice,
        "spread": spread,
        "trade_intensity": trade_intensity,
        "volatility_30s": volatility_30s,
    })

    print(f"   - Built features for {len(feat)} seconds")
    return feat.reset_index(drop=True)


def build_features_student2(df):
    """
    Convert Student 2's bid/ask Coinbase ticker dataset into 4 features.
    Columns include: timestamp, best_bid, best_ask, price, ...
    """
    print("🔧 Building features for Student 2 (bid/ask dataset)...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    df["midprice"] = (df["best_bid"] + df["best_ask"]) / 2
    df["spread"] = df["best_ask"] - df["best_bid"]

    df["count"] = 1
    trade_intensity = df["count"].resample("1s").sum().fillna(0)

    midprice_1s = df["midprice"].resample("1s").last().ffill()
    spread_1s = df["spread"].resample("1s").last().fillna(0)

    ret = np.log(midprice_1s).diff()
    volatility_30s = ret.rolling(30).std().fillna(0)

    feat = pd.DataFrame({
        "midprice": midprice_1s,
        "spread": spread_1s,
        "trade_intensity": trade_intensity,
        "volatility_30s": volatility_30s,
    })

    print(f"   - Built features for {len(feat)} seconds")
    return feat.reset_index(drop=True)


# =========================================================
#  UNIVERSAL FEATURE CHECKER / DISPATCHER
# =========================================================

def ensure_features(df_raw: pd.DataFrame, dataset_name: str) -> pd.DataFrame:

    # Already complete?
    if all(col in df_raw.columns for col in REQUIRED_FEATURES):
        return df_raw

    # --- YOUR DATASET ---
    if dataset_name == "mine" and set(df_raw.columns) == {"timestamp", "price", "volume"}:
        print("🔧 Dataset 'mine' looks like raw ticks — building features...")
        return build_features_from_ticks_mine(df_raw)

    # --- STUDENT 1 ---
    if dataset_name == "student1" and set(df_raw.columns) == {"timestamp", "price"}:
        print("🔧 Dataset 'student1' is price-only — building features...")
        return build_features_student1(df_raw)

    # --- STUDENT 2 ---
    if dataset_name == "student2" and "best_bid" in df_raw.columns and "best_ask" in df_raw.columns:
        print("🔧 Dataset 'student2' is Coinbase bid/ask — building features...")
        return build_features_student2(df_raw)

    # --- FALLBACK ---
    print(f"🔧 Dataset '{dataset_name}' missing features — running preprocess_data()...")
    df = preprocess_data(df_raw.copy())
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"❌ After preprocess_data(), '{dataset_name}' still missing: {missing}")
    return df


# =========================================================
# LOAD DATA HELPERS
# =========================================================
DATA_ROOT = "/app/data"
MODEL_PATH = os.environ.get("MODEL_OUTPUT_PATH", "/app/models/gbm_volatility.pkl")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def load_model_artifact(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    return artifact["model"], artifact["feature_cols"], artifact["high_vol_threshold"]


def load_my_data():
    path = f"{DATA_ROOT}/btcusd_ticks_10min.csv"
    print(f"\n📂 Loading YOUR data from: {path}")
    return pd.read_csv(path)


def load_student1_data():
    path = f"{DATA_ROOT}/raw_10min_slice.parquet"
    print(f"\n📂 Loading STUDENT 1 data from: {path}")
    return pd.read_parquet(path)


def load_student2_data():
    path = "/app/data/processed/ticks_BTCUSD_20251109_130539 (1).ndjson"
    print(f"\n📂 Loading STUDENT 2 NDJSON file: {path}")
    return pd.read_json(path, lines=True)


# =========================================================
# BUILD X, y USING SAME THRESHOLD AS TRAINING
# =========================================================

def build_xy(df_feat, feature_cols, tau, dataset_name):
    print(f"\n🛠 Preprocessing dataset: {dataset_name}")
    df = preprocess_data(df_feat.copy())

    X = df[feature_cols].fillna(0)
    vol = df["volatility_30s"].fillna(0)
    y = (vol > tau).astype(int)

    print(f"   - Rows: {len(y)}")
    print(f"   - Positives: {y.sum()} ({100*y.mean():.2f}%)")

    return X, y


# =========================================================
# EVALUATE ONE DATASET
# =========================================================

def evaluate(model, feature_cols, tau, df_raw, name):
    df_feat = ensure_features(df_raw, name)
    X, y = build_xy(df_feat, feature_cols, tau, name)

    if len(np.unique(y)) < 2:
        print(f"⚠ {name}: Only one class → ROC/PR skipped.")
        pred = model.predict(X)
        return {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "n": len(y),
            "positive_rate": float(y.mean())
        }

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "n": len(y),
        "positive_rate": float(y.mean())
    }


# =========================================================
# MAIN
# =========================================================

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("crypto_volatility_eval")

    print("📥 Loading trained model artifact...")
    model, feature_cols, tau = load_model_artifact(MODEL_PATH)
    print("   ✓ Loaded model")
    print("   ✓ feature_cols:", feature_cols)
    print("   ✓ tau:", tau)

    df_mine = load_my_data()
    df_s1 = load_student1_data()
    df_s2 = load_student2_data()

    with mlflow.start_run(run_name="dataset_comparison"):

        results = {}

        for name, df in [
            ("mine", df_mine),
            ("student1", df_s1),
            ("student2", df_s2)
        ]:
            print(f"\n===== Evaluating {name} =====")
            metrics = evaluate(model, feature_cols, tau, df, name)
            results[name] = metrics

            # Log to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(f"{name}_{k}", v)

        mlflow.log_param("tau_used", float(tau))
        mlflow.log_param("feature_cols", ",".join(feature_cols))

    # Print comparison table
    print("\n================ COMPARISON TABLE ================\n")
    comp = pd.DataFrame([
        {"dataset": k, **v} for k, v in results.items()
    ])
    print(comp.to_string(index=False))

    out_path = "/app/models/dataset_comparison.csv"
    comp.to_csv(out_path, index=False)
    print(f"\n💾 Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
