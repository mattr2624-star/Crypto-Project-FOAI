#!/usr/bin/env python3
"""
Train models on the HARD realistic volatility dataset
and save them with metadata so the leaderboard can load them.

Models saved as:
    models/artifacts/<model>_hard_<timestamp>.joblib

Metadata stored:
    {model, imputer, feature_names}

"""

import joblib, numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ---------------------------- Paths ----------------------------
DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

LABEL_COL = "volatility_spike_future"


# ---------------------------- Helpers ----------------------------
def latest(pattern):
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Missing: {pattern}. Run make_harder_dataset.py first.")
    return files[-1]

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_model_bundle(model_name, model, imputer, feature_names):
    out_path = MODEL_DIR / f"{model_name}_hard_{timestamp()}.joblib"
    bundle = {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names
    }
    joblib.dump(bundle, out_path)
    print(f"💾 Saved {model_name} ➜ {out_path}\n")


# ---------------------------- Load HARD data ----------------------------
print("\n📌 Loading HARD TRAIN + TEST...")
train_path = latest("features_hard_train_*.parquet")
test_path  = latest("features_hard_test_*.parquet")

train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)

print(f"   ✔ TRAIN rows: {len(train_df):,}")
print(f"   ✔ TEST  rows: {len(test_df):,}")

# Extract label and numeric features
y_train = train_df[LABEL_COL].astype(int)
X_train = train_df.drop(columns=[LABEL_COL, "timestamp"], errors="ignore")

feature_names = list(X_train.columns)
print(f"   ✔ Features = {len(feature_names)}")

# ---------------------------- Input validation ----------------------------
if len(np.unique(y_train)) < 2:
    raise ValueError("❌ Training set has only one class. Need more spikes or a lower label threshold.")

# ---------------------------- Impute ----------------------------
print("\n🧹 Imputing missing values...")
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)

# ---------------------------- Train models ----------------------------
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=5,
        class_weight="balanced_subsample", random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(),
    "xgboost": XGBClassifier(
        eval_metric="logloss", max_depth=6, n_estimators=400,
        subsample=0.8, colsample_bytree=0.8, learning_rate=0.05,
        scale_pos_weight=float(len(y_train) - y_train.sum()) / y_train.sum()
    ),
}

print("\n🚀 Training HARD models...\n")

for name, model in MODELS.items():
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🚀 {name.upper()}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    model.fit(X_train_imp, y_train)
    save_model_bundle(name, model, imputer, feature_names)

print("\n🎉 DONE! Models trained using realistic hard conditions.\n")
