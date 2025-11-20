#!/usr/bin/env python3
"""
Train 4 models on the HARD time-based dataset:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (if available)

Inputs:
    data/processed/features_hard_train_*.parquet
    data/processed/features_hard_test_*.parquet

Outputs (per model):
    models/artifacts/<model_name>_hard_<ts>.joblib
    models/artifacts/<model_name>_hard_<ts>_features.txt
"""

import os
import glob
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ---------- Paths ----------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest(pattern: str) -> Path:
    candidates = sorted(DATA_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files match pattern {pattern} in {DATA_DIR}")
    return candidates[-1]


# ---------- Load HARD train/test ----------
print("\n📌 Locating HARD dataset files...")
train_path = latest("features_hard_train_*.parquet")
test_path = latest("features_hard_test_*.parquet")
print(f"   Using TRAIN: {train_path.name}")
print(f"   Using TEST : {test_path.name}")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# label + timestamp handling
LABEL_COL = "volatility_spike_future"

if LABEL_COL not in train_df.columns or LABEL_COL not in test_df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' missing from hard datasets.")

# Drop timestamp from features
feature_cols = [c for c in train_df.columns if c not in [LABEL_COL, "timestamp"]]

X_train_raw = train_df[feature_cols].copy()
y_train = train_df[LABEL_COL].astype(int)

X_test_raw = test_df[feature_cols].copy()
y_test = test_df[LABEL_COL].astype(int)

print("\n📊 HARD Dataset Summary:")
print(f"   Train rows: {len(train_df):,}")
print(f"   Test  rows: {len(test_df):,}")
print(f"   Features : {len(feature_cols)}")
print(f"   Train Spike Rate: {y_train.mean()*100:.2f}%")
print(f"   Test  Spike Rate: {y_test.mean()*100:.2f}%")

# ---------- Impute missing values ----------
print("\n🧹 Imputing missing values with median...")
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

# we’ll store feature names for later reuse
feature_names = feature_cols[:]


def train_and_save(model, model_name: str):
    print(f"\n🚀 Training {model_name} on HARD dataset…")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas) if probas is not None else float("nan")

    print(f"   ✔ Accuracy: {acc:.4f}")
    print(f"   ✔ AUC:      {auc:.4f}" if not np.isnan(auc) else "   ✔ AUC:      N/A")
    print("\n🔎 Classification Report:")
    print(classification_report(y_test, preds))

    # Attach feature names to the model object for later compatibility
    if not hasattr(model, "feature_names_in_"):
        # set as numpy array for sklearn compatibility style
        model.feature_names_in_ = np.array(feature_names)

    ts = timestamp()
    base_name = model_name.replace(" ", "_").lower() + f"_hard_{ts}"

    model_path = MODEL_DIR / f"{base_name}.joblib"
    feats_path = MODEL_DIR / f"{base_name}_features.txt"

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_names": feature_names,
        },
        model_path,
    )
    with open(feats_path, "w", encoding="utf-8") as f:
        for c in feature_names:
            f.write(c + "\n")

    print(f"💾 Saved model ➜ {model_path}")
    print(f"💾 Saved feature list ➜ {feats_path}")


# ---------- Train all models ----------
train_and_save(LogisticRegression(max_iter=1000), "Logistic Regression")
train_and_save(
    RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
    ),
    "Random Forest",
)
train_and_save(
    GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
    "Gradient Boosting",
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    train_and_save(xgb, "XGBoost")
except Exception as e:
    print("\n⚠ XGBoost not installed or failed to import. Skipping.")
    print(f"   Reason: {e}")

print("\n🎉 DONE! Hard-mode models trained & saved.\n")
