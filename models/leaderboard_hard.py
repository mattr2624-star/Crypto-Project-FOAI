#!/usr/bin/env python3
"""
Leaderboard with optimized probability thresholds for rare spike detection.
- Finds best threshold on TRAIN (max F1)
- Applies it on TEST
- Works with our saved *_hard_*.joblib models
- Reports: AUC, Precision, Recall, F1, Optimal Threshold
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support
)

# ==============================
# 📌 Paths
# ==============================
DATA_DIR    = Path("data/processed")
MODEL_DIR   = Path("models/artifacts")
RESULTS_CSV = MODEL_DIR / "leaderboard_hard_results.csv"
LABEL_COL   = "volatility_spike_future"

# ==============================
# 📌 Load hard train + test
# ==============================
def latest(pattern):
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No matching {pattern} in {DATA_DIR}")
    return files[-1]

train_path = latest("features_hard_train_*.parquet")
test_path  = latest("features_hard_test_*.parquet")

print("\n📌 Loading HARD DATASET...")
train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)

y_train = train_df[LABEL_COL].astype(int)
y_test  = test_df[LABEL_COL].astype(int)

X_train = train_df.drop(columns=[LABEL_COL, "timestamp"], errors="ignore")
X_test  = test_df.drop(columns=[LABEL_COL, "timestamp"], errors="ignore")

print(f"   ✔ TRAIN rows: {len(train_df):,}")
print(f"   ✔ TEST rows : {len(test_df):,}")
print(f"   ✔ Spike rate (TEST): {y_test.mean()*100:.2f}%")

# ==============================
# 🔍 Feature-Intersection Helper
# ==============================
def align_features(model, X):
    # our saved models include metadata
    if isinstance(model, dict) and "feature_names" in model:
        expected = model["feature_names"]
    else:
        expected = getattr(model, "feature_names_in_", list(X.columns))

    usable = [c for c in expected if c in X.columns]
    return X[usable].copy(), usable

# ==============================
# 🧪 Threshold Optimizer
# ==============================
def best_threshold(model, X_train, y_train):
    if not hasattr(model, "predict_proba"):
        return 0.5  # fallback

    prob = model.predict_proba(X_train)[:, 1]

    best_thr = 0.5
    best_f1 = -1
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (prob >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_train, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr

# ==============================
# 🔎 Load & Evaluate Models
# ==============================
results = []

print("\n🔎 Evaluating models with optimized thresholds...\n")
for p in sorted(MODEL_DIR.glob("*_hard_*.joblib")):
    name = p.stem
    try:
        bundle = joblib.load(p)

        # Unpack saved dicts
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            imputer = bundle.get("imputer", None)
        else:
            model = bundle
            imputer = None

        # Align features
        X_tr, feats = align_features(bundle, X_train)
        X_ts, _     = align_features(bundle, X_test)

        # Impute if needed
        if imputer:
            X_tr = imputer.transform(X_tr.values)
            X_ts = imputer.transform(X_ts.values)
        else:
            X_tr = X_tr.fillna(X_tr.median()).values
            X_ts = X_ts.fillna(X_ts.median()).values

        # Find best threshold on TRAIN
        thr = best_threshold(model, X_tr, y_train)

        # Test predictions
        if hasattr(model, "predict_proba"):
            prob_test = model.predict_proba(X_ts)[:, 1]
            preds = (prob_test >= thr).astype(int)
            auc = roc_auc_score(y_test, prob_test)
        else:
            preds = model.predict(X_ts)
            auc = float("nan")

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )

        print(f"   ✔ {name}: thr={thr:.3f}, AUC={auc:.3f}, F1={f1:.3f}, Rec={rec:.3f}")

        results.append({
            "Model": name,
            "Threshold": thr,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Features_used": len(feats)
        })

    except Exception as e:
        print(f"   ❌ Failed {name}: {e}")

# ==============================
# 💾 Save Leaderboard
# ==============================
df = pd.DataFrame(results).sort_values(by=["F1","AUC"], ascending=False)
df.to_csv(RESULTS_CSV, index=False)
print(f"\n🏆 Leaderboard saved → {RESULTS_CSV}\n")
print(df.to_string(index=False))
print("\n🎉 DONE! Threshold-optimized comparison complete.\n")
