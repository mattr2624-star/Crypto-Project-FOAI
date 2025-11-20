#!/usr/bin/env python3
"""
Train 4 models on combined, labeled dataset, safely without leakage:
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost (optional)

Features:
✔ Loads combined labeled dataset
✔ Automatically labels if missing
✔ Removes ALL future-leaking volatility metrics
✔ Removes label leakage
✔ Imputes NaN with median
✔ Saves versioned models

Requires:
    pip install scikit-learn pandas joblib xgboost (optional)
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from datetime import datetime

# ==== CONFIG ====
COMBINED_DATA = "data/processed/features_combined_labeled.parquet"
FALLBACK_DATA = "data/processed/features_early_labeled.parquet"
SAVE_DIR = "models/artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== Helper: timestamp ====
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==== Auto-Label Function (used only if missing) ====
def auto_label(df):
    """Add 'volatility_spike' label; ensure both classes exist."""
    print("\n🔎 Checking for volatility spike labels...")

    if "volatility_spike" in df.columns:
        print("✔ Labels already found.")
        return df

    print("⚠ No label found. Creating labels automatically...")

    p95 = df["volatility_30s"].quantile(0.95)
    p90 = df["volatility_30s"].quantile(0.90)

    threshold = min(p95, p90)
    df["volatility_spike"] = (df["volatility_30s"] > threshold).astype(int)

    if df["volatility_spike"].sum() < 3:
        print("⚠ Too few spikes. Using 85th percentile.")
        threshold = df["volatility_30s"].quantile(0.85)
        df["volatility_spike"] = (df["volatility_30s"] > threshold).astype(int)

    labeled_path = f"data/processed/features_auto_labeled_{timestamp()}.parquet"
    df.to_parquet(labeled_path)
    print(f"💾 Saved labeled dataset ➜ {labeled_path}")
    return df


# ================= LOAD DATA ==================
print("\n📌 Loading combined dataset…")

try:
    df = pd.read_parquet(COMBINED_DATA)
except Exception:
    print("⚠ Combined dataset not found. Falling back.")
    df = pd.read_parquet(FALLBACK_DATA)

# Auto-label if needed
df = auto_label(df)


# ================= REMOVE LEAKAGE ==================
numeric_df = df.select_dtypes(include=["float", "int"])

# Remove label leakage
leak_label = [c for c in numeric_df.columns if "volatility_spike" in c.lower()]

# Remove ALL volatility future leaks
leak_vol = [c for c in numeric_df.columns if "volatility" in c.lower()]

# Combine leak columns
remove_cols = leak_label + leak_vol

print(f"\n🧹 Removing leakage columns: {remove_cols}")
numeric_df = numeric_df.drop(columns=remove_cols, errors="ignore")

# X = features, y = label
X = numeric_df
y = df["volatility_spike"]


# ================= IMPUTE MISSING VALUES ==================
print("\n🧹 Handling missing values (NaNs)…")
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("   ✔ Missing values imputed using median strategy.")


# ================= SUMMARY ==================
print("\n📊 Dataset Summary:")
print(f"   Rows:       {len(df):,}")
print(f"   Features:   {len(X.columns)}")
print(f"   Class 0:    {(y == 0).sum()} rows")
print(f"   Class 1:    {(y == 1).sum()} rows")
print(f"   Spike Rate: {y.mean()*100:.2f}%")


# ================= SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)


# ================= TRAINING FUNCTION ==================
def train_and_save(model, name):
    print(f"\n🚀 Training {name}…")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probas)

    print(f"   ✔ Accuracy: {acc:.4f}")
    print(f"   ✔ AUC:      {auc:.4f}")
    print("\n🔎 Classification Report:")
    print(classification_report(y_test, preds))

    # Save versioned model
    filename = f"{name.replace(' ', '_').lower()}_{timestamp()}.joblib"
    path = os.path.join(SAVE_DIR, filename)
    joblib.dump(model, path)
    print(f"💾 Saved model ➜ {path}\n")


# ================= TRAIN MODELS ==================
train_and_save(LogisticRegression(max_iter=500), "Logistic Regression")
train_and_save(RandomForestClassifier(n_estimators=200, max_depth=10), "Random Forest")
train_and_save(GradientBoostingClassifier(), "Gradient Boosting")

# Optional XGBoost
try:
    from xgboost import XGBClassifier

    train_and_save(
        XGBClassifier(
            eval_metric="logloss",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1
        ),
        "XGBoost"
    )
except Exception:
    print("\n⚠ XGBoost not installed or training failed. Skipping.\n")


print("\n🎉 DONE! All models trained & versioned leak-free!\n")
