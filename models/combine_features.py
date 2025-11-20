#!/usr/bin/env python3
"""
Combine feature datasets from 3 students, align numeric feature space,
impute missing features, and create a unified volatility spike label.
"""

import pandas as pd
from pathlib import Path

# ================== FILE PATHS ==================
YOU = "data/processed/features_early.parquet"
STUDENT1 = "C:/Users/mattr/OneDrive/Downloads/handoff_Asli/handoff/data/features_10min_slice.parquet"
STUDENT2 = "C:/Users/mattr/OneDrive/Downloads/handoff/handoff/data/processed/features_sample.parquet"

SAVE_PATH = "data/processed/features_combined_labeled.parquet"

# ================== LOAD FILES ==================
print("📌 Loading datasets...")

dfs = []
paths = [YOU, STUDENT1, STUDENT2]
for path in paths:
    try:
        df = pd.read_parquet(path)
        print(f"   ✔ Loaded {path} -> {df.shape}")
        dfs.append(df)
    except Exception as e:
        print(f"   ⚠ Failed to load {path}: {e}")

# ================== ALIGN NUMERIC COLUMNS ==================
print("\n📎 Aligning numeric feature columns...")

# Get union of all numeric columns (not intersection)
numeric_cols_union = set()
for df in dfs:
    numeric_cols_union |= set(df.select_dtypes(include=["float", "int"]).columns)

numeric_cols_union = sorted(list(numeric_cols_union))
print(f"   ➕ Total numeric features in union: {len(numeric_cols_union)}")

# Reindex each df to have the same numeric cols (fill missing with NaN)
aligned_dfs = [df.reindex(columns=numeric_cols_union) for df in dfs]
combined = pd.concat(aligned_dfs, ignore_index=True)

print(f"\n🔗 Combined dataset size: {combined.shape}")

# ================== LABEL CREATION ==================
print("\n🏷 Creating volatility spike labels...")

if "volatility_30s" not in combined.columns:
    raise ValueError("❌ ERROR: No 'volatility_30s' column exists in combined features!")

# Use global threshold
threshold = combined["volatility_30s"].quantile(0.95)
combined["volatility_spike"] = (combined["volatility_30s"] > threshold).astype(int)

print(f"   ✔ Spike threshold = {threshold}")
print(f"   ✔ Spike ratio     = {combined['volatility_spike'].mean():.4f}")

# ================== SAVE ==================
combined.to_parquet(SAVE_PATH)

print(f"\n💾 Saved combined labeled dataset ➜ {SAVE_PATH}")
print(f"📊 Final shape: {combined.shape}")
print("\n🎉 COMBINATION COMPLETE! Ready for training.\n")
