#!/usr/bin/env python3
"""
Build a *harder*, more realistic dataset for volatility spike prediction.

Changes vs previous setup:
- Label = future volatility spike in next window
- Features = only past-looking, no direct volatility targets
- Time-based split (train on past, test on future)
- Auto-detects timestamp column
- Prevents class collapse (guarantees some spikes)
- No overwriting existing files: always writes new parquet files

Input:
    data/processed/features_combined_labeled.parquet

Outputs:
    data/processed/features_hard_train.parquet
    data/processed/features_hard_test.parquet
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

INPUT_PATH = Path("data/processed/features_combined_labeled_with_timestamp.parquet")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(exist_ok=True)

# Prediction horizon (future window size)
FUTURE_HORIZON = 60
TRAIN_FRACTION = 0.8


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


print("\n📌 Loading combined dataset...")
df = pd.read_parquet(INPUT_PATH)

# ---------- Detect timestamp automatically ----------
time_candidates = [
    c for c in df.columns
    if any(k in c.lower() for k in ["timestamp", "time", "datetime", "date"])
]

if not time_candidates:
    raise ValueError(
        "❌ No time-like column found for chronological splitting.\n"
        f"Columns available: {list(df.columns)}"
    )

timestamp_col = time_candidates[0]
df = df.rename(columns={timestamp_col: "timestamp"})
print(f"🕒 Using time column: '{timestamp_col}' → standardized as 'timestamp'")

# Ensure ordering
print("⏱ Sorting by timestamp...")
df = df.sort_values("timestamp").reset_index(drop=True)

# ---------- 1. Base volatility proxy selection ----------
candidate_vol_cols = [
    col for col in df.columns
    if any(k in col.lower() for k in ["volatility", "return", "mid_return"])
]

if not candidate_vol_cols:
    raise ValueError("❌ No volatility/return-like column found to construct future label.")

# Pick best candidate (prefers volatility > return > mid_return)
priority_keywords = ["volatility", "return", "mid_return"]
base_col = next(
    (c for keyword in priority_keywords for c in candidate_vol_cols if keyword in c.lower()),
    candidate_vol_cols[0]
)

print(f"📈 Using '{base_col}' as base for future volatility proxy...")

# Build rolling future-window metric
df["future_volatility_proxy"] = (
    df[base_col]
    .rolling(window=FUTURE_HORIZON, min_periods=max(10, FUTURE_HORIZON // 4))
    .std()
    .shift(-FUTURE_HORIZON)
)

before = len(df)
df = df.dropna(subset=["future_volatility_proxy"]).reset_index(drop=True)
after = len(df)
print(f"🧹 Dropped {before - after} incomplete tail rows where future window was unavailable.")

# ---------- 2. Create future spike label ----------
threshold = df["future_volatility_proxy"].quantile(0.98)
df["volatility_spike_future"] = (df["future_volatility_proxy"] > threshold).astype(int)

# Guarantee at least 10 spikes
if df["volatility_spike_future"].sum() < 10:
    print("⚠ Too few spikes! Lowering threshold to 95th percentile.")
    threshold = df["future_volatility_proxy"].quantile(0.95)
    df["volatility_spike_future"] = (df["future_volatility_proxy"] > threshold).astype(int)

spike_rate = df["volatility_spike_future"].mean()
print(f"🏷 Future spike threshold = {threshold:.6f}")
print(f"   Future spike rate      = {spike_rate*100:.2f}%")

# ---------- 3. Build features without leakage ----------
num_df = df.select_dtypes(include=["float", "int"]).copy()

leak_keywords = [
    "future_vol", "future", "volatility_30", "volatility_60", "volatility_120",
    "volatility_spike", "volatility"
]

leak_cols = [c for c in num_df.columns if any(k in c.lower() for k in leak_keywords)]

print(f"\n🧹 Removing leakage-like columns from features: {leak_cols}")

for col in leak_cols:
    if col in num_df.columns:
        num_df.drop(columns=[col], inplace=True)

X = num_df
y = df["volatility_spike_future"]

print("\n📊 Hard Dataset Summary:")
print(f"   Rows:       {len(df):,}")
print(f"   Features:   {X.shape[1]}")
print(f"   Class 0:    {(y == 0).sum():,}")
print(f"   Class 1:    {(y == 1).sum():,}")
print(f"   Spike Rate: {spike_rate*100:.2f}%")

# ---------- 4. Final dataset assembly ----------
hard_df = X.copy()
hard_df["volatility_spike_future"] = y
hard_df["timestamp"] = df["timestamp"]

# Time-based split
split_idx = int(len(hard_df) * TRAIN_FRACTION)
train_df = hard_df.iloc[:split_idx].reset_index(drop=True)
test_df = hard_df.iloc[split_idx:].reset_index(drop=True)

print("\n⏱ Time-based split (no shuffle):")
print(f"   Train rows: {len(train_df):,}")
print(f"   Test rows:  {len(test_df):,}")
print(f"   Train time: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
print(f"   Test  time: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}")

# ---------- 5. Save versioned datasets ----------
train_path = OUTPUT_DIR / f"features_hard_train_{timestamp()}.parquet"
test_path  = OUTPUT_DIR / f"features_hard_test_{timestamp()}.parquet"

train_df.to_parquet(train_path)
test_df.to_parquet(test_path)

print(f"\n💾 Saved HARD TRAIN ➜ {train_path}")
print(f"💾 Saved HARD TEST  ➜ {test_path}")
print("\n🎯 Harder, more realistic dataset created successfully.\n")
