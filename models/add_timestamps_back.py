#!/usr/bin/env python3
"""
Restore timestamps into combined features using the ORIGINAL SOURCES.

Order MUST match the original combination:
    1) features_early.parquet            (~50 rows)
    2) Asli 10-min parquet               (~6446 rows)
    3) features_sample.parquet           (~1000 rows)

Outputs new file:
    data/processed/features_combined_labeled_with_timestamp.parquet
"""

import pandas as pd
from pathlib import Path

# --- Combined features WITHOUT timestamp ---
COMBINED = Path("data/processed/features_combined_labeled.parquet")

# --- The ORIGINAL inputs (must match your combine script) ---
S1 = Path("data/processed/features_early.parquet")  # Your early dataset
S2 = Path(r"C:/Users/mattr/OneDrive/Downloads/handoff_Asli/handoff/data/features_10min_slice.parquet")
S3 = Path(r"C:/Users/mattr/OneDrive/Downloads/handoff/handoff/data/processed/features_sample.parquet")

print("\n📌 Restoring timestamps using original datasets...")

# Load combined dataset
df = pd.read_parquet(COMBINED)
print(f"   ✔ Combined dataset loaded: {df.shape}")

# Load raw sources WITH timestamp
dfs = []
for name, path in [("Student1", S1), ("Asli", S2), ("Student3", S3)]:
    raw = pd.read_parquet(path)
    if "timestamp" not in raw.columns:
        raise ValueError(f"❌ Input {name} at {path} does NOT contain timestamp column.")
    dfs.append(raw[["timestamp"]])
    print(f"   ✔ Loaded timestamps from {name}: {len(raw)} rows")

# Combine timestamps
timestamps = pd.concat(dfs, ignore_index=True)

if len(timestamps) != len(df):
    raise ValueError(
        f"❌ Row mismatch: timestamps={len(timestamps)}, combined={len(df)}. "
        "Cannot merge. Check source file order or dataset paths."
    )

# Attach timestamp
df["timestamp"] = timestamps["timestamp"].values

# Save output
OUT = Path("data/processed/features_combined_labeled_with_timestamp.parquet")
df.to_parquet(OUT)

print(f"\n💾 Saved WITH timestamps ➜ {OUT}")
print("\n🎉 Timestamp restoration complete.\n")
