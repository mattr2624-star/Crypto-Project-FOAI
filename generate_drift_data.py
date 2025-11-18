"""
generate_drift_data.py
Utility script to create mock reference.csv and current.csv for Evidently drift testing.
"""

import pandas as pd
import numpy as np
import os

# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# ----------------------------
# Generate Reference Dataset
# ----------------------------
n = 500
reference_data = pd.DataFrame({
    "price": np.random.normal(100, 10, n),
    "volume": np.random.normal(1000, 200, n),
    "sentiment_score": np.random.normal(0, 1, n),
    "target": np.random.choice([0, 1], n)
})

reference_data.to_csv("data/reference.csv", index=False)
print("✅ Created data/reference.csv")

# ----------------------------
# Generate Current Dataset (with mild drift)
# ----------------------------
m = 500
current_data = pd.DataFrame({
    "price": np.random.normal(105, 15, m),           # Slight upward drift in price
    "volume": np.random.normal(900, 250, m),         # Slight drop in volume
    "sentiment_score": np.random.normal(0.2, 1.1, m),# Small sentiment drift
    "target": np.random.choice([0, 1], m)
})

current_data.to_csv("data/current.csv", index=False)
print("✅ Created data/current.csv")

# ----------------------------
# Verify and preview
# ----------------------------
print("\n📊 Reference sample:")
print(reference_data.head())

print("\n📊 Current sample:")
print(current_data.head())

print("\n✅ Drift test datasets ready for Evidently.")
