import pandas as pd
import glob

files = glob.glob("data/processed/features_*.parquet")
dfs = [pd.read_parquet(f) for f in files]

df = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
df.to_parquet("data/processed/all_features.parquet")
print("Saved combined dataset with rows:", len(df))
