import pandas as pd

df = pd.read_parquet("data/processed/features_combined_labeled.parquet")

# Correlation check
corr = df.corr(numeric_only=True)["volatility_spike"].sort_values(ascending=False)
print("\n🔍 Correlation with spike:\n", corr.head(15))

# Check if any feature equals label directly
direct_matches = [col for col in df.columns if (df[col] == df["volatility_spike"]).mean() > 0.95]
print("\n⚠ Direct matches found:", direct_matches if direct_matches else "None")

# Check if any feature only exists during spikes
bad_features = [col for col in df.columns if df[df["volatility_spike"] == 1][col].nunique() == 1]
print("\n⚠ Features constant during spikes:", bad_features if bad_features else "None")
