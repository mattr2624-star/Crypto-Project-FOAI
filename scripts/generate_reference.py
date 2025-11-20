import pandas as pd

DATA_PATH = "/app/data/raw_10min_slice.csv"
OUT_PATH = "/app/data/reference.csv"
WINDOW = 10

print("📥 Loading raw data...")
df = pd.read_csv(DATA_PATH)

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

df["ret"] = df["price"].pct_change()
df["ret_mean"] = df["ret"].rolling(WINDOW).mean()
df["ret_std"] = df["ret"].rolling(WINDOW).std()
df["n"] = WINDOW

out = df[["ret_mean", "ret_std", "n"]].dropna()

print(f"📝 Saving reference → {OUT_PATH} (rows={len(out)})")
out.to_csv(OUT_PATH, index=False)
print("🎉 Reference created!")
