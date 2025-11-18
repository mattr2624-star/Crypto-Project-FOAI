import pandas as pd

LIVE_FILE = r"C:\cp\data\btcusd_ticks_10min.csv"

df = pd.read_csv(LIVE_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

print(df.head())
print(df.tail())
print(df.shape)
