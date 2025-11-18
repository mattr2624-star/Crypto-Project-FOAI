import time
from datetime import datetime

import pandas as pd

from producer import CryptoProducer  # reuses your existing class

TOPIC = "crypto_ticks"
BOOTSTRAP = ["localhost:29092"]

# Path to your 10-minute historical dataset
TICKS_FILE = r"..\data\btcusd_ticks_10min.csv"

# 1.0 = real time, 10.0 = 10x faster than real time, 0 = no waiting
SPEEDUP = 10.0


def load_ticks(path: str) -> pd.DataFrame:
    """
    Expects a CSV with at least: timestamp, price, volume
    timestamp can be ISO8601 or anything pandas can parse.
    """
    df = pd.read_csv(path)

    # Normalize column names in case they're slightly different
    df.columns = [c.strip().lower() for c in df.columns]

    # Require basic columns
    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError("CSV must contain at least 'timestamp' and 'price' columns.")

    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def replay_ticks():
    producer = CryptoProducer(bootstrap_servers=BOOTSTRAP)
    df = load_ticks(TICKS_FILE)

    print(f"Loaded {len(df)} ticks from {TICKS_FILE}")
    print(f"Replaying to topic '{TOPIC}' with speedup factor {SPEEDUP}x")

    prev_ts = None

    for _, row in df.iterrows():
        ts = row["timestamp"]

        # Sleep to simulate time gaps between ticks
        if prev_ts is not None and SPEEDUP > 0:
            real_gap = (ts - prev_ts).total_seconds()
            sleep_for = real_gap / SPEEDUP
            if sleep_for > 0:
                time.sleep(sleep_for)

        tick = {
            "symbol": "BTC/USD",
            "timestamp": ts.isoformat(),
            "price": float(row["price"]),
            "volume": float(row["volume"]),
        }

        producer.send(TOPIC, tick)
        prev_ts = ts

    print("✅ Finished replaying all ticks.")


if __name__ == "__main__":
    replay_ticks()
