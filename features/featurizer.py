"""
featurizer.py
Consumes raw tick data from Kafka (ticks.raw),
computes windowed features, and publishes to ticks.features.
Also saves processed data to data/processed/features.parquet
"""

import os
import json
import pandas as pd
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime

# Kafka config
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:29092")
TOPIC_IN = os.environ.get("TOPIC_IN", "ticks.raw")
TOPIC_OUT = os.environ.get("TOPIC_OUT", "ticks.features")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Connecting KafkaConsumer to:", KAFKA_BROKER)

# Kafka connections
consumer = KafkaConsumer(
    TOPIC_IN,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="featurizer-group"
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8")
)

def compute_features(df):
    """Compute windowed features with robust numeric handling."""
    for col in ["price", "best_bid", "best_ask", "last_size"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").ffill()

    # Derived fields
    df["midprice"] = (df["best_bid"] + df["best_ask"]) / 2
    df["mid_return"] = df["midprice"].pct_change().fillna(0)
    df["spread"] = df["best_ask"] - df["best_bid"]

    # Optional: trade intensity
    if "last_size" in df.columns:
        df["trade_intensity"] = (
            pd.to_numeric(df["last_size"], errors="coerce")
            .fillna(0)
            .rolling(window=5, min_periods=1)
            .sum()
        )
    else:
        df["trade_intensity"] = 0

    # Rolling volatility
    df["volatility_30s"] = df["mid_return"].rolling(window=30, min_periods=5).std().fillna(0)

    df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)

    return df


def main():
    print(f"Featurizer started: consuming from {TOPIC_IN}...")

    buffer = []
    for msg in consumer:
        data = msg.value
        if data.get("type") != "ticker":
            continue
        buffer.append(data)

        if len(buffer) >= 50:  # batch
            df = pd.DataFrame(buffer)
            features = compute_features(df)

            # Ensure JSON-safe values
            features = features.replace({np.nan: None})

            # Publish & save
            for _, row in features.iterrows():
                record = row.to_dict()
                producer.send(TOPIC_OUT, record)

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            features.to_parquet(f"{OUTPUT_DIR}/features_{ts}.parquet", index=False)
            print(f"✅ Published and saved {len(features)} feature rows.")
            buffer = []


if __name__ == "__main__":
    main()
