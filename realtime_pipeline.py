import csv
import json
import time
import requests
from datetime import datetime, timedelta
from kafka import KafkaProducer

API_URL = "http://127.0.0.1:8000/predict"
KAFKA_BOOTSTRAP = "127.0.0.1:29092"
TOPIC = "btc_ticks"

class RealTimePipeline:

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BOOTSTRAP],
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )

        # rolling window for volatility + intensity
        self.prices = []
        self.timestamps = []

    def compute_features(self, ts, price):
        now = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        window_start = now - timedelta(seconds=30)

        # keep only last 30 seconds
        self.timestamps.append(now)
        self.prices.append(price)

        while self.timestamps and self.timestamps[0] < window_start:
            self.timestamps.pop(0)
            self.prices.pop(0)

        # midprice = price
        midprice = price

        # static spread assumption
        spread = 0.5

        # trade_intensity = trades per 30s
        trade_intensity = len(self.prices)

        # volatility_30s = std of log returns
        if len(self.prices) > 1:
            import numpy as np
            rets = np.diff(np.log(self.prices))
            volatility = float(np.std(rets))
        else:
            volatility = 0.0

        return {
            "ts": ts,
            "price": price,
            "midprice": midprice,
            "spread": spread,
            "trade_intensity": trade_intensity,
            "volatility_30s": volatility
        }

    def send_to_kafka(self, tick):
        self.producer.send(TOPIC, tick)
        print(f"[Producer] Sent → {tick}")

    def predict(self, tick):
        try:
            response = requests.post(API_URL, json=tick)
            response.raise_for_status()
            print("[Consumer] Prediction:", response.json())
        except Exception as e:
            print("[Consumer] Prediction failed:", e)

    def replay_csv(self, path="crypto-producer/sample_ticks.csv"):
        print("[Pipeline] Starting replay + prediction mode…")
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row["ts"]
                price = float(row["price"])

                tick = self.compute_features(ts, price)

                self.send_to_kafka(tick)
                self.predict(tick)

                time.sleep(0.5)


if __name__ == "__main__":
    pipeline = RealTimePipeline()
    pipeline.replay_csv()
