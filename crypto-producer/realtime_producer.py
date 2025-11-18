import json
import time
import pandas as pd
from dataclasses import dataclass
from kafka import KafkaProducer


@dataclass
class ProducerConfig:
    bootstrap_servers: list[str] = ("127.0.0.1:29092",)
    topic: str = "btc_ticks"
    replay_file: str = "crypto-producer/sample_ticks.csv"
    sleep_seconds: float = 0.1  # interval between ticks


class CryptoProducer:
    def __init__(self, cfg: ProducerConfig):
        self.cfg = cfg
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=cfg.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"[Producer] Connected to Kafka @ {cfg.bootstrap_servers}")
        except Exception as e:
            print(f"[Producer] Kafka connection failed: {e}")
            raise

    def send(self, message: dict):
        try:
            self.producer.send(self.cfg.topic, value=message)
            self.producer.flush()
            print(f"[Producer] Sent → {self.cfg.topic}: {message}")
        except Exception as e:
            print(f"[Producer] Failed to send message: {e}")

    def replay(self):
        """Replay ticks from CSV continuously."""
        df = pd.read_csv(self.cfg.replay_file)
        while True:
            for _, row in df.iterrows():
                tick = {
                    "ts": row["ts"],
                    "price": float(row["price"]),
                    "midprice": float(row.get("midprice", row["price"])),
                    "spread": float(row.get("spread", 0)),
                    "trade_intensity": float(row.get("trade_intensity", 0)),
                    "volatility_30s": float(row.get("volatility_30s", 0)),
                }
                self.send(tick)
                time.sleep(self.cfg.sleep_seconds)
            # optional: small pause between CSV loops
            print("[Producer] Completed one full replay, looping again...")
            time.sleep(1)


if __name__ == "__main__":
    cfg = ProducerConfig()
    producer = CryptoProducer(cfg)
    print("[Producer] Starting real-time replay loop…")
    producer.replay()
