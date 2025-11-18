import json
import time
import pandas as pd
from dataclasses import dataclass
from kafka import KafkaProducer, errors


# ---------------------------
# Configuration (simple Phase-1)
# ---------------------------
@dataclass
class ProducerConfig:
    # Force IPv4 to avoid ::1 / IPv6 connection issues on Windows
    bootstrap_servers: list[str] = ("127.0.0.1:29092",)
    topic: str = "btc_ticks"
    replay_file: str = "crypto-producer/sample_ticks.csv"
    sleep_seconds: float = 0.10  # 100 ms between ticks


# ---------------------------
# Producer
# ---------------------------
class CryptoProducer:
    def __init__(self, cfg: ProducerConfig):
        self.cfg = cfg

        print(f"[Producer] Connecting to Kafka @ {cfg.bootstrap_servers} ...")

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=cfg.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000,
                api_version_auto_timeout_ms=5000
            )
            print(f"[Producer] Connected to {cfg.bootstrap_servers}")

        except errors.NoBrokersAvailable:
            print("\n❌ ERROR: No Kafka brokers available at "
                  f"{cfg.bootstrap_servers}\n"
                  "→ Make sure Docker Kafka is running:\n"
                  "  docker compose up kafka zookeeper\n")
            raise

    def send(self, message: dict):
        """Send a single tick to Kafka."""
        self.producer.send(self.cfg.topic, value=message)
        # Flush immediately for reliability
        self.producer.flush()
        print(f"[Producer] Sent → {self.cfg.topic}: {message}")

    def replay(self):
        """Replay historical ticks from CSV."""
        print(f"[Producer] Loading replay file: {self.cfg.replay_file}")
        df = pd.read_csv(self.cfg.replay_file)

        for _, row in df.iterrows():
            msg = {
                "ts": row["ts"],
                "price": float(row["price"]),
                "volume": float(row["volume"]),
            }
            self.send(msg)
            time.sleep(self.cfg.sleep_seconds)


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    cfg = ProducerConfig()
    producer = CryptoProducer(cfg)

    print("[Producer] Starting replay mode…")
    producer.replay()
