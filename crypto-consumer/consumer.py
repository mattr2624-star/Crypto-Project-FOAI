import os
import json
import time
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from dotenv import load_dotenv

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
load_dotenv()

TOPIC = os.getenv("KAFKA_TOPIC", "crypto_ticks")
BOOTSTRAP = os.getenv("KAFKA_BROKER", "localhost:29092")
BOOTSTRAP = [BOOTSTRAP] if isinstance(BOOTSTRAP, str) else BOOTSTRAP

OUTPUT_DIR = os.getenv("TICK_OUTPUT_DIR", "./")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
# Connect to Kafka with retry logic
# -------------------------------------------------------
def connect_consumer(max_retries=10, wait=3):
    for i in range(max_retries):
        try:
            print(f"🔌 Connecting to Kafka ({i+1}/{max_retries})...")
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=BOOTSTRAP,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                group_id="tick_exporter_group",
                enable_auto_commit=False,
                consumer_timeout_ms=10000
            )
            print("✅ Kafka connected.")
            return consumer

        except NoBrokersAvailable:
            print(f"⚠️ Kafka not available. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("❌ Could not connect to Kafka after retries.")


# -------------------------------------------------------
# Main consumer loop
# -------------------------------------------------------
def run_consumer():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"ticks_BTCUSD_{timestamp}.ndjson")

    print(f"📥 Writing tick data to: {output_file}")

    consumer = connect_consumer()
    count = 0

    try:
        with open(output_file, "w") as f:
            for msg in consumer:
                json.dump(msg.value, f)
                f.write("\n")
                count += 1

                if count % 10 == 0:
                    print(f"Saved {count} ticks...")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Closing consumer...")

    except Exception as e:
        print(f"❌ Consumer error: {e}")

    finally:
        consumer.close()
        print(f"📦 Done. Total messages saved: {count}")


if __name__ == "__main__":
    run_consumer()
