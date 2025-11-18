import asyncio
import json
import logging

import httpx
from aiokafka import AIOKafkaConsumer

KAFKA_BOOTSTRAP = "127.0.0.1:29092"
KAFKA_TOPIC = "btc_ticks"
MODEL_SERVER_URL = "http://127.0.0.1:8000/predict"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consumer_predictor")


async def consume_and_predict():
    consumer = AIOKafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="consumer-predictor",
        auto_offset_reset="latest",
    )
    await consumer.start()
    logger.info(f"Connected to Kafka @ {KAFKA_BOOTSTRAP}, consuming topic '{KAFKA_TOPIC}'")

    async with httpx.AsyncClient() as client:
        try:
            async for msg in consumer:
                try:
                    tick = json.loads(msg.value)
                except json.JSONDecodeError:
                    tick = msg.value
                # Send tick to model server
                try:
                    resp = await client.post(MODEL_SERVER_URL, json=tick)
                    if resp.status_code == 200:
                        prediction = resp.json()
                        logger.info(f"Tick: {tick} → Prediction: {prediction}")
                    else:
                        logger.error(f"Prediction failed for tick {tick}: {resp.text}")
                except Exception as e:
                    logger.error(f"HTTP request failed for tick {tick}: {e}")
        finally:
            await consumer.stop()


if __name__ == "__main__":
    asyncio.run(consume_and_predict())
