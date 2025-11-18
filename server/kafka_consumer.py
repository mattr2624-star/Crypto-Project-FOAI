import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from aiokafka import AIOKafkaConsumer, ConsumerRecord
from .config import get_settings

logger = logging.getLogger(__name__)


class TickConsumer:
    def __init__(self):
        settings = get_settings()
        self.bootstrap_servers = settings.kafka_bootstrap_servers
        self.topic = settings.kafka_ticks_topic
        self.group_id = settings.kafka_consumer_group

        self.consumer: Optional[AIOKafkaConsumer] = None
        self._task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        if self.running:
            return

        logger.info(
            "Starting Kafka consumer: bootstrap_servers=%s topic=%s group_id=%s",
            self.bootstrap_servers,
            self.topic,
            self.group_id,
        )

        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )

        await self.consumer.start()
        self.running = True

        # background loop
        self._task = asyncio.create_task(self._consume_loop())

    async def stop(self):
        logger.info("Stopping Kafka consumer")
        self.running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self.consumer:
            await self.consumer.stop()
            self.consumer = None

    async def _consume_loop(self):
        assert self.consumer is not None

        try:
            async for record in self.consumer:
                self._handle_record(record)
        except asyncio.CancelledError:
            logger.info("Kafka consume loop cancelled")
        except Exception as exc:
            logger.exception("Error in Kafka consume loop: %s", exc)

    def _handle_record(self, record: ConsumerRecord):
        try:
            value = json.loads(record.value)
        except Exception:
            value = record.value

        logger.debug("Received Kafka tick: %s", value)

    async def iter_ticks(self) -> AsyncIterator[bytes]:
        """Optional iterator for structured pipelines."""
        if self.consumer is None:
            raise RuntimeError("Consumer not started")
        async for msg in self.consumer:
            yield msg.value


# ---- REQUIRED BY main.py ----
_singleton_consumer = TickConsumer()

def get_consumer() -> TickConsumer:
    """Return singleton consumer instance."""
    return _singleton_consumer
