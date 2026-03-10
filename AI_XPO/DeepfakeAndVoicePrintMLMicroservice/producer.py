"""
SentinelAI ML Service — Kafka Threat Score Producer

Publishes ThreatScore results to the 'threat_scores' Kafka topic.
Downstream consumers (alerting engine, SIEM connector, frontend gateway)
subscribe to this topic for real-time threat processing.

Production Features:
- AIOKafkaProducer with idempotent delivery (exactly-once semantics).
- Per-tenant Kafka key for ordered delivery within a tenant's stream.
- Retry with exponential backoff on transient producer errors.
- Structured logging with message size and partition telemetry.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import ThreatScore

logger = get_logger("kafka.producer")
settings = get_settings()


class ThreatScoreProducer:
    """
    Async Kafka producer for the threat_scores topic.

    Keying strategy: tenant_id — ensures all events for a tenant
    land on the same partition (ordered processing guarantee for
    downstream alerting correlation).
    """

    _MAX_RETRIES = 3
    _BASE_BACKOFF_S = 0.2

    def __init__(self) -> None:
        self._producer: Optional[AIOKafkaProducer] = None
        self._initialized = False

    async def start(self) -> None:
        """Initialize and start the AIOKafkaProducer."""
        producer_config: dict = {
            "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "value_serializer": lambda v: v,
            "key_serializer": lambda k: k.encode("utf-8") if k else None,
            # Idempotent delivery: prevents duplicate messages on producer retry
            "enable_idempotence": True,
            # Strongest durability guarantee: all ISR replicas must ack
            "acks": "all",
            "compression_type": "gzip",
            "max_batch_size": 65536,
            "linger_ms": 5,
            "request_timeout_ms": 30000,
            "retry_backoff_ms": 200,
        }

        # SASL/SSL for production clusters
        if settings.KAFKA_SASL_MECHANISM:
            producer_config.update({
                "security_protocol": "SASL_SSL",
                "sasl_mechanism": settings.KAFKA_SASL_MECHANISM,
                "sasl_plain_username": settings.KAFKA_SASL_USERNAME,
                "sasl_plain_password": settings.KAFKA_SASL_PASSWORD.get_secret_value()
                    if settings.KAFKA_SASL_PASSWORD else "",
                "ssl_cafile": settings.KAFKA_SSL_CA_LOCATION,
            })

        self._producer = AIOKafkaProducer(**producer_config)
        await self._producer.start()
        self._initialized = True

        logger.info(
            "Kafka producer started.",
            extra={"topic": settings.KAFKA_THREAT_TOPIC},
        )

    async def stop(self) -> None:
        """Flush pending messages and stop the producer."""
        if self._producer:
            await self._producer.stop()
            self._initialized = False
            logger.info("Kafka producer stopped.")

    async def publish_threat_score(self, threat_score: ThreatScore) -> None:
        """
        Serialize and publish a ThreatScore to the threat_scores topic.

        Args:
            threat_score: Aggregated threat assessment from the orchestrator.
        """
        if not self._initialized or not self._producer:
            raise RuntimeError(
                "ThreatScoreProducer.start() must be called before publishing."
            )

        message_bytes = self._serialize(threat_score)
        partition_key = threat_score.tenant_id  # Ordered per tenant

        for attempt in range(self._MAX_RETRIES):
            try:
                record_metadata = await self._producer.send_and_wait(
                    topic=settings.KAFKA_THREAT_TOPIC,
                    key=partition_key,
                    value=message_bytes,
                    headers=[
                        ("correlation_id", threat_score.correlation_id.encode("utf-8")),
                        ("event_id", threat_score.event_id.encode("utf-8")),
                        ("threat_level", threat_score.threat_level.value.encode("utf-8")),
                        ("service_version", settings.SERVICE_VERSION.encode("utf-8")),
                    ],
                )

                logger.info(
                    "ThreatScore published.",
                    extra={
                        "event_id": threat_score.event_id,
                        "topic": record_metadata.topic,
                        "partition": record_metadata.partition,
                        "offset": record_metadata.offset,
                        "threat_level": threat_score.threat_level.value,
                        "message_bytes": len(message_bytes),
                        "combined_score": round(threat_score.combined_threat_score, 4),
                    },
                )
                return

            except KafkaTimeoutError as exc:
                logger.warning(
                    "Kafka publish timeout; retrying.",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                await asyncio.sleep(self._BASE_BACKOFF_S * (2 ** attempt))

            except KafkaError as exc:
                logger.error(
                    "Kafka publish error; retrying.",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                    exc_info=True,
                )
                await asyncio.sleep(self._BASE_BACKOFF_S * (2 ** attempt))

        logger.error(
            "Failed to publish ThreatScore after all retries.",
            extra={
                "event_id": threat_score.event_id,
                "retries": self._MAX_RETRIES,
            },
        )
        raise RuntimeError(
            f"Kafka publish failed for event {threat_score.event_id} "
            f"after {self._MAX_RETRIES} retries."
        )

    async def send_raw(self, topic: str, value: bytes, key: Optional[str] = None) -> None:
        """
        Low-level raw message send — used by the DLQ publisher.
        Does not apply ThreatScore-specific serialization logic.
        """
        if not self._initialized or not self._producer:
            logger.warning("Producer not initialized; cannot send raw message.")
            return
        try:
            await self._producer.send_and_wait(topic=topic, value=value, key=key)
        except KafkaError as exc:
            logger.error(
                "Raw send failed.",
                extra={"topic": topic, "error": str(exc)},
            )
            raise

    @staticmethod
    def _serialize(threat_score: ThreatScore) -> bytes:
        """
        Serialize ThreatScore to UTF-8 JSON bytes.
        Uses Pydantic's model_dump with mode='json' to handle datetime/enum serialization.
        """
        payload = threat_score.model_dump(mode="json")
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_producer_instance: Optional[ThreatScoreProducer] = None


def get_kafka_producer() -> ThreatScoreProducer:
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = ThreatScoreProducer()
    return _producer_instance
