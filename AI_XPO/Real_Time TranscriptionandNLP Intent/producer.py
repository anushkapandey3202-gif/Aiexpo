"""
SentinelAI NLP Service — Kafka NLP Threat Score Producer

Publishes NLPThreatScore results to the 'threat_scores' Kafka topic.
Uses an enrichment side-channel ('nlp_enrichments') for full transcript
storage (downstream case management / SIEM ingestion).

Design:
  - Primary topic (threat_scores):    lightweight scoring envelope.
  - Enrichment topic (nlp_enrichments): full transcript + intent breakdown.
  - Idempotent producer (exactly-once within a session).
  - Per-tenant partition key for ordered processing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import NLPThreatScore

logger   = get_logger("kafka.producer")
settings = get_settings()


class NLPThreatScoreProducer:
    """
    Async Kafka producer for NLPThreatScore publication.

    Publishes to two topics:
      1. threat_scores      — envelope with score + intent label (lightweight).
      2. nlp_enrichments    — full payload including transcript text (for SIEM/case).
    """

    _MAX_RETRIES  = 3
    _BASE_BACKOFF = 0.2

    def __init__(self) -> None:
        self._producer:    Optional[AIOKafkaProducer] = None
        self._initialized: bool = False

    async def start(self) -> None:
        config: dict = {
            "bootstrap_servers":  settings.KAFKA_BOOTSTRAP_SERVERS,
            "value_serializer":   lambda v: v,
            "key_serializer":     lambda k: k.encode("utf-8") if k else None,
            "enable_idempotence": True,
            "acks":               "all",
            "compression_type":   "gzip",
            "max_batch_size":     65536,
            "linger_ms":          5,
            "request_timeout_ms": 30000,
        }

        if settings.KAFKA_SASL_MECHANISM:
            config.update({
                "security_protocol":   "SASL_SSL",
                "sasl_mechanism":      settings.KAFKA_SASL_MECHANISM,
                "sasl_plain_username": settings.KAFKA_SASL_USERNAME,
                "sasl_plain_password": (
                    settings.KAFKA_SASL_PASSWORD.get_secret_value()
                    if settings.KAFKA_SASL_PASSWORD else ""
                ),
                "ssl_cafile": settings.KAFKA_SSL_CA_LOCATION,
            })

        self._producer    = AIOKafkaProducer(**config)
        await self._producer.start()
        self._initialized = True
        logger.info(
            "NLP Kafka producer started.",
            extra={
                "primary_topic":    settings.KAFKA_THREAT_TOPIC,
                "enrichment_topic": settings.KAFKA_NLP_ENRICHMENT_TOPIC,
            },
        )

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._initialized = False
            logger.info("NLP Kafka producer stopped.")

    async def publish_nlp_threat_score(self, score: NLPThreatScore) -> None:
        """
        Publish NLPThreatScore to the primary threat_scores topic.
        Also publishes enrichment payload to nlp_enrichments if transcript is non-empty.
        """
        if not self._initialized or not self._producer:
            raise RuntimeError("NLPThreatScoreProducer.start() not called.")

        primary_payload    = self._serialize_primary(score)
        enrichment_payload = self._serialize_enrichment(score)
        key                = score.tenant_id

        # Publish to threat_scores (primary)
        await self._publish_with_retry(
            topic=settings.KAFKA_THREAT_TOPIC,
            key=key,
            value=primary_payload,
            headers={
                "correlation_id":  score.correlation_id,
                "event_id":        score.event_id,
                "threat_level":    score.threat_level.value,
                "top_intent":      score.top_intent_label or "none",
                "source_service":  score.source_service,
                "schema_version":  score.schema_version,
            },
        )

        # Publish to nlp_enrichments (full transcript + intent breakdown)
        if score.transcript_text:
            await self._publish_with_retry(
                topic=settings.KAFKA_NLP_ENRICHMENT_TOPIC,
                key=key,
                value=enrichment_payload,
                headers={
                    "correlation_id": score.correlation_id,
                    "event_id":       score.event_id,
                },
            )

    async def send_raw(self, topic: str, value: bytes, key: Optional[str] = None) -> None:
        """Low-level send for DLQ messages."""
        if not self._initialized or not self._producer:
            return
        try:
            await self._producer.send_and_wait(topic=topic, value=value, key=key)
        except KafkaError as exc:
            logger.error("Raw send failed.", extra={"topic": topic, "error": str(exc)})
            raise

    async def _publish_with_retry(
        self,
        topic:   str,
        key:     str,
        value:   bytes,
        headers: dict,
    ) -> None:
        for attempt in range(self._MAX_RETRIES):
            try:
                encoded_headers = [
                    (k, v.encode("utf-8")) for k, v in headers.items()
                ]
                meta = await self._producer.send_and_wait(
                    topic=topic,
                    key=key,
                    value=value,
                    headers=encoded_headers,
                )
                logger.info(
                    "NLP payload published.",
                    extra={
                        "topic":       meta.topic,
                        "partition":   meta.partition,
                        "offset":      meta.offset,
                        "bytes":       len(value),
                    },
                )
                return
            except KafkaTimeoutError as exc:
                wait = self._BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "NLP produce timeout; retrying.",
                    extra={"attempt": attempt + 1, "error": str(exc), "wait_s": wait},
                )
                await asyncio.sleep(wait)
            except KafkaError as exc:
                wait = self._BASE_BACKOFF * (2 ** attempt)
                logger.error(
                    "NLP produce error; retrying.",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Failed to publish to {topic} after {self._MAX_RETRIES} retries."
        )

    @staticmethod
    def _serialize_primary(score: NLPThreatScore) -> bytes:
        """
        Lightweight envelope for the threat_scores topic.
        Excludes full transcript to keep message size small for downstream consumers
        that only need scoring signals (alerting engine, real-time dashboard).
        """
        payload = score.model_dump(
            mode="json",
            exclude={"transcription": {"segments"}, "transcript_text": True},
        )
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _serialize_enrichment(score: NLPThreatScore) -> bytes:
        """
        Full payload for nlp_enrichments topic.
        Includes transcript text, word timestamps, and intent breakdown
        for SIEM ingestion and case management.
        """
        payload = score.model_dump(mode="json")
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# ── Module-level singleton ────────────────────────────────────────────────────

_producer_instance: Optional[NLPThreatScoreProducer] = None


def get_nlp_producer() -> NLPThreatScoreProducer:
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = NLPThreatScoreProducer()
    return _producer_instance
