"""
SentinelAI NLP Service — Kafka Audio Consumer

Consumes AudioEvent messages from 'raw_audio_events', resolves audio payloads,
dispatches to the NLP orchestrator, publishes results to 'threat_scores',
and commits offsets only after successful end-to-end processing.

Identical structural contract to the ML service consumer but scoped to
NLP workloads (different consumer group → reads same events independently).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from aiokafka import AIOKafkaConsumer, ConsumerRecord
from aiokafka.errors import KafkaError
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging import get_logger, set_correlation_id
from app.models.schemas import AudioEvent, NLPInferenceRequest
from app.services.nlp_orchestrator import NLPOrchestrator, decode_audio_payload

logger = get_logger("kafka.consumer")
settings = get_settings()


class NLPKafkaConsumer:
    """
    Async Kafka consumer for the raw_audio_events topic (NLP consumer group).

    Key differences from ML service consumer:
      - Consumer group: 'nlp-inference-group' (independent offset tracking).
      - Lower KAFKA_MAX_POLL_RECORDS (transcription is heavier than embedding).
      - NLPOrchestrator replaces InferenceOrchestrator.
    """

    def __init__(
        self,
        orchestrator: NLPOrchestrator,
        producer_publish_fn,
    ) -> None:
        self._orchestrator  = orchestrator
        self._publish        = producer_publish_fn
        self._consumer:      Optional[AIOKafkaConsumer] = None
        self._redis:         Optional[aioredis.Redis]   = None
        self._shutdown_event = asyncio.Event()
        self._inflight_tasks: set[asyncio.Task]         = set()

    async def start(self) -> None:
        """Connect to Kafka and Redis, then run the poll loop."""
        self._redis = await aioredis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
        )

        consumer_config: dict = {
            "bootstrap_servers":     settings.KAFKA_BOOTSTRAP_SERVERS,
            "group_id":              settings.KAFKA_CONSUMER_GROUP,
            "auto_offset_reset":     settings.KAFKA_AUTO_OFFSET_RESET,
            "enable_auto_commit":    settings.KAFKA_ENABLE_AUTO_COMMIT,
            "max_poll_records":      settings.KAFKA_MAX_POLL_RECORDS,
            "session_timeout_ms":    settings.KAFKA_SESSION_TIMEOUT_MS,
            "heartbeat_interval_ms": settings.KAFKA_HEARTBEAT_INTERVAL_MS,
            "value_deserializer":    lambda v: v,
            "key_deserializer":      lambda k: k.decode("utf-8") if k else None,
        }

        if settings.KAFKA_SASL_MECHANISM:
            consumer_config.update({
                "security_protocol":  "SASL_SSL",
                "sasl_mechanism":     settings.KAFKA_SASL_MECHANISM,
                "sasl_plain_username": settings.KAFKA_SASL_USERNAME,
                "sasl_plain_password": (
                    settings.KAFKA_SASL_PASSWORD.get_secret_value()
                    if settings.KAFKA_SASL_PASSWORD else ""
                ),
                "ssl_cafile": settings.KAFKA_SSL_CA_LOCATION,
            })

        self._consumer = AIOKafkaConsumer(
            settings.KAFKA_AUDIO_TOPIC,
            **consumer_config,
        )
        await self._consumer.start()
        logger.info(
            "NLP Kafka consumer started.",
            extra={
                "topic":             settings.KAFKA_AUDIO_TOPIC,
                "group_id":          settings.KAFKA_CONSUMER_GROUP,
                "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            },
        )

        try:
            await self._poll_loop()
        finally:
            await self._drain_inflight()
            await self._consumer.stop()
            if self._redis:
                await self._redis.aclose()
            logger.info("NLP Kafka consumer shut down cleanly.")

    async def stop(self) -> None:
        logger.info("Shutdown signal received by NLP Kafka consumer.")
        self._shutdown_event.set()

    # ── Poll Loop ──────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        async for record in self._consumer:
            if self._shutdown_event.is_set():
                break

            task = asyncio.create_task(
                self._handle_record(record),
                name=f"nlp-{record.partition}-{record.offset}",
            )
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)

            # Soft backpressure gate
            if len(self._inflight_tasks) >= settings.MAX_CONCURRENT_TRANSCRIPTIONS:
                await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)

    async def _drain_inflight(self) -> None:
        if self._inflight_tasks:
            logger.info("Draining NLP inflight tasks.", extra={"count": len(self._inflight_tasks)})
            await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)

    # ── Per-Record Handler ─────────────────────────────────────────────────────

    async def _handle_record(self, record: ConsumerRecord) -> None:
        offset_str = f"{record.topic}[{record.partition}]@{record.offset}"

        try:
            # ── 1. Parse ────────────────────────────────────────────────
            try:
                payload = json.loads(record.value.decode("utf-8"))
                event   = AudioEvent(**payload)
            except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as exc:
                logger.error(
                    "Malformed NLP Kafka message; routing to DLQ.",
                    extra={"offset": offset_str, "error": str(exc)},
                )
                await self._publish_dlq(record, str(exc))
                await self._commit(record)
                return

            set_correlation_id(event.correlation_id)

            # ── 2. Deduplication ─────────────────────────────────────────
            if await self._is_duplicate(event.event_id):
                logger.info("Duplicate event; skipping.", extra={"event_id": event.event_id})
                await self._commit(record)
                return

            # ── 3. Check if NLP pipelines are requested ──────────────────
            if not event.run_transcription and not event.run_intent:
                logger.info(
                    "Neither transcription nor intent requested; skipping.",
                    extra={"event_id": event.event_id},
                )
                await self._commit(record)
                return

            # ── 4. Resolve audio ─────────────────────────────────────────
            try:
                audio_bytes = await self._resolve_audio(event)
            except Exception as exc:
                logger.error(
                    "Audio resolution failed; routing to DLQ.",
                    extra={"event_id": event.event_id, "error": str(exc)},
                )
                await self._publish_dlq(record, str(exc))
                await self._commit(record)
                return

            # ── 5. Duration guard ────────────────────────────────────────
            if event.metadata.duration_seconds < settings.WHISPER_MIN_AUDIO_SECONDS:
                logger.warning(
                    "Audio too short for NLP pipeline; skipping.",
                    extra={
                        "event_id":   event.event_id,
                        "duration_s": event.metadata.duration_seconds,
                    },
                )
                await self._commit(record)
                return

            # ── 6. NLP Orchestration ─────────────────────────────────────
            nlp_request = NLPInferenceRequest(
                audio_event=event,
                audio_bytes=audio_bytes,
            )
            response = await self._orchestrator.process(nlp_request)

            # ── 7. Publish NLP threat score ──────────────────────────────
            await self._publish(response.nlp_threat_score)

            # ── 8. Mark processed ────────────────────────────────────────
            await self._mark_processed(event.event_id)

            # ── 9. Commit offset ─────────────────────────────────────────
            await self._commit(record)

            logger.info(
                "NLP record processed and committed.",
                extra={
                    "event_id":     event.event_id,
                    "offset":       offset_str,
                    "threat_level": response.nlp_threat_score.threat_level.value,
                    "top_intent":   response.nlp_threat_score.top_intent_label,
                    "processing_ms": response.processing_ms,
                },
            )

        except KafkaError as exc:
            logger.error(
                "Unrecoverable Kafka error; not committing.",
                extra={"offset": offset_str, "error": str(exc)},
                exc_info=True,
            )
        except Exception as exc:
            logger.error(
                "Unexpected error; committing to avoid poison-pill loop.",
                extra={"offset": offset_str, "error": str(exc)},
                exc_info=True,
            )
            await self._commit(record)

    # ── Audio Resolution ───────────────────────────────────────────────────────

    async def _resolve_audio(self, event: AudioEvent) -> bytes:
        if event.audio_b64:
            return decode_audio_payload(event)
        if event.audio_s3_uri:
            return await self._fetch_from_s3(event.audio_s3_uri)
        raise ValueError("No audio source in AudioEvent.")

    async def _fetch_from_s3(self, s3_uri: str) -> bytes:
        try:
            import aioboto3
        except ImportError:
            raise RuntimeError("aioboto3 not installed; cannot fetch from S3.")

        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        bucket, _, key = s3_uri[5:].partition("/")
        session = aioboto3.Session()
        async with session.client("s3", region_name=settings.AWS_REGION) as s3:
            resp = await s3.get_object(Bucket=bucket, Key=key)
            return await resp["Body"].read()

    # ── Deduplication ──────────────────────────────────────────────────────────

    async def _is_duplicate(self, event_id: str) -> bool:
        if not self._redis:
            return False
        return bool(await self._redis.exists(f"dedup:nlp:{event_id}"))

    async def _mark_processed(self, event_id: str) -> None:
        if not self._redis:
            return
        await self._redis.setex(
            f"dedup:nlp:{event_id}",
            settings.REDIS_DEDUP_TTL_SECONDS,
            datetime.now(timezone.utc).isoformat(),
        )

    # ── Kafka Utilities ────────────────────────────────────────────────────────

    async def _commit(self, record: ConsumerRecord) -> None:
        try:
            await self._consumer.commit({record.topic_partition: record.offset + 1})
        except KafkaError as exc:
            logger.warning(
                "NLP offset commit failed.",
                extra={"partition": record.partition, "offset": record.offset, "error": str(exc)},
            )

    async def _publish_dlq(self, record: ConsumerRecord, reason: str) -> None:
        dlq_topic = f"{settings.KAFKA_AUDIO_TOPIC}.nlp.dlq"
        try:
            dlq_payload = json.dumps({
                "original_topic":    record.topic,
                "partition":         record.partition,
                "offset":            record.offset,
                "timestamp_ms":      record.timestamp,
                "error_reason":      reason,
                "service":           settings.SERVICE_NAME,
                "raw_value_preview": record.value[:512].decode("utf-8", errors="replace"),
            }).encode("utf-8")

            from app.kafka.producer import get_nlp_producer
            producer = get_nlp_producer()
            await producer.send_raw(topic=dlq_topic, value=dlq_payload)
        except Exception as exc:
            logger.error("Failed to publish to NLP DLQ.", extra={"error": str(exc)})
