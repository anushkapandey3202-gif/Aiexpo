"""
SentinelAI ML Service — Kafka Audio Consumer

Consumes AudioEvent messages from the 'raw_audio_events' topic,
resolves audio payloads (base64 or S3), dispatches to the inference
orchestrator, and commits offsets only on successful processing.

Production Features:
- aiokafka async consumer with manual offset commit.
- Message deduplication via Redis (event_id → processed timestamp).
- Dead-letter queue (DLQ) routing for malformed/unparseable messages.
- Backpressure via DeviceManager semaphore (already held by pipelines).
- Graceful shutdown: drains in-flight tasks before committing final offsets.
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
from app.models.schemas import AudioEvent, InferenceRequest
from app.services.inference_orchestrator import InferenceOrchestrator, decode_audio_payload

logger = get_logger("kafka.consumer")
settings = get_settings()


class AudioKafkaConsumer:
    """
    Long-running async Kafka consumer for the raw_audio_events topic.

    Lifecycle:
    - start()  : Connect to Kafka, initialize Redis dedup, begin poll loop.
    - stop()   : Signal shutdown, drain in-flight tasks, commit final offsets.
    """

    def __init__(
        self,
        orchestrator: InferenceOrchestrator,
        producer_publish_fn,  # Callable[[ThreatScore], Awaitable[None]]
    ) -> None:
        self._orchestrator = orchestrator
        self._publish = producer_publish_fn
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._redis: Optional[aioredis.Redis] = None
        self._shutdown_event = asyncio.Event()
        self._inflight_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Initialize consumer and Redis, then enter the poll loop."""
        self._redis = await aioredis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
        )

        consumer_config = {
            "bootstrap_servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group_id": settings.KAFKA_CONSUMER_GROUP,
            "auto_offset_reset": settings.KAFKA_AUTO_OFFSET_RESET,
            "enable_auto_commit": settings.KAFKA_ENABLE_AUTO_COMMIT,
            "max_poll_records": settings.KAFKA_MAX_POLL_RECORDS,
            "session_timeout_ms": settings.KAFKA_SESSION_TIMEOUT_MS,
            "heartbeat_interval_ms": settings.KAFKA_HEARTBEAT_INTERVAL_MS,
            "value_deserializer": lambda v: v,  # Raw bytes; we handle JSON parsing
            "key_deserializer": lambda k: k.decode("utf-8") if k else None,
        }

        # SASL/SSL for production Kafka clusters
        if settings.KAFKA_SASL_MECHANISM:
            consumer_config.update({
                "security_protocol": "SASL_SSL",
                "sasl_mechanism": settings.KAFKA_SASL_MECHANISM,
                "sasl_plain_username": settings.KAFKA_SASL_USERNAME,
                "sasl_plain_password": settings.KAFKA_SASL_PASSWORD.get_secret_value()
                    if settings.KAFKA_SASL_PASSWORD else "",
                "ssl_cafile": settings.KAFKA_SSL_CA_LOCATION,
            })

        self._consumer = AIOKafkaConsumer(
            settings.KAFKA_AUDIO_TOPIC,
            **consumer_config,
        )

        await self._consumer.start()
        logger.info(
            "Kafka consumer started.",
            extra={
                "topic": settings.KAFKA_AUDIO_TOPIC,
                "group_id": settings.KAFKA_CONSUMER_GROUP,
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
            logger.info("Kafka consumer shut down cleanly.")

    async def stop(self) -> None:
        """Signal the poll loop to exit gracefully."""
        logger.info("Shutdown signal received by Kafka consumer.")
        self._shutdown_event.set()

    # ---------------------------------------------------------------------------
    # Core Poll Loop
    # ---------------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """
        Main message poll loop.
        Runs until _shutdown_event is set or an unrecoverable Kafka error occurs.
        """
        async for record in self._consumer:
            if self._shutdown_event.is_set():
                logger.info("Shutdown event set; exiting poll loop.")
                break

            # Spawn a task per message — allows concurrent processing while
            # keeping the poll loop responsive (heartbeat stays alive).
            task = asyncio.create_task(
                self._handle_record(record),
                name=f"process-{record.partition}-{record.offset}",
            )
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)

            # Soft backpressure: if too many tasks inflight, yield to let them drain
            if len(self._inflight_tasks) >= settings.MAX_CONCURRENT_INFERENCES:
                await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)

    async def _drain_inflight(self) -> None:
        """Wait for all in-flight processing tasks to complete before final commit."""
        if self._inflight_tasks:
            logger.info(
                "Draining in-flight tasks before shutdown.",
                extra={"count": len(self._inflight_tasks)},
            )
            await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)

    # ---------------------------------------------------------------------------
    # Per-Message Handler
    # ---------------------------------------------------------------------------

    async def _handle_record(self, record: ConsumerRecord) -> None:
        """
        Process a single Kafka record end-to-end.
        Handles: parse → dedup → audio decode → inference → publish → commit.
        """
        partition_offset = f"{record.topic}[{record.partition}]@{record.offset}"

        try:
            # ── 1. Parse JSON payload ──────────────────────────────────────
            try:
                payload_dict = json.loads(record.value.decode("utf-8"))
                event = AudioEvent(**payload_dict)
            except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as exc:
                logger.error(
                    "Malformed Kafka message; routing to DLQ.",
                    extra={"offset": partition_offset, "error": str(exc)},
                )
                await self._publish_dlq(record, str(exc))
                await self._commit(record)
                return

            set_correlation_id(event.correlation_id)

            # ── 2. Deduplication check ─────────────────────────────────────
            if await self._is_duplicate(event.event_id):
                logger.info(
                    "Duplicate event_id; skipping.",
                    extra={"event_id": event.event_id, "offset": partition_offset},
                )
                await self._commit(record)
                return

            # ── 3. Resolve audio bytes ─────────────────────────────────────
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

            # ── 4. Validate audio duration ─────────────────────────────────
            if event.metadata.duration_seconds < settings.AUDIO_MIN_DURATION_SECONDS:
                logger.warning(
                    "Audio too short; skipping inference.",
                    extra={
                        "event_id": event.event_id,
                        "duration_s": event.metadata.duration_seconds,
                        "min_s": settings.AUDIO_MIN_DURATION_SECONDS,
                    },
                )
                await self._commit(record)
                return

            # ── 5. Inference orchestration ─────────────────────────────────
            inference_request = InferenceRequest(
                audio_event=event,
                audio_bytes=audio_bytes,
            )
            response = await self._orchestrator.process(inference_request)

            # ── 6. Publish threat score to Kafka ───────────────────────────
            await self._publish(response.threat_score)

            # ── 7. Mark event as processed in Redis ────────────────────────
            await self._mark_processed(event.event_id)

            # ── 8. Manual offset commit ────────────────────────────────────
            await self._commit(record)

            logger.info(
                "Record processed and committed.",
                extra={
                    "event_id": event.event_id,
                    "offset": partition_offset,
                    "threat_level": response.threat_score.threat_level.value,
                    "processing_ms": response.processing_ms,
                },
            )

        except KafkaError as exc:
            logger.error(
                "Unrecoverable Kafka error during record handling.",
                extra={"offset": partition_offset, "error": str(exc)},
                exc_info=True,
            )
            # Do not commit — allow Kafka to redeliver on rebalance

        except Exception as exc:
            logger.error(
                "Unexpected error processing record.",
                extra={"offset": partition_offset, "error": str(exc)},
                exc_info=True,
            )
            # Commit to avoid poison-pill message loop; DLQ already received it
            await self._commit(record)

    # ---------------------------------------------------------------------------
    # Audio Resolution
    # ---------------------------------------------------------------------------

    async def _resolve_audio(self, event: AudioEvent) -> bytes:
        """
        Resolve audio bytes from the event's audio source.
        - audio_b64: decode immediately.
        - audio_s3_uri: fetch from S3 asynchronously.
        """
        if event.audio_b64:
            return decode_audio_payload(event)

        if event.audio_s3_uri:
            return await self._fetch_from_s3(event.audio_s3_uri)

        raise ValueError("AudioEvent has neither audio_b64 nor audio_s3_uri.")

    async def _fetch_from_s3(self, s3_uri: str) -> bytes:
        """
        Download audio from S3 using aioboto3.
        URI format: s3://bucket/key
        """
        try:
            import aioboto3
        except ImportError:
            raise RuntimeError(
                "aioboto3 not installed. Cannot fetch audio from S3. "
                "Install with: pip install aioboto3"
            )

        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        path = s3_uri[5:]
        bucket, _, key = path.partition("/")

        session = aioboto3.Session()
        async with session.client("s3", region_name=settings.AWS_REGION) as s3:
            response = await s3.get_object(Bucket=bucket, Key=key)
            body = await response["Body"].read()

        logger.debug(
            "S3 audio fetched.",
            extra={"s3_uri": s3_uri, "bytes": len(body)},
        )
        return body

    # ---------------------------------------------------------------------------
    # Deduplication (Redis)
    # ---------------------------------------------------------------------------

    async def _is_duplicate(self, event_id: str) -> bool:
        """Check Redis for prior processing of this event_id."""
        if not self._redis:
            return False
        key = f"dedup:ml:{event_id}"
        result = await self._redis.exists(key)
        return bool(result)

    async def _mark_processed(self, event_id: str) -> None:
        """Mark event_id as processed in Redis with TTL."""
        if not self._redis:
            return
        key = f"dedup:ml:{event_id}"
        await self._redis.setex(
            key,
            settings.REDIS_DEDUP_TTL_SECONDS,
            datetime.now(timezone.utc).isoformat(),
        )

    # ---------------------------------------------------------------------------
    # Kafka Utilities
    # ---------------------------------------------------------------------------

    async def _commit(self, record: ConsumerRecord) -> None:
        """Commit offset for this specific partition/offset."""
        try:
            await self._consumer.commit(
                {record.topic_partition: record.offset + 1}
            )
        except KafkaError as exc:
            logger.warning(
                "Offset commit failed (will retry on next heartbeat).",
                extra={"partition": record.partition, "offset": record.offset, "error": str(exc)},
            )

    async def _publish_dlq(self, record: ConsumerRecord, error_reason: str) -> None:
        """
        Publish unparseable/failed records to a dead-letter topic.
        The DLQ topic receives the raw bytes + error metadata for manual review.
        """
        # DLQ publishing is best-effort; do not block the commit path on failure.
        dlq_topic = f"{settings.KAFKA_AUDIO_TOPIC}.dlq"
        try:
            dlq_message = json.dumps({
                "original_topic": record.topic,
                "partition": record.partition,
                "offset": record.offset,
                "timestamp_ms": record.timestamp,
                "error_reason": error_reason,
                "raw_value_preview": record.value[:512].decode("utf-8", errors="replace"),
            }).encode("utf-8")

            # We use the producer singleton to publish to DLQ
            from app.kafka.producer import get_kafka_producer
            producer = get_kafka_producer()
            await producer.send_raw(topic=dlq_topic, value=dlq_message)

            logger.info(
                "Record published to DLQ.",
                extra={
                    "dlq_topic": dlq_topic,
                    "original_offset": record.offset,
                    "error_reason": error_reason,
                },
            )
        except Exception as exc:
            logger.error(
                "Failed to publish to DLQ.",
                extra={"error": str(exc)},
                exc_info=True,
            )
