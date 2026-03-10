"""
SentinelAI - Threat Score Kafka Consumer
==========================================
Consumes ThreatScoreEvent messages from the `threat_scores` Kafka topic
and drives the fusion pipeline.

Consumer design:
- Uses confluent-kafka-python (librdkafka) for production performance.
- Group ID: `sentinelai-risk-fusion` — all Risk Fusion service instances
  share one consumer group; partitions are distributed across pods.
- MANUAL offset commit (enable.auto.commit=false):
    Offsets are committed ONLY after the fusion pipeline has persisted
    the result to PostgreSQL. This guarantees at-least-once processing.
    Duplicate events (after rebalance/crash recovery) are deduplicated
    via the Redis CAS status check in FusionSessionStore.try_claim_for_fusion().
- Deserialization errors are forwarded to a Dead-Letter Topic (DLT)
  `threat_scores.DLT` with full original message bytes and error context.
- Blocking poll() is offloaded to a ThreadPoolExecutor so the asyncio
  event loop remains responsive during broker timeouts.
- A rebalance callback logs partition assignment changes for monitoring.

Backpressure:
  MAX_POLL_RECORDS controls how many messages are processed per poll batch.
  The consumer pauses partition consumption if the in-flight fusion queue
  exceeds PAUSE_THRESHOLD, resuming once it drains below RESUME_THRESHOLD.
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Coroutine, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer

from sentinel_ai.services.risk_fusion.schemas.fusion import ThreatScoreEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consumer configuration constants
# ---------------------------------------------------------------------------
CONSUMER_GROUP_ID    = "sentinelai-risk-fusion"
TOPIC_THREAT_SCORES  = "sentinelai.threat.scores"
TOPIC_DLT            = "sentinelai.threat.scores.DLT"
MAX_POLL_RECORDS      = 50
POLL_TIMEOUT_S        = 1.0        # Seconds to block waiting for messages
PAUSE_THRESHOLD       = 500        # Pause if fusion queue exceeds this
RESUME_THRESHOLD      = 100        # Resume when fusion queue drains below this

_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="kafka-consumer",
)


class ThreatScoreConsumer:
    """
    Async-safe Kafka consumer for the `threat_scores` ML pipeline topic.

    Lifecycle:
        consumer = ThreatScoreConsumer(message_handler=my_handler)
        await consumer.start()     # subscribe and begin polling
        ...
        await consumer.stop()      # graceful drain and shutdown
    """

    MessageHandler = Callable[[ThreatScoreEvent], Coroutine[Any, Any, None]]

    def __init__(self, message_handler: "ThreatScoreConsumer.MessageHandler") -> None:
        self._handler       = message_handler
        self._consumer: Optional[Consumer] = None
        self._dlt_producer: Optional[Producer] = None
        self._running       = False
        self._poll_task: Optional[asyncio.Task] = None
        self._in_flight     = 0
        self._paused        = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Creates the Kafka consumer, subscribes to the topic, and starts polling."""
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()

        consumer_config = self._build_consumer_config(cfg)
        dlt_config      = self._build_dlt_producer_config(cfg)

        loop = asyncio.get_event_loop()

        self._consumer = await loop.run_in_executor(
            _EXECUTOR, lambda: Consumer(consumer_config)
        )
        self._dlt_producer = await loop.run_in_executor(
            _EXECUTOR, lambda: Producer(dlt_config)
        )

        # Subscribe with rebalance callback
        self._consumer.subscribe(
            [TOPIC_THREAT_SCORES],
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
        )

        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="kafka-threat-scores-consumer"
        )

        logger.info(
            "ThreatScoreConsumer started",
            extra={
                "topic": TOPIC_THREAT_SCORES,
                "group_id": CONSUMER_GROUP_ID,
                "max_poll_records": MAX_POLL_RECORDS,
            },
        )

    async def stop(self) -> None:
        """Signals the poll loop to stop, drains in-flight events, commits offsets."""
        self._running = False

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._consumer is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _EXECUTOR,
                self._consumer.close,  # Commits offsets and leaves the consumer group
            )
            self._consumer = None

        if self._dlt_producer is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _EXECUTOR,
                lambda: self._dlt_producer.flush(10),  # type: ignore[union-attr]
            )
            self._dlt_producer = None

        logger.info("ThreatScoreConsumer stopped and offsets committed")

    # ------------------------------------------------------------------
    # Poll Loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """
        Core polling loop. Runs as a background asyncio task.
        Polls Kafka for messages and dispatches them to the fusion handler.
        """
        logger.info("Kafka poll loop started")
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Backpressure: pause if fusion queue is saturated
                await self._manage_backpressure()

                # Poll is blocking — offload to executor
                msg: Optional[Message] = await loop.run_in_executor(
                    _EXECUTOR,
                    partial(self._consumer.poll, POLL_TIMEOUT_S),  # type: ignore[union-attr]
                )

                if msg is None:
                    continue  # Timeout — no messages available

                if msg.error():
                    await self._handle_kafka_error(msg)
                    continue

                await self._process_message(msg)

            except asyncio.CancelledError:
                logger.info("Kafka poll loop cancelled")
                return
            except Exception:
                logger.error(
                    "Unexpected error in Kafka poll loop",
                    exc_info=True,
                )
                await asyncio.sleep(1)  # Brief pause before retrying

    async def _process_message(self, msg: Message) -> None:
        """
        Deserializes a single Kafka message and invokes the fusion handler.
        Commits the offset AFTER successful handler execution (at-least-once).
        Forwards malformed messages to the Dead-Letter Topic.
        """
        topic     = msg.topic()
        partition = msg.partition()
        offset    = msg.offset()
        raw_value = msg.value()

        logger.debug(
            "Kafka message received",
            extra={"topic": topic, "partition": partition, "offset": offset},
        )

        # Deserialize
        try:
            payload_dict = json.loads(raw_value)
            event        = ThreatScoreEvent.model_validate(payload_dict)
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            logger.error(
                "Failed to deserialise ThreatScoreEvent — sending to DLT",
                extra={
                    "topic": topic,
                    "partition": partition,
                    "offset": offset,
                    "error": str(exc),
                },
            )
            await self._send_to_dlt(msg, error=str(exc))
            # Commit the bad offset to avoid infinite retry loops
            await self._commit_offset(msg)
            return

        # Invoke fusion handler
        self._in_flight += 1
        try:
            await self._handler(event)
        except Exception:
            logger.error(
                "Fusion handler raised an unhandled exception",
                extra={
                    "session_id": event.session_id,
                    "model_type": event.model_type.value,
                    "offset": offset,
                },
                exc_info=True,
            )
            # Send to DLT so the message isn't silently dropped
            await self._send_to_dlt(
                msg,
                error="Fusion handler exception",
                event_id=event.event_id,
            )
        finally:
            self._in_flight -= 1

        # Manual commit — only after successful or DLT-forwarded processing
        await self._commit_offset(msg)

    async def _commit_offset(self, msg: Message) -> None:
        """Commits a single message offset asynchronously."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _EXECUTOR,
                partial(self._consumer.commit, message=msg, asynchronous=False),  # type: ignore[union-attr]
            )
        except KafkaException:
            logger.error(
                "Offset commit failed",
                extra={
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                },
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Error Handling
    # ------------------------------------------------------------------

    async def _handle_kafka_error(self, msg: Message) -> None:
        """Handles partition EOF and other Kafka errors from polled messages."""
        err = msg.error()
        if err.code() == KafkaError._PARTITION_EOF:
            logger.debug(
                "Reached end of partition",
                extra={"partition": msg.partition(), "offset": msg.offset()},
            )
        else:
            logger.error(
                "Kafka consumer error",
                extra={
                    "code": err.code(),
                    "name": err.name(),
                    "str": err.str(),
                    "partition": msg.partition(),
                },
            )

    async def _send_to_dlt(
        self,
        original_msg: Message,
        error: str,
        event_id: Optional[str] = None,
    ) -> None:
        """Forwards an unprocessable message to the Dead-Letter Topic with error context."""
        if self._dlt_producer is None:
            return

        dlt_value = json.dumps({
            "original_topic":     original_msg.topic(),
            "original_partition": original_msg.partition(),
            "original_offset":    original_msg.offset(),
            "original_value":     original_msg.value().decode("utf-8", errors="replace"),
            "error":              error,
            "event_id":           event_id,
        }).encode("utf-8")

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _EXECUTOR,
                partial(
                    self._dlt_producer.produce,
                    topic=TOPIC_DLT,
                    value=dlt_value,
                    key=original_msg.key(),
                ),
            )
            await loop.run_in_executor(_EXECUTOR, partial(self._dlt_producer.poll, 0))
        except Exception:
            logger.error("Failed to publish to DLT", exc_info=True)

    # ------------------------------------------------------------------
    # Backpressure
    # ------------------------------------------------------------------

    async def _manage_backpressure(self) -> None:
        """
        Pauses / resumes partition consumption based on in-flight count.
        When paused, this coroutine blocks until the queue drains.
        """
        if not self._paused and self._in_flight >= PAUSE_THRESHOLD:
            self._paused = True
            logger.warning(
                "Kafka consumer paused — fusion queue saturated",
                extra={"in_flight": self._in_flight, "threshold": PAUSE_THRESHOLD},
            )

        if self._paused:
            while self._in_flight >= RESUME_THRESHOLD:
                await asyncio.sleep(0.1)
            self._paused = False
            logger.info(
                "Kafka consumer resumed — fusion queue drained",
                extra={"in_flight": self._in_flight},
            )

    # ------------------------------------------------------------------
    # Rebalance Callbacks (called from librdkafka thread — synchronous)
    # ------------------------------------------------------------------

    def _on_assign(self, consumer: Consumer, partitions: list) -> None:
        logger.info(
            "Kafka partitions assigned",
            extra={
                "partitions": [
                    f"{p.topic}[{p.partition}]@{p.offset}" for p in partitions
                ]
            },
        )

    def _on_revoke(self, consumer: Consumer, partitions: list) -> None:
        logger.info(
            "Kafka partitions revoked",
            extra={
                "partitions": [f"{p.topic}[{p.partition}]" for p in partitions]
            },
        )

    # ------------------------------------------------------------------
    # Config Builders
    # ------------------------------------------------------------------

    def _build_consumer_config(self, cfg: Any) -> dict[str, Any]:
        return {
            "bootstrap.servers":           getattr(cfg, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "group.id":                    CONSUMER_GROUP_ID,
            "auto.offset.reset":           "earliest",
            "enable.auto.commit":          False,    # MANUAL COMMIT — core reliability guarantee
            "max.poll.interval.ms":        300_000,  # 5 min — avoid rebalance on slow fusions
            "session.timeout.ms":          30_000,
            "heartbeat.interval.ms":       3_000,
            "fetch.max.bytes":             52_428_800,  # 50 MB
            "max.partition.fetch.bytes":   10_485_760,  # 10 MB per partition
            "security.protocol":           getattr(cfg, "KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "ssl.ca.location":             getattr(cfg, "KAFKA_SSL_CA_LOCATION", "/etc/ssl/certs/ca-certificates.crt"),
            "sasl.mechanism":              getattr(cfg, "KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
            "sasl.username":               getattr(cfg, "KAFKA_SASL_USERNAME", ""),
            "sasl.password":               (
                getattr(cfg, "KAFKA_SASL_PASSWORD", None).get_secret_value()
                if hasattr(getattr(cfg, "KAFKA_SASL_PASSWORD", None), "get_secret_value")
                else getattr(cfg, "KAFKA_SASL_PASSWORD", "")
            ),
        }

    def _build_dlt_producer_config(self, cfg: Any) -> dict[str, Any]:
        return {
            "bootstrap.servers": getattr(cfg, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "acks":              "1",   # DLT: leader ack only — best effort
            "security.protocol": getattr(cfg, "KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "ssl.ca.location":   getattr(cfg, "KAFKA_SSL_CA_LOCATION", "/etc/ssl/certs/ca-certificates.crt"),
            "sasl.mechanism":    getattr(cfg, "KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
            "sasl.username":     getattr(cfg, "KAFKA_SASL_USERNAME", ""),
            "sasl.password":     (
                getattr(cfg, "KAFKA_SASL_PASSWORD", None).get_secret_value()
                if hasattr(getattr(cfg, "KAFKA_SASL_PASSWORD", None), "get_secret_value")
                else getattr(cfg, "KAFKA_SASL_PASSWORD", "")
            ),
        }
