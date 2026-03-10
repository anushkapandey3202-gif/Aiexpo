"""
SentinelAI - Kafka Audio Event Producer
=========================================
Publishes assembled audio session payloads to the Kafka ML processing topic.

Design decisions:
- confluent-kafka-python (librdkafka) is used for production-grade performance.
  The synchronous produce() + async delivery callback pattern is the correct
  approach — confluent-kafka does NOT have a native asyncio API.
- produce() is called from a ThreadPoolExecutor to prevent event loop blocking.
- flush() is also offloaded; a configurable timeout prevents indefinite blocking.
- Delivery reports are surfaced via asyncio.Future resolved in the callback.
- Per-message headers carry: session_id, user_id, org_id, event_type.
  These allow Kafka Streams consumers to filter/route without deserializing payloads.
- The topic is partitioned by organization_id (keyed) so all messages from one
  org land on the same partition, preserving ordering guarantees.
- Circuit breaker: After CIRCUIT_OPEN_AFTER_FAILURES consecutive delivery
  failures, the producer enters OPEN state and fast-fails for CIRCUIT_OPEN_DURATION_S
  seconds before attempting to probe again.
- Large audio sessions (> INLINE_THRESHOLD) reference an S3 key in the payload
  rather than embedding raw bytes. S3 upload is handled in the WebSocket handler
  before calling publish_audio_session().

Retry policy:
- Delivery failures (broker unreachable, timeout) are retried up to MAX_RETRIES.
- Each retry uses exponential backoff (BASE_BACKOFF_S * 2^attempt).
- After MAX_RETRIES the error is re-raised to the caller.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Optional

from confluent_kafka import KafkaError, KafkaException, Producer

from sentinel_ai.services.ingestion.core.audio_buffer import INLINE_THRESHOLD_BYTES
from sentinel_ai.services.ingestion.schemas.audio import (
    AudioFormat,
    AudioSessionState,
    KafkaAudioEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry & circuit-breaker constants
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 3
BASE_BACKOFF_S: float = 0.5
PRODUCE_TIMEOUT_S: float = 30.0
FLUSH_TIMEOUT_S: float = 10.0

CIRCUIT_OPEN_AFTER_FAILURES: int = 5
CIRCUIT_OPEN_DURATION_S: float = 60.0

# Kafka message headers
HEADER_SESSION_ID = "sentinel-session-id"
HEADER_USER_ID = "sentinel-user-id"
HEADER_ORG_ID = "sentinel-org-id"
HEADER_EVENT_TYPE = "sentinel-event-type"
HEADER_CONTENT_TYPE = "content-type"

# Audio sessions ≤ 512 KB are inlined in Kafka payload; larger → S3 reference
INLINE_THRESHOLD = INLINE_THRESHOLD_BYTES

_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="kafka-producer",
)


class CircuitState(str, Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Fast-failing all produce attempts
    HALF_OPEN = "half_open"  # One probe allowed to test recovery


class KafkaProducerError(Exception):
    """Raised when a Kafka publish fails after all retries."""


class KafkaCircuitOpenError(KafkaProducerError):
    """Raised when the circuit breaker is OPEN — broker is unreachable."""


class AudioEventKafkaProducer:
    """
    Async-safe Kafka producer for SentinelAI audio pipeline events.

    Lifecycle:
        producer = AudioEventKafkaProducer()
        await producer.start()             # creates confluent Producer
        ...
        await producer.publish_audio_session(...)
        ...
        await producer.stop()              # flush + close
    """

    def __init__(self) -> None:
        self._producer: Optional[Producer] = None
        self._topic: Optional[str] = None
        self._consecutive_failures: int = 0
        self._circuit_state: CircuitState = CircuitState.CLOSED
        self._circuit_open_since: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initializes the confluent Producer with production settings."""
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()

        kafka_cfg = self._build_producer_config(cfg)
        loop = asyncio.get_event_loop()
        self._producer = await loop.run_in_executor(
            _EXECUTOR,
            lambda: Producer(kafka_cfg),
        )
        self._topic = getattr(cfg, "KAFKA_AUDIO_TOPIC", "sentinelai.audio.raw")

        logger.info(
            "Kafka audio producer started",
            extra={
                "topic": self._topic,
                "bootstrap_servers": kafka_cfg.get("bootstrap.servers"),
            },
        )

    def _build_producer_config(self, cfg: Any) -> dict[str, Any]:
        """
        Builds the librdkafka configuration dictionary.
        All sensitive values (SASL password) come from Settings.
        """
        base: dict[str, Any] = {
            # Delivery reliability: wait for all ISR replicas to acknowledge
            "acks": "all",
            # Maximum in-flight unacknowledged requests per broker connection
            # Set to 1 to ensure ordering (combined with max.in.flight.requests.per.connection=1
            # and enable.idempotence=true for exactly-once delivery at the producer)
            "max.in.flight.requests.per.connection": 1,
            "enable.idempotence": True,
            # Retry configuration
            "retries": 5,
            "retry.backoff.ms": 500,
            # Message compression (snappy is a good balance of speed vs. ratio for audio)
            "compression.type": "snappy",
            # Batch settings — tune for throughput vs. latency
            "linger.ms": 5,
            "batch.size": 65536,
            # Socket / network
            "socket.timeout.ms": 10000,
            "message.timeout.ms": int(PRODUCE_TIMEOUT_S * 1000),
            # TLS
            "security.protocol": getattr(cfg, "KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "ssl.ca.location": getattr(cfg, "KAFKA_SSL_CA_LOCATION", "/etc/ssl/certs/ca-certificates.crt"),
        }

        bootstrap = getattr(cfg, "KAFKA_BOOTSTRAP_SERVERS", None)
        if bootstrap:
            base["bootstrap.servers"] = bootstrap

        sasl_mechanism = getattr(cfg, "KAFKA_SASL_MECHANISM", None)
        if sasl_mechanism:
            base["sasl.mechanism"] = sasl_mechanism
            sasl_user = getattr(cfg, "KAFKA_SASL_USERNAME", None)
            sasl_pass = getattr(cfg, "KAFKA_SASL_PASSWORD", None)
            if sasl_user:
                base["sasl.username"] = sasl_user
            if sasl_pass:
                base["sasl.password"] = (
                    sasl_pass.get_secret_value()
                    if hasattr(sasl_pass, "get_secret_value")
                    else sasl_pass
                )

        return base

    async def stop(self) -> None:
        """Flushes in-flight messages and releases producer resources."""
        if self._producer is None:
            return
        loop = asyncio.get_event_loop()
        try:
            remaining = await loop.run_in_executor(
                _EXECUTOR,
                lambda: self._producer.flush(FLUSH_TIMEOUT_S),  # type: ignore[union-attr]
            )
            if remaining > 0:
                logger.warning(
                    "Kafka producer flush timed out — messages may be lost",
                    extra={"unflushed_messages": remaining},
                )
        except Exception:
            logger.error("Error during Kafka producer flush", exc_info=True)
        finally:
            self._producer = None
            logger.info("Kafka audio producer stopped")

    # ------------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------------

    def _check_circuit(self) -> None:
        """
        Inspects circuit breaker state.
        Raises KafkaCircuitOpenError if OPEN.
        Transitions OPEN → HALF_OPEN after CIRCUIT_OPEN_DURATION_S.
        """
        if self._circuit_state == CircuitState.CLOSED:
            return

        if self._circuit_state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._circuit_open_since
            if elapsed >= CIRCUIT_OPEN_DURATION_S:
                self._circuit_state = CircuitState.HALF_OPEN
                logger.info("Kafka circuit breaker transitioned OPEN → HALF_OPEN")
            else:
                raise KafkaCircuitOpenError(
                    f"Kafka circuit breaker is OPEN — "
                    f"retry in {CIRCUIT_OPEN_DURATION_S - elapsed:.1f}s"
                )

    def _on_delivery_success(self) -> None:
        """Resets failure counter and closes circuit on successful delivery."""
        self._consecutive_failures = 0
        if self._circuit_state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            self._circuit_state = CircuitState.CLOSED
            logger.info("Kafka circuit breaker CLOSED after successful delivery")

    def _on_delivery_failure(self) -> None:
        """Increments failure counter; opens circuit after threshold."""
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= CIRCUIT_OPEN_AFTER_FAILURES
            and self._circuit_state == CircuitState.CLOSED
        ):
            self._circuit_state = CircuitState.OPEN
            self._circuit_open_since = time.monotonic()
            logger.error(
                "Kafka circuit breaker OPENED after consecutive failures",
                extra={"failures": self._consecutive_failures},
            )

    # ------------------------------------------------------------------
    # Core Publish
    # ------------------------------------------------------------------

    async def publish_audio_session(
        self,
        audio_bytes: bytes,
        session_state: AudioSessionState,
    ) -> KafkaAudioEvent:
        """
        Publishes an assembled audio session to the Kafka ML pipeline topic.

        For sessions ≤ INLINE_THRESHOLD bytes, audio_bytes is base64-encoded
        and embedded in the KafkaAudioEvent payload.
        For larger sessions, callers must upload to S3 first and pass the S3 key
        via session_state metadata (key: 's3_object_key').

        Args:
            audio_bytes:   The fully assembled audio buffer from AudioBufferManager.
            session_state: Final session state (user_id, org_id, format, etc.).

        Returns:
            KafkaAudioEvent with Kafka offset populated after delivery.

        Raises:
            KafkaProducerError:      After MAX_RETRIES failures.
            KafkaCircuitOpenError:   If circuit breaker is OPEN.
            ValueError:              On schema validation failure.
        """
        if self._producer is None:
            raise KafkaProducerError(
                "Producer not started — call await producer.start() before publishing"
            )

        self._check_circuit()

        # Build the Kafka event payload
        s3_key = session_state.client_metadata.get("s3_object_key")
        is_inline = len(audio_bytes) <= INLINE_THRESHOLD and s3_key is None

        event = KafkaAudioEvent(
            session_id=session_state.session_id,
            user_id=session_state.user_id,
            organization_id=session_state.organization_id,
            audio_format=session_state.audio_format or AudioFormat(),
            total_chunks=session_state.chunk_count,
            total_bytes=len(audio_bytes),
            s3_object_key=s3_key if not is_inline else None,
            audio_bytes_b64=base64.b64encode(audio_bytes).decode() if is_inline else None,
        )

        payload_bytes = event.model_dump_json().encode("utf-8")
        partition_key = session_state.organization_id.encode("utf-8")

        headers = [
            (HEADER_SESSION_ID, session_state.session_id.encode()),
            (HEADER_USER_ID, session_state.user_id.encode()),
            (HEADER_ORG_ID, session_state.organization_id.encode()),
            (HEADER_EVENT_TYPE, b"audio_session_ready"),
            (HEADER_CONTENT_TYPE, b"application/json"),
        ]

        # Retry loop with exponential backoff
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                delivery_future: asyncio.Future = asyncio.get_event_loop().create_future()
                offset, partition = await self._produce_with_callback(
                    payload=payload_bytes,
                    key=partition_key,
                    headers=headers,
                    delivery_future=delivery_future,
                )
                self._on_delivery_success()

                logger.info(
                    "Audio event published to Kafka",
                    extra={
                        "session_id": session_state.session_id,
                        "topic": self._topic,
                        "partition": partition,
                        "offset": offset,
                        "total_bytes": len(audio_bytes),
                        "inline": is_inline,
                    },
                )
                return event

            except (KafkaException, asyncio.TimeoutError) as exc:
                last_error = exc
                self._on_delivery_failure()

                if attempt < MAX_RETRIES - 1:
                    backoff = BASE_BACKOFF_S * (2 ** attempt)
                    logger.warning(
                        "Kafka produce failed — retrying",
                        extra={
                            "attempt": attempt + 1,
                            "max_retries": MAX_RETRIES,
                            "backoff_seconds": backoff,
                            "error": str(exc),
                            "session_id": session_state.session_id,
                        },
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "Kafka produce failed after all retries",
                        extra={
                            "session_id": session_state.session_id,
                            "attempts": MAX_RETRIES,
                        },
                        exc_info=True,
                    )

        raise KafkaProducerError(
            f"Failed to publish audio session '{session_state.session_id}' "
            f"after {MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    async def _produce_with_callback(
        self,
        payload: bytes,
        key: bytes,
        headers: list[tuple[str, bytes]],
        delivery_future: asyncio.Future,
    ) -> tuple[int, int]:
        """
        Calls Producer.produce() on the executor thread, then awaits the delivery
        callback result. Returns (offset, partition) on success.
        """
        loop = asyncio.get_event_loop()

        def _delivery_callback(err: Optional[KafkaError], msg: Any) -> None:
            """librdkafka delivery callback — resolves or rejects the asyncio Future."""
            if err is not None:
                loop.call_soon_threadsafe(
                    delivery_future.set_exception,
                    KafkaException(err),
                )
            else:
                loop.call_soon_threadsafe(
                    delivery_future.set_result,
                    (msg.offset(), msg.partition()),
                )

        def _produce() -> None:
            assert self._producer is not None
            self._producer.produce(
                topic=self._topic,
                value=payload,
                key=key,
                headers=headers,
                on_delivery=_delivery_callback,
            )
            # Poll is required to trigger delivery callbacks in the librdkafka thread
            self._producer.poll(0)

        await loop.run_in_executor(_EXECUTOR, _produce)

        # Await delivery confirmation with a timeout
        try:
            offset, partition = await asyncio.wait_for(
                delivery_future,
                timeout=PRODUCE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Kafka delivery callback not received within {PRODUCE_TIMEOUT_S}s"
            )

        return offset, partition

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """
        Returns the producer health status including circuit breaker state.
        Used by the /health endpoint.
        """
        return {
            "status": "healthy" if self._circuit_state == CircuitState.CLOSED else "degraded",
            "circuit_state": self._circuit_state.value,
            "consecutive_failures": self._consecutive_failures,
            "topic": self._topic,
            "producer_initialized": self._producer is not None,
        }
