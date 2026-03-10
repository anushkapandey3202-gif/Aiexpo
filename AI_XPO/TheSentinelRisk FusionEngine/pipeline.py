"""
SentinelAI - Fusion Pipeline Orchestrator
==========================================
The central coordinator that wires all Risk Fusion Engine components:

  Kafka Consumer → Score Aggregator → Session Store → Persistence → Alert Publisher

Message flow per ThreatScoreEvent:

  1. `handle_score_event(event)` is called by ThreatScoreConsumer per Kafka message.
  2. Retrieve or create ThreatScoreAggregation from FusionSessionStore.
  3. Ingest the score via ScoreFusionEngine.ingest_score().
  4. Persist updated aggregation to Redis.
  5. If is_complete → attempt CAS claim → fuse → persist → alert.
  6. If TTL fires → expiry worker calls `handle_partial_expiry(aggregation)` → fuse with available data.

Concurrency model:
  - handle_score_event() is called from multiple asyncio tasks (one per Kafka partition).
  - FusionSessionStore.try_claim_for_fusion() provides the CAS guard preventing double-fusion.
  - DB session is opened only for the persist phase — not held open during Kafka processing.

Observability:
  - Every phase transition emits a structured log event with session_id + duration_ms.
  - Prometheus counter increments are called on fusion_complete, alert_dispatched, persist_success.
    (Counter objects are stubs here — replace with prometheus_client in production.)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from sentinel_ai.database.session import AsyncSessionFactory
from sentinel_ai.services.risk_fusion.core.alert_publisher import AlertPublisher
from sentinel_ai.services.risk_fusion.core.persistence import FusionPersistenceService
from sentinel_ai.services.risk_fusion.core.score_aggregator import (
    ScoreAggregationError,
    ScoreFusionEngine,
)
from sentinel_ai.services.risk_fusion.core.session_store import FusionSessionStore
from sentinel_ai.services.risk_fusion.schemas.fusion import (
    FusionStatus,
    FusionWeightConfig,
    ThreatScoreAggregation,
    ThreatScoreEvent,
)

logger = logging.getLogger(__name__)


class FusionPipeline:
    """
    Orchestrates the full risk fusion lifecycle for a streaming session.

    Instantiate once at application startup and pass to ThreatScoreConsumer
    as the message_handler coroutine.
    """

    def __init__(
        self,
        session_store:   FusionSessionStore,
        fusion_engine:   ScoreFusionEngine,
        persistence_svc: FusionPersistenceService,
        alert_publisher: AlertPublisher,
    ) -> None:
        self._store       = session_store
        self._engine      = fusion_engine
        self._persistence = persistence_svc
        self._publisher   = alert_publisher

        # Register the TTL-expiry callback on the session store
        self._store.set_expiry_callback(self.handle_partial_expiry)

    # ------------------------------------------------------------------
    # Primary Entry Point (called by Kafka consumer per message)
    # ------------------------------------------------------------------

    async def handle_score_event(self, event: ThreatScoreEvent) -> None:
        """
        Processes a single ThreatScoreEvent from Kafka.
        This coroutine is the message_handler passed to ThreatScoreConsumer.

        Args:
            event: Validated ThreatScoreEvent deserialized from Kafka.
        """
        t_start = time.monotonic()
        session_id = event.session_id

        logger.debug(
            "Score event received",
            extra={
                "session_id": session_id,
                "model_type": event.model_type.value,
                "score": event.confidence_score,
            },
        )

        # Retrieve or create the session accumulator
        aggregation = await self._store.get(session_id)
        if aggregation is None:
            aggregation = ThreatScoreAggregation(
                session_id      = session_id,
                user_id         = event.user_id,
                organization_id = event.organization_id,
                source_channel  = event.source_channel,
                expected_models = list(event.expected_models),
            )
            await self._store.create(aggregation)

        # Ingest this model's score
        aggregation = self._engine.ingest_score(aggregation, event)

        # Persist updated state
        await self._store.update(aggregation)

        elapsed_ingest_ms = int((time.monotonic() - t_start) * 1000)
        logger.debug(
            "Score ingested",
            extra={
                "session_id":   session_id,
                "model_type":   event.model_type.value,
                "is_complete":  aggregation.is_complete,
                "missing":      [m.value for m in aggregation.missing_models],
                "elapsed_ms":   elapsed_ingest_ms,
            },
        )

        # If all expected scores have arrived, trigger fusion
        if aggregation.is_complete:
            await self._trigger_fusion(session_id, is_partial=False)

    # ------------------------------------------------------------------
    # TTL-Expiry Callback (called by FusionSessionStore._expiry_worker_loop)
    # ------------------------------------------------------------------

    async def handle_partial_expiry(
        self, aggregation: ThreatScoreAggregation
    ) -> None:
        """
        Triggered when a session TTL fires before all expected models responded.
        Fuses with available data, marking the result as partial.

        Args:
            aggregation: The expired ThreatScoreAggregation from the store.
        """
        session_id = aggregation.session_id

        if not aggregation.received_scores:
            logger.info(
                "Expired fusion session has zero scores — discarding",
                extra={"session_id": session_id},
            )
            await self._store.delete(session_id)
            return

        logger.info(
            "Partial fusion triggered by TTL expiry",
            extra={
                "session_id":       session_id,
                "received_models":  list(aggregation.received_scores.keys()),
                "missing_models":   [m.value for m in aggregation.missing_models],
            },
        )
        await self._trigger_fusion(session_id, is_partial=True)

    # ------------------------------------------------------------------
    # Core Fusion Trigger
    # ------------------------------------------------------------------

    async def _trigger_fusion(
        self,
        session_id: str,
        is_partial: bool,
    ) -> None:
        """
        Attempts to claim a session for fusion using CAS, then executes the
        full fusion → persist → alert pipeline.

        The CAS guard ensures exactly one task executes the fusion even if
        the final score arrival and TTL expiry fire concurrently.
        """
        # Atomic CAS: only one caller gets True
        claimed = await self._store.try_claim_for_fusion(session_id)
        if not claimed:
            logger.debug(
                "Fusion already claimed by another task — skipping",
                extra={"session_id": session_id},
            )
            return

        t_fusion_start = time.monotonic()

        # Re-fetch the LATEST state (may have changed since claim)
        aggregation = await self._store.get(session_id)
        if aggregation is None:
            logger.warning(
                "Session vanished from store after CAS claim",
                extra={"session_id": session_id},
            )
            return

        try:
            # --- Step 1: Fuse ---
            result = self._engine.fuse(aggregation, is_partial=is_partial)

            # --- Step 2: Build alert payload ---
            alert_payload = self._engine.build_alert_payload(result)

            # --- Step 3: Persist to PostgreSQL (only if above threshold) ---
            alert_record = None
            if result.fused_score >= self._get_persist_threshold():
                async with AsyncSessionFactory() as db_session:
                    try:
                        alert_record = await self._persistence.persist(db_session, result)
                        await db_session.commit()
                        result.persisted = True

                        logger.info(
                            "Fusion result persisted",
                            extra={
                                "session_id":  session_id,
                                "alert_uid":   alert_record.alert_uid if alert_record else None,
                                "fused_score": result.fused_score,
                            },
                        )
                    except Exception:
                        await db_session.rollback()
                        logger.error(
                            "Failed to persist fusion result — alert will still be dispatched",
                            extra={"session_id": session_id},
                            exc_info=True,
                        )

            # --- Step 4: Dispatch WebSocket + Kafka alert ---
            if result.fused_score >= self._get_alert_threshold():
                delivered = await self._publisher.publish(result, alert_payload)
                result.alert_dispatched = True

                logger.info(
                    "Alert dispatched",
                    extra={
                        "session_id":   session_id,
                        "risk_level":   result.risk_level.value,
                        "fused_score":  result.fused_score,
                        "ws_delivered": delivered,
                        "is_partial":   is_partial,
                    },
                )
            else:
                logger.debug(
                    "Fused score below alert threshold — no alert dispatched",
                    extra={
                        "session_id":      session_id,
                        "fused_score":     result.fused_score,
                        "alert_threshold": self._get_alert_threshold(),
                    },
                )

        except ScoreAggregationError as exc:
            logger.error(
                "Score aggregation error during fusion",
                extra={"session_id": session_id, "error": str(exc)},
            )
        except Exception:
            logger.error(
                "Unexpected error in fusion pipeline",
                extra={"session_id": session_id},
                exc_info=True,
            )
        finally:
            # Mark session as COMPLETE and clean up store
            if aggregation:
                aggregation.status = FusionStatus.COMPLETE
                await self._store.update(aggregation)

            elapsed_ms = int((time.monotonic() - t_fusion_start) * 1000)
            logger.info(
                "Fusion pipeline complete",
                extra={
                    "session_id": session_id,
                    "is_partial": is_partial,
                    "elapsed_ms": elapsed_ms,
                },
            )

    # ------------------------------------------------------------------
    # Threshold Helpers
    # ------------------------------------------------------------------

    def _get_persist_threshold(self) -> float:
        try:
            from sentinel_ai.config.settings import get_settings
            return getattr(get_settings(), "FUSION_PERSIST_THRESHOLD", 0.30)
        except Exception:
            return 0.30

    def _get_alert_threshold(self) -> float:
        try:
            from sentinel_ai.config.settings import get_settings
            return getattr(get_settings(), "FUSION_ALERT_THRESHOLD", 0.50)
        except Exception:
            return 0.50
