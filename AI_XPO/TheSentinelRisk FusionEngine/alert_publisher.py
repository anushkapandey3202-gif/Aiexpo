"""
SentinelAI - Alert Publisher (WebSocket + Kafka Fanout)
=========================================================
Dispatches real-time RiskAlertPayload to two destinations simultaneously:

1. WebSocket (mobile client):
   - Clients authenticate and register via `GET /ws/alerts?token=<JWT>`.
   - Alerts are fan-out broadcast to ALL active WebSocket connections for
     the target user_id and the target organization_id.
   - Organization-level broadcast enables SOC dashboard clients to receive
     all org alerts without per-user channel subscriptions.
   - Dead connections are silently removed from the registry.

2. Kafka `threat_alerts` topic:
   - A KafkaAlertEvent is published for downstream consumers:
     SOC dashboard, push notification service, SIEM export, email/SMS notifier.
   - The Kafka publish is non-blocking and best-effort — WebSocket delivery
     does NOT depend on Kafka success.

Alert Registry:
   - `_user_connections`:  user_id → set of (session_id, WebSocket) pairs
   - `_org_connections`:   org_id  → set of (session_id, WebSocket) pairs
   - Registry updates are protected by asyncio.Lock for concurrent safety.

Connection lifecycle:
   CONNECT → register() → [alerts pushed] → unregister() on disconnect/close
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Optional

from confluent_kafka import Producer
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sentinel_ai.services.risk_fusion.schemas.fusion import (
    FusedRiskResult,
    KafkaAlertEvent,
    RiskAlertPayload,
    RiskLevel,
)

logger = logging.getLogger(__name__)

TOPIC_THREAT_ALERTS = "sentinelai.threat.alerts"

_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="alert-kafka-producer"
)


@dataclass
class AlertConnection:
    """Tracks a single registered alert WebSocket connection."""
    connection_id: str
    user_id:       str
    org_id:        str
    websocket:     WebSocket
    connected_at:  datetime

    @property
    def is_alive(self) -> bool:
        return self.websocket.client_state == WebSocketState.CONNECTED


class AlertPublisher:
    """
    Singleton service that broadcasts RiskAlertPayload objects to
    registered WebSocket clients and publishes to Kafka for fanout.

    One instance is shared across the FastAPI app (app.state.alert_publisher).
    """

    def __init__(self) -> None:
        # user_id  → set of connection_ids
        self._user_connections: dict[str, set[str]]   = {}
        # org_id   → set of connection_ids
        self._org_connections:  dict[str, set[str]]   = {}
        # connection_id → AlertConnection
        self._connections:      dict[str, AlertConnection] = {}

        self._lock                = asyncio.Lock()
        self._kafka_producer: Optional[Producer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initializes the Kafka alert producer."""
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()

        loop = asyncio.get_event_loop()
        producer_cfg = self._build_producer_cfg(cfg)
        self._kafka_producer = await loop.run_in_executor(
            _EXECUTOR, lambda: Producer(producer_cfg)
        )
        logger.info(
            "AlertPublisher started",
            extra={"kafka_topic": TOPIC_THREAT_ALERTS},
        )

    async def stop(self) -> None:
        """Closes all WebSocket connections and flushes the Kafka producer."""
        async with self._lock:
            conn_ids = list(self._connections.keys())

        for conn_id in conn_ids:
            await self._close_connection(conn_id, reason="Server shutting down")

        if self._kafka_producer:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _EXECUTOR,
                lambda: self._kafka_producer.flush(10),  # type: ignore[union-attr]
            )
        logger.info("AlertPublisher stopped")

    # ------------------------------------------------------------------
    # Connection Registry
    # ------------------------------------------------------------------

    async def register(
        self,
        user_id:   str,
        org_id:    str,
        websocket: WebSocket,
    ) -> str:
        """
        Registers a new alert WebSocket connection.

        Args:
            user_id:   Authenticated user UUID string.
            org_id:    Organisation UUID string.
            websocket: Accepted (post-accept()) FastAPI WebSocket.

        Returns:
            connection_id string for this registration.
        """
        conn_id = str(uuid.uuid4())
        conn    = AlertConnection(
            connection_id = conn_id,
            user_id       = user_id,
            org_id        = org_id,
            websocket     = websocket,
            connected_at  = datetime.utcnow(),
        )

        async with self._lock:
            self._connections[conn_id] = conn

            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(conn_id)

            if org_id not in self._org_connections:
                self._org_connections[org_id] = set()
            self._org_connections[org_id].add(conn_id)

        logger.info(
            "Alert WebSocket connection registered",
            extra={
                "connection_id": conn_id,
                "user_id": user_id,
                "org_id": org_id,
                "total_alert_connections": len(self._connections),
            },
        )
        return conn_id

    async def unregister(self, connection_id: str) -> None:
        """Removes a connection from all registries. Idempotent."""
        async with self._lock:
            conn = self._connections.pop(connection_id, None)
            if conn is None:
                return

            self._user_connections.get(conn.user_id, set()).discard(connection_id)
            if not self._user_connections.get(conn.user_id):
                self._user_connections.pop(conn.user_id, None)

            self._org_connections.get(conn.org_id, set()).discard(connection_id)
            if not self._org_connections.get(conn.org_id):
                self._org_connections.pop(conn.org_id, None)

        logger.debug(
            "Alert connection unregistered",
            extra={"connection_id": connection_id},
        )

    # ------------------------------------------------------------------
    # Alert Dispatch
    # ------------------------------------------------------------------

    async def publish(
        self,
        result: FusedRiskResult,
        alert_payload: RiskAlertPayload,
    ) -> int:
        """
        Broadcasts a risk alert to all relevant WebSocket connections
        and publishes a KafkaAlertEvent to the `threat_alerts` topic.

        Targeting:
          - Direct user channel:    all connections for result.user_id
          - Organization channel:   all connections for result.organization_id
          - Deduplication:          a connection that is in BOTH sets receives
                                    the alert exactly once (set union before fanout)

        Args:
            result:        The FusedRiskResult driving the alert.
            alert_payload: Pre-built RiskAlertPayload for WebSocket delivery.

        Returns:
            Number of WebSocket connections successfully delivered to.
        """
        payload_json = alert_payload.model_dump_json()

        # Collect unique connection IDs (user + org, deduped)
        async with self._lock:
            user_conn_ids = set(self._user_connections.get(result.user_id, set()))
            org_conn_ids  = set(self._org_connections.get(result.organization_id, set()))
            target_ids    = user_conn_ids | org_conn_ids
            snapshot      = {
                cid: self._connections[cid]
                for cid in target_ids
                if cid in self._connections
            }

        if not snapshot:
            logger.debug(
                "No active alert connections for session — alert queued in Kafka only",
                extra={
                    "session_id": result.session_id,
                    "user_id": result.user_id,
                    "org_id": result.organization_id,
                },
            )

        # Fan-out WebSocket delivery (concurrent)
        delivery_tasks = [
            self._send_to_connection(cid, conn, payload_json)
            for cid, conn in snapshot.items()
        ]
        delivery_results = await asyncio.gather(*delivery_tasks, return_exceptions=True)

        delivered       = sum(1 for r in delivery_results if r is True)
        dead_connections = [
            cid for cid, r in zip(snapshot.keys(), delivery_results)
            if r is False
        ]

        # Clean up dead connections
        for cid in dead_connections:
            await self.unregister(cid)

        logger.info(
            "Risk alert dispatched",
            extra={
                "session_id":       result.session_id,
                "risk_level":       result.risk_level.value,
                "fused_score":      result.fused_score,
                "ws_targets":       len(snapshot),
                "ws_delivered":     delivered,
                "dead_connections": len(dead_connections),
            },
        )

        # Non-blocking Kafka fanout (best-effort)
        asyncio.create_task(
            self._publish_to_kafka(result, alert_payload),
            name=f"kafka-alert-{result.session_id}",
        )

        return delivered

    async def _send_to_connection(
        self,
        connection_id: str,
        conn: AlertConnection,
        payload_json: str,
    ) -> bool:
        """
        Sends a JSON payload to a single WebSocket connection.
        Returns True on success, False if the connection is dead/gone.
        """
        if not conn.is_alive:
            return False

        try:
            await conn.websocket.send_text(payload_json)
            return True
        except Exception:
            logger.debug(
                "Failed to send alert — connection closed",
                extra={"connection_id": connection_id},
            )
            return False

    async def _close_connection(self, connection_id: str, reason: str) -> None:
        """Forcibly closes and unregisters a WebSocket connection."""
        async with self._lock:
            conn = self._connections.get(connection_id)
        if conn and conn.is_alive:
            try:
                await conn.websocket.close(code=1001, reason=reason)
            except Exception:
                pass
        await self.unregister(connection_id)

    # ------------------------------------------------------------------
    # Kafka Fanout
    # ------------------------------------------------------------------

    async def _publish_to_kafka(
        self,
        result: FusedRiskResult,
        alert: RiskAlertPayload,
    ) -> None:
        """Publishes a KafkaAlertEvent to the threat_alerts topic (best-effort)."""
        if self._kafka_producer is None:
            return

        event = KafkaAlertEvent(
            fusion_id        = result.fusion_id,
            session_id       = result.session_id,
            user_id          = result.user_id,
            organization_id  = result.organization_id,
            fused_score      = result.fused_score,
            risk_level       = result.risk_level,
            threat_summary   = alert.threat_summary,
            fused_at         = result.fused_at,
            source_channel   = result.source_channel,
            is_partial       = result.is_partial_fusion,
            component_scores = alert.component_scores,
            active_boosters  = result.active_boosters,
        )

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                _EXECUTOR,
                partial(
                    self._kafka_producer.produce,
                    topic=TOPIC_THREAT_ALERTS,
                    value=event.model_dump_json().encode("utf-8"),
                    key=result.organization_id.encode("utf-8"),
                    headers=[
                        ("sentinel-event-type", b"risk_alert"),
                        ("sentinel-risk-level", result.risk_level.value.encode()),
                        ("sentinel-org-id",     result.organization_id.encode()),
                    ],
                ),
            )
            await loop.run_in_executor(
                _EXECUTOR, partial(self._kafka_producer.poll, 0)
            )
        except Exception:
            logger.error(
                "Failed to publish KafkaAlertEvent — WebSocket delivery was not affected",
                extra={"session_id": result.session_id},
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "total_alert_connections": len(self._connections),
            "unique_users_subscribed": len(self._user_connections),
            "unique_orgs_subscribed":  len(self._org_connections),
        }

    def _build_producer_cfg(self, cfg: Any) -> dict[str, Any]:
        return {
            "bootstrap.servers": getattr(cfg, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "acks":              "1",
            "compression.type":  "snappy",
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
