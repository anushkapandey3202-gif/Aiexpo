"""
SentinelAI - Risk Alert WebSocket Router
==========================================
Provides the WebSocket endpoint mobile clients connect to for real-time
threat alert push notifications.

This endpoint is separate from the ingestion WebSocket (Phase 2):
  - Ingestion WS:  wss://api.sentinelai.io/ws/audio/stream    (push audio IN)
  - Alert WS:      wss://api.sentinelai.io/ws/alerts           (receive alerts OUT)

Mobile clients maintain a persistent alert WebSocket connection for the
duration of an active call session. When the Risk Fusion Engine determines
a threat exceeds the alert threshold, RiskAlertPayload is pushed through
this channel in < 100ms of the Kafka consumer processing the final score.

Connection flow:
  1. Client connects: wss://api.sentinelai.io/ws/alerts?token=<JWT>
  2. Server validates JWT (same authenticator as Phase 2)
  3. Server registers the connection in AlertPublisher
  4. Server sends CONNECTED_ACK
  5. Server pushes RiskAlertPayload messages as threats are detected
  6. Client sends PING; server responds with PONG (application heartbeat)
  7. Client disconnects → server unregisters

HTTP Endpoints:
  GET /alerts/history      — paginated alert history from PostgreSQL
  GET /alerts/{alert_uid}  — single alert detail with full analysis metadata
  GET /alerts/stats        — aggregated stats for the authenticated org
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketState

from sentinel_ai.database.models.threat_alert import AlertSeverity, AlertStatus, ThreatAlert
from sentinel_ai.database.session import get_db
from sentinel_ai.services.ingestion.core.auth import WebSocketAuthError, WebSocketAuthenticator
from sentinel_ai.services.risk_fusion.core.alert_publisher import AlertPublisher
from sentinel_ai.services.risk_fusion.schemas.fusion import RiskLevel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["Risk Alert WebSocket"])
http_router = APIRouter(prefix="/alerts", tags=["Threat Alerts"])


# ---------------------------------------------------------------------------
# Dependency Injectors
# ---------------------------------------------------------------------------

def _get_alert_publisher(request: Request) -> AlertPublisher:
    return request.app.state.alert_publisher


def _get_authenticator(request: Request) -> WebSocketAuthenticator:
    return request.app.state.ws_authenticator


# ---------------------------------------------------------------------------
# WebSocket Alert Subscription Endpoint
# ---------------------------------------------------------------------------

@router.websocket("/alerts")
async def alert_subscription_endpoint(
    websocket:     WebSocket,
    alert_publisher: AlertPublisher     = Depends(_get_alert_publisher),
    authenticator:   WebSocketAuthenticator = Depends(_get_authenticator),
) -> None:
    """
    Real-time risk alert subscription WebSocket.

    Connect with: wss://api.sentinelai.io/ws/alerts?token=<JWT>

    The client receives RiskAlertPayload JSON objects whenever the Risk
    Fusion Engine detects a threat above the configured alert threshold.
    """
    await websocket.accept()

    # --- JWT Authentication ---
    try:
        claims = await authenticator.authenticate(websocket)
    except WebSocketAuthError as exc:
        await websocket.send_text(
            f'{{"type":"error","code":"AUTH_FAILED","message":"{exc}"}}'
        )
        await websocket.close(code=1008, reason="Authentication failed")
        return

    # --- Register with alert publisher ---
    connection_id = await alert_publisher.register(
        user_id   = claims.sub,
        org_id    = claims.org,
        websocket = websocket,
    )

    # --- Send CONNECTED_ACK ---
    await websocket.send_text(
        f'{{"type":"connected_ack",'
        f'"connection_id":"{connection_id}",'
        f'"user_id":"{claims.sub}",'
        f'"org_id":"{claims.org}",'
        f'"server_timestamp_ms":{int(time.time() * 1000)}}}'
    )

    logger.info(
        "Alert WebSocket client connected",
        extra={
            "connection_id": connection_id,
            "user_id": claims.sub,
            "org_id": claims.org,
        },
    )

    # --- Message loop (PING/PONG + clean disconnect detection) ---
    try:
        while websocket.client_state == WebSocketState.CONNECTED:
            try:
                message = await websocket.receive_text()

                if '"type":"ping"' in message:
                    await websocket.send_text(
                        f'{{"type":"pong","server_timestamp_ms":{int(time.time() * 1000)}}}'
                    )

            except WebSocketDisconnect:
                break
            except Exception:
                break

    finally:
        await alert_publisher.unregister(connection_id)
        logger.info(
            "Alert WebSocket client disconnected",
            extra={"connection_id": connection_id, "user_id": claims.sub},
        )


# ---------------------------------------------------------------------------
# HTTP: Alert History
# ---------------------------------------------------------------------------

@http_router.get("/history")
async def get_alert_history(
    request:   Request,
    page:      int          = Query(1, ge=1),
    page_size: int          = Query(20, ge=1, le=100),
    severity:  Optional[AlertSeverity] = Query(None),
    status:    Optional[AlertStatus]   = Query(None),
    db:        AsyncSession = Depends(get_db),
) -> dict:
    """
    Returns paginated threat alert history for the authenticated organisation.
    Ordered by created_at descending (most recent first).
    """
    # Extract org from JWT (resolved from request state in production via middleware)
    # Placeholder — in production, resolve from JWT middleware on request.state.claims
    org_id: Optional[str] = request.headers.get("X-Org-Id")

    query = select(ThreatAlert).order_by(desc(ThreatAlert.created_at))

    if org_id:
        from sqlalchemy import cast
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID
        query = query.where(
            ThreatAlert.organization_id == cast(org_id, PG_UUID)
        )
    if severity:
        query = query.where(ThreatAlert.severity == severity)
    if status:
        query = query.where(ThreatAlert.status == status)

    offset = (page - 1) * page_size
    query  = query.offset(offset).limit(page_size)

    result = await db.execute(query)
    alerts = result.scalars().all()

    return {
        "page":       page,
        "page_size":  page_size,
        "items":      [a.to_dict() for a in alerts],
        "count":      len(alerts),
    }


@http_router.get("/{alert_uid}")
async def get_alert_detail(
    alert_uid: str,
    db:        AsyncSession = Depends(get_db),
) -> dict:
    """Returns the full detail for a single threat alert by its UID (e.g. SA-2024-000042)."""
    result = await db.execute(
        select(ThreatAlert).where(ThreatAlert.alert_uid == alert_uid)
    )
    alert = result.scalar_one_or_none()
    if alert is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Alert '{alert_uid}' not found")

    detail = alert.to_dict()
    # Include full analysis_metadata (stripped by to_dict only for encrypted blobs)
    detail["analysis_metadata"] = alert.analysis_metadata
    return detail


@http_router.get("/stats/summary")
async def get_alert_stats(
    request: Request,
    db:      AsyncSession = Depends(get_db),
) -> dict:
    """Returns alert counts aggregated by severity and status for the org."""
    from sqlalchemy import func

    org_id = request.headers.get("X-Org-Id")

    # Count by severity
    sev_query = select(
        ThreatAlert.severity,
        func.count(ThreatAlert.id).label("count"),
    ).group_by(ThreatAlert.severity)

    if org_id:
        from sqlalchemy import cast
        from sqlalchemy.dialects.postgresql import UUID as PG_UUID
        sev_query = sev_query.where(
            ThreatAlert.organization_id == cast(org_id, PG_UUID)
        )

    sev_result = await db.execute(sev_query)

    # Count by status
    status_query = select(
        ThreatAlert.status,
        func.count(ThreatAlert.id).label("count"),
    ).group_by(ThreatAlert.status)

    if org_id:
        status_query = status_query.where(
            ThreatAlert.organization_id == cast(org_id, PG_UUID)
        )

    status_result = await db.execute(status_query)

    return {
        "by_severity": {
            row.severity.value: row.count for row in sev_result
        },
        "by_status": {
            row.status.value: row.count for row in status_result
        },
    }
