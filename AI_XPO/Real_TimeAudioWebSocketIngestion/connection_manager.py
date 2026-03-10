"""
SentinelAI - WebSocket Connection Manager
==========================================
Manages the full lifecycle of active WebSocket connections to the audio
ingestion endpoint. Designed as a singleton shared across all WebSocket sessions.

Responsibilities:
1. Tracks all active WebSocket connections by session_id
2. Provides structured JSON send helpers (ServerMessage wrapping)
3. Enforces per-user and global concurrent connection limits
4. Drives per-connection heartbeat / ping-pong health checks
5. Handles graceful disconnect with proper WebSocket close codes
6. Provides an iterator over all active connections for broadcast/monitoring

Connection lifecycle:
  CONNECTING → ACTIVE → [DRAINING] → CLOSED

Rate limits enforced:
  - MAX_CONNECTIONS_PER_USER: Max simultaneous WebSocket connections per user_id.
    Mobile apps typically hold 1; burst during app restart may briefly hit 2.
  - MAX_GLOBAL_CONNECTIONS: Hard ceiling across all tenants (K8s resource guard).

WebSocket close codes used:
  1000 — Normal closure (stream complete, session_abort)
  1008 — Policy violation (auth failure, rate limit, size exceeded)
  1011 — Internal error (Kafka failure, buffer error)
  4000 — Application-level: session expired
  4001 — Application-level: duplicate connection
  4002 — Application-level: circuit open
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from sentinel_ai.services.ingestion.schemas.audio import (
    JWTClaims,
    ServerMessage,
    ServerMessageType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_CONNECTIONS_PER_USER: int = 3
MAX_GLOBAL_CONNECTIONS: int = 10_000
HEARTBEAT_INTERVAL_S: float = 30.0
HEARTBEAT_TIMEOUT_S: float = 10.0


# ---------------------------------------------------------------------------
# WebSocket Close Codes
# ---------------------------------------------------------------------------
class WSCloseCode(int, Enum):
    NORMAL = 1000
    POLICY_VIOLATION = 1008
    INTERNAL_ERROR = 1011
    SESSION_EXPIRED = 4000
    DUPLICATE_CONNECTION = 4001
    CIRCUIT_OPEN = 4002


# ---------------------------------------------------------------------------
# Connection Record
# ---------------------------------------------------------------------------

@dataclass
class WebSocketConnection:
    """Tracks the state of a single active WebSocket session."""

    session_id: str
    websocket: WebSocket
    claims: JWTClaims
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_ping_at: float = field(default_factory=time.monotonic)
    last_pong_at: float = field(default_factory=time.monotonic)
    chunks_received: int = 0
    bytes_received: int = 0
    is_draining: bool = False     # True when graceful shutdown has been initiated

    def touch(self) -> None:
        """Updates last_message_at to now. Called on every received frame."""
        self.last_message_at = datetime.now(timezone.utc)

    @property
    def user_id(self) -> str:
        return self.claims.sub

    @property
    def organization_id(self) -> str:
        return self.claims.org

    @property
    def is_alive(self) -> bool:
        return self.websocket.client_state == WebSocketState.CONNECTED


class ConnectionLimitError(Exception):
    """Raised when a new connection would violate a rate/limit policy."""


class ConnectionManager:
    """
    Singleton WebSocket connection manager for the audio ingestion service.

    Thread safety:
        All state mutations use asyncio.Lock. This is an asyncio-first class —
        never call mutating methods from a thread without scheduling them onto
        the event loop.
    """

    def __init__(self) -> None:
        # session_id → WebSocketConnection
        self._connections: dict[str, WebSocketConnection] = {}
        # user_id → set of session_ids
        self._user_sessions: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        # Background heartbeat task (started in start())
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Starts the background heartbeat loop."""
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name="ws-heartbeat",
        )
        logger.info("ConnectionManager started — heartbeat loop active")

    async def stop(self) -> None:
        """Cancels heartbeat loop and forcibly closes all active connections."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all remaining connections
        async with self._lock:
            session_ids = list(self._connections.keys())

        for session_id in session_ids:
            await self._force_close(
                session_id,
                WSCloseCode.NORMAL,
                "Server shutting down",
            )

        logger.info("ConnectionManager stopped — all connections closed")

    # ------------------------------------------------------------------
    # Connection Registration
    # ------------------------------------------------------------------

    async def register(
        self,
        session_id: str,
        websocket: WebSocket,
        claims: JWTClaims,
    ) -> WebSocketConnection:
        """
        Registers a new authenticated WebSocket connection.

        Enforces per-user and global connection limits before accepting.
        The WebSocket must already be accepted (websocket.accept() called) before
        calling register().

        Args:
            session_id: Server-allocated UUID string for this session.
            websocket:  Accepted FastAPI WebSocket.
            claims:     Validated JWT claims for the connection owner.

        Returns:
            WebSocketConnection record for the new connection.

        Raises:
            ConnectionLimitError: If per-user or global limit would be exceeded.
        """
        async with self._lock:
            # Global limit
            if len(self._connections) >= MAX_GLOBAL_CONNECTIONS:
                raise ConnectionLimitError(
                    f"Global WebSocket connection limit ({MAX_GLOBAL_CONNECTIONS}) reached. "
                    "Try again later."
                )

            # Per-user limit
            user_sessions = self._user_sessions.get(claims.sub, set())
            if len(user_sessions) >= MAX_CONNECTIONS_PER_USER:
                raise ConnectionLimitError(
                    f"User '{claims.sub}' already has {MAX_CONNECTIONS_PER_USER} active "
                    "WebSocket connections. Close an existing session before opening a new one."
                )

            conn = WebSocketConnection(
                session_id=session_id,
                websocket=websocket,
                claims=claims,
            )
            self._connections[session_id] = conn

            if claims.sub not in self._user_sessions:
                self._user_sessions[claims.sub] = set()
            self._user_sessions[claims.sub].add(session_id)

        logger.info(
            "WebSocket connection registered",
            extra={
                "session_id": session_id,
                "user_id": claims.sub,
                "org_id": claims.org,
                "total_connections": len(self._connections),
            },
        )
        return conn

    async def unregister(self, session_id: str) -> None:
        """
        Removes a connection from tracking.
        Safe to call multiple times (idempotent).
        """
        async with self._lock:
            conn = self._connections.pop(session_id, None)
            if conn is None:
                return

            user_sessions = self._user_sessions.get(conn.user_id, set())
            user_sessions.discard(session_id)
            if not user_sessions:
                self._user_sessions.pop(conn.user_id, None)

        logger.info(
            "WebSocket connection unregistered",
            extra={
                "session_id": session_id,
                "total_connections": len(self._connections),
            },
        )

    def get_connection(self, session_id: str) -> Optional[WebSocketConnection]:
        """Returns the connection record for a session_id, or None."""
        return self._connections.get(session_id)

    # ------------------------------------------------------------------
    # Structured Message Senders
    # ------------------------------------------------------------------

    async def send_json(
        self,
        session_id: str,
        msg_type: ServerMessageType,
        payload: Any,
    ) -> bool:
        """
        Sends a structured JSON ServerMessage to a specific session.

        Returns True if sent successfully, False if the connection was gone.
        Never raises — caller should check the return value.
        """
        conn = self._connections.get(session_id)
        if conn is None or not conn.is_alive:
            return False

        message = ServerMessage(type=msg_type, payload=payload)
        try:
            await conn.websocket.send_text(message.model_dump_json())
            return True
        except (WebSocketDisconnect, RuntimeError):
            logger.debug(
                "Failed to send — connection already closed",
                extra={"session_id": session_id, "msg_type": msg_type},
            )
            return False
        except Exception:
            logger.error(
                "Unexpected error sending WebSocket message",
                extra={"session_id": session_id},
                exc_info=True,
            )
            return False

    async def send_error_and_close(
        self,
        session_id: str,
        payload: Any,
        close_code: WSCloseCode = WSCloseCode.POLICY_VIOLATION,
        close_reason: str = "",
    ) -> None:
        """
        Sends an ERROR message then closes the WebSocket with the given code.
        Always unregisters the connection after closing.
        """
        await self.send_json(session_id, ServerMessageType.ERROR, payload)
        await self._force_close(session_id, close_code, close_reason)

    async def _force_close(
        self,
        session_id: str,
        code: WSCloseCode,
        reason: str,
    ) -> None:
        """Closes the WebSocket and removes it from tracking."""
        conn = self._connections.get(session_id)
        if conn is None:
            return

        if conn.is_alive:
            try:
                await conn.websocket.close(code=int(code), reason=reason[:123])
            except Exception:
                pass  # Connection may already be closed

        await self.unregister(session_id)

    # ------------------------------------------------------------------
    # Heartbeat Loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """
        Background task: sends periodic application-level PINGs to all connections.
        Connections that do not respond within HEARTBEAT_TIMEOUT_S are closed.

        Note: This is an application-level heartbeat, separate from the WebSocket
        protocol-level ping/pong. Both are used: the WS protocol ping detects TCP
        connection drops; the application ping detects hung clients.
        """
        logger.info("WebSocket heartbeat loop started")
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)

                now = time.monotonic()
                sessions_to_close: list[str] = []

                async with self._lock:
                    snapshot = dict(self._connections)

                for session_id, conn in snapshot.items():
                    # Check pong timeout from previous ping
                    if (
                        conn.last_ping_at > conn.last_pong_at
                        and (now - conn.last_ping_at) > HEARTBEAT_TIMEOUT_S
                    ):
                        logger.warning(
                            "WebSocket connection timed out — no pong received",
                            extra={
                                "session_id": session_id,
                                "elapsed_s": now - conn.last_ping_at,
                            },
                        )
                        sessions_to_close.append(session_id)
                        continue

                    # Send application-level PONG challenge
                    sent = await self.send_json(
                        session_id,
                        ServerMessageType.PONG,
                        {
                            "server_timestamp_ms": int(time.time() * 1000),
                            "session_id": session_id,
                        },
                    )
                    if sent:
                        conn.last_ping_at = now
                    else:
                        sessions_to_close.append(session_id)

                for session_id in sessions_to_close:
                    logger.info(
                        "Closing timed-out WebSocket connection",
                        extra={"session_id": session_id},
                    )
                    await self._force_close(
                        session_id,
                        WSCloseCode.POLICY_VIOLATION,
                        "Heartbeat timeout",
                    )

            except asyncio.CancelledError:
                logger.info("WebSocket heartbeat loop cancelled")
                return
            except Exception:
                logger.error("Unexpected error in heartbeat loop", exc_info=True)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def connection_count(self) -> int:
        return len(self._connections)

    def user_connection_count(self, user_id: str) -> int:
        return len(self._user_sessions.get(user_id, set()))

    def stats(self) -> dict[str, Any]:
        return {
            "total_connections": len(self._connections),
            "unique_users": len(self._user_sessions),
            "max_connections_per_user": MAX_CONNECTIONS_PER_USER,
            "max_global_connections": MAX_GLOBAL_CONNECTIONS,
        }
