"""
SentinelAI - Audio Ingestion WebSocket Router
===============================================
Implements the primary WebSocket endpoint for real-time audio streaming.

Endpoint:
    GET /ws/audio/stream
    Upgrade: websocket
    Query: ?token=<JWT>

Session flow per connection:
    1.  IP rate limit check (pre-accept)
    2.  WebSocket accept (upgrade completes)
    3.  JWT authentication
    4.  User rate limit check
    5.  Connection registration with ConnectionManager
    6.  Session allocation (AudioBufferManager + Redis state)
    7.  SESSION_READY message sent to client
    8.  Audio chunk receive loop:
        a.  Parse and validate AudioChunkMessage or ControlMessage
        b.  Per-session chunk rate limit check
        c.  append_chunk() → AudioBufferManager
        d.  CHUNK_ACK sent to client
        e.  Loop until STREAM_END / is_final=True / disconnect
    9.  flush_session() → assembles complete audio buffer
    10. publish_audio_session() → Kafka
    11. PIPELINE_QUEUED message sent to client
    12. SESSION_COMPLETE message sent to client
    13. WebSocket closed normally (1000)
    14. Connection unregistered

Error handling:
    All errors within the chunk loop are caught and converted to ErrorCode-tagged
    ServerErrorPayload messages. The connection is then closed with the appropriate
    WebSocket close code. The AudioBufferManager aborts the session to clean up
    Redis/memory buffers on any error path.

Important: The WebSocket endpoint must be on an isolated router
    with no HTTP authentication middleware applied — auth is handled inline
    at step 3. HTTP middleware that tries to read Authorization headers
    will hang on WebSocket upgrade requests.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sentinel_ai.services.ingestion.core.audio_buffer import (
    AudioBufferManager,
    AudioBufferError,
    ChunkIntegrityError,
    ChunkOutOfOrderError,
    SessionAudioLimitExceededError,
)
from sentinel_ai.services.ingestion.core.auth import WebSocketAuthenticator, WebSocketAuthError
from sentinel_ai.services.ingestion.core.connection_manager import (
    ConnectionLimitError,
    ConnectionManager,
    WSCloseCode,
)
from sentinel_ai.services.ingestion.core.kafka_producer import (
    AudioEventKafkaProducer,
    KafkaCircuitOpenError,
    KafkaProducerError,
)
from sentinel_ai.services.ingestion.middleware.rate_limiter import (
    RateLimitExceededError,
    WebSocketRateLimiter,
)
from sentinel_ai.services.ingestion.schemas.audio import (
    AudioChunkMessage,
    ChunkAckPayload,
    ClientMessageType,
    ControlMessage,
    ErrorCode,
    PipelineQueuedPayload,
    ServerErrorPayload,
    ServerMessageType,
    SessionReadyPayload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["Audio Ingestion WebSocket"])

# ---------------------------------------------------------------------------
# Request-scoped helpers (resolved from app.state by dependency functions)
# ---------------------------------------------------------------------------

def _get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


def _get_buffer_manager(request: Request) -> AudioBufferManager:
    return request.app.state.audio_buffer_manager


def _get_kafka_producer(request: Request) -> AudioEventKafkaProducer:
    return request.app.state.kafka_producer


def _get_authenticator(request: Request) -> WebSocketAuthenticator:
    return request.app.state.ws_authenticator


def _get_rate_limiter(request: Request) -> WebSocketRateLimiter:
    return request.app.state.rate_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_client_ip(websocket: WebSocket) -> str:
    """
    Resolves the real client IP from X-Forwarded-For (load balancer / Nginx)
    or falls back to websocket.client.host.
    The first entry in X-Forwarded-For is the original client IP (trust the proxy).
    """
    forwarded = websocket.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if websocket.client:
        return websocket.client.host
    return "unknown"


async def _parse_client_frame(raw: str) -> Optional[AudioChunkMessage | ControlMessage]:
    """
    Parses an incoming WebSocket text frame.
    Returns the typed message object or None on parse failure (caller sends error).

    Tries AudioChunkMessage first (most common); falls back to ControlMessage.
    Returns None if the JSON is malformed or doesn't match either schema.
    """
    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Received non-JSON WebSocket frame")
        return None

    msg_type: str = data.get("type", "")

    if msg_type == ClientMessageType.AUDIO_CHUNK:
        try:
            return AudioChunkMessage.model_validate(data)
        except ValidationError as exc:
            logger.warning("AudioChunkMessage validation failed", extra={"errors": exc.errors()})
            return None

    if msg_type in (
        ClientMessageType.STREAM_END,
        ClientMessageType.PING,
        ClientMessageType.SESSION_ABORT,
    ):
        try:
            return ControlMessage.model_validate(data)
        except ValidationError as exc:
            logger.warning("ControlMessage validation failed", extra={"errors": exc.errors()})
            return None

    logger.warning("Unknown client message type received", extra={"type": msg_type})
    return None


# ---------------------------------------------------------------------------
# Main WebSocket Endpoint
# ---------------------------------------------------------------------------

@router.websocket("/audio/stream")
async def audio_stream_endpoint(
    websocket: WebSocket,
    connection_manager: ConnectionManager = Depends(_get_connection_manager),
    buffer_manager: AudioBufferManager = Depends(_get_buffer_manager),
    kafka_producer: AudioEventKafkaProducer = Depends(_get_kafka_producer),
    authenticator: WebSocketAuthenticator = Depends(_get_authenticator),
    rate_limiter: WebSocketRateLimiter = Depends(_get_rate_limiter),
) -> None:
    """
    Real-time audio streaming WebSocket endpoint.

    Clients connect with a valid JWT as a query parameter:
        wss://api.sentinelai.io/ws/audio/stream?token=<JWT>

    The server buffers incoming audio chunks and publishes the assembled
    session to Kafka when the client signals end-of-stream.
    """
    session_id: str = str(uuid.uuid4())
    client_ip: str = _extract_client_ip(websocket)
    request_id: str = str(uuid.uuid4())

    logger.info(
        "WebSocket connection attempt",
        extra={
            "session_id": session_id,
            "client_ip": client_ip,
            "request_id": request_id,
        },
    )

    # ------------------------------------------------------------------
    # Step 1: IP-level rate limit (BEFORE accepting the upgrade)
    # ------------------------------------------------------------------
    try:
        await rate_limiter.check_ip_connection_limit(client_ip)
    except RateLimitExceededError as exc:
        # Reject the upgrade with 429 — we must accept() first in FastAPI
        await websocket.accept()
        await connection_manager.send_json(
            session_id,
            ServerMessageType.RATE_LIMITED,
            ServerErrorPayload(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message=str(exc),
                retry_after_ms=exc.retry_after_ms,
            ).model_dump(),
        )
        await websocket.close(code=int(WSCloseCode.POLICY_VIOLATION), reason="Rate limited")
        logger.warning(
            "WebSocket upgrade rejected — IP rate limit exceeded",
            extra={"client_ip": client_ip, "retry_after_ms": exc.retry_after_ms},
        )
        return

    # ------------------------------------------------------------------
    # Step 2: Accept WebSocket upgrade
    # ------------------------------------------------------------------
    await websocket.accept()

    # ------------------------------------------------------------------
    # Step 3: JWT Authentication
    # ------------------------------------------------------------------
    try:
        claims = await authenticator.authenticate(websocket)
    except WebSocketAuthError as exc:
        error_code = (
            ErrorCode.AUTH_EXPIRED if exc.is_expired else ErrorCode.AUTH_INVALID
        )
        await websocket.send_text(
            ServerErrorPayload(
                error_code=error_code,
                message=str(exc),
            ).model_dump_json()
        )
        await websocket.close(code=int(WSCloseCode.POLICY_VIOLATION), reason="Authentication failed")
        logger.warning(
            "WebSocket authentication failed",
            extra={"client_ip": client_ip, "error": str(exc)},
        )
        return

    # ------------------------------------------------------------------
    # Step 4: User-level rate limit (post-auth)
    # ------------------------------------------------------------------
    try:
        await rate_limiter.check_user_connection_limit(claims.sub)
    except RateLimitExceededError as exc:
        await websocket.send_text(
            ServerErrorPayload(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message=str(exc),
                retry_after_ms=exc.retry_after_ms,
            ).model_dump_json()
        )
        await websocket.close(code=int(WSCloseCode.POLICY_VIOLATION), reason="Rate limited")
        return

    # ------------------------------------------------------------------
    # Step 5: Register connection
    # ------------------------------------------------------------------
    try:
        conn = await connection_manager.register(session_id, websocket, claims)
    except ConnectionLimitError as exc:
        await websocket.send_text(
            ServerErrorPayload(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message=str(exc),
            ).model_dump_json()
        )
        await websocket.close(
            code=int(WSCloseCode.DUPLICATE_CONNECTION),
            reason="Connection limit reached",
        )
        return

    # ------------------------------------------------------------------
    # Step 6: Initialize audio buffer session
    # ------------------------------------------------------------------
    from sentinel_ai.config.settings import get_settings
    cfg = get_settings()

    await buffer_manager.create_session(
        session_id=session_id,
        user_id=claims.sub,
        organization_id=claims.org,
        jti=claims.jti,
        client_metadata={
            "client_ip": client_ip,
            "user_agent": websocket.headers.get("user-agent", ""),
        },
    )

    # ------------------------------------------------------------------
    # Step 7: Send SESSION_READY
    # ------------------------------------------------------------------
    kafka_topic = getattr(cfg, "KAFKA_AUDIO_TOPIC", "sentinelai.audio.raw")
    await connection_manager.send_json(
        session_id,
        ServerMessageType.SESSION_READY,
        SessionReadyPayload(
            session_id=uuid.UUID(session_id),
            user_id=claims.sub,
            organization_id=claims.org,
            max_chunk_bytes=65_536,
            max_session_bytes=50 * 1024 * 1024,
            max_session_duration_seconds=600,
            server_timestamp_ms=int(time.time() * 1000),
            kafka_topic=kafka_topic,
        ).model_dump(),
    )

    logger.info(
        "Audio ingestion session opened",
        extra={
            "session_id": session_id,
            "user_id": claims.sub,
            "org_id": claims.org,
            "jti": claims.jti,
        },
    )

    # ------------------------------------------------------------------
    # Step 8: Audio chunk receive loop
    # ------------------------------------------------------------------
    stream_complete = False
    try:
        async for raw_frame in _receive_frames(websocket):
            conn.touch()

            parsed = await _parse_client_frame(raw_frame)
            if parsed is None:
                await connection_manager.send_json(
                    session_id,
                    ServerMessageType.ERROR,
                    ServerErrorPayload(
                        session_id=uuid.UUID(session_id),
                        error_code=ErrorCode.INTERNAL_ERROR,
                        message="Malformed or unrecognized message type. Frame ignored.",
                    ).model_dump(),
                )
                continue

            # ---- Control messages ----
            if isinstance(parsed, ControlMessage):
                if parsed.type == ClientMessageType.PING:
                    await connection_manager.send_json(
                        session_id,
                        ServerMessageType.PONG,
                        {"server_timestamp_ms": int(time.time() * 1000)},
                    )
                    continue

                if parsed.type == ClientMessageType.SESSION_ABORT:
                    logger.info(
                        "Client requested session abort",
                        extra={
                            "session_id": session_id,
                            "reason": parsed.reason,
                        },
                    )
                    await buffer_manager.abort_session(
                        session_id, reason=parsed.reason or "client_abort"
                    )
                    await websocket.close(code=int(WSCloseCode.NORMAL), reason="Session aborted")
                    return

                if parsed.type == ClientMessageType.STREAM_END:
                    stream_complete = True
                    break

            # ---- Audio chunk messages ----
            elif isinstance(parsed, AudioChunkMessage):
                # Chunk rate limit
                try:
                    await rate_limiter.check_chunk_limit(session_id)
                except RateLimitExceededError as exc:
                    await connection_manager.send_error_and_close(
                        session_id,
                        ServerErrorPayload(
                            session_id=uuid.UUID(session_id),
                            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                            message=str(exc),
                            retry_after_ms=exc.retry_after_ms,
                        ).model_dump(),
                        close_code=WSCloseCode.POLICY_VIOLATION,
                        close_reason="Chunk rate limit exceeded",
                    )
                    return

                # Validate session_id matches authenticated session
                if str(parsed.session_id) != session_id:
                    await connection_manager.send_error_and_close(
                        session_id,
                        ServerErrorPayload(
                            session_id=uuid.UUID(session_id),
                            error_code=ErrorCode.SESSION_NOT_FOUND,
                            message=(
                                f"chunk session_id '{parsed.session_id}' does not match "
                                f"authenticated session '{session_id}'"
                            ),
                        ).model_dump(),
                        close_code=WSCloseCode.POLICY_VIOLATION,
                        close_reason="Session ID mismatch",
                    )
                    return

                # Append to buffer
                try:
                    bytes_appended, state = await buffer_manager.append_chunk(parsed)
                    conn.chunks_received += 1
                    conn.bytes_received += bytes_appended
                except ChunkOutOfOrderError as exc:
                    await connection_manager.send_error_and_close(
                        session_id,
                        ServerErrorPayload(
                            session_id=uuid.UUID(session_id),
                            error_code=ErrorCode.CHUNK_OUT_OF_ORDER,
                            message=str(exc),
                        ).model_dump(),
                        close_code=WSCloseCode.POLICY_VIOLATION,
                        close_reason="Out-of-order chunk",
                    )
                    return
                except ChunkIntegrityError as exc:
                    # Non-fatal — client must retransmit the chunk
                    await connection_manager.send_json(
                        session_id,
                        ServerMessageType.ERROR,
                        ServerErrorPayload(
                            session_id=uuid.UUID(session_id),
                            error_code=ErrorCode.INTERNAL_ERROR,
                            message=f"CRC mismatch: {exc}. Please retransmit chunk {parsed.chunk_index}.",
                        ).model_dump(),
                    )
                    continue
                except SessionAudioLimitExceededError as exc:
                    await connection_manager.send_error_and_close(
                        session_id,
                        ServerErrorPayload(
                            session_id=uuid.UUID(session_id),
                            error_code=ErrorCode.SESSION_AUDIO_LIMIT_EXCEEDED,
                            message=str(exc),
                        ).model_dump(),
                        close_code=WSCloseCode.POLICY_VIOLATION,
                        close_reason="Session audio size limit exceeded",
                    )
                    return

                # Send CHUNK_ACK
                await connection_manager.send_json(
                    session_id,
                    ServerMessageType.CHUNK_ACK,
                    ChunkAckPayload(
                        session_id=uuid.UUID(session_id),
                        chunk_index=parsed.chunk_index,
                        bytes_received=bytes_appended,
                        total_bytes_buffered=state.total_bytes,
                        server_timestamp_ms=int(time.time() * 1000),
                    ).model_dump(),
                )

                # Support inline is_final signal (no separate STREAM_END required)
                if parsed.is_final:
                    stream_complete = True
                    break

    except WebSocketDisconnect as exc:
        logger.info(
            "WebSocket client disconnected during stream",
            extra={
                "session_id": session_id,
                "code": exc.code,
                "chunks_received": conn.chunks_received,
                "bytes_received": conn.bytes_received,
            },
        )
        await buffer_manager.abort_session(session_id, reason="client_disconnect")
        await connection_manager.unregister(session_id)
        return

    except Exception as exc:
        logger.error(
            "Unexpected error in WebSocket receive loop",
            extra={"session_id": session_id},
            exc_info=True,
        )
        await buffer_manager.abort_session(session_id, reason=f"internal_error: {type(exc).__name__}")
        await connection_manager.send_error_and_close(
            session_id,
            ServerErrorPayload(
                session_id=uuid.UUID(session_id),
                error_code=ErrorCode.INTERNAL_ERROR,
                message="An internal error occurred during audio ingestion.",
            ).model_dump(),
            close_code=WSCloseCode.INTERNAL_ERROR,
            close_reason="Internal server error",
        )
        return

    # ------------------------------------------------------------------
    # Steps 9–12: Flush, publish to Kafka, notify client
    # ------------------------------------------------------------------
    if not stream_complete:
        # Stream was not properly terminated — clean up silently
        await buffer_manager.abort_session(session_id, reason="stream_incomplete")
        await connection_manager.unregister(session_id)
        return

    try:
        audio_bytes, final_state = await buffer_manager.flush_session(session_id)

        if not audio_bytes:
            logger.warning(
                "Empty audio buffer after flush — nothing to publish",
                extra={"session_id": session_id},
            )
            await connection_manager.send_error_and_close(
                session_id,
                ServerErrorPayload(
                    session_id=uuid.UUID(session_id),
                    error_code=ErrorCode.INTERNAL_ERROR,
                    message="Audio session contains no data.",
                ).model_dump(),
                close_code=WSCloseCode.INTERNAL_ERROR,
            )
            return

        event = await kafka_producer.publish_audio_session(
            audio_bytes=audio_bytes,
            session_state=final_state,
        )

    except KafkaCircuitOpenError as exc:
        logger.error(
            "Kafka circuit breaker open — cannot publish audio session",
            extra={"session_id": session_id},
        )
        await connection_manager.send_error_and_close(
            session_id,
            ServerErrorPayload(
                session_id=uuid.UUID(session_id),
                error_code=ErrorCode.KAFKA_PUBLISH_FAILED,
                message="ML pipeline is temporarily unavailable. Your audio was not processed.",
                retry_after_ms=60_000,
            ).model_dump(),
            close_code=WSCloseCode.CIRCUIT_OPEN,
            close_reason="Pipeline unavailable",
        )
        return

    except KafkaProducerError as exc:
        logger.error(
            "Kafka publish failed for audio session",
            extra={"session_id": session_id},
            exc_info=True,
        )
        await connection_manager.send_error_and_close(
            session_id,
            ServerErrorPayload(
                session_id=uuid.UUID(session_id),
                error_code=ErrorCode.KAFKA_PUBLISH_FAILED,
                message="Failed to queue audio for processing. Please retry.",
            ).model_dump(),
            close_code=WSCloseCode.INTERNAL_ERROR,
            close_reason="Kafka publish failed",
        )
        return

    # Notify client: pipeline queued
    await connection_manager.send_json(
        session_id,
        ServerMessageType.PIPELINE_QUEUED,
        PipelineQueuedPayload(
            session_id=uuid.UUID(session_id),
            kafka_topic=getattr(cfg, "KAFKA_AUDIO_TOPIC", "sentinelai.audio.raw"),
            kafka_partition=0,
            kafka_offset=0,
            total_chunks=final_state.chunk_count,
            total_bytes=final_state.total_bytes,
            pipeline_job_id=event.event_id,
            server_timestamp_ms=int(time.time() * 1000),
        ).model_dump(),
    )

    # Final session complete acknowledgement
    await connection_manager.send_json(
        session_id,
        ServerMessageType.SESSION_COMPLETE,
        {
            "session_id": session_id,
            "pipeline_job_id": event.event_id,
            "server_timestamp_ms": int(time.time() * 1000),
        },
    )

    logger.info(
        "Audio ingestion session completed successfully",
        extra={
            "session_id": session_id,
            "user_id": claims.sub,
            "org_id": claims.org,
            "total_chunks": final_state.chunk_count,
            "total_bytes": final_state.total_bytes,
            "pipeline_job_id": event.event_id,
        },
    )

    # Clean close
    await websocket.close(code=int(WSCloseCode.NORMAL), reason="Session complete")
    await connection_manager.unregister(session_id)


# ---------------------------------------------------------------------------
# Async Frame Iterator
# ---------------------------------------------------------------------------

async def _receive_frames(websocket: WebSocket):
    """
    Async generator that yields raw text frames from the WebSocket.
    Handles binary frames (rejects with logged warning).
    Terminates cleanly on WebSocketDisconnect.
    """
    while websocket.client_state == WebSocketState.CONNECTED:
        try:
            message = await websocket.receive()
        except WebSocketDisconnect:
            return

        if message["type"] == "websocket.disconnect":
            return

        if "text" in message:
            yield message["text"]
        elif "bytes" in message:
            logger.warning(
                "Received binary WebSocket frame — expected JSON text. Frame discarded.",
                extra={"bytes": len(message["bytes"])},
            )
        else:
            logger.debug("Received unhandled WebSocket message type", extra={"type": message.get("type")})
