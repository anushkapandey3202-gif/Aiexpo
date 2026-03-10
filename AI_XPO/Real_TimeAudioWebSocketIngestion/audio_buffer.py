"""
SentinelAI - Audio Buffer Manager
===================================
Manages per-session audio chunk assembly with a two-tier buffering strategy.

Buffering strategy (auto-selected based on total session bytes):
  TIER 1 — In-memory (asyncio.Lock + bytearray):
      Used when total_bytes ≤ INLINE_THRESHOLD (default: 512 KB).
      Fastest path — no serialization, no network I/O to Redis.
      Suitable for short voice analysis windows (< 30s @ 16 kHz mono 16-bit).

  TIER 2 — Redis (APPEND to a single binary key):
      Used when total_bytes > INLINE_THRESHOLD.
      Key: `audio_buf:{session_id}` — SET on first chunk, APPEND on subsequent.
      TTL is reset on every APPEND to prevent orphan keys on client disconnect.
      Suitable for long calls (minutes) that exceed in-memory budget.

  Overflow guard:
      Sessions exceeding MAX_SESSION_AUDIO_BYTES (default: 50 MB) are hard-rejected
      to prevent memory exhaustion attacks. The connection is forcibly closed with
      SESSION_AUDIO_LIMIT_EXCEEDED error.

Ordering:
      Chunks are tracked by index. Out-of-order chunks are rejected (not buffered).
      The client MUST send chunks in monotonically increasing chunk_index order.
      Gap detection (missing chunk_index) triggers a CHUNK_OUT_OF_ORDER error and
      aborts the session — partial audio is not forwarded to the ML pipeline.

Integrity:
      If the client provides checksum_crc32 in AudioChunkMessage, each chunk is
      verified before appending to the buffer. A CRC mismatch triggers rejection
      of that chunk only (not the entire session) — the client must retransmit.
"""
from __future__ import annotations

import asyncio
import binascii
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from redis.asyncio import Redis

from sentinel_ai.services.ingestion.schemas.audio import (
    AudioChunkMessage,
    AudioFormat,
    AudioSessionState,
    AudioStreamStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier thresholds and limits
# ---------------------------------------------------------------------------

# Sessions ≤ 512 KB stay in memory
INLINE_THRESHOLD_BYTES: int = 512 * 1024

# Hard cap per session: 50 MB
MAX_SESSION_AUDIO_BYTES: int = 50 * 1024 * 1024

# Redis key TTL for audio buffers (seconds) — reset on every chunk append
REDIS_BUFFER_TTL_SECONDS: int = 300  # 5 minutes idle = abandon

# Redis key prefix
_AUDIO_BUF_PREFIX = "audio_buf:"
_SESSION_STATE_PREFIX = "audio_session:"

# Session state TTL in Redis (longer than buffer — for post-publish audit)
SESSION_STATE_TTL_SECONDS: int = 3600  # 1 hour


class AudioBufferError(Exception):
    """Base exception for buffer management errors."""


class ChunkOutOfOrderError(AudioBufferError):
    """Raised when a chunk_index arrives out of sequential order."""


class SessionAudioLimitExceededError(AudioBufferError):
    """Raised when total session audio exceeds MAX_SESSION_AUDIO_BYTES."""


class ChunkIntegrityError(AudioBufferError):
    """Raised when a chunk's CRC-32 checksum does not match its data."""


class AudioBufferManager:
    """
    Manages per-session audio chunk assembly for the WebSocket ingestion service.

    One instance is shared across all sessions (stateless except for in-memory buffers).
    In-memory buffers are stored in self._memory_buffers keyed by session_id.

    Thread safety:
        Each session's in-memory buffer is protected by a per-session asyncio.Lock.
        Redis operations are inherently async-safe via the asyncio Redis client.
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client
        # session_id → bytearray (Tier 1 only)
        self._memory_buffers: dict[str, bytearray] = {}
        # session_id → asyncio.Lock (per-session concurrency guard)
        self._buffer_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Session Lifecycle
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        organization_id: str,
        jti: str,
        client_metadata: Optional[dict] = None,
    ) -> AudioSessionState:
        """
        Initializes a new buffering session in Redis.
        Creates the per-session lock and memory buffer placeholder.

        Args:
            session_id:      Unique session UUID string.
            user_id:         Authenticated user UUID string.
            organization_id: Tenant UUID string.
            jti:             JWT ID for token correlation.
            client_metadata: Optional dict from the WebSocket handshake headers.

        Returns:
            Initialized AudioSessionState (also persisted to Redis).
        """
        state = AudioSessionState(
            session_id=session_id,
            user_id=user_id,
            organization_id=organization_id,
            jti=jti,
            status=AudioStreamStatus.ACTIVE,
            client_metadata=client_metadata or {},
        )

        # Initialize per-session lock and memory buffer
        self._buffer_locks[session_id] = asyncio.Lock()
        self._memory_buffers[session_id] = bytearray()

        await self._persist_session_state(state)

        logger.info(
            "Audio buffer session created",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "org": organization_id,
            },
        )
        return state

    async def get_session_state(self, session_id: str) -> Optional[AudioSessionState]:
        """
        Retrieves session state from Redis.
        Returns None if the session does not exist (expired or never created).
        """
        raw = await self._redis.get(f"{_SESSION_STATE_PREFIX}{session_id}")
        if raw is None:
            return None
        return AudioSessionState.model_validate_json(raw)

    async def _persist_session_state(self, state: AudioSessionState) -> None:
        """Serializes AudioSessionState to Redis with TTL."""
        await self._redis.setex(
            f"{_SESSION_STATE_PREFIX}{state.session_id}",
            SESSION_STATE_TTL_SECONDS,
            state.model_dump_json(),
        )

    # ------------------------------------------------------------------
    # Chunk Ingestion
    # ------------------------------------------------------------------

    async def append_chunk(
        self, chunk_msg: AudioChunkMessage
    ) -> Tuple[int, AudioSessionState]:
        """
        Validates and appends a single audio chunk to the session buffer.

        Validation pipeline (in order):
        1. Session existence check
        2. Sequential chunk_index enforcement
        3. Optional CRC-32 integrity verification
        4. Per-session size limit enforcement
        5. Tier selection and buffer append

        Args:
            chunk_msg: Validated AudioChunkMessage from the WebSocket frame.

        Returns:
            Tuple of (bytes_appended, updated_session_state).

        Raises:
            AudioBufferError:               Session not found.
            ChunkOutOfOrderError:           chunk_index is not expected_next_index.
            SessionAudioLimitExceededError: Buffer would exceed MAX_SESSION_AUDIO_BYTES.
            ChunkIntegrityError:            CRC-32 mismatch.
        """
        session_id = str(chunk_msg.session_id)

        async with self._get_session_lock(session_id):
            state = await self.get_session_state(session_id)
            if state is None:
                raise AudioBufferError(
                    f"Session '{session_id}' not found — may have expired or never been created"
                )

            # 1. Sequential ordering enforcement
            expected_index = state.last_chunk_index + 1
            if chunk_msg.chunk_index != expected_index:
                raise ChunkOutOfOrderError(
                    f"Expected chunk_index {expected_index}, "
                    f"received {chunk_msg.chunk_index} for session '{session_id}'"
                )

            # 2. Decode raw bytes and validate CRC-32 if provided
            raw_audio = chunk_msg.decode_audio_bytes()
            if chunk_msg.checksum_crc32 is not None:
                computed_crc = binascii.crc32(raw_audio) & 0xFFFFFFFF
                if computed_crc != chunk_msg.checksum_crc32:
                    raise ChunkIntegrityError(
                        f"CRC-32 mismatch on chunk {chunk_msg.chunk_index}: "
                        f"expected {chunk_msg.checksum_crc32:#010x}, "
                        f"computed {computed_crc:#010x}"
                    )

            # 3. Session audio size limit
            new_total = state.total_bytes + len(raw_audio)
            if new_total > MAX_SESSION_AUDIO_BYTES:
                raise SessionAudioLimitExceededError(
                    f"Session '{session_id}' would exceed maximum audio size "
                    f"({MAX_SESSION_AUDIO_BYTES // (1024*1024)} MB). "
                    f"Current: {state.total_bytes} bytes, "
                    f"chunk: {len(raw_audio)} bytes."
                )

            # 4. Latch audio format on first chunk
            if chunk_msg.chunk_index == 0 and chunk_msg.format is not None:
                state.audio_format = chunk_msg.format
            elif chunk_msg.chunk_index == 0 and chunk_msg.format is None:
                logger.warning(
                    "No audio format provided on first chunk — using default",
                    extra={"session_id": session_id},
                )

            # 5. Append to appropriate tier
            if new_total <= INLINE_THRESHOLD_BYTES:
                await self._append_memory(session_id, raw_audio)
            else:
                # Promote from memory to Redis if crossing the threshold
                if session_id in self._memory_buffers and self._memory_buffers[session_id]:
                    await self._promote_to_redis(session_id)
                await self._append_redis(session_id, raw_audio)

            # 6. Update session state
            state.last_chunk_index = chunk_msg.chunk_index
            state.chunk_count += 1
            state.total_bytes = new_total
            state.last_activity_at = datetime.now(timezone.utc)

            await self._persist_session_state(state)

            logger.debug(
                "Audio chunk appended",
                extra={
                    "session_id": session_id,
                    "chunk_index": chunk_msg.chunk_index,
                    "chunk_bytes": len(raw_audio),
                    "total_bytes": new_total,
                    "tier": "memory" if new_total <= INLINE_THRESHOLD_BYTES else "redis",
                },
            )
            return len(raw_audio), state

    # ------------------------------------------------------------------
    # Buffer Flush (called when stream is complete)
    # ------------------------------------------------------------------

    async def flush_session(
        self, session_id: str
    ) -> Tuple[bytes, AudioSessionState]:
        """
        Assembles and returns the complete audio buffer for a finished session.

        This method:
        1. Reads assembled audio from memory or Redis
        2. Updates session status to FLUSHING
        3. Cleans up the buffer (Redis key deleted, memory buffer freed)
        4. Does NOT delete session STATE (kept for audit trail)

        Args:
            session_id: Session UUID string to flush.

        Returns:
            Tuple of (assembled_audio_bytes, final_session_state).

        Raises:
            AudioBufferError: If session not found.
        """
        async with self._get_session_lock(session_id):
            state = await self.get_session_state(session_id)
            if state is None:
                raise AudioBufferError(f"Cannot flush: session '{session_id}' not found")

            state.status = AudioStreamStatus.FLUSHING
            state.completed_at = datetime.now(timezone.utc)
            await self._persist_session_state(state)

            # Read assembled bytes from the active tier
            if state.total_bytes <= INLINE_THRESHOLD_BYTES:
                audio_bytes = bytes(self._memory_buffers.get(session_id, bytearray()))
            else:
                raw = await self._redis.get(f"{_AUDIO_BUF_PREFIX}{session_id}")
                audio_bytes = bytes(raw) if raw else b""

            # Validate assembled length matches expected total
            if len(audio_bytes) != state.total_bytes:
                logger.error(
                    "Buffer size mismatch during flush",
                    extra={
                        "session_id": session_id,
                        "expected_bytes": state.total_bytes,
                        "actual_bytes": len(audio_bytes),
                    },
                )
                # Log and continue — partial data is better than no data for ML

            # Clean up buffers (not state — state is kept for audit)
            await self._cleanup_buffers(session_id)

            logger.info(
                "Audio session buffer flushed",
                extra={
                    "session_id": session_id,
                    "total_bytes": len(audio_bytes),
                    "chunk_count": state.chunk_count,
                },
            )
            return audio_bytes, state

    async def abort_session(self, session_id: str, reason: str) -> None:
        """
        Marks a session as aborted and cleans up all associated buffers.
        Called on client disconnect, validation failure, or size limit exceeded.
        """
        async with self._get_session_lock(session_id):
            state = await self.get_session_state(session_id)
            if state is not None:
                state.status = AudioStreamStatus.ABORTED
                state.error = reason
                state.completed_at = datetime.now(timezone.utc)
                await self._persist_session_state(state)

            await self._cleanup_buffers(session_id)

        logger.info(
            "Audio session aborted",
            extra={"session_id": session_id, "reason": reason},
        )

    # ------------------------------------------------------------------
    # Tier-specific I/O
    # ------------------------------------------------------------------

    async def _append_memory(self, session_id: str, data: bytes) -> None:
        """Appends audio bytes to the in-memory bytearray buffer."""
        if session_id not in self._memory_buffers:
            self._memory_buffers[session_id] = bytearray()
        self._memory_buffers[session_id].extend(data)

    async def _append_redis(self, session_id: str, data: bytes) -> None:
        """
        APPENDs audio bytes to the Redis buffer key.
        Uses a pipeline to atomically append and reset the TTL.
        """
        buf_key = f"{_AUDIO_BUF_PREFIX}{session_id}"
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.append(buf_key, data)
            pipe.expire(buf_key, REDIS_BUFFER_TTL_SECONDS)
            await pipe.execute()

    async def _promote_to_redis(self, session_id: str) -> None:
        """
        Promotes the in-memory buffer to Redis when the INLINE_THRESHOLD is crossed.
        The existing in-memory bytes are SET as the initial Redis value.
        """
        mem_data = bytes(self._memory_buffers.pop(session_id, bytearray()))
        if mem_data:
            buf_key = f"{_AUDIO_BUF_PREFIX}{session_id}"
            await self._redis.setex(buf_key, REDIS_BUFFER_TTL_SECONDS, mem_data)
            logger.debug(
                "Audio buffer promoted from memory to Redis",
                extra={"session_id": session_id, "promoted_bytes": len(mem_data)},
            )

    async def _cleanup_buffers(self, session_id: str) -> None:
        """Frees in-memory buffer and deletes the Redis buffer key."""
        self._memory_buffers.pop(session_id, None)
        self._buffer_locks.pop(session_id, None)
        await self._redis.delete(f"{_AUDIO_BUF_PREFIX}{session_id}")

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """
        Returns the asyncio.Lock for a session, creating it if absent.
        Creating a lock for an unknown session is valid — create_session
        may not have been called yet in a race condition (defensive pattern).
        """
        if session_id not in self._buffer_locks:
            self._buffer_locks[session_id] = asyncio.Lock()
        return self._buffer_locks[session_id]
