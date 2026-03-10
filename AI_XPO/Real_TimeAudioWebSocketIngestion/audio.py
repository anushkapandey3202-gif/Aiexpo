"""
SentinelAI - Audio Ingestion Schemas
======================================
All Pydantic v2 message contracts for the WebSocket audio ingestion protocol.

Protocol Overview:
  CLIENT → SERVER: AudioChunkMessage | ControlMessage
  SERVER → CLIENT: ServerAckMessage | ServerErrorMessage | ServerEventMessage

Session lifecycle:
  1. Client connects with ?token=<JWT> query param
  2. Server validates JWT, sends SESSION_READY
  3. Client streams AUDIO_CHUNK messages
  4. Server sends CHUNK_ACK per chunk, SESSION_COMPLETE on stream end
  5. Client sends STREAM_END control message to signal completion
  6. Server publishes assembled buffer to Kafka, sends SESSION_COMPLETE
  7. Server sends PIPELINE_QUEUED when Kafka publish is confirmed

Strict validation rules (enforced in AudioChunkMessage):
  - chunk_index must be monotonically non-decreasing (enforced by ConnectionManager)
  - chunk_data must be base64-encoded raw PCM or compressed audio bytes
  - session_id must match the JWT-authenticated session on the connection
  - Total session audio hard-capped at MAX_SESSION_AUDIO_BYTES
"""
from __future__ import annotations

import base64
import enum
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AudioEncoding(str, enum.Enum):
    """Wire-level audio encoding of each chunk payload."""

    PCM_S16LE = "pcm_s16le"   # Raw signed 16-bit little-endian PCM
    PCM_F32LE = "pcm_f32le"   # Raw 32-bit float PCM
    OPUS = "opus"             # Opus compressed (preferred for mobile)
    WEBM_OPUS = "webm_opus"   # WebM container with Opus (browser WebRTC)
    AAC = "aac"               # AAC compressed


class ClientMessageType(str, enum.Enum):
    """All message type tags the mobile client may send."""

    AUDIO_CHUNK = "audio_chunk"
    STREAM_END = "stream_end"
    PING = "ping"
    SESSION_ABORT = "session_abort"


class ServerMessageType(str, enum.Enum):
    """All message type tags the server may push to the client."""

    SESSION_READY = "session_ready"
    CHUNK_ACK = "chunk_ack"
    PIPELINE_QUEUED = "pipeline_queued"
    SESSION_COMPLETE = "session_complete"
    ERROR = "error"
    PONG = "pong"
    RATE_LIMITED = "rate_limited"


class AudioStreamStatus(str, enum.Enum):
    """Lifecycle state of an audio ingestion session."""

    ACTIVE = "active"
    BUFFERING = "buffering"
    FLUSHING = "flushing"
    PUBLISHED = "published"
    ABORTED = "aborted"
    EXPIRED = "expired"
    ERROR = "error"


class ErrorCode(str, enum.Enum):
    """Structured error codes for client-side error handling."""

    AUTH_INVALID = "AUTH_INVALID"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_INSUFFICIENT_SCOPE = "AUTH_INSUFFICIENT_SCOPE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    CHUNK_OUT_OF_ORDER = "CHUNK_OUT_OF_ORDER"
    CHUNK_TOO_LARGE = "CHUNK_TOO_LARGE"
    CHUNK_INVALID_ENCODING = "CHUNK_INVALID_ENCODING"
    SESSION_AUDIO_LIMIT_EXCEEDED = "SESSION_AUDIO_LIMIT_EXCEEDED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    KAFKA_PUBLISH_FAILED = "KAFKA_PUBLISH_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Audio Format Specification
# ---------------------------------------------------------------------------

class AudioFormat(BaseModel):
    """Audio stream format negotiated at session open time."""

    model_config = {"frozen": True}

    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    sample_rate_hz: int = Field(
        16000,
        ge=8000,
        le=48000,
        description="Sample rate in Hz. Whisper and ECAPA-TDNN require 16 kHz.",
    )
    channels: int = Field(1, ge=1, le=2, description="1=mono (required by ML models), 2=stereo")
    bit_depth: int = Field(16, ge=8, le=32, description="Bit depth for PCM encodings")
    chunk_duration_ms: int = Field(
        200,
        ge=20,
        le=2000,
        description="Expected audio duration per chunk in ms (informational, not enforced)",
    )

    @model_validator(mode="after")
    def _validate_pcm_bit_depth(self) -> "AudioFormat":
        if self.encoding in (AudioEncoding.OPUS, AudioEncoding.WEBM_OPUS, AudioEncoding.AAC):
            if self.bit_depth not in (0, 16, 32):
                pass  # bit_depth is ignored for compressed formats
        return self


# ---------------------------------------------------------------------------
# JWT Claims (extracted from validated token)
# ---------------------------------------------------------------------------

class JWTClaims(BaseModel):
    """
    Typed representation of claims extracted from a validated JWT.
    Sub (subject) is the platform user UUID.
    """

    model_config = {"frozen": True}

    sub: str = Field(..., description="User UUID (platform user ID)")
    org: str = Field(..., description="Organization UUID (tenant isolation key)")
    jti: str = Field(..., description="JWT ID — used as session correlation token")
    iss: str = Field(..., description="Issuer — must match settings.JWT_ISSUER")
    aud: str = Field(..., description="Audience — must match settings.JWT_AUDIENCE")
    exp: int = Field(..., description="Expiry Unix timestamp")
    iat: int = Field(..., description="Issued-at Unix timestamp")
    roles: list[str] = Field(default_factory=list, description="Granted role names")
    scopes: list[str] = Field(
        default_factory=list,
        description="Fine-grained scopes. 'audio:ingest' required for WebSocket ingestion.",
    )


# ---------------------------------------------------------------------------
# Client → Server Messages
# ---------------------------------------------------------------------------

class AudioChunkMessage(BaseModel):
    """
    A single audio chunk sent from the mobile client.

    Fields:
        type:         Must be 'audio_chunk'
        session_id:   Must match the server-allocated session UUID
        chunk_index:  Zero-based sequential index for ordering / gap detection
        chunk_data:   Base64-encoded raw audio bytes for the chunk
        format:       Audio format descriptor (only required on chunk_index=0)
        timestamp_ms: Client-side monotonic timestamp of chunk capture start
        is_final:     True on the last chunk (acts as inline STREAM_END signal)
    """

    type: Literal[ClientMessageType.AUDIO_CHUNK] = ClientMessageType.AUDIO_CHUNK
    session_id: UUID
    chunk_index: int = Field(..., ge=0, description="Zero-based sequential chunk number")
    chunk_data: str = Field(..., min_length=4, description="Base64-encoded audio bytes")
    format: Optional[AudioFormat] = Field(
        None,
        description="Required on chunk_index=0; optional (server caches) on subsequent chunks",
    )
    timestamp_ms: int = Field(
        ...,
        ge=0,
        description="Client-side monotonic capture timestamp in milliseconds",
    )
    is_final: bool = Field(
        False,
        description="Set True on the last chunk to trigger server-side flush",
    )
    checksum_crc32: Optional[int] = Field(
        None,
        description="CRC-32 of raw (pre-base64) audio bytes for corruption detection",
    )

    @field_validator("chunk_data")
    @classmethod
    def _validate_base64(cls, v: str) -> str:
        """Reject non-base64 data before it reaches the buffer."""
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("chunk_data must be valid base64-encoded bytes") from exc

        # Hard cap: 64 KB per chunk (prevents single-chunk memory attacks)
        max_bytes = 65_536
        if len(decoded) > max_bytes:
            raise ValueError(
                f"Decoded chunk size {len(decoded)} exceeds maximum {max_bytes} bytes per chunk"
            )
        if len(decoded) == 0:
            raise ValueError("chunk_data must not be empty after base64 decoding")
        return v

    def decode_audio_bytes(self) -> bytes:
        """Returns the raw audio bytes decoded from the base64 chunk_data field."""
        return base64.b64decode(self.chunk_data)


class ControlMessage(BaseModel):
    """
    Client-side control signal (stream end, ping, abort).
    """

    type: Literal[
        ClientMessageType.STREAM_END,
        ClientMessageType.PING,
        ClientMessageType.SESSION_ABORT,
    ]
    session_id: Optional[UUID] = None
    reason: Optional[str] = Field(
        None,
        max_length=256,
        description="Human-readable reason string for ABORT events",
    )
    client_timestamp_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# Server → Client Messages
# ---------------------------------------------------------------------------

class SessionReadyPayload(BaseModel):
    """Sent immediately after successful JWT auth and session allocation."""

    session_id: UUID
    user_id: str
    organization_id: str
    max_chunk_bytes: int = Field(65_536, description="Server-enforced max bytes per chunk")
    max_session_bytes: int
    max_session_duration_seconds: int
    server_timestamp_ms: int
    kafka_topic: str


class ChunkAckPayload(BaseModel):
    """Per-chunk acknowledgement confirming receipt and buffer write."""

    session_id: UUID
    chunk_index: int
    bytes_received: int
    total_bytes_buffered: int
    server_timestamp_ms: int


class PipelineQueuedPayload(BaseModel):
    """Confirmation that the assembled audio was published to Kafka."""

    session_id: UUID
    kafka_topic: str
    kafka_partition: int
    kafka_offset: int
    total_chunks: int
    total_bytes: int
    pipeline_job_id: str
    server_timestamp_ms: int


class ServerErrorPayload(BaseModel):
    """Structured error payload — always sent before the server closes the WebSocket."""

    session_id: Optional[UUID] = None
    error_code: ErrorCode
    message: str
    retry_after_ms: Optional[int] = Field(
        None,
        description="Set for RATE_LIMITED errors — client must wait before reconnecting",
    )


# ---------------------------------------------------------------------------
# Envelope Wrappers (top-level WebSocket JSON frames)
# ---------------------------------------------------------------------------

class ServerMessage(BaseModel):
    """Root envelope for all server → client WebSocket messages."""

    type: ServerMessageType
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    payload: Any
    server_time: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Internal Kafka Payload (published to ML processing topic)
# ---------------------------------------------------------------------------

class KafkaAudioEvent(BaseModel):
    """
    Schema of the Kafka message published to the ML processing topic.
    Consumed by: Whisper transcription, ECAPA-TDNN, DeBERTa-v3 services.
    """

    model_config = {"frozen": True}

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    organization_id: str
    audio_format: AudioFormat
    total_chunks: int
    total_bytes: int
    s3_object_key: Optional[str] = Field(
        None,
        description="S3 key of the assembled audio file (set for large sessions, None for small inline)",
    )
    audio_bytes_b64: Optional[str] = Field(
        None,
        description="Base64 of assembled audio inline (set when total_bytes < INLINE_THRESHOLD)",
    )
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source_channel: str = "real_time_stream"
    client_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload_exclusivity(self) -> "KafkaAudioEvent":
        if self.s3_object_key is None and self.audio_bytes_b64 is None:
            raise ValueError(
                "KafkaAudioEvent must have either s3_object_key or audio_bytes_b64"
            )
        if self.s3_object_key is not None and self.audio_bytes_b64 is not None:
            raise ValueError(
                "KafkaAudioEvent must have s3_object_key OR audio_bytes_b64, not both"
            )
        return self


# ---------------------------------------------------------------------------
# Internal Session State (stored in Redis)
# ---------------------------------------------------------------------------

class AudioSessionState(BaseModel):
    """
    Redis-persisted session state for a streaming audio ingestion connection.
    Serialized as JSON; stored at key: `audio_session:{session_id}`
    """

    session_id: str
    user_id: str
    organization_id: str
    jti: str
    status: AudioStreamStatus = AudioStreamStatus.ACTIVE
    audio_format: Optional[AudioFormat] = None
    chunk_count: int = 0
    total_bytes: int = 0
    last_chunk_index: int = -1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    kafka_offset: Optional[int] = None
    error: Optional[str] = None
    client_metadata: dict[str, Any] = Field(default_factory=dict)
