"""
SentinelAI ML Service — Domain Schemas & DTOs
Strict Pydantic v2 models enforcing the full data contract
between Kafka messages, inference pipelines, and downstream consumers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ThreatLevel(str, Enum):
    CLEAR = "CLEAR"
    WARN = "WARN"
    ALERT = "ALERT"
    CRITICAL = "CRITICAL"


class PipelineType(str, Enum):
    VOICEPRINT_AUTH = "voiceprint_auth"
    DEEPFAKE_DETECTION = "deepfake_detection"
    COMBINED = "combined"


class AudioChannel(str, Enum):
    VOIP = "voip"
    MOBILE = "mobile"
    WEB_RTC = "webrtc"
    UPLOAD = "upload"


class AuthDecision(str, Enum):
    AUTHENTICATED = "AUTHENTICATED"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    UNKNOWN_SPEAKER = "UNKNOWN_SPEAKER"
    INSUFFICIENT_AUDIO = "INSUFFICIENT_AUDIO"


class InferenceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Inbound Kafka Message — raw_audio_events topic
# ---------------------------------------------------------------------------


class AudioMetadata(BaseModel):
    """Metadata about the audio segment's capture context."""

    channel: AudioChannel
    sample_rate: int = Field(ge=8000, le=48000)
    duration_seconds: float = Field(ge=0.1, le=300.0)
    encoding: str = Field(default="pcm_16bit")
    num_channels: int = Field(default=1, ge=1, le=2)
    snr_db: Optional[float] = Field(default=None, description="Signal-to-noise ratio")
    device_id: Optional[str] = Field(default=None, max_length=128)
    geo_region: Optional[str] = Field(default=None, max_length=64)


class AudioEvent(BaseModel):
    """
    Inbound event from the raw_audio_events Kafka topic.
    Audio payload is base64-encoded PCM or a pre-signed S3 URI.
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier",
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Cross-service tracing ID",
    )
    session_id: str = Field(..., min_length=1, max_length=256)
    user_id: str = Field(..., min_length=1, max_length=256)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: AudioMetadata
    audio_s3_uri: Optional[str] = Field(
        default=None,
        description="S3 URI for large audio files (preferred for >1MB)",
    )
    audio_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded raw audio for small payloads",
    )
    reference_speaker_id: Optional[str] = Field(
        default=None,
        description="Known speaker ID to authenticate against (voiceprint lookup)",
    )
    run_deepfake: bool = Field(default=True)
    run_voiceprint: bool = Field(default=True)
    priority: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_audio_source(self) -> "AudioEvent":
        if not self.audio_s3_uri and not self.audio_b64:
            raise ValueError(
                "AudioEvent must contain either 'audio_s3_uri' or 'audio_b64'."
            )
        return self

    @field_validator("audio_b64")
    @classmethod
    def validate_b64_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.strip()) == 0:
            raise ValueError("audio_b64 cannot be an empty string.")
        return v


# ---------------------------------------------------------------------------
# ECAPA-TDNN / Voiceprint Pipeline Results
# ---------------------------------------------------------------------------


class VoiceprintMatch(BaseModel):
    """A single candidate match returned from Pinecone similarity search."""

    speaker_id: str
    cosine_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VoiceprintResult(BaseModel):
    """Full output from the Voiceprint Authentication pipeline."""

    status: InferenceStatus
    decision: AuthDecision
    speaker_id: Optional[str] = None
    top_matches: List[VoiceprintMatch] = Field(default_factory=list)
    best_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    embedding_dim: int = 192
    inference_device: str = "cpu"
    inference_latency_ms: float = 0.0
    error_message: Optional[str] = None

    @property
    def is_authenticated(self) -> bool:
        return self.decision == AuthDecision.AUTHENTICATED


# ---------------------------------------------------------------------------
# RawNet3 / Deepfake Detection Pipeline Results
# ---------------------------------------------------------------------------


class DeepfakeResult(BaseModel):
    """Full output from the Deepfake Detection pipeline."""

    status: InferenceStatus
    spoof_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    genuine_probability: float = Field(ge=0.0, le=1.0, default=1.0)
    is_deepfake: bool = False
    model_version: str = "rawnet3-v1"
    inference_device: str = "cpu"
    inference_latency_ms: float = 0.0
    confidence_band: str = "low"  # low / medium / high
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def compute_confidence_band(self) -> "DeepfakeResult":
        p = self.spoof_probability
        if p < 0.40:
            self.confidence_band = "low"
        elif p < 0.70:
            self.confidence_band = "medium"
        else:
            self.confidence_band = "high"
        return self


# ---------------------------------------------------------------------------
# Combined Threat Score — published to threat_scores topic
# ---------------------------------------------------------------------------


class ThreatScore(BaseModel):
    """
    Aggregated threat assessment published to Kafka 'threat_scores' topic.
    Downstream consumers (alerting, SIEM, frontend) subscribe to this topic.
    """

    # Tracing
    event_id: str
    correlation_id: str
    session_id: str
    user_id: str
    tenant_id: str
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    processing_completed_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Pipeline outputs
    voiceprint: Optional[VoiceprintResult] = None
    deepfake: Optional[DeepfakeResult] = None

    # Combined scoring
    combined_threat_score: float = Field(ge=0.0, le=1.0)
    threat_level: ThreatLevel
    threat_factors: List[str] = Field(
        default_factory=list,
        description="Human-readable list of detected threat indicators",
    )

    # Latency metrics
    total_processing_ms: float = 0.0
    audio_duration_seconds: float = 0.0

    # Service provenance
    service_version: str = "1.0.0"
    model_versions: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal Worker Messages
# ---------------------------------------------------------------------------


class InferenceRequest(BaseModel):
    """Internal DTO passed from Kafka consumer to inference workers."""

    audio_event: AudioEvent
    audio_bytes: bytes = Field(exclude=True)  # Raw decoded audio

    model_config = {"arbitrary_types_allowed": True}


class InferenceResponse(BaseModel):
    """Internal DTO returned from the orchestrator to the Kafka producer."""

    event_id: str
    correlation_id: str
    threat_score: ThreatScore
    processing_ms: float


# ---------------------------------------------------------------------------
# Health & Readiness
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    service: str
    version: str
    environment: str
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    components: List[ComponentHealth] = Field(default_factory=list)
