"""
SentinelAI NLP Service — Domain Schemas & DTOs

Strict Pydantic v2 models enforcing the complete data contract:
  AudioEvent     → inbound from Kafka raw_audio_events
  TranscriptResult → output of Whisper pipeline
  IntentResult     → output of DeBERTa classifier
  NLPThreatScore   → published to Kafka threat_scores topic
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class ThreatLevel(str, Enum):
    CLEAR    = "CLEAR"
    WARN     = "WARN"
    ALERT    = "ALERT"
    CRITICAL = "CRITICAL"


class InferenceStatus(str, Enum):
    SUCCESS            = "success"
    FAILED             = "failed"
    SKIPPED            = "skipped"
    TIMEOUT            = "timeout"
    INSUFFICIENT_AUDIO = "insufficient_audio"


class AudioChannel(str, Enum):
    VOIP    = "voip"
    MOBILE  = "mobile"
    WEB_RTC = "webrtc"
    UPLOAD  = "upload"


class TranscriptionQuality(str, Enum):
    HIGH       = "high"       # avg_logprob > -0.3
    MEDIUM     = "medium"     # avg_logprob > -0.6
    LOW        = "low"        # avg_logprob > -1.0
    UNRELIABLE = "unreliable" # avg_logprob <= -1.0 or no_speech_prob > 0.8


class IntentLabel(str, Enum):
    BENIGN               = "BENIGN"
    OTP_HARVESTING       = "OTP_HARVESTING"
    FINANCIAL_FRAUD      = "FINANCIAL_FRAUD"
    IMPERSONATION        = "IMPERSONATION"
    PHISHING_LINK        = "PHISHING_LINK"
    CREDENTIAL_THEFT     = "CREDENTIAL_THEFT"
    VISHING              = "VISHING"
    ACCOUNT_TAKEOVER     = "ACCOUNT_TAKEOVER"
    SOCIAL_ENGINEERING   = "SOCIAL_ENGINEERING"
    URGENCY_MANIPULATION = "URGENCY_MANIPULATION"
    UNCERTAIN            = "UNCERTAIN"


# ── Inbound Kafka Message ─────────────────────────────────────────────────────

class AudioMetadata(BaseModel):
    channel: AudioChannel
    sample_rate: int = Field(ge=8000, le=48000)
    duration_seconds: float = Field(ge=0.1, le=600.0)
    encoding: str = Field(default="pcm_16bit")
    num_channels: int = Field(default=1, ge=1, le=2)
    snr_db: Optional[float] = None
    device_id: Optional[str] = Field(default=None, max_length=128)
    geo_region: Optional[str] = Field(default=None, max_length=64)
    language_hint: Optional[str] = Field(
        default=None,
        description="BCP-47 language code hint (e.g. 'en', 'es')",
    )


class AudioEvent(BaseModel):
    """
    Inbound event from the raw_audio_events Kafka topic.
    Identical schema to the ML service; both services consume the same topic.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(..., min_length=1, max_length=256)
    user_id: str = Field(..., min_length=1, max_length=256)
    tenant_id: str = Field(..., min_length=1, max_length=128)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: AudioMetadata
    audio_s3_uri: Optional[str] = None
    audio_b64: Optional[str] = None
    reference_speaker_id: Optional[str] = None
    run_transcription: bool = True
    run_intent: bool = True
    priority: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def require_audio_source(self) -> "AudioEvent":
        if not self.audio_s3_uri and not self.audio_b64:
            raise ValueError("AudioEvent must contain either 'audio_s3_uri' or 'audio_b64'.")
        return self

    @field_validator("audio_b64")
    @classmethod
    def validate_b64(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("audio_b64 cannot be empty.")
        return v


# ── Whisper Transcription Output ──────────────────────────────────────────────

class TranscriptionSegment(BaseModel):
    """A single time-aligned speech segment from Whisper."""
    id: int
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(ge=0.0)
    text: str
    avg_log_prob: float = Field(default=0.0)
    no_speech_prob: float = Field(ge=0.0, le=1.0, default=0.0)
    compression_ratio: float = Field(ge=0.0, default=1.0)
    words: Optional[List[Dict[str, Any]]] = None  # Word-level timestamps

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    @property
    def is_speech(self) -> bool:
        return self.no_speech_prob < 0.5


class TranscriptResult(BaseModel):
    """Full output of the Whisper transcription pipeline."""
    status: InferenceStatus
    full_text: str = ""
    language_detected: Optional[str] = None
    language_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    word_count: int = 0
    avg_log_prob: float = 0.0
    quality: TranscriptionQuality = TranscriptionQuality.MEDIUM
    model_size: str = "large-v3"
    inference_device: str = "cpu"
    inference_latency_ms: float = 0.0
    audio_duration_seconds: float = 0.0
    real_time_factor: float = 0.0   # latency_ms / (duration_s * 1000)
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def compute_derived_fields(self) -> "TranscriptResult":
        if self.full_text:
            self.word_count = len(self.full_text.split())
        if self.audio_duration_seconds > 0 and self.inference_latency_ms > 0:
            self.real_time_factor = round(
                self.inference_latency_ms / (self.audio_duration_seconds * 1000), 3
            )
        return self

    @property
    def is_usable(self) -> bool:
        return (
            self.status == InferenceStatus.SUCCESS
            and self.word_count >= 3
            and self.quality != TranscriptionQuality.UNRELIABLE
        )


# ── DeBERTa Intent Classification Output ─────────────────────────────────────

class IntentPrediction(BaseModel):
    """A single label prediction from the classifier."""
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_threat: bool = False
    rank: int = Field(ge=1)


class ThreatIndicator(BaseModel):
    """Specific threat pattern detected via keyword/regex heuristics or classifier."""
    indicator_type: str       # e.g. "OTP_KEYWORD", "URGENCY_PHRASE", "FINANCIAL_KEYWORD"
    matched_text: str         # The triggering text (PII-safe; no raw numbers)
    severity: str             # "low" | "medium" | "high"
    confidence: float = Field(ge=0.0, le=1.0)


class IntentResult(BaseModel):
    """Full output of the DeBERTa intent classification pipeline."""
    status: InferenceStatus
    primary_intent: str = IntentLabel.UNCERTAIN.value
    primary_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    top_predictions: List[IntentPrediction] = Field(default_factory=list)
    threat_indicators: List[ThreatIndicator] = Field(default_factory=list)
    is_threat: bool = False
    threat_signal: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Normalized threat signal: max(threat_intent_confidences)",
    )
    model_name: str = "deberta-v3-base"
    inference_device: str = "cpu"
    inference_latency_ms: float = 0.0
    input_token_count: int = 0
    error_message: Optional[str] = None


# ── NLP Threat Score: published to threat_scores topic ───────────────────────

class NLPThreatScore(BaseModel):
    """
    Aggregated NLP threat assessment published to Kafka 'threat_scores' topic.
    Carries both structured signals and the raw transcript for downstream SIEM correlation.
    """
    # Tracing
    event_id: str
    correlation_id: str
    session_id: str
    user_id: str
    tenant_id: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_completed_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Source signal: what NLP service detected
    source: str = "nlp_service"

    # Pipeline outputs
    transcription: Optional[TranscriptResult] = None
    intent: Optional[IntentResult] = None

    # Derived threat assessment
    combined_threat_score: float = Field(ge=0.0, le=1.0)
    threat_level: ThreatLevel
    threat_factors: List[str] = Field(default_factory=list)

    # Transcript payload (conditionally PII-redacted)
    transcript_text: str = ""
    transcript_language: Optional[str] = None
    transcript_quality: Optional[str] = None

    # Performance metrics
    total_processing_ms: float = 0.0
    audio_duration_seconds: float = 0.0

    # Provenance
    service_version: str = "1.0.0"
    model_versions: Dict[str, str] = Field(default_factory=dict)


# ── Internal Worker DTOs ──────────────────────────────────────────────────────

class NLPInferenceRequest(BaseModel):
    """Internal DTO: Kafka consumer → NLP orchestrator."""
    audio_event: AudioEvent
    audio_bytes: bytes = Field(exclude=True)

    model_config = {"arbitrary_types_allowed": True}


class NLPInferenceResponse(BaseModel):
    """Internal DTO: NLP orchestrator → Kafka producer."""
    event_id: str
    correlation_id: str
    nlp_threat_score: NLPThreatScore
    processing_ms: float


# ── Health & Readiness ────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: List[ComponentHealth] = Field(default_factory=list)
