"""
SentinelAI - Risk Fusion Engine Schemas
========================================
All Pydantic v2 contracts for the Risk Fusion Engine.

Data flow:
  Kafka topic `threat_scores`  ──►  ThreatScoreEvent (one per ML model per session)
  ThreatScoreAggregation       ──►  accumulates partial scores for a session
  FusedRiskResult              ──►  final weighted aggregate once all scores arrive
  RiskAlertPayload             ──►  WebSocket push to mobile client

Score taxonomy:
  DEEPFAKE_VIDEO:  ViT frame-level confidence — 0.0 (authentic) → 1.0 (deepfake)
  DEEPFAKE_VOICE:  ECAPA-TDNN voice-clone confidence — 0.0 → 1.0
  NLP_INTENT:      DeBERTa-v3 social-engineering intent — 0.0 (benign) → 1.0 (malicious)
  VOICEPRINT_SIM:  Cosine similarity to known enrolled voiceprint — 0.0 → 1.0
                   NOTE: HIGH similarity = LOWER risk if user is verified;
                         HIGH similarity to a DIFFERENT enrolled user = risk.
                         The ML pipeline normalises this to a risk score before publishing.

Session completeness:
  The engine waits for all expected model scores before fusing.
  Each session's Kafka header `sentinel-expected-models` lists which models
  were invoked (e.g. ["deepfake_video","nlp_intent"] for a video call).
  A session is fused once all expected scores arrive OR the session TTL fires.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelType(str, enum.Enum):
    """Identifies which ML model produced a threat score."""
    DEEPFAKE_VIDEO   = "deepfake_video"
    DEEPFAKE_VOICE   = "deepfake_voice"
    NLP_INTENT       = "nlp_intent"
    VOICEPRINT_SIM   = "voiceprint_sim"


class RiskLevel(str, enum.Enum):
    """
    Derived risk classification from the fused score.

    Thresholds (configurable in FusionWeightConfig):
      CRITICAL  ≥ 0.85
      HIGH      ≥ 0.70
      MEDIUM    ≥ 0.50
      LOW       ≥ 0.30
      MINIMAL   <  0.30
    """
    CRITICAL  = "critical"
    HIGH      = "high"
    MEDIUM    = "medium"
    LOW       = "low"
    MINIMAL   = "minimal"


class AlertChannel(str, enum.Enum):
    """Delivery channel for a risk alert."""
    WEBSOCKET  = "websocket"
    PUSH       = "push_notification"
    EMAIL      = "email"
    WEBHOOK    = "webhook"
    SOC_QUEUE  = "soc_queue"


class FusionStatus(str, enum.Enum):
    """Processing state of a session in the fusion engine."""
    ACCUMULATING = "accumulating"   # Waiting for more model scores
    FUSING       = "fusing"         # All scores received, computing result
    COMPLETE     = "complete"       # Fused, persisted, alerts dispatched
    TIMED_OUT    = "timed_out"      # TTL expired before all scores arrived
    ERROR        = "error"          # Processing failure


# ---------------------------------------------------------------------------
# Kafka Inbound: Individual ML Model Score
# ---------------------------------------------------------------------------

class ThreatScoreEvent(BaseModel):
    """
    Single model score event consumed from the `threat_scores` Kafka topic.
    Each ML pipeline stage publishes exactly one event per session per model.

    Published by:
      - ViT deepfake detection service    → model_type = DEEPFAKE_VIDEO
      - ECAPA-TDNN voice biometrics       → model_type = DEEPFAKE_VOICE / VOICEPRINT_SIM
      - DeBERTa-v3 intent classifier      → model_type = NLP_INTENT
    """
    event_id:          str       = Field(default_factory=lambda: str(uuid4()))
    session_id:        str       = Field(..., description="Originating ingestion session UUID")
    user_id:           str       = Field(..., description="Platform user UUID")
    organization_id:   str       = Field(..., description="Tenant UUID")
    model_type:        ModelType = Field(..., description="Which model produced this score")
    model_version:     str       = Field(..., description="e.g. 'vit-deepfake-v2.1'")
    confidence_score:  float     = Field(..., ge=0.0, le=1.0, description="Raw model output")
    processing_time_ms: int      = Field(..., ge=0, description="Model inference latency")
    expected_models:   list[ModelType] = Field(
        ...,
        min_length=1,
        description="All model types expected for this session — used for completeness check",
    )
    source_channel:    str       = Field("real_time_stream")
    model_metadata:    dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Model-specific diagnostics: frame scores (ViT), "
            "embedding distance (ECAPA), token attention weights (DeBERTa)"
        ),
    )
    kafka_timestamp_ms: Optional[int] = Field(None, description="Kafka message timestamp")
    published_at:      datetime       = Field(default_factory=datetime.utcnow)

    @field_validator("session_id", "user_id", "organization_id")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ID fields must not be empty strings")
        return v.strip()


# ---------------------------------------------------------------------------
# Fusion Weight Configuration
# ---------------------------------------------------------------------------

class FusionWeightConfig(BaseModel):
    """
    Configurable weight matrix for the weighted risk fusion algorithm.

    Weights must sum to 1.0 (enforced by model validator).
    Defaults represent the threat model for audio+video deepfake calls:
      - Deepfake video is the primary signal (highest weight)
      - Deepfake voice is the secondary signal
      - NLP intent adds urgency/manipulation signal
      - Voiceprint similarity is a supporting signal

    Booster coefficients amplify the fused score when correlated signals
    appear together (e.g., HIGH deepfake + HIGH urgency language).
    """
    model_config = {"frozen": True}

    # --- Base weights ---
    weight_deepfake_video:  float = Field(0.40, ge=0.0, le=1.0)
    weight_deepfake_voice:  float = Field(0.30, ge=0.0, le=1.0)
    weight_nlp_intent:      float = Field(0.20, ge=0.0, le=1.0)
    weight_voiceprint_sim:  float = Field(0.10, ge=0.0, le=1.0)

    # --- Booster coefficients (multiplicative, applied to fused score) ---
    # Applied when BOTH deepfake signals exceed their booster thresholds
    av_deepfake_booster:               float = Field(1.15, ge=1.0, le=2.0,
        description="Audio+video deepfake correlation booster")
    av_deepfake_booster_threshold:     float = Field(0.70, ge=0.0, le=1.0,
        description="Both deepfake scores must exceed this to apply AV booster")

    # Applied when deepfake score is high AND NLP urgency/manipulation is high
    urgency_deepfake_booster:          float = Field(1.20, ge=1.0, le=2.0,
        description="High deepfake + high urgency language booster")
    urgency_deepfake_threshold:        float = Field(0.65, ge=0.0, le=1.0,
        description="Both scores must exceed this to apply urgency booster")

    # Applied when all available signals are simultaneously high
    all_signals_booster:               float = Field(1.25, ge=1.0, le=2.0,
        description="All signals high simultaneously booster")
    all_signals_threshold:             float = Field(0.75, ge=0.0, le=1.0,
        description="All signals must exceed this to apply the all-signals booster")

    # --- Risk level thresholds ---
    threshold_critical: float = Field(0.85, ge=0.0, le=1.0)
    threshold_high:     float = Field(0.70, ge=0.0, le=1.0)
    threshold_medium:   float = Field(0.50, ge=0.0, le=1.0)
    threshold_low:      float = Field(0.30, ge=0.0, le=1.0)

    # --- Persistence threshold ---
    # Only sessions with fused_score >= persist_threshold are written to PostgreSQL
    persist_threshold:  float = Field(0.30, ge=0.0, le=1.0,
        description="Minimum fused score to trigger DB write and alerting")

    # --- Session completeness TTL ---
    session_ttl_seconds: int  = Field(120, ge=10, le=600,
        description="Max seconds to wait for all model scores before fusing with available data")

    @model_validator(mode="after")
    def _validate_weights_sum(self) -> "FusionWeightConfig":
        total = (
            self.weight_deepfake_video
            + self.weight_deepfake_voice
            + self.weight_nlp_intent
            + self.weight_voiceprint_sim
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fusion weights must sum to 1.0; current sum = {total:.6f}"
            )
        return self

    @model_validator(mode="after")
    def _validate_thresholds_descending(self) -> "FusionWeightConfig":
        if not (
            self.threshold_critical > self.threshold_high
            > self.threshold_medium > self.threshold_low >= 0.0
        ):
            raise ValueError(
                "Risk thresholds must be strictly descending: critical > high > medium > low >= 0"
            )
        return self


# ---------------------------------------------------------------------------
# In-Flight Session Accumulator (Redis-persisted)
# ---------------------------------------------------------------------------

class ThreatScoreAggregation(BaseModel):
    """
    Accumulates partial model scores for a session awaiting completeness.
    Persisted in Redis at key: `fusion:session:{session_id}`
    """
    session_id:       str
    user_id:          str
    organization_id:  str
    source_channel:   str                       = "real_time_stream"
    expected_models:  list[ModelType]           = Field(default_factory=list)
    received_scores:  dict[str, float]          = Field(default_factory=dict,
        description="ModelType.value → confidence_score")
    model_versions:   dict[str, str]            = Field(default_factory=dict,
        description="ModelType.value → model_version")
    model_metadata:   dict[str, dict[str, Any]] = Field(default_factory=dict,
        description="ModelType.value → model_metadata dict")
    processing_times: dict[str, int]            = Field(default_factory=dict,
        description="ModelType.value → processing_time_ms")
    status:           FusionStatus              = FusionStatus.ACCUMULATING
    first_score_at:   datetime                  = Field(default_factory=datetime.utcnow)
    last_score_at:    datetime                  = Field(default_factory=datetime.utcnow)

    @property
    def is_complete(self) -> bool:
        """True when all expected model scores have been received."""
        return (
            len(self.expected_models) > 0
            and all(m.value in self.received_scores for m in self.expected_models)
        )

    @property
    def missing_models(self) -> list[ModelType]:
        """Returns models that have not yet published a score."""
        return [m for m in self.expected_models if m.value not in self.received_scores]

    @property
    def received_count(self) -> int:
        return len(self.received_scores)


# ---------------------------------------------------------------------------
# Fused Risk Result
# ---------------------------------------------------------------------------

class FusedRiskResult(BaseModel):
    """
    Output of the Risk Fusion Engine after all model scores are aggregated.
    Written to PostgreSQL and broadcast as a WebSocket alert.
    """
    fusion_id:        str      = Field(default_factory=lambda: str(uuid4()))
    session_id:       str
    user_id:          str
    organization_id:  str
    source_channel:   str

    # --- Individual component scores (0.0–1.0 each) ---
    score_deepfake_video:  Optional[float] = Field(None, ge=0.0, le=1.0)
    score_deepfake_voice:  Optional[float] = Field(None, ge=0.0, le=1.0)
    score_nlp_intent:      Optional[float] = Field(None, ge=0.0, le=1.0)
    score_voiceprint_sim:  Optional[float] = Field(None, ge=0.0, le=1.0)

    # --- Fusion output ---
    fused_score:      float    = Field(..., ge=0.0, le=1.0,
        description="Final weighted + boosted composite risk score")
    risk_level:       RiskLevel
    active_boosters:  list[str] = Field(default_factory=list,
        description="Names of booster coefficients that were applied")
    weighted_components: dict[str, float] = Field(default_factory=dict,
        description="Each model's weighted contribution to the fused score")

    # --- Session metadata ---
    model_versions:     dict[str, str]            = Field(default_factory=dict)
    model_metadata:     dict[str, dict[str, Any]] = Field(default_factory=dict)
    processing_time_ms: dict[str, int]            = Field(default_factory=dict)
    expected_models:    list[ModelType]           = Field(default_factory=list)
    received_models:    list[ModelType]           = Field(default_factory=list)
    is_partial_fusion:  bool                      = Field(False,
        description="True when TTL fired before all expected scores arrived")

    fused_at:         datetime = Field(default_factory=datetime.utcnow)
    alert_dispatched: bool     = False
    persisted:        bool     = False

    @property
    def exceeds_persist_threshold(self) -> bool:
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()
        threshold = getattr(cfg, "FUSION_PERSIST_THRESHOLD", 0.30)
        return self.fused_score >= threshold


# ---------------------------------------------------------------------------
# WebSocket Alert Payload (Client-facing)
# ---------------------------------------------------------------------------

class RiskAlertPayload(BaseModel):
    """
    Real-time risk alert pushed to the mobile client via WebSocket.
    Contains enough detail for the client to display a threat warning.
    No raw model metadata (internal) is included — only actionable fields.
    """
    alert_id:         str       = Field(default_factory=lambda: str(uuid4()))
    session_id:       str
    fusion_id:        str
    user_id:          str

    risk_level:       RiskLevel
    fused_score:      float     = Field(..., ge=0.0, le=1.0)
    threat_summary:   str       = Field(..., description="Human-readable threat description")
    recommended_action: str     = Field(..., description="Suggested action for the user")

    # Component scores (rounded, safe to expose)
    component_scores: dict[str, float] = Field(default_factory=dict)
    active_boosters:  list[str]        = Field(default_factory=list)

    # Alert delivery metadata
    channels:         list[AlertChannel] = Field(default_factory=list)
    timestamp_ms:     int                = Field(
        default_factory=lambda: int(__import__("time").time() * 1000)
    )
    expires_at_ms:    int = Field(
        default_factory=lambda: int((__import__("time").time() + 300) * 1000),
        description="Alert expiry — client should dismiss if not acted on within 5 minutes"
    )


# ---------------------------------------------------------------------------
# Kafka Alert Event (published back to `threat_alerts` topic for fanout)
# ---------------------------------------------------------------------------

class KafkaAlertEvent(BaseModel):
    """
    Serialized FusedRiskResult published to the `threat_alerts` Kafka topic
    for downstream consumers (SOC dashboard, email/SMS notifier, SIEM export).
    """
    model_config = {"frozen": True}

    event_id:       str            = Field(default_factory=lambda: str(uuid4()))
    fusion_id:      str
    session_id:     str
    user_id:        str
    organization_id: str
    fused_score:    float
    risk_level:     RiskLevel
    threat_summary: str
    fused_at:       datetime
    source_channel: str
    is_partial:     bool
    component_scores: dict[str, float]
    active_boosters:  list[str]
