"""
SentinelAI - Threat Alert ORM Model
=====================================
Primary threat detection record produced by the SentinelAI ML pipeline.

Each alert is generated when a detection model (ViT, ECAPA-TDNN, DeBERTa-v3)
exceeds its configured confidence threshold. Alerts pass through a SOC
workflow: OPEN → INVESTIGATING → ESCALATED → RESOLVED | FALSE_POSITIVE.

Security design:
- Raw evidence S3 object keys are AES-256-GCM encrypted at rest.
- analysis_metadata stores model-specific output (frame scores, token weights,
  spectrogram heatmaps) as structured JSONB for flexible querying.
- alert_uid provides human-readable IDs for SOC operators (SA-YYYY-NNNNNN).
- Correlated alerts in the same attack campaign are linked via related_alert_id.
- Evidence retention is configurable per-alert for compliance flexibility.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel_ai.database.base import Base
from sentinel_ai.database.encryption import decrypt_field, encrypt_field

if TYPE_CHECKING:
    from sentinel_ai.database.models.user import User


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ThreatAlertType(str, enum.Enum):
    """Classification of detected threat category."""

    DEEPFAKE_VOICE = "deepfake_voice"
    DEEPFAKE_VIDEO = "deepfake_video"
    DEEPFAKE_AUDIO_VISUAL = "deepfake_audio_visual"
    VOICE_CLONE = "voice_clone"
    FACE_SWAP = "face_swap"
    SOCIAL_ENGINEERING_CALL = "social_engineering_call"
    SOCIAL_ENGINEERING_EMAIL = "social_engineering_email"
    SOCIAL_ENGINEERING_CHAT = "social_engineering_chat"
    VISHING = "vishing"
    IMPERSONATION = "impersonation"
    UNKNOWN = "unknown"


class AlertSeverity(str, enum.Enum):
    """
    Operational severity of a threat alert.

    CRITICAL  — Immediate action required; high confidence, high impact
    HIGH      — Prompt investigation required within 1 hour
    MEDIUM    — Standard queue; investigate within 4 hours
    LOW       — Background noise; investigate within 24 hours
    INFO      — Telemetry / audit record; no action required
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"


class AlertStatus(str, enum.Enum):
    """SOC workflow status for a threat alert."""

    OPEN = "open"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    SUPPRESSED = "suppressed"


class SourceChannel(str, enum.Enum):
    """Communication channel where the threat was detected."""

    PHONE_CALL = "phone_call"
    VIDEO_CALL = "video_call"
    VOIP = "voip"
    EMAIL = "email"
    LIVE_CHAT = "live_chat"
    SMS = "sms"
    SOCIAL_MEDIA = "social_media"
    UPLOAD = "upload"        # Manual file submission via API
    REAL_TIME_STREAM = "real_time_stream"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ThreatAlert(Base):
    """
    Core threat detection record linking an ML model's output to a user,
    organization, and raw evidence stored in S3.

    Lifecycle: Created by the detection pipeline → Triaged by SOC analysts
    → Resolved or classified as false positive.
    """

    __tablename__ = "threat_alerts"

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
    alert_uid: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        nullable=False,
        index=True,
        comment="Human-readable SOC ID, e.g. 'SA-2024-001234' — generated at creation",
    )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    alert_type: Mapped[ThreatAlertType] = mapped_column(
        Enum(ThreatAlertType, name="threat_alert_type", create_type=True),
        nullable=False,
        index=True,
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity, name="alert_severity", create_type=True),
        nullable=False,
        index=True,
    )
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status", create_type=True),
        default=AlertStatus.OPEN,
        nullable=False,
        index=True,
    )
    source_channel: Mapped[SourceChannel] = mapped_column(
        Enum(SourceChannel, name="source_channel", create_type=True),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Target & Tenant
    # ------------------------------------------------------------------
    target_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User targeted by the detected social engineering attempt",
    )
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Tenant isolation key for multi-org deployments",
    )

    # ------------------------------------------------------------------
    # Detection Metadata
    # ------------------------------------------------------------------
    detected_by_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Model identifier, e.g. 'vit-deepfake-v2.1', 'ecapa-tdnn-v1.0', 'deberta-v3-intent'",
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        comment="Semantic version of the detecting model for drift tracking",
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Raw model output confidence 0.0–1.0",
    )
    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Composite risk score 0.0–1.0 combining confidence + contextual signals",
    )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    raw_evidence_s3_key_encrypted: Mapped[Optional[bytes]] = mapped_column(
        BYTEA,
        nullable=True,
        comment="AES-256-GCM encrypted S3 object key pointing to raw audio/video",
    )
    evidence_retention_days: Mapped[int] = mapped_column(
        Integer,
        default=90,
        nullable=False,
        comment="Days to retain raw S3 evidence before lifecycle expiry",
    )

    # Whisper transcription of audio content (if applicable)
    transcription: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Whisper ASR transcription of audio content for NLP analysis",
    )

    # Structured model output: frame-level scores, token attention weights, etc.
    analysis_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "Model-specific structured output. "
            "ViT: per-frame scores; ECAPA: embedding distances; DeBERTa: token weights."
        ),
    )

    # ------------------------------------------------------------------
    # SOC Workflow
    # ------------------------------------------------------------------
    assigned_to: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="SOC analyst currently owning this alert",
    )
    escalated_to: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Senior analyst or manager to whom alert was escalated",
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    resolved_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    false_positive_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Analyst explanation when closing as FALSE_POSITIVE",
    )
    remediation_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Actions taken to remediate or contain the threat",
    )

    # ------------------------------------------------------------------
    # Campaign Correlation
    # ------------------------------------------------------------------
    related_alert_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("threat_alerts.id", ondelete="SET NULL"),
        nullable=True,
        comment="Links alerts believed to belong to the same attack campaign",
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        index=True,
        comment="Source session/call ID from the ingest pipeline",
    )
    external_ticket_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="JIRA / ServiceNow ticket ID for external workflow integration",
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    target_user: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[target_user_id],
        back_populates="threat_alerts_targeted",
    )
    assigned_analyst: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[assigned_to],
    )
    escalation_target: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[escalated_to],
    )
    resolver: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[resolved_by],
    )
    related_alert: Mapped[Optional["ThreatAlert"]] = relationship(
        "ThreatAlert",
        remote_side="ThreatAlert.id",
        foreign_keys=[related_alert_id],
    )

    # ------------------------------------------------------------------
    # Composite Indexes
    # ------------------------------------------------------------------
    __table_args__ = (
        Index("ix_threat_alerts_org_status", "organization_id", "status"),
        Index("ix_threat_alerts_org_severity", "organization_id", "severity"),
        Index("ix_threat_alerts_type_created", "alert_type", "created_at"),
        Index("ix_threat_alerts_confidence_risk", "confidence_score", "risk_score"),
        Index("ix_threat_alerts_assigned_status", "assigned_to", "status"),
        {
            "comment": "Threat detection records from SentinelAI ML pipeline",
        },
    )

    # ------------------------------------------------------------------
    # Encrypted Property Accessors
    # ------------------------------------------------------------------

    @property
    def raw_evidence_s3_key(self) -> Optional[str]:
        """Decrypts and returns the S3 object key for raw evidence."""
        if not self.raw_evidence_s3_key_encrypted:
            return None
        return decrypt_field(bytes(self.raw_evidence_s3_key_encrypted))

    @raw_evidence_s3_key.setter
    def raw_evidence_s3_key(self, value: str) -> None:
        """Encrypts the S3 key before storage."""
        self.raw_evidence_s3_key_encrypted = encrypt_field(value)

    # ------------------------------------------------------------------
    # Derived Properties
    # ------------------------------------------------------------------

    @property
    def is_resolved(self) -> bool:
        return self.status in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE)

    @property
    def is_critical(self) -> bool:
        return self.severity == AlertSeverity.CRITICAL

    def __repr__(self) -> str:
        return (
            f"<ThreatAlert uid={self.alert_uid!r} type={self.alert_type} "
            f"severity={self.severity} status={self.status} "
            f"confidence={self.confidence_score:.3f}>"
        )
