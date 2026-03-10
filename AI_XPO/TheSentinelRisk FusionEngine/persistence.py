"""
SentinelAI - Risk Fusion Persistence Layer
============================================
Writes FusedRiskResult objects to the PostgreSQL `threat_alerts` table
(defined in Phase 1: sentinel_ai.database.models.threat_alert.ThreatAlert).

Design decisions:
- Only sessions exceeding `persist_threshold` (default: 0.30) are written.
  Sub-threshold sessions are discarded — no PII storage without actionable signal.
- alert_uid is generated in the format 'SA-YYYY-NNNNNN' using a PostgreSQL
  sequence-backed counter to guarantee global uniqueness across all Risk Fusion pods.
- analysis_metadata stores the full FusedRiskResult JSON for SOC explainability
  and model-drift auditing.
- The raw evidence S3 key (if set) is AES-256-GCM encrypted via the ThreatAlert
  model's property accessor (inherited from Phase 1).
- AlertSeverity is derived from RiskLevel via the mapping table below.
- All DB writes are wrapped in the AsyncSession context manager — commits happen
  only after the WebSocket alert is dispatched to avoid partial state.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel_ai.database.models.threat_alert import (
    AlertSeverity,
    AlertStatus,
    SourceChannel,
    ThreatAlert,
    ThreatAlertType,
)
from sentinel_ai.services.risk_fusion.schemas.fusion import (
    FusedRiskResult,
    ModelType,
    RiskLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk Level → Alert Severity mapping
# ---------------------------------------------------------------------------
_SEVERITY_MAP: dict[RiskLevel, AlertSeverity] = {
    RiskLevel.CRITICAL: AlertSeverity.CRITICAL,
    RiskLevel.HIGH:     AlertSeverity.HIGH,
    RiskLevel.MEDIUM:   AlertSeverity.MEDIUM,
    RiskLevel.LOW:      AlertSeverity.LOW,
    RiskLevel.MINIMAL:  AlertSeverity.INFO,
}

# Risk Level → ThreatAlertType heuristic
# The definitive type is determined by which model scored highest
def _infer_alert_type(result: FusedRiskResult) -> ThreatAlertType:
    """
    Infers the primary threat type from the component scores.
    Falls back to IMPERSONATION if no clear dominant signal.
    """
    scores: dict[str, Optional[float]] = {
        ModelType.DEEPFAKE_VIDEO.value: result.score_deepfake_video,
        ModelType.DEEPFAKE_VOICE.value: result.score_deepfake_voice,
        ModelType.NLP_INTENT.value:     result.score_nlp_intent,
    }
    # Filter out None scores and find highest
    valid = {k: v for k, v in scores.items() if v is not None}
    if not valid:
        return ThreatAlertType.UNKNOWN

    dominant = max(valid, key=lambda k: valid[k])  # type: ignore[arg-type]

    if dominant == ModelType.DEEPFAKE_VIDEO.value:
        # Check if voice is also high — if so, audio+visual
        if (
            result.score_deepfake_voice is not None
            and result.score_deepfake_voice >= 0.50
        ):
            return ThreatAlertType.DEEPFAKE_AUDIO_VISUAL
        return ThreatAlertType.DEEPFAKE_VIDEO

    if dominant == ModelType.DEEPFAKE_VOICE.value:
        return ThreatAlertType.VOICE_CLONE

    if dominant == ModelType.NLP_INTENT.value:
        channel = result.source_channel.lower()
        if "call" in channel or "voice" in channel:
            return ThreatAlertType.VISHING
        if "email" in channel:
            return ThreatAlertType.SOCIAL_ENGINEERING_EMAIL
        if "chat" in channel:
            return ThreatAlertType.SOCIAL_ENGINEERING_CHAT
        return ThreatAlertType.SOCIAL_ENGINEERING_CALL

    return ThreatAlertType.IMPERSONATION


def _infer_source_channel(channel_str: str) -> SourceChannel:
    """Maps a raw channel string to the SourceChannel enum."""
    channel_map: dict[str, SourceChannel] = {
        "real_time_stream": SourceChannel.REAL_TIME_STREAM,
        "phone_call":       SourceChannel.PHONE_CALL,
        "video_call":       SourceChannel.VIDEO_CALL,
        "voip":             SourceChannel.VOIP,
        "email":            SourceChannel.EMAIL,
        "live_chat":        SourceChannel.LIVE_CHAT,
        "sms":              SourceChannel.SMS,
        "upload":           SourceChannel.UPLOAD,
    }
    return channel_map.get(channel_str.lower(), SourceChannel.REAL_TIME_STREAM)


# ---------------------------------------------------------------------------
# Persistence Service
# ---------------------------------------------------------------------------

class FusionPersistenceService:
    """
    Writes FusedRiskResult objects to PostgreSQL and returns the created ThreatAlert.

    Thread safety: Stateless — safe for concurrent use across asyncio tasks.
    """

    async def persist(
        self,
        session: AsyncSession,
        result: FusedRiskResult,
    ) -> Optional[ThreatAlert]:
        """
        Persists a FusedRiskResult as a ThreatAlert record.

        Skips records below persist_threshold (already checked upstream but
        checked again here as a defence-in-depth guard).

        Args:
            session: Active SQLAlchemy AsyncSession (NOT auto-committed).
            result:  Validated FusedRiskResult from the fusion engine.

        Returns:
            The newly created ThreatAlert ORM object, or None if below threshold.
        """
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()
        persist_threshold = getattr(cfg, "FUSION_PERSIST_THRESHOLD", 0.30)

        if result.fused_score < persist_threshold:
            logger.debug(
                "Fusion result below persist threshold — skipping DB write",
                extra={
                    "session_id": result.session_id,
                    "fused_score": result.fused_score,
                    "threshold": persist_threshold,
                },
            )
            return None

        alert_uid = await self._generate_alert_uid(session)

        alert = ThreatAlert(
            alert_uid           = alert_uid,
            alert_type          = _infer_alert_type(result),
            severity            = _SEVERITY_MAP[result.risk_level],
            status              = AlertStatus.OPEN,
            source_channel      = _infer_source_channel(result.source_channel),
            # Target
            target_user_id      = uuid.UUID(result.user_id) if result.user_id else None,
            organization_id     = uuid.UUID(result.organization_id) if result.organization_id else None,
            # Detection
            detected_by_model   = self._format_model_list(result.received_models),
            model_version       = ", ".join(set(result.model_versions.values()))[:32],
            confidence_score    = result.fused_score,
            risk_score          = result.fused_score,
            # Structured analysis output
            analysis_metadata   = {
                "fusion_id":           result.fusion_id,
                "fused_score":         result.fused_score,
                "risk_level":          result.risk_level.value,
                "component_scores": {
                    ModelType.DEEPFAKE_VIDEO.value: result.score_deepfake_video,
                    ModelType.DEEPFAKE_VOICE.value: result.score_deepfake_voice,
                    ModelType.NLP_INTENT.value:     result.score_nlp_intent,
                    ModelType.VOICEPRINT_SIM.value: result.score_voiceprint_sim,
                },
                "weighted_components": result.weighted_components,
                "active_boosters":     result.active_boosters,
                "model_versions":      result.model_versions,
                "processing_time_ms":  result.processing_time_ms,
                "expected_models":     [m.value for m in result.expected_models],
                "received_models":     [m.value for m in result.received_models],
                "is_partial_fusion":   result.is_partial_fusion,
                "fused_at":            result.fused_at.isoformat(),
            },
            # Session correlation
            session_id          = result.session_id,
        )

        session.add(alert)
        await session.flush()  # Flush to get the DB-generated ID without committing

        logger.info(
            "ThreatAlert persisted to PostgreSQL",
            extra={
                "alert_uid":    alert_uid,
                "alert_id":     str(alert.id),
                "session_id":   result.session_id,
                "fused_score":  result.fused_score,
                "risk_level":   result.risk_level.value,
                "alert_type":   alert.alert_type.value,
            },
        )
        return alert

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _generate_alert_uid(session: AsyncSession) -> str:
        """
        Generates a globally unique, human-readable alert UID.
        Format: 'SA-YYYY-NNNNNN'

        Uses a PostgreSQL sequence for atomically incrementing counter.
        The sequence is created once via migration (see notes below).

        Migration note:
            CREATE SEQUENCE IF NOT EXISTS threat_alert_uid_seq START 1;
        """
        result = await session.execute(
            text("SELECT nextval('threat_alert_uid_seq')")
        )
        seq_val: int = result.scalar_one()
        year = datetime.now(timezone.utc).year
        return f"SA-{year}-{seq_val:06d}"

    @staticmethod
    def _format_model_list(models: list) -> str:
        """Formats the list of received models into a compact identifier string."""
        return ", ".join(m.value if hasattr(m, "value") else str(m) for m in models)[:128]
