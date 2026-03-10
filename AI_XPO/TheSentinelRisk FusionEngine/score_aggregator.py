"""
SentinelAI - Score Aggregator & Weighted Fusion Engine
=======================================================
Core algorithmic module. Stateless — all session state lives in Redis
and is passed in via ThreatScoreAggregation objects.

Algorithm: Weighted Average → Booster Matrix → Clamp → Classify

Step 1 — Weighted Base Score:
    For each received model score s_i with weight w_i (from FusionWeightConfig),
    compute a weight-normalised average over RECEIVED scores only:

        base = Σ(s_i × w_i) / Σ(w_i)   for all received models

    Normalising the denominator by only the weights of received models handles
    partial fusion (TTL-fired) gracefully — the relative weights of available
    signals are preserved without artificially deflating the score.

Step 2 — Booster Matrix:
    Three multiplicative boosters are evaluated independently.
    Only boosters whose signal thresholds are ALL met are applied.
    Boosters are applied multiplicatively (not additively) to avoid
    unbounded score inflation:

        boosted = base × Π(b_i)   for all active boosters b_i

    Max theoretical score: 1.0 × 1.15 × 1.20 × 1.25 = 1.725 → clamped to 1.0

Step 3 — Clamp:
    fused_score = min(1.0, max(0.0, boosted))

Step 4 — Classify:
    Risk level determined by descending threshold comparison.

Audit Trail:
    FusedRiskResult records exactly which boosters fired, each model's
    weighted_contribution, and whether the fusion was partial — giving
    SOC analysts full explainability of every score.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sentinel_ai.services.risk_fusion.schemas.fusion import (
    FusedRiskResult,
    FusionWeightConfig,
    ModelType,
    RiskLevel,
    ThreatScoreAggregation,
    ThreatScoreEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threat Narrative Templates
# Keyed by RiskLevel — used to build client-facing threat_summary strings
# ---------------------------------------------------------------------------
_THREAT_SUMMARIES: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: (
        "CRITICAL THREAT DETECTED: High-confidence deepfake and/or social engineering signals "
        "across multiple detection channels. This communication is likely malicious."
    ),
    RiskLevel.HIGH: (
        "HIGH RISK: Strong indicators of synthetic media or manipulative intent detected. "
        "Exercise extreme caution before acting on any instructions from this source."
    ),
    RiskLevel.MEDIUM: (
        "ELEVATED RISK: Moderate threat signals detected. Verify the identity of this "
        "contact through an independent channel before sharing sensitive information."
    ),
    RiskLevel.LOW: (
        "LOW RISK: Weak anomaly signals detected. Communication appears largely authentic "
        "but some caution is advised."
    ),
    RiskLevel.MINIMAL: (
        "MINIMAL RISK: No significant threat signals detected. Communication appears authentic."
    ),
}

_RECOMMENDED_ACTIONS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: (
        "TERMINATE this call immediately. Do not share any information. "
        "Report to your security team and contact SentinelAI SOC."
    ),
    RiskLevel.HIGH: (
        "Suspend the call. Verify the caller's identity via a known, independent channel "
        "(e.g., call back on a registered number). Do not proceed until verified."
    ),
    RiskLevel.MEDIUM: (
        "Proceed with caution. Request identity verification before sharing any "
        "sensitive data. Alert your security contact if in doubt."
    ),
    RiskLevel.LOW: (
        "Stay alert. If the caller requests sensitive information or unusual actions, "
        "verify through an independent channel."
    ),
    RiskLevel.MINIMAL: "No immediate action required. Continue monitoring.",
}


class ScoreAggregationError(Exception):
    """Raised when the fusion engine encounters an unrecoverable error."""


class ScoreFusionEngine:
    """
    Stateless weighted score fusion engine.

    Usage:
        engine = ScoreFusionEngine(config=FusionWeightConfig())

        # Ingest each model's score as it arrives from Kafka
        aggregation = engine.ingest_score(aggregation, score_event)

        # When complete (or TTL fires), fuse
        if aggregation.is_complete:
            result = engine.fuse(aggregation)
    """

    def __init__(self, config: Optional[FusionWeightConfig] = None) -> None:
        self._config: FusionWeightConfig = config or FusionWeightConfig()

        # Static weight map — keyed by ModelType.value for O(1) lookups
        self._weights: dict[str, float] = {
            ModelType.DEEPFAKE_VIDEO.value:  self._config.weight_deepfake_video,
            ModelType.DEEPFAKE_VOICE.value:  self._config.weight_deepfake_voice,
            ModelType.NLP_INTENT.value:      self._config.weight_nlp_intent,
            ModelType.VOICEPRINT_SIM.value:  self._config.weight_voiceprint_sim,
        }

        logger.info(
            "ScoreFusionEngine initialized",
            extra={
                "weights": self._weights,
                "thresholds": {
                    "critical": self._config.threshold_critical,
                    "high":     self._config.threshold_high,
                    "medium":   self._config.threshold_medium,
                    "low":      self._config.threshold_low,
                },
                "persist_threshold": self._config.persist_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Score Ingestion
    # ------------------------------------------------------------------

    def ingest_score(
        self,
        aggregation: ThreatScoreAggregation,
        event: ThreatScoreEvent,
    ) -> ThreatScoreAggregation:
        """
        Applies a single model's score to an accumulator.

        Idempotent for the same model_type: a duplicate event overwrites
        the previous score (last-write-wins) and logs a warning.

        Args:
            aggregation: Current accumulator state for the session.
            event:       Validated ThreatScoreEvent from Kafka.

        Returns:
            Updated ThreatScoreAggregation (caller must persist to Redis).
        """
        model_key = event.model_type.value

        if model_key in aggregation.received_scores:
            logger.warning(
                "Duplicate model score received — overwriting",
                extra={
                    "session_id": event.session_id,
                    "model_type": model_key,
                    "previous_score": aggregation.received_scores[model_key],
                    "new_score": event.confidence_score,
                },
            )

        aggregation.received_scores[model_key]  = event.confidence_score
        aggregation.model_versions[model_key]   = event.model_version
        aggregation.model_metadata[model_key]   = event.model_metadata
        aggregation.processing_times[model_key] = event.processing_time_ms
        aggregation.last_score_at               = datetime.now(timezone.utc)

        # Merge expected_models from each event (union across all received)
        for m in event.expected_models:
            if m not in aggregation.expected_models:
                aggregation.expected_models.append(m)

        logger.debug(
            "Score ingested into aggregation",
            extra={
                "session_id": event.session_id,
                "model_type": model_key,
                "score": event.confidence_score,
                "received": aggregation.received_count,
                "expected": len(aggregation.expected_models),
                "is_complete": aggregation.is_complete,
            },
        )
        return aggregation

    # ------------------------------------------------------------------
    # Weighted Fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        aggregation: ThreatScoreAggregation,
        is_partial: bool = False,
    ) -> FusedRiskResult:
        """
        Computes the fused risk score from the accumulated model scores.

        Args:
            aggregation: Completed (or TTL-fired partial) accumulator state.
            is_partial:  True when fusing before all expected scores arrived.

        Returns:
            FusedRiskResult with score, risk level, booster audit, and narratives.

        Raises:
            ScoreAggregationError: If no scores have been received at all.
        """
        if not aggregation.received_scores:
            raise ScoreAggregationError(
                f"Cannot fuse session '{aggregation.session_id}': no scores received"
            )

        cfg = self._config

        # --- Step 1: Weighted base score (normalised over received models only) ---
        weighted_sum = 0.0
        weight_sum   = 0.0
        weighted_components: dict[str, float] = {}

        for model_key, score in aggregation.received_scores.items():
            weight = self._weights.get(model_key, 0.0)
            contribution = score * weight
            weighted_components[model_key] = round(contribution, 6)
            weighted_sum += contribution
            weight_sum   += weight

        if weight_sum == 0.0:
            raise ScoreAggregationError(
                f"All received model keys have zero weight: {list(aggregation.received_scores.keys())}"
            )

        base_score: float = weighted_sum / weight_sum

        # --- Step 2: Booster matrix ---
        active_boosters: list[str] = []
        boosted_score    = base_score

        scores = aggregation.received_scores

        dv = scores.get(ModelType.DEEPFAKE_VIDEO.value, 0.0)
        da = scores.get(ModelType.DEEPFAKE_VOICE.value, 0.0)
        nl = scores.get(ModelType.NLP_INTENT.value, 0.0)

        # Booster 1: Audio+Video deepfake correlation
        if (
            dv >= cfg.av_deepfake_booster_threshold
            and da >= cfg.av_deepfake_booster_threshold
        ):
            boosted_score *= cfg.av_deepfake_booster
            active_boosters.append(
                f"av_deepfake_booster(×{cfg.av_deepfake_booster:.2f})"
            )
            logger.debug(
                "AV deepfake booster applied",
                extra={
                    "session_id": aggregation.session_id,
                    "dv": dv, "da": da,
                    "booster": cfg.av_deepfake_booster,
                },
            )

        # Booster 2: High deepfake + high urgency language
        max_deepfake = max(dv, da)
        if (
            max_deepfake >= cfg.urgency_deepfake_threshold
            and nl >= cfg.urgency_deepfake_threshold
        ):
            boosted_score *= cfg.urgency_deepfake_booster
            active_boosters.append(
                f"urgency_deepfake_booster(×{cfg.urgency_deepfake_booster:.2f})"
            )
            logger.debug(
                "Urgency+deepfake booster applied",
                extra={
                    "session_id": aggregation.session_id,
                    "max_deepfake": max_deepfake, "nlp": nl,
                    "booster": cfg.urgency_deepfake_booster,
                },
            )

        # Booster 3: All available signals simultaneously high
        all_scores = list(aggregation.received_scores.values())
        if (
            len(all_scores) >= 3
            and all(s >= cfg.all_signals_threshold for s in all_scores)
        ):
            boosted_score *= cfg.all_signals_booster
            active_boosters.append(
                f"all_signals_booster(×{cfg.all_signals_booster:.2f})"
            )
            logger.debug(
                "All-signals booster applied",
                extra={
                    "session_id": aggregation.session_id,
                    "scores": aggregation.received_scores,
                    "booster": cfg.all_signals_booster,
                },
            )

        # --- Step 3: Clamp ---
        fused_score = round(min(1.0, max(0.0, boosted_score)), 6)

        # --- Step 4: Classify ---
        risk_level = self._classify(fused_score)

        # --- Assemble result ---
        result = FusedRiskResult(
            session_id       = aggregation.session_id,
            user_id          = aggregation.user_id,
            organization_id  = aggregation.organization_id,
            source_channel   = aggregation.source_channel,
            # Component scores
            score_deepfake_video = scores.get(ModelType.DEEPFAKE_VIDEO.value),
            score_deepfake_voice = scores.get(ModelType.DEEPFAKE_VOICE.value),
            score_nlp_intent     = scores.get(ModelType.NLP_INTENT.value),
            score_voiceprint_sim = scores.get(ModelType.VOICEPRINT_SIM.value),
            # Fusion output
            fused_score          = fused_score,
            risk_level           = risk_level,
            active_boosters      = active_boosters,
            weighted_components  = weighted_components,
            # Session metadata
            model_versions       = dict(aggregation.model_versions),
            model_metadata       = dict(aggregation.model_metadata),
            processing_time_ms   = dict(aggregation.processing_times),
            expected_models      = list(aggregation.expected_models),
            received_models      = [
                ModelType(k) for k in aggregation.received_scores
            ],
            is_partial_fusion    = is_partial,
        )

        logger.info(
            "Risk fusion complete",
            extra={
                "session_id":     aggregation.session_id,
                "fused_score":    fused_score,
                "risk_level":     risk_level.value,
                "base_score":     round(base_score, 6),
                "active_boosters": active_boosters,
                "is_partial":     is_partial,
                "component_scores": {
                    k: round(v, 4) for k, v in aggregation.received_scores.items()
                },
            },
        )
        return result

    # ------------------------------------------------------------------
    # Risk Classification
    # ------------------------------------------------------------------

    def _classify(self, score: float) -> RiskLevel:
        """Maps a fused score to its RiskLevel via descending threshold comparison."""
        cfg = self._config
        if score >= cfg.threshold_critical:
            return RiskLevel.CRITICAL
        if score >= cfg.threshold_high:
            return RiskLevel.HIGH
        if score >= cfg.threshold_medium:
            return RiskLevel.MEDIUM
        if score >= cfg.threshold_low:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    # ------------------------------------------------------------------
    # Alert Payload Builder
    # ------------------------------------------------------------------

    def build_alert_payload(self, result: FusedRiskResult) -> "RiskAlertPayload":
        """
        Constructs the client-facing RiskAlertPayload from a FusedRiskResult.
        Strips internal metadata (model_metadata, versions) from the response.
        """
        from sentinel_ai.services.risk_fusion.schemas.fusion import RiskAlertPayload, AlertChannel

        component_scores = {
            k: round(v, 3)
            for k, v in {
                ModelType.DEEPFAKE_VIDEO.value:  result.score_deepfake_video,
                ModelType.DEEPFAKE_VOICE.value:  result.score_deepfake_voice,
                ModelType.NLP_INTENT.value:      result.score_nlp_intent,
                ModelType.VOICEPRINT_SIM.value:  result.score_voiceprint_sim,
            }.items()
            if v is not None
        }

        return RiskAlertPayload(
            session_id        = result.session_id,
            fusion_id         = result.fusion_id,
            user_id           = result.user_id,
            risk_level        = result.risk_level,
            fused_score       = round(result.fused_score, 3),
            threat_summary    = _THREAT_SUMMARIES[result.risk_level],
            recommended_action= _RECOMMENDED_ACTIONS[result.risk_level],
            component_scores  = component_scores,
            active_boosters   = result.active_boosters,
            channels          = [AlertChannel.WEBSOCKET],
        )
