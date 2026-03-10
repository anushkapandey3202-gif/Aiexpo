"""
SentinelAI ML Service — Inference Orchestrator

The central coordinator that:
1. Decodes and validates the incoming audio payload.
2. Dispatches Voiceprint Auth and Deepfake Detection pipelines concurrently.
3. Aggregates individual scores into a single weighted ThreatScore.
4. Computes ThreatLevel using configurable threshold bands.
5. Returns the ThreatScore to the Kafka producer for downstream routing.
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime, timezone
from typing import Optional

from app.core.config import get_settings
from app.core.logging import PerformanceLogger, get_logger, set_correlation_id
from app.models.schemas import (
    AudioEvent,
    DeepfakeResult,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ThreatLevel,
    ThreatScore,
    VoiceprintResult,
)
from app.pipelines.deepfake import DeepfakePipeline
from app.pipelines.voiceprint import VoiceprintPipeline

logger = get_logger("services.inference_orchestrator")


class InferenceOrchestrator:
    """
    Coordinates all ML pipelines for a single audio inference request.

    Design decisions:
    - Pipelines execute concurrently via asyncio.gather().
    - Pipeline failures are isolated — one failure never blocks the other.
    - Combined score uses configurable weighted sum of deepfake and voiceprint signals.
    - Audio decoding is done here once, then passed to both pipelines.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._voiceprint_pipeline = VoiceprintPipeline()
        self._deepfake_pipeline = DeepfakePipeline()

    async def process(self, request: InferenceRequest) -> InferenceResponse:
        """
        Run all requested pipelines for one AudioEvent and aggregate results.

        Args:
            request: InferenceRequest containing the AudioEvent and raw audio bytes.

        Returns:
            InferenceResponse with the computed ThreatScore.
        """
        event = request.audio_event
        audio_bytes = request.audio_bytes

        set_correlation_id(event.correlation_id)
        orchestration_start = time.perf_counter()

        logger.info(
            "Orchestrator processing event.",
            extra={
                "event_id": event.event_id,
                "run_voiceprint": event.run_voiceprint,
                "run_deepfake": event.run_deepfake,
                "audio_duration_s": event.metadata.duration_seconds,
                "channel": event.metadata.channel.value,
            },
        )

        # ── Concurrent pipeline dispatch ───────────────────────────────────
        voiceprint_task: Optional[asyncio.Task] = None
        deepfake_task: Optional[asyncio.Task] = None

        tasks = {}
        if event.run_voiceprint:
            tasks["voiceprint"] = asyncio.create_task(
                self._run_voiceprint_safe(audio_bytes, event),
                name=f"voiceprint-{event.event_id}",
            )
        if event.run_deepfake:
            tasks["deepfake"] = asyncio.create_task(
                self._run_deepfake_safe(audio_bytes, event),
                name=f"deepfake-{event.event_id}",
            )

        # Gather with return_exceptions to prevent one failure killing the other
        results = {}
        if tasks:
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), gathered):
                results[key] = result

        voiceprint_result: Optional[VoiceprintResult] = results.get("voiceprint")
        deepfake_result: Optional[DeepfakeResult] = results.get("deepfake")

        # ── Aggregate threat score ─────────────────────────────────────────
        threat_score = self._compute_threat_score(
            event=event,
            voiceprint_result=voiceprint_result,
            deepfake_result=deepfake_result,
        )

        total_ms = (time.perf_counter() - orchestration_start) * 1000
        threat_score.total_processing_ms = round(total_ms, 3)
        threat_score.audio_duration_seconds = event.metadata.duration_seconds
        threat_score.processing_completed_utc = datetime.now(timezone.utc)

        logger.info(
            "Orchestration complete.",
            extra={
                "event_id": event.event_id,
                "threat_level": threat_score.threat_level.value,
                "combined_threat_score": round(threat_score.combined_threat_score, 4),
                "total_ms": round(total_ms, 2),
                "threat_factors": threat_score.threat_factors,
            },
        )

        return InferenceResponse(
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            threat_score=threat_score,
            processing_ms=round(total_ms, 3),
        )

    async def _run_voiceprint_safe(
        self, audio_bytes: bytes, event: AudioEvent
    ) -> VoiceprintResult:
        """Isolated voiceprint pipeline execution with timeout guard."""
        try:
            return await asyncio.wait_for(
                self._voiceprint_pipeline.run(audio_bytes, event),
                timeout=self._settings.INFERENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Voiceprint pipeline timed out.",
                extra={"event_id": event.event_id, "timeout": self._settings.INFERENCE_TIMEOUT_SECONDS},
            )
            from app.models.schemas import AuthDecision
            return VoiceprintResult(
                status=InferenceStatus.TIMEOUT,
                decision=AuthDecision.INSUFFICIENT_AUDIO,
                inference_device="unknown",
                error_message="Pipeline timeout",
            )
        except Exception as exc:
            logger.error(
                "Voiceprint pipeline raised unexpected exception.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            from app.models.schemas import AuthDecision
            return VoiceprintResult(
                status=InferenceStatus.FAILED,
                decision=AuthDecision.INSUFFICIENT_AUDIO,
                inference_device="unknown",
                error_message=str(exc),
            )

    async def _run_deepfake_safe(
        self, audio_bytes: bytes, event: AudioEvent
    ) -> DeepfakeResult:
        """Isolated deepfake pipeline execution with timeout guard."""
        try:
            return await asyncio.wait_for(
                self._deepfake_pipeline.run(audio_bytes, event),
                timeout=self._settings.INFERENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Deepfake pipeline timed out.",
                extra={"event_id": event.event_id, "timeout": self._settings.INFERENCE_TIMEOUT_SECONDS},
            )
            return DeepfakeResult(
                status=InferenceStatus.TIMEOUT,
                spoof_probability=0.0,
                genuine_probability=1.0,
                is_deepfake=False,
                model_version="rawnet3-v1",
                inference_device="unknown",
                error_message="Pipeline timeout",
            )
        except Exception as exc:
            logger.error(
                "Deepfake pipeline raised unexpected exception.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            return DeepfakeResult(
                status=InferenceStatus.FAILED,
                spoof_probability=0.0,
                genuine_probability=1.0,
                is_deepfake=False,
                model_version="rawnet3-v1",
                inference_device="unknown",
                error_message=str(exc),
            )

    def _compute_threat_score(
        self,
        event: AudioEvent,
        voiceprint_result: Optional[VoiceprintResult],
        deepfake_result: Optional[DeepfakeResult],
    ) -> ThreatScore:
        """
        Aggregate individual pipeline outputs into a single ThreatScore.

        Weighting:
            combined_score = (deepfake_weight * deepfake_signal)
                           + (voice_weight   * voice_signal)

        Deepfake signal: spoof_probability (0→1)
        Voice signal:    1 - best_cosine_score when unauthenticated, else 0
        """
        settings = self._settings
        threat_factors: list[str] = []
        deepfake_signal: float = 0.0
        voice_signal: float = 0.0

        # ── Deepfake signal ────────────────────────────────────────────────
        if deepfake_result and deepfake_result.status == InferenceStatus.SUCCESS:
            deepfake_signal = deepfake_result.spoof_probability
            if deepfake_result.spoof_probability >= settings.DEEPFAKE_THRESHOLD_ALERT:
                threat_factors.append("HIGH_SPOOF_PROBABILITY")
            elif deepfake_result.spoof_probability >= settings.DEEPFAKE_THRESHOLD_WARN:
                threat_factors.append("ELEVATED_SPOOF_PROBABILITY")
        elif deepfake_result and deepfake_result.status in (
            InferenceStatus.FAILED, InferenceStatus.TIMEOUT
        ):
            threat_factors.append("DEEPFAKE_PIPELINE_ERROR")

        # ── Voiceprint signal ──────────────────────────────────────────────
        if voiceprint_result and voiceprint_result.status == InferenceStatus.SUCCESS:
            from app.models.schemas import AuthDecision

            if voiceprint_result.decision == AuthDecision.AUTHENTICATED:
                voice_signal = 0.0
            elif voiceprint_result.decision == AuthDecision.UNAUTHENTICATED:
                best = voiceprint_result.best_score or 0.0
                voice_signal = max(0.0, 1.0 - best)
                threat_factors.append("SPEAKER_AUTHENTICATION_FAILED")
            elif voiceprint_result.decision == AuthDecision.UNKNOWN_SPEAKER:
                voice_signal = settings.VOICEPRINT_MISMATCH_THRESHOLD
                threat_factors.append("UNKNOWN_SPEAKER")
            else:
                voice_signal = 0.0
        elif voiceprint_result and voiceprint_result.status in (
            InferenceStatus.FAILED, InferenceStatus.TIMEOUT
        ):
            threat_factors.append("VOICEPRINT_PIPELINE_ERROR")

        # ── Weighted combination ───────────────────────────────────────────
        active_deepfake = deepfake_result is not None and deepfake_result.status == InferenceStatus.SUCCESS
        active_voice = voiceprint_result is not None and voiceprint_result.status == InferenceStatus.SUCCESS

        if active_deepfake and active_voice:
            combined = (
                settings.COMBINED_THREAT_WEIGHT_DEEPFAKE * deepfake_signal
                + settings.COMBINED_THREAT_WEIGHT_VOICE * voice_signal
            )
        elif active_deepfake:
            combined = deepfake_signal
        elif active_voice:
            combined = voice_signal
        else:
            combined = 0.0

        combined = min(max(combined, 0.0), 1.0)

        # ── Threat level classification ────────────────────────────────────
        threat_level = self._classify_threat(combined, threat_factors)

        # ── Model version provenance ───────────────────────────────────────
        model_versions: dict[str, str] = {}
        if deepfake_result:
            model_versions["deepfake"] = deepfake_result.model_version
        if voiceprint_result:
            model_versions["voiceprint"] = "ecapa-tdnn-v1"

        return ThreatScore(
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            user_id=event.user_id,
            tenant_id=event.tenant_id,
            timestamp_utc=event.timestamp_utc,
            voiceprint=voiceprint_result,
            deepfake=deepfake_result,
            combined_threat_score=round(combined, 6),
            threat_level=threat_level,
            threat_factors=threat_factors,
            model_versions=model_versions,
            service_version=settings.SERVICE_VERSION,
        )

    def _classify_threat(
        self,
        score: float,
        factors: list[str],
    ) -> ThreatLevel:
        """Map combined score to a discrete ThreatLevel."""
        # Critical: extremely high score OR specific high-confidence factor combos
        if score >= 0.85 or (
            "HIGH_SPOOF_PROBABILITY" in factors and "SPEAKER_AUTHENTICATION_FAILED" in factors
        ):
            return ThreatLevel.CRITICAL
        if score >= self._settings.DEEPFAKE_THRESHOLD_ALERT:
            return ThreatLevel.ALERT
        if score >= self._settings.DEEPFAKE_THRESHOLD_WARN:
            return ThreatLevel.WARN
        return ThreatLevel.CLEAR


# ---------------------------------------------------------------------------
# Audio Byte Decoder (shared utility)
# ---------------------------------------------------------------------------


def decode_audio_payload(event: AudioEvent) -> bytes:
    """
    Decode audio bytes from an AudioEvent.
    Prioritizes base64 payload; S3 download handled separately in consumer.

    Raises:
        ValueError: If audio_b64 is present but malformed.
    """
    if event.audio_b64:
        try:
            return base64.b64decode(event.audio_b64)
        except Exception as exc:
            raise ValueError(f"Failed to decode audio_b64: {exc}") from exc
    raise ValueError(
        "decode_audio_payload called but audio_b64 is not set. "
        "S3 audio should be resolved by the Kafka consumer before calling this function."
    )
