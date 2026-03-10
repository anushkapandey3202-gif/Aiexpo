"""
SentinelAI NLP Service — NLP Inference Orchestrator

Coordinates the full NLP pipeline for one AudioEvent:
  1. Decode audio payload (base64 or pre-fetched S3 bytes).
  2. Run Transcription + Intent pipelines — sequentially by necessity
     (intent requires the transcript text), but each internally async.
  3. Assemble and return a typed NLPThreatScore.

Design decisions:
  - Transcription runs first; intent is gated on its output.
  - Each pipeline has an independent timeout guard.
  - A failed transcription still yields a partial threat score
    (keyword heuristics on empty text return no indicators, not an error).
  - Threat level classification mirrors the ML service bands for
    consistency across the shared threat_scores topic.
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger, set_correlation_id
from app.models.schemas import (
    AudioEvent,
    InferenceStatus,
    IntentLabel,
    IntentResult,
    NLPInferenceRequest,
    NLPInferenceResponse,
    NLPThreatScore,
    THREAT_INTENT_LABELS,
    ThreatLevel,
    TranscriptResult,
)
from app.pipelines.intent import IntentPipeline
from app.pipelines.transcription import TranscriptionPipeline

logger = get_logger("services.nlp_orchestrator")


class NLPOrchestrator:
    """
    Stateless NLP inference coordinator.
    One instance lives for the lifetime of the service process.
    Redis client is injected post-construction (after async context available).
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._transcription_pipeline = TranscriptionPipeline()
        self._intent_pipeline        = IntentPipeline()

    def set_redis(self, redis_client: aioredis.Redis) -> None:
        """Inject Redis after the async loop is running."""
        self._transcription_pipeline.set_redis(redis_client)

    async def process(self, request: NLPInferenceRequest) -> NLPInferenceResponse:
        """
        Execute the full NLP pipeline for one AudioEvent.

        Args:
            request: NLPInferenceRequest with AudioEvent + pre-decoded audio bytes.

        Returns:
            NLPInferenceResponse wrapping the NLPThreatScore.
        """
        event      = request.audio_event
        audio_bytes = request.audio_bytes

        set_correlation_id(event.correlation_id)
        t_start = time.perf_counter()

        logger.info(
            "NLP orchestrator processing event.",
            extra={
                "event_id":        event.event_id,
                "run_transcription": event.run_transcription,
                "run_intent":      event.run_intent,
                "duration_s":      event.metadata.duration_seconds,
                "channel":         event.metadata.channel.value,
            },
        )

        # ── Step 1: Transcription ─────────────────────────────────────────────
        transcript_result: Optional[TranscriptResult] = None
        if event.run_transcription:
            transcript_result = await self._run_transcription_safe(audio_bytes, event)
        else:
            logger.info("Transcription skipped per event flag.",
                        extra={"event_id": event.event_id})

        # ── Step 2: Intent classification ─────────────────────────────────────
        intent_result: Optional[IntentResult] = None
        if event.run_intent and transcript_result is not None:
            intent_result = await self._run_intent_safe(transcript_result, event)
        elif event.run_intent and transcript_result is None:
            logger.info("Intent skipped — no transcript available.",
                        extra={"event_id": event.event_id})

        # ── Step 3: Assemble NLPThreatScore ───────────────────────────────────
        threat_score = self._assemble_threat_score(
            event=event,
            transcript=transcript_result,
            intent=intent_result,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        threat_score.total_processing_ms    = round(total_ms, 3)
        threat_score.audio_duration_seconds = event.metadata.duration_seconds
        threat_score.processing_completed_utc = datetime.now(timezone.utc)

        logger.info(
            "NLP orchestration complete.",
            extra={
                "event_id":             event.event_id,
                "threat_level":         threat_score.threat_level.value,
                "combined_score":       round(threat_score.combined_threat_score, 4),
                "primary_intent":       intent_result.primary_intent if intent_result else "N/A",
                "transcript_words":     transcript_result.word_count if transcript_result else 0,
                "total_ms":             round(total_ms, 2),
            },
        )

        return NLPInferenceResponse(
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            nlp_threat_score=threat_score,
            processing_ms=round(total_ms, 3),
        )

    # ── Safe pipeline wrappers ────────────────────────────────────────────────

    async def _run_transcription_safe(
        self, audio_bytes: bytes, event: AudioEvent
    ) -> TranscriptResult:
        """Transcription with independent timeout and exception isolation."""
        from app.models.schemas import TranscriptionQuality
        try:
            return await asyncio.wait_for(
                self._transcription_pipeline.run(audio_bytes, event),
                timeout=self._settings.WHISPER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Transcription pipeline timed out.",
                extra={"event_id": event.event_id,
                       "timeout_s": self._settings.WHISPER_TIMEOUT_SECONDS},
            )
            return TranscriptResult(
                status=InferenceStatus.TIMEOUT,
                model_size=self._settings.WHISPER_MODEL_SIZE.value,
                inference_device="unknown",
                error_message="Transcription timeout.",
            )
        except Exception as exc:
            logger.error(
                "Transcription pipeline exception.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            return TranscriptResult(
                status=InferenceStatus.FAILED,
                model_size=self._settings.WHISPER_MODEL_SIZE.value,
                inference_device="unknown",
                error_message=str(exc),
            )

    async def _run_intent_safe(
        self, transcript: TranscriptResult, event: AudioEvent
    ) -> IntentResult:
        """Intent classification with independent timeout and exception isolation."""
        try:
            return await asyncio.wait_for(
                self._intent_pipeline.run(transcript, event),
                timeout=self._settings.INTENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Intent pipeline timed out.",
                extra={"event_id": event.event_id,
                       "timeout_s": self._settings.INTENT_TIMEOUT_SECONDS},
            )
            return IntentResult(
                status=InferenceStatus.TIMEOUT,
                primary_intent=IntentLabel.UNCERTAIN.value,
                inference_device="unknown",
                error_message="Intent timeout.",
            )
        except Exception as exc:
            logger.error(
                "Intent pipeline exception.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            return IntentResult(
                status=InferenceStatus.FAILED,
                primary_intent=IntentLabel.UNCERTAIN.value,
                inference_device="unknown",
                error_message=str(exc),
            )

    # ── Threat score assembly ─────────────────────────────────────────────────

    def _assemble_threat_score(
        self,
        event: AudioEvent,
        transcript: Optional[TranscriptResult],
        intent: Optional[IntentResult],
    ) -> NLPThreatScore:
        """
        Derive a NLPThreatScore from pipeline outputs.

        Scoring rules:
        - Base signal = intent.threat_signal (0–1).
        - Penalty: UNRELIABLE transcription quality → signal × 0.7.
        - Bonus:   multiple independent threat indicators → signal × 1.1 (capped at 1.0).
        - Keyword-only path: if intent unavailable but indicators present,
          use max(indicator.confidence) × 0.8 as signal.
        """
        factors: list[str] = []
        signal: float      = 0.0

        # ── Compute threat signal ─────────────────────────────────────────────
        if intent and intent.status == InferenceStatus.SUCCESS:
            signal = intent.threat_signal

            if intent.primary_intent in THREAT_INTENT_LABELS:
                factors.append(f"INTENT_{intent.primary_intent}")

            for ind in intent.threat_indicators:
                if ind.severity == "high":
                    factors.append(f"KW_{ind.indicator_type}")

            # Corroboration bonus: neural model + keyword agree
            if (
                intent.primary_intent in THREAT_INTENT_LABELS
                and len(intent.threat_indicators) >= 2
            ):
                signal = min(signal * 1.10, 1.0)
                factors.append("MULTI_INDICATOR_CORROBORATION")

        elif intent and intent.status in (InferenceStatus.FAILED, InferenceStatus.TIMEOUT):
            factors.append("INTENT_PIPELINE_ERROR")

        elif intent and intent.status == InferenceStatus.SKIPPED:
            # Keyword-only fallback
            if intent.threat_indicators:
                kw_max = max(i.confidence for i in intent.threat_indicators)
                signal = kw_max * 0.80
                factors.append("KEYWORD_ONLY_SIGNAL")

        # ── Transcription quality adjustment ─────────────────────────────────
        if transcript:
            if transcript.quality.value == "unreliable":
                signal = signal * 0.70
                factors.append("UNRELIABLE_TRANSCRIPT_QUALITY")
            if transcript.status == InferenceStatus.FAILED:
                factors.append("TRANSCRIPTION_FAILED")
            if transcript.status == InferenceStatus.TIMEOUT:
                factors.append("TRANSCRIPTION_TIMEOUT")

        signal = min(max(signal, 0.0), 1.0)

        # ── Classify threat level ─────────────────────────────────────────────
        threat_level = self._classify_level(signal, factors)

        # ── Model versions provenance ─────────────────────────────────────────
        model_versions: dict[str, str] = {}
        if transcript:
            model_versions["whisper"] = f"faster-whisper-{transcript.model_size}"
        if intent:
            model_versions["intent"] = intent.model_name

        # ── Denormalised transcript fields ────────────────────────────────────
        transcript_text     = transcript.full_text    if transcript else ""
        transcript_language = transcript.language_detected if transcript else None
        transcript_quality  = transcript.quality.value    if transcript else None
        word_count          = transcript.word_count        if transcript else 0

        return NLPThreatScore(
            event_id=event.event_id,
            correlation_id=event.correlation_id,
            session_id=event.session_id,
            user_id=event.user_id,
            tenant_id=event.tenant_id,
            timestamp_utc=event.timestamp_utc,
            transcription=transcript,
            intent=intent,
            combined_threat_score=round(signal, 6),
            threat_level=threat_level,
            threat_factors=factors,
            transcript_text=transcript_text,
            transcript_language=transcript_language,
            transcript_quality=transcript_quality,
            word_count=word_count,
            model_versions=model_versions,
            service_version=self._settings.SERVICE_VERSION,
        )

    def _classify_level(self, score: float, factors: list[str]) -> ThreatLevel:
        """Map combined NLP threat score to a discrete ThreatLevel."""
        s = self._settings
        if score >= s.INTENT_THREAT_CRITICAL_THRESHOLD or (
            "MULTI_INDICATOR_CORROBORATION" in factors
            and score >= s.INTENT_THREAT_ALERT_THRESHOLD
        ):
            return ThreatLevel.CRITICAL
        if score >= s.INTENT_THREAT_ALERT_THRESHOLD:
            return ThreatLevel.ALERT
        if score >= s.INTENT_THREAT_WARN_THRESHOLD:
            return ThreatLevel.WARN
        return ThreatLevel.CLEAR


# ── Audio decode utility ──────────────────────────────────────────────────────

def decode_audio_payload(event: AudioEvent) -> bytes:
    """
    Decode base64 audio payload from an AudioEvent.
    S3-sourced audio must be pre-fetched by the consumer before calling this.

    Raises:
        ValueError: On malformed base64.
    """
    if event.audio_b64:
        try:
            return base64.b64decode(event.audio_b64)
        except Exception as exc:
            raise ValueError(f"Failed to decode audio_b64: {exc}") from exc
    raise ValueError(
        "decode_audio_payload called but audio_b64 is not set. "
        "Resolve S3 audio in the consumer before calling this function."
    )
