"""
SentinelAI NLP Service — Transcription Pipeline

Async orchestration layer for Whisper inference:
1. Check Redis transcript cache (identical audio_b64 hash → skip re-inference).
2. Run faster-whisper via WhisperEngine.
3. Quality-gate the result (UNRELIABLE transcripts are flagged, not failed).
4. Cache the transcript in Redis for downstream reuse.
5. Return a typed TranscriptResult.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.logging import get_logger, set_request_context
from app.inference.whisper_engine import get_whisper_engine
from app.models.schemas import (
    AudioEvent,
    InferenceStatus,
    TranscriptResult,
    TranscriptionQuality,
)

logger = get_logger("pipelines.transcription")


class TranscriptionPipeline:
    """
    Stateless transcription pipeline with Redis-backed caching.
    Caching key = SHA256 of (audio_bytes + language_hint) to handle
    the same audio with different language overrides separately.
    """

    def __init__(self, redis_client: Optional[aioredis.Redis] = None) -> None:
        self._settings = get_settings()
        self._whisper = get_whisper_engine()
        self._redis: Optional[aioredis.Redis] = redis_client

    def set_redis(self, redis_client: aioredis.Redis) -> None:
        """Inject Redis client (provided at startup after connection is established)."""
        self._redis = redis_client

    async def run(
        self,
        audio_bytes: bytes,
        event: AudioEvent,
    ) -> TranscriptResult:
        """
        Execute the full transcription pipeline.

        Args:
            audio_bytes: Decoded raw PCM-16 16kHz mono audio bytes.
            event:       The originating AudioEvent for metadata.

        Returns:
            TranscriptResult with full text, segments, quality, and latency.
        """
        set_request_context(
            session_id=event.session_id,
            user_id=event.user_id,
            pipeline="transcription",
        )

        pipeline_start = time.perf_counter()

        # ── 1. Cache lookup ────────────────────────────────────────────────
        cache_key = self._make_cache_key(audio_bytes, event)
        cached = await self._get_cached_transcript(cache_key)
        if cached is not None:
            logger.info(
                "Transcript cache hit.",
                extra={"event_id": event.event_id, "cache_key": cache_key[:16]},
            )
            return cached

        # ── 2. Whisper inference ────────────────────────────────────────────
        language = event.metadata.language_hint or self._settings.WHISPER_LANGUAGE
        try:
            result, latency_ms = await self._whisper.transcribe(
                audio_bytes=audio_bytes,
                source_sample_rate=event.metadata.sample_rate,
                num_channels=event.metadata.num_channels,
                encoding=event.metadata.encoding,
                language_hint=language,
            )
        except Exception as exc:
            logger.error(
                "Whisper transcription raised unexpected error.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            return TranscriptResult(
                status=InferenceStatus.FAILED,
                model_size=self._whisper.model_size,
                inference_device=self._whisper.device,
                error_message=str(exc),
            )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # ── 3. Quality gating ──────────────────────────────────────────────
        if result.status == InferenceStatus.SUCCESS:
            if result.quality == TranscriptionQuality.UNRELIABLE:
                logger.warning(
                    "Transcript quality UNRELIABLE; flagging but not dropping.",
                    extra={
                        "event_id": event.event_id,
                        "avg_log_prob": result.avg_log_prob,
                        "word_count": result.word_count,
                    },
                )
            elif result.word_count < self._settings.TRANSCRIPT_MIN_WORDS:
                logger.info(
                    "Transcript below minimum word count.",
                    extra={
                        "event_id": event.event_id,
                        "word_count": result.word_count,
                        "min_words": self._settings.TRANSCRIPT_MIN_WORDS,
                    },
                )
                result.status = InferenceStatus.INSUFFICIENT_AUDIO

        # ── 4. Cache the result ────────────────────────────────────────────
        if result.status == InferenceStatus.SUCCESS:
            await self._cache_transcript(cache_key, result)

        logger.info(
            "Transcription pipeline complete.",
            extra={
                "event_id": event.event_id,
                "word_count": result.word_count,
                "language": result.language_detected,
                "quality": result.quality.value if result.quality else None,
                "total_ms": round(total_ms, 2),
                "rtf": result.real_time_factor,
                "transcript_snippet": (result.full_text[:80] + "…")
                if result.full_text and len(result.full_text) > 80
                else result.full_text,
            },
        )

        return result

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _make_cache_key(self, audio_bytes: bytes, event: AudioEvent) -> str:
        """
        Deterministic cache key: SHA256 of audio bytes + language override.
        Tenant-scoped to prevent cross-tenant cache collisions.
        """
        lang = event.metadata.language_hint or self._settings.WHISPER_LANGUAGE or "auto"
        fingerprint = hashlib.sha256(audio_bytes + lang.encode()).hexdigest()
        return f"transcript:{event.tenant_id}:{fingerprint}"

    async def _get_cached_transcript(self, key: str) -> Optional[TranscriptResult]:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(key)
            if raw:
                data = json.loads(raw)
                return TranscriptResult(**data)
        except Exception as exc:
            logger.warning(
                "Redis transcript cache read failed.",
                extra={"error": str(exc)},
            )
        return None

    async def _cache_transcript(self, key: str, result: TranscriptResult) -> None:
        if not self._redis:
            return
        try:
            payload = result.model_dump_json()
            await self._redis.setex(
                key,
                self._settings.REDIS_TRANSCRIPT_TTL_SECONDS,
                payload,
            )
        except Exception as exc:
            logger.warning(
                "Redis transcript cache write failed.",
                extra={"error": str(exc)},
            )
