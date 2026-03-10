"""
SentinelAI ML Service — Voiceprint Authentication Pipeline

Orchestrates the full voiceprint authentication flow:
1. Extract 192-dim speaker embedding via ECAPA-TDNN.
2. Query Pinecone for top-K cosine-similar enrolled voiceprints.
3. Apply threshold decision logic to produce an AuthDecision.
4. Return a fully typed VoiceprintResult.
"""

from __future__ import annotations

import time
from typing import Optional

from app.core.config import get_settings
from app.core.logging import PerformanceLogger, get_logger, set_request_context
from app.inference.ecapa_tdnn import get_ecapa_inference
from app.models.schemas import (
    AudioEvent,
    AuthDecision,
    InferenceStatus,
    VoiceprintMatch,
    VoiceprintResult,
)
from app.vector_store.pinecone_client import get_pinecone_client

logger = get_logger("pipelines.voiceprint")


class VoiceprintPipeline:
    """
    Stateless voiceprint authentication pipeline.
    Instantiated once; all state flows through method arguments.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._ecapa = get_ecapa_inference()
        self._pinecone = get_pinecone_client()

    async def run(
        self,
        audio_bytes: bytes,
        event: AudioEvent,
    ) -> VoiceprintResult:
        """
        Execute the full voiceprint authentication pipeline.

        Args:
            audio_bytes: Decoded raw PCM-16 16kHz mono audio.
            event:       The originating AudioEvent (for speaker ID and context).

        Returns:
            VoiceprintResult with decision, top matches, and latency.
        """
        set_request_context(
            session_id=event.session_id,
            user_id=event.user_id,
            pipeline="voiceprint_auth",
        )

        pipeline_start = time.perf_counter()

        # ── Step 1: ECAPA-TDNN embedding extraction ────────────────────────
        try:
            embedding, ecapa_latency_ms = await self._ecapa.extract_embedding(audio_bytes)
        except Exception as exc:
            logger.error(
                "ECAPA-TDNN embedding extraction failed.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            return VoiceprintResult(
                status=InferenceStatus.FAILED,
                decision=AuthDecision.INSUFFICIENT_AUDIO,
                inference_device=self._ecapa._device_manager.device_name,
                error_message=f"Embedding extraction failed: {exc}",
            )

        logger.debug(
            "Embedding extracted.",
            extra={
                "event_id": event.event_id,
                "embedding_norm": float((embedding ** 2).sum() ** 0.5),
                "ecapa_latency_ms": round(ecapa_latency_ms, 2),
            },
        )

        # ── Step 2: Pinecone cosine similarity search ──────────────────────
        # If a reference speaker_id is provided, filter the search space.
        speaker_filter: Optional[str] = event.reference_speaker_id
        try:
            top_matches = await self._pinecone.query(
                embedding=embedding,
                speaker_id_filter=speaker_filter,
            )
        except Exception as exc:
            logger.error(
                "Pinecone query failed.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            top_matches: list[VoiceprintMatch] = []

        # ── Step 3: Threshold decision ─────────────────────────────────────
        decision, best_score, matched_speaker = self._make_decision(
            matches=top_matches,
            reference_speaker_id=event.reference_speaker_id,
        )

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            "Voiceprint pipeline complete.",
            extra={
                "event_id": event.event_id,
                "decision": decision.value,
                "best_score": round(best_score or 0.0, 4),
                "total_latency_ms": round(total_latency_ms, 2),
                "num_matches": len(top_matches),
            },
        )

        return VoiceprintResult(
            status=InferenceStatus.SUCCESS,
            decision=decision,
            speaker_id=matched_speaker,
            top_matches=top_matches,
            best_score=best_score,
            embedding_dim=len(embedding),
            inference_device=self._ecapa._device_manager.device_name,
            inference_latency_ms=round(total_latency_ms, 3),
        )

    def _make_decision(
        self,
        matches: list[VoiceprintMatch],
        reference_speaker_id: Optional[str],
    ) -> tuple[AuthDecision, Optional[float], Optional[str]]:
        """
        Apply threshold logic to Pinecone matches.

        Decision rules:
        - If no matches returned → UNKNOWN_SPEAKER.
        - If best cosine score ≥ PINECONE_SCORE_THRESHOLD:
            - If reference_speaker_id provided, verify the top match IS that speaker.
            - Otherwise accept highest-scoring enrolled speaker → AUTHENTICATED.
        - If best cosine score < threshold → UNAUTHENTICATED.
        """
        if not matches:
            return AuthDecision.UNKNOWN_SPEAKER, None, None

        best_match = matches[0]
        best_score = best_match.cosine_score

        if best_score < self._settings.PINECONE_SCORE_THRESHOLD:
            return AuthDecision.UNAUTHENTICATED, best_score, None

        # Score above threshold — check identity constraint if reference provided
        if reference_speaker_id is not None:
            if best_match.speaker_id == reference_speaker_id:
                return AuthDecision.AUTHENTICATED, best_score, best_match.speaker_id
            else:
                # Matches *someone* enrolled, but not the expected speaker
                logger.warning(
                    "Speaker mismatch: high-score match is not the reference speaker.",
                    extra={
                        "reference_speaker_id": reference_speaker_id,
                        "matched_speaker_id": best_match.speaker_id,
                        "score": round(best_score, 4),
                    },
                )
                return AuthDecision.UNAUTHENTICATED, best_score, best_match.speaker_id

        return AuthDecision.AUTHENTICATED, best_score, best_match.speaker_id
