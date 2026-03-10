"""
SentinelAI NLP Service — Intent Classification Pipeline

Two-layer intent detection:
  1. DeBERTa-v3 sequence classifier (probabilistic, learns from training data)
  2. KeywordDetector rule engine (deterministic, zero-shot explainability)

Fusion strategy:
  - If DeBERTa confidence ≥ threshold AND KeywordDetector agrees → high trust
  - If DeBERTa confidence < threshold AND keywords fire → elevate to keyword score
  - If DeBERTa flags threat but no keywords → moderate trust (may be paraphrase)
  - BENIGN requires both layers to agree
"""

from __future__ import annotations

import time
from typing import List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger, set_request_context
from app.inference.deberta_classifier import get_deberta_classifier, get_keyword_detector
from app.models.schemas import (
    AudioEvent,
    InferenceStatus,
    IntentLabel,
    IntentPrediction,
    IntentResult,
    ThreatIndicator,
    TranscriptResult,
)

logger = get_logger("pipelines.intent")


class IntentClassificationPipeline:
    """
    Stateless intent classification pipeline.
    Fuses DeBERTa-v3 model output with keyword heuristics.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._classifier = get_deberta_classifier()
        self._keyword_detector = get_keyword_detector()

    async def run(
        self,
        transcript: TranscriptResult,
        event: AudioEvent,
    ) -> IntentResult:
        """
        Execute intent classification on a transcribed text.

        Args:
            transcript: Output from the TranscriptionPipeline.
            event:      Originating AudioEvent for context logging.

        Returns:
            IntentResult with primary intent, top-K predictions, and threat indicators.
        """
        set_request_context(
            session_id=event.session_id,
            user_id=event.user_id,
            pipeline="intent_classification",
        )

        # ── Guard: skip if transcript is not usable ────────────────────────
        if not transcript.is_usable:
            logger.info(
                "Transcript not usable; skipping intent classification.",
                extra={
                    "event_id": event.event_id,
                    "transcript_status": transcript.status.value,
                    "word_count": transcript.word_count,
                    "quality": transcript.quality.value if transcript.quality else None,
                },
            )
            return IntentResult(
                status=InferenceStatus.SKIPPED,
                primary_intent=IntentLabel.UNCERTAIN.value,
                model_name=self._classifier.model_name,
                error_message="Transcript not usable for classification.",
            )

        pipeline_start = time.perf_counter()
        text = transcript.full_text

        # ── Layer 1: DeBERTa-v3 classification ─────────────────────────────
        model_predictions: List[IntentPrediction] = []
        token_count: int = 0
        model_latency_ms: float = 0.0

        try:
            model_predictions, token_count, model_latency_ms = (
                await self._classifier.classify(text)
            )
        except Exception as exc:
            logger.error(
                "DeBERTa classification failed.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            model_predictions = []

        # ── Layer 2: Keyword / heuristic detection ──────────────────────────
        try:
            threat_indicators: List[ThreatIndicator] = self._keyword_detector.detect(text)
        except Exception as exc:
            logger.warning(
                "Keyword detector failed.",
                extra={"event_id": event.event_id, "error": str(exc)},
            )
            threat_indicators = []

        # ── Fusion: combine both signals ────────────────────────────────────
        primary_intent, primary_confidence, is_threat, threat_signal = (
            self._fuse_signals(model_predictions, threat_indicators)
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            "Intent classification complete.",
            extra={
                "event_id": event.event_id,
                "primary_intent": primary_intent,
                "primary_confidence": round(primary_confidence, 4),
                "is_threat": is_threat,
                "threat_signal": round(threat_signal, 4),
                "keyword_hits": len(threat_indicators),
                "token_count": token_count,
                "total_ms": round(total_ms, 2),
            },
        )

        return IntentResult(
            status=InferenceStatus.SUCCESS,
            primary_intent=primary_intent,
            primary_confidence=round(primary_confidence, 6),
            top_predictions=model_predictions,
            threat_indicators=threat_indicators,
            is_threat=is_threat,
            threat_signal=round(threat_signal, 6),
            model_name=self._classifier.model_name,
            inference_device=str(self._classifier._device),
            inference_latency_ms=round(total_ms, 3),
            input_token_count=token_count,
        )

    # ── Signal fusion logic ───────────────────────────────────────────────────

    def _fuse_signals(
        self,
        model_preds: List[IntentPrediction],
        keyword_indicators: List[ThreatIndicator],
    ) -> tuple[str, float, bool, float]:
        """
        Combine DeBERTa predictions and keyword indicators into a final decision.

        Returns:
            (primary_intent, primary_confidence, is_threat, threat_signal)
        """
        settings = self._settings
        threshold = settings.INTENT_CONFIDENCE_THRESHOLD
        threat_set = settings.threat_label_set

        # ── DeBERTa signal ─────────────────────────────────────────────────
        model_primary_label = IntentLabel.UNCERTAIN.value
        model_primary_conf = 0.0
        model_threat_signal = 0.0

        if model_preds:
            top = model_preds[0]
            model_primary_label = top.label
            model_primary_conf = top.confidence

            # Aggregate confidence across all threat-labeled predictions
            model_threat_signal = max(
                (p.confidence for p in model_preds if p.label in threat_set),
                default=0.0,
            )

        # ── Keyword signal ─────────────────────────────────────────────────
        keyword_threat_signal = 0.0
        keyword_primary_label = None
        if keyword_indicators:
            best_indicator = max(keyword_indicators, key=lambda i: i.confidence)
            keyword_threat_signal = best_indicator.confidence
            keyword_primary_label = best_indicator.indicator_type

        # ── Fusion rules ───────────────────────────────────────────────────

        # Rule 1: Model is confident AND model says threat
        if model_primary_conf >= threshold and model_primary_label in threat_set:
            # Keyword corroboration boosts confidence slightly
            boost = 0.05 if keyword_threat_signal > 0.5 else 0.0
            final_conf = min(model_primary_conf + boost, 1.0)
            final_signal = max(model_threat_signal, keyword_threat_signal * 0.5)
            return model_primary_label, final_conf, True, final_signal

        # Rule 2: Model is uncertain BUT keywords fire strongly
        if model_primary_conf < threshold and keyword_threat_signal >= 0.70:
            # Use keyword label as primary, model as supporting evidence
            label = keyword_primary_label or IntentLabel.SOCIAL_ENGINEERING.value
            conf = keyword_threat_signal
            signal = max(keyword_threat_signal, model_threat_signal * 0.6)
            return label, conf, True, signal

        # Rule 3: Model says BENIGN with high confidence, no keyword hits
        if (
            model_primary_label == IntentLabel.BENIGN.value
            and model_primary_conf >= threshold
            and keyword_threat_signal < 0.5
        ):
            return IntentLabel.BENIGN.value, model_primary_conf, False, 0.0

        # Rule 4: Model flags threat below threshold — low-confidence warning
        if model_primary_label in threat_set and model_primary_conf > 0.40:
            return (
                model_primary_label,
                model_primary_conf,
                True,
                model_primary_conf,
            )

        # Rule 5: Keyword-only, moderate confidence
        if keyword_threat_signal >= 0.50:
            label = keyword_primary_label or IntentLabel.SOCIAL_ENGINEERING.value
            return label, keyword_threat_signal, True, keyword_threat_signal

        # Default: UNCERTAIN / BENIGN with combined low signal
        combined_signal = max(model_threat_signal, keyword_threat_signal) * 0.5
        final_label = (
            model_primary_label
            if model_primary_label != IntentLabel.UNCERTAIN.value
            else IntentLabel.BENIGN.value
        )
        return final_label, model_primary_conf, False, combined_signal
