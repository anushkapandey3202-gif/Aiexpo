"""
SentinelAI ML Service — Deepfake / Anti-Spoofing Detection Pipeline

Orchestrates RawNet3 anti-spoofing inference and maps raw logits
to a typed DeepfakeResult with threat banding and threshold decisions.
"""

from __future__ import annotations

import time

from app.core.config import get_settings
from app.core.logging import get_logger, set_request_context
from app.inference.rawnet3 import get_rawnet3_inference
from app.models.schemas import AudioEvent, DeepfakeResult, InferenceStatus

logger = get_logger("pipelines.deepfake")


class DeepfakePipeline:
    """
    Stateless deepfake detection pipeline backed by RawNet3.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._rawnet3 = get_rawnet3_inference()

    async def run(self, audio_bytes: bytes, event: AudioEvent) -> DeepfakeResult:
        """
        Execute RawNet3 anti-spoofing inference.

        Args:
            audio_bytes: Raw 16kHz mono PCM-16 audio.
            event:       Originating AudioEvent for logging context.

        Returns:
            DeepfakeResult with spoof probability and is_deepfake flag.
        """
        set_request_context(
            session_id=event.session_id,
            user_id=event.user_id,
            pipeline="deepfake_detection",
        )

        pipeline_start = time.perf_counter()

        # ── RawNet3 inference ──────────────────────────────────────────────
        try:
            spoof_prob, genuine_prob, rawnet_latency_ms = await self._rawnet3.predict(
                audio_bytes
            )
        except Exception as exc:
            logger.error(
                "RawNet3 inference failed.",
                extra={"event_id": event.event_id, "error": str(exc)},
                exc_info=True,
            )
            return DeepfakeResult(
                status=InferenceStatus.FAILED,
                spoof_probability=0.0,
                genuine_probability=1.0,
                is_deepfake=False,
                model_version=self._rawnet3.model_version,
                inference_device=self._rawnet3._device_manager.device_name,
                error_message=f"RawNet3 inference error: {exc}",
            )

        # ── Threshold classification ───────────────────────────────────────
        is_deepfake = spoof_prob >= self._settings.DEEPFAKE_THRESHOLD_WARN
        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            "Deepfake pipeline complete.",
            extra={
                "event_id": event.event_id,
                "spoof_probability": round(spoof_prob, 4),
                "genuine_probability": round(genuine_prob, 4),
                "is_deepfake": is_deepfake,
                "rawnet_latency_ms": round(rawnet_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
            },
        )

        return DeepfakeResult(
            status=InferenceStatus.SUCCESS,
            spoof_probability=round(spoof_prob, 6),
            genuine_probability=round(genuine_prob, 6),
            is_deepfake=is_deepfake,
            model_version=self._rawnet3.model_version,
            inference_device=self._rawnet3._device_manager.device_name,
            inference_latency_ms=round(total_latency_ms, 3),
        )
