"""
SentinelAI NLP Service — Whisper Transcription Engine

Uses faster-whisper (CTranslate2 backend) which provides:
  • 2–4× faster inference than the original OpenAI Whisper
  • Lower memory footprint with INT8 quantization on CPU
  • Built-in Voice Activity Detection (VAD) via Silero VAD
  • Word-level timestamps for downstream NLP alignment
  • Identical accuracy to the original Whisper at equivalent model sizes

Architecture:
  WhisperEngine  — singleton managing model lifecycle and warm-up
  AudioNormalizer — validates and resamples raw PCM bytes
  QualityEvaluator — maps Whisper's avg_logprob → TranscriptionQuality
  WhisperPipeline  — async public interface consumed by the orchestrator
"""

from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.core.logging import PerformanceLogger, get_logger
from app.models.schemas import (
    InferenceStatus,
    TranscriptResult,
    TranscriptionQuality,
    TranscriptionSegment,
)

logger = get_logger("inference.whisper")


# ── Quality Evaluator ─────────────────────────────────────────────────────────

class QualityEvaluator:
    """
    Maps Whisper's segment-level log-probability statistics to a
    discrete TranscriptionQuality band.

    Thresholds calibrated against Whisper's own filtering in
    whisper/transcribe.py (logprob_threshold=-1.0, no_speech_threshold=0.6).
    """

    @staticmethod
    def evaluate(avg_log_prob: float, no_speech_prob: float) -> TranscriptionQuality:
        if no_speech_prob > 0.80 or avg_log_prob <= -1.0:
            return TranscriptionQuality.UNRELIABLE
        if avg_log_prob > -0.3 and no_speech_prob < 0.3:
            return TranscriptionQuality.HIGH
        if avg_log_prob > -0.6 and no_speech_prob < 0.5:
            return TranscriptionQuality.MEDIUM
        return TranscriptionQuality.LOW


# ── Audio Normalizer ──────────────────────────────────────────────────────────

class AudioNormalizer:
    """
    Converts raw audio bytes into a float32 numpy array normalized to
    the sample rate and channel count required by faster-whisper (16kHz mono).

    Supports PCM-16, PCM-32, and float32 raw encoding.
    Multi-channel audio is mixed down to mono via channel averaging.
    """

    TARGET_SR = 16000

    def normalize(
        self,
        audio_bytes: bytes,
        source_sample_rate: int = 16000,
        num_channels: int = 1,
        encoding: str = "pcm_16bit",
    ) -> np.ndarray:
        """
        Returns float32 numpy array, shape [n_samples], range [-1.0, 1.0].
        """
        audio_np = self._decode(audio_bytes, encoding)
        audio_np = self._to_mono(audio_np, num_channels)
        if source_sample_rate != self.TARGET_SR:
            audio_np = self._resample(audio_np, source_sample_rate, self.TARGET_SR)
        return audio_np.astype(np.float32)

    @staticmethod
    def _decode(audio_bytes: bytes, encoding: str) -> np.ndarray:
        """Decode raw bytes to float32 based on encoding format."""
        if encoding in ("pcm_16bit", "pcm_s16le"):
            arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            return arr / 32768.0
        if encoding in ("pcm_32bit", "pcm_s32le"):
            arr = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32)
            return arr / 2147483648.0
        if encoding == "float32":
            return np.frombuffer(audio_bytes, dtype=np.float32).copy()
        # Fallback: attempt pcm_16bit
        logger.warning("Unknown encoding '%s'; attempting pcm_16bit decode.", encoding)
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        return arr / 32768.0

    @staticmethod
    def _to_mono(audio: np.ndarray, num_channels: int) -> np.ndarray:
        """Mix down multi-channel audio to mono."""
        if num_channels <= 1:
            return audio
        if len(audio) % num_channels != 0:
            trim = len(audio) - (len(audio) % num_channels)
            audio = audio[:trim]
        return audio.reshape(-1, num_channels).mean(axis=1)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Linear-interpolation resampling (scipy preferred; numpy fallback)."""
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
        except ImportError:
            # numpy fallback: linear interpolation
            target_len = int(len(audio) * target_sr / orig_sr)
            return np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)


# ── Whisper Engine (faster-whisper backend) ───────────────────────────────────

class WhisperEngine:
    """
    Singleton managing faster-whisper model lifecycle:
    loading, warm-up, and thread-pool inference.

    faster-whisper uses CTranslate2 for efficient CPU/GPU inference.
    INT8 quantization on CPU reduces latency by ~3× vs. FP32.
    FP16 on GPU matches OpenAI Whisper accuracy with 2× speed.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model = None
        self._normalizer = AudioNormalizer()
        self._quality_eval = QualityEvaluator()
        self._device: str = "cpu"
        self._compute_type: str = "int8"
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._model_size = self._settings.WHISPER_MODEL_SIZE.value

    def load(self) -> None:
        """
        Load the faster-whisper WhisperModel.
        Resolves compute device and quantization type automatically.
        Called once at service startup in a thread executor.
        """
        import torch

        # Resolve device
        pref = self._settings.DEVICE_PREFERENCE.value
        if pref == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
            self._compute_type = "float16"
        elif pref == "mps" and torch.backends.mps.is_available():
            # faster-whisper doesn't support MPS natively; fall back to CPU
            self._device = "cpu"
            self._compute_type = "int8"
        elif pref in ("cpu", "auto"):
            if pref == "auto" and torch.cuda.is_available():
                self._device = "cuda"
                self._compute_type = "float16"
            else:
                self._device = "cpu"
                self._compute_type = "int8"
        else:
            self._device = "cpu"
            self._compute_type = "int8"

        # Override with explicit config if provided
        if self._settings.WHISPER_COMPUTE_TYPE:
            self._compute_type = self._settings.WHISPER_COMPUTE_TYPE

        logger.info(
            "Loading faster-whisper model.",
            extra={
                "model_size": self._model_size,
                "device": self._device,
                "compute_type": self._compute_type,
                "model_path": self._settings.WHISPER_MODEL_PATH,
            },
        )

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                model_size_or_path=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._settings.WHISPER_MODEL_PATH,
                num_workers=self._settings.WHISPER_NUM_WORKERS,
                # CPU thread tuning
                cpu_threads=4,
            )
            logger.info("faster-whisper model loaded successfully.")
        except ImportError:
            logger.warning(
                "faster-whisper not installed. Using mock transcription engine. "
                "Install: pip install faster-whisper",
            )
            self._model = None

        self._semaphore = asyncio.Semaphore(self._settings.MAX_CONCURRENT_INFERENCES)
        self._warmup()

    def _warmup(self) -> None:
        """
        Single short transcription to prime CTranslate2 kernel cache
        and avoid first-request latency spike.
        """
        if self._model is None:
            logger.debug("Whisper warm-up skipped (mock mode).")
            return
        try:
            # 0.5s of silence at 16kHz = 8000 samples
            dummy_audio = np.zeros(8000, dtype=np.float32)
            segments, _ = self._model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                vad_filter=False,
            )
            # Consume the generator to actually run inference
            list(segments)
            logger.debug("Whisper warm-up complete.", extra={"device": self._device})
        except Exception as exc:
            logger.warning("Whisper warm-up failed.", extra={"error": str(exc)})

    async def transcribe(
        self,
        audio_bytes: bytes,
        source_sample_rate: int = 16000,
        num_channels: int = 1,
        encoding: str = "pcm_16bit",
        language_hint: Optional[str] = None,
    ) -> Tuple[TranscriptResult, float]:
        """
        Async transcription of raw audio bytes.

        Args:
            audio_bytes:        Raw PCM audio bytes.
            source_sample_rate: Original sample rate of the audio.
            num_channels:       Number of audio channels (1=mono, 2=stereo).
            encoding:           PCM encoding type.
            language_hint:      BCP-47 language code override (None = auto-detect).

        Returns:
            (TranscriptResult, latency_ms)
        """
        if self._semaphore is None:
            raise RuntimeError("WhisperEngine.load() must be called before transcription.")

        loop = asyncio.get_event_loop()
        async with self._semaphore:
            start = time.perf_counter()

            # Normalize audio in thread executor (CPU-bound numpy ops)
            audio_array = await loop.run_in_executor(
                None,
                lambda: self._normalizer.normalize(
                    audio_bytes, source_sample_rate, num_channels, encoding
                ),
            )

            result = await loop.run_in_executor(
                None,
                lambda: self._sync_transcribe(
                    audio_array, language_hint or self._settings.WHISPER_LANGUAGE
                ),
            )

            latency_ms = (time.perf_counter() - start) * 1000

        result.inference_latency_ms = round(latency_ms, 3)
        result.audio_duration_seconds = len(audio_array) / AudioNormalizer.TARGET_SR
        return result, latency_ms

    def _sync_transcribe(
        self, audio: np.ndarray, language: Optional[str]
    ) -> TranscriptResult:
        """
        Synchronous faster-whisper transcription — runs in thread executor.
        Returns TranscriptResult without latency fields (set by caller).
        """
        if self._model is None:
            return self._mock_transcribe(audio)

        settings = self._settings
        try:
            segments_generator, info = self._model.transcribe(
                audio,
                language=language,
                task=settings.WHISPER_TASK,
                beam_size=settings.WHISPER_BEAM_SIZE,
                best_of=settings.WHISPER_BEST_OF,
                temperature=settings.WHISPER_TEMPERATURE,
                condition_on_previous_text=settings.WHISPER_CONDITION_ON_PREV_TOKENS,
                word_timestamps=settings.WHISPER_WORD_TIMESTAMPS,
                vad_filter=settings.WHISPER_VAD_FILTER,
                vad_parameters={
                    "threshold": settings.WHISPER_VAD_THRESHOLD,
                    "min_silence_duration_ms": settings.WHISPER_MIN_SILENCE_DURATION_MS,
                },
            )

            # Materialize the lazy generator (actual GPU/CPU compute happens here)
            raw_segments = list(segments_generator)

            segments: List[TranscriptionSegment] = []
            all_log_probs: List[float] = []
            all_no_speech: List[float] = []

            for i, seg in enumerate(raw_segments):
                t_seg = TranscriptionSegment(
                    id=i,
                    start_seconds=round(seg.start, 3),
                    end_seconds=round(seg.end, 3),
                    text=seg.text.strip(),
                    avg_log_prob=round(float(seg.avg_logprob), 4),
                    no_speech_prob=round(float(seg.no_speech_prob), 4),
                    compression_ratio=round(float(seg.compression_ratio), 4),
                    words=[
                        {
                            "word": w.word,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(float(w.probability), 4),
                        }
                        for w in (seg.words or [])
                    ] if settings.WHISPER_WORD_TIMESTAMPS and seg.words else None,
                )
                segments.append(t_seg)
                all_log_probs.append(t_seg.avg_log_prob)
                all_no_speech.append(t_seg.no_speech_prob)

            full_text = " ".join(s.text for s in segments if s.text).strip()
            avg_log_prob = float(np.mean(all_log_probs)) if all_log_probs else -0.5
            avg_no_speech = float(np.mean(all_no_speech)) if all_no_speech else 0.0
            quality = self._quality_eval.evaluate(avg_log_prob, avg_no_speech)

            return TranscriptResult(
                status=InferenceStatus.SUCCESS,
                full_text=full_text,
                language_detected=info.language,
                language_probability=round(float(info.language_probability), 4),
                segments=segments,
                avg_log_prob=round(avg_log_prob, 4),
                quality=quality,
                model_size=self._model_size,
                inference_device=self._device,
            )

        except Exception as exc:
            logger.error(
                "Whisper inference failed.",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return TranscriptResult(
                status=InferenceStatus.FAILED,
                model_size=self._model_size,
                inference_device=self._device,
                error_message=str(exc),
            )

    def _mock_transcribe(self, audio: np.ndarray) -> TranscriptResult:
        """
        Graceful mock for development/testing without faster-whisper installed.
        Returns a plausible-structure result with clear mock indicators.
        """
        duration = len(audio) / AudioNormalizer.TARGET_SR
        mock_text = (
            "[MOCK TRANSCRIPT] Please send me your one-time password immediately. "
            "Your account will be suspended unless you verify your identity right now."
        )
        return TranscriptResult(
            status=InferenceStatus.SUCCESS,
            full_text=mock_text,
            language_detected="en",
            language_probability=0.99,
            segments=[
                TranscriptionSegment(
                    id=0,
                    start_seconds=0.0,
                    end_seconds=duration,
                    text=mock_text,
                    avg_log_prob=-0.25,
                    no_speech_prob=0.02,
                    compression_ratio=1.1,
                )
            ],
            avg_log_prob=-0.25,
            quality=TranscriptionQuality.HIGH,
            model_size=f"mock-{self._model_size}",
            inference_device=self._device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._semaphore is not None

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def device(self) -> str:
        return self._device


# ── Module-level singleton ────────────────────────────────────────────────────

_whisper_engine: Optional[WhisperEngine] = None


def get_whisper_engine() -> WhisperEngine:
    global _whisper_engine
    if _whisper_engine is None:
        _whisper_engine = WhisperEngine()
    return _whisper_engine
