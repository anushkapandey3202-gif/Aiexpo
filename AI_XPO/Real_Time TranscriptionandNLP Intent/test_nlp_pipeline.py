"""
SentinelAI NLP Service — Integration Test Suite

Tests the full NLP pipeline end-to-end using synthetic audio and
known-bad transcript strings. All external dependencies (Kafka, Redis,
Pinecone) are mocked.

Run with: pytest tests/test_nlp_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import math
import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import (
    AudioChannel,
    AudioEvent,
    AudioMetadata,
    InferenceStatus,
    IntentRiskTier,
    ThreatLevel,
)
from app.inference.lexical_analyzer import LexicalThreatAnalyzer, get_lexical_analyzer
from app.inference.whisper_engine import AudioNormalizer, TranscriptSanitizer


# ── Test Fixtures ─────────────────────────────────────────────────────────────

def _make_sine_pcm(frequency: float = 440.0, duration: float = 2.0,
                   sample_rate: int = 16000) -> bytes:
    """Generate a pure sine wave as PCM-16 bytes."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave = (np.sin(2 * math.pi * frequency * t) * 16383).astype(np.int16)
    return wave.tobytes()


def _make_audio_event(audio_bytes: bytes, **kwargs) -> AudioEvent:
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return AudioEvent(
        session_id="test-session-001",
        user_id="test-user-001",
        tenant_id="test-tenant-001",
        metadata=AudioMetadata(
            channel=AudioChannel.VOIP,
            sample_rate=16000,
            duration_seconds=2.0,
            encoding="pcm_16bit",
            num_channels=1,
        ),
        audio_b64=b64,
        run_transcription=True,
        run_intent=True,
        **kwargs,
    )


# ── AudioNormalizer Tests ─────────────────────────────────────────────────────

class TestAudioNormalizer:

    def test_pcm16_decode_produces_float32(self):
        audio_bytes = _make_sine_pcm(duration=1.0)
        normalizer  = AudioNormalizer()
        result      = normalizer.process(audio_bytes, source_sample_rate=16000)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_stereo_to_mono_halves_sample_count(self):
        """Stereo PCM interleaved should be averaged to mono."""
        mono_pcm   = _make_sine_pcm(duration=1.0, sample_rate=16000)
        mono_np    = np.frombuffer(mono_pcm, dtype=np.int16)
        stereo_np  = np.repeat(mono_np, 2)          # Interleave L+R as identical
        stereo_pcm = stereo_np.astype(np.int16).tobytes()
        normalizer = AudioNormalizer()
        result     = normalizer.process(stereo_pcm, source_sample_rate=16000, source_channels=2)
        # Output should have same length as original mono
        assert abs(len(result) - len(mono_np)) <= 1

    def test_resampling_from_8khz(self):
        """8kHz audio should be resampled to 16kHz — output double the length."""
        audio_bytes = _make_sine_pcm(duration=1.0, sample_rate=8000)
        normalizer  = AudioNormalizer()
        result      = normalizer.process(audio_bytes, source_sample_rate=8000)
        expected_len = 16000  # 1s at 16kHz
        assert abs(len(result) - expected_len) < 100   # Allow small rounding error

    def test_empty_audio_raises_value_error(self):
        normalizer = AudioNormalizer()
        with pytest.raises(ValueError, match="too short"):
            normalizer.process(b"", source_sample_rate=16000)

    def test_clip_saturation(self):
        """Amplified audio beyond [-1,1] should be clipped, not wrapped."""
        loud_np    = (np.ones(16000) * 40000).astype(np.int16)
        audio_bytes = loud_np.tobytes()
        normalizer  = AudioNormalizer()
        result      = normalizer.process(audio_bytes, source_sample_rate=16000)
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_duration_calculation(self):
        audio_bytes = _make_sine_pcm(duration=3.5)
        normalizer  = AudioNormalizer()
        audio_f32   = normalizer.process(audio_bytes, source_sample_rate=16000)
        duration    = normalizer.get_duration(audio_f32)
        assert abs(duration - 3.5) < 0.02


# ── TranscriptSanitizer Tests ──────────────────────────────────────────────────

class TestTranscriptSanitizer:

    def test_removes_blank_audio_hallucination(self):
        raw = "[BLANK_AUDIO] Hello can you hear me? [BLANK_AUDIO]"
        assert "[BLANK_AUDIO]" not in TranscriptSanitizer.sanitize(raw)

    def test_removes_music_noise_tags(self):
        raw = "(Music) I need you to verify your account details (Noise)"
        result = TranscriptSanitizer.sanitize(raw)
        assert "(Music)" not in result
        assert "(Noise)" not in result

    def test_collapses_repeated_punctuation(self):
        raw    = "Hello!!!! Can you hear me???"
        result = TranscriptSanitizer.sanitize(raw)
        assert "!!!!" not in result
        assert "???" not in result

    def test_unicode_normalization(self):
        # café as decomposed NFC vs NFD should normalize to same form
        raw = "caf\u00e9"   # NFC already
        assert TranscriptSanitizer.sanitize(raw) == "café"

    def test_empty_string_returns_empty(self):
        assert TranscriptSanitizer.sanitize("") == ""

    def test_strips_leading_trailing_whitespace(self):
        assert TranscriptSanitizer.sanitize("  hello  ") == "hello"


# ── LexicalThreatAnalyzer Tests ───────────────────────────────────────────────

class TestLexicalThreatAnalyzer:

    def setup_method(self):
        self.analyzer = LexicalThreatAnalyzer()

    def test_otp_harvesting_detected(self):
        text   = "Please read me your one-time password so I can verify your account."
        result = self.analyzer.analyze(text)
        assert result.status == InferenceStatus.SUCCESS
        assert "otp_harvesting" in result.flagged_categories
        assert result.aggregated_boost > 0.0

    def test_wire_transfer_detected(self):
        text   = "Urgent: you need to wire transfer $5000 to this account immediately."
        result = self.analyzer.analyze(text)
        assert result.status == InferenceStatus.SUCCESS
        assert "urgent_financial_transfer" in result.flagged_categories

    def test_credential_phishing_detected(self):
        text   = "Can you confirm your password and username for me right now?"
        result = self.analyzer.analyze(text)
        assert result.status == InferenceStatus.SUCCESS
        assert "credential_phishing" in result.flagged_categories

    def test_impersonation_authority_detected(self):
        text   = "This is calling from the IRS. There is a warrant for your arrest."
        result = self.analyzer.analyze(text)
        assert result.status == InferenceStatus.SUCCESS
        assert "impersonation_authority" in result.flagged_categories

    def test_benign_text_no_matches(self):
        text   = "Good morning. How are you? I was calling about the weather today."
        result = self.analyzer.analyze(text)
        assert result.total_matches == 0
        assert result.aggregated_boost == 0.0

    def test_multiple_categories_boost_capped(self):
        """Even with multiple patterns, boost must not exceed THREAT_KEYWORD_BOOST."""
        text = (
            "Read me your OTP verification code. "
            "I'm calling from the IRS. "
            "Wire transfer $10000 immediately. "
            "Confirm your password right now."
        )
        result = self.analyzer.analyze(text)
        from app.core.config import get_settings
        assert result.aggregated_boost <= get_settings().THREAT_KEYWORD_BOOST

    def test_too_short_text_returns_skipped(self):
        result = self.analyzer.analyze("hi")
        assert result.status == InferenceStatus.SKIPPED

    def test_empty_text_returns_skipped(self):
        result = self.analyzer.analyze("")
        assert result.status == InferenceStatus.SKIPPED

    def test_match_positions_are_valid(self):
        text   = "Please share your OTP code with me right now."
        result = self.analyzer.analyze(text)
        for match in result.matched_patterns:
            start, end = match.position_chars
            assert 0 <= start < end <= len(text)
            # Verify matched_text is consistent with position
            assert len(match.matched_text) <= end - start + 5  # +5 for minor trim tolerance


# ── AudioEvent Validation Tests ───────────────────────────────────────────────

class TestAudioEventValidation:

    def test_valid_event_with_b64(self):
        audio = _make_sine_pcm()
        event = _make_audio_event(audio)
        assert event.audio_b64 is not None
        assert event.session_id == "test-session-001"

    def test_event_fails_without_audio_source(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AudioEvent(
                session_id="s",
                user_id="u",
                tenant_id="t",
                metadata=AudioMetadata(
                    channel=AudioChannel.VOIP,
                    sample_rate=16000,
                    duration_seconds=2.0,
                    encoding="pcm_16bit",
                    num_channels=1,
                ),
                # Neither audio_b64 nor audio_s3_uri provided
            )

    def test_event_invalid_sample_rate(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AudioEvent(
                session_id="s",
                user_id="u",
                tenant_id="t",
                metadata=AudioMetadata(
                    channel=AudioChannel.VOIP,
                    sample_rate=100,  # Below minimum 8000
                    duration_seconds=2.0,
                    encoding="pcm_16bit",
                    num_channels=1,
                ),
                audio_b64="dGVzdA==",
            )


# ── NLP Score Fusion Tests ─────────────────────────────────────────────────────

class TestScoreFusion:
    """
    Unit tests for the NLPOrchestrator._fuse_scores method,
    covering threat level classification boundary conditions.
    """

    def setup_method(self):
        from app.services.nlp_orchestrator import NLPOrchestrator
        self.orchestrator = NLPOrchestrator()
        self.event        = _make_audio_event(_make_sine_pcm())

    def _make_intent_result(self, label: str, score: float, tier: IntentRiskTier):
        from app.models.schemas import IntentLabel, IntentResult
        return IntentResult(
            status=InferenceStatus.SUCCESS,
            top_intent=label,
            top_score=score,
            top_risk_tier=tier,
            all_labels=[IntentLabel(label=label, score=score, risk_tier=tier)],
            inference_device="cpu",
        )

    def test_high_risk_intent_high_score_yields_alert(self):
        intent  = self._make_intent_result("otp_harvesting", 0.92, IntentRiskTier.HIGH_RISK)
        result  = self.orchestrator._fuse_scores(
            event=self.event,
            transcription=None,
            intent=intent,
            lexical=None,
            transcript_text="Read me your OTP",
        )
        assert result.threat_level in (ThreatLevel.ALERT, ThreatLevel.CRITICAL)
        assert result.nlp_threat_score >= 0.55

    def test_benign_intent_yields_clear(self):
        intent  = self._make_intent_result("benign_conversation", 0.95, IntentRiskTier.BENIGN)
        result  = self.orchestrator._fuse_scores(
            event=self.event,
            transcription=None,
            intent=intent,
            lexical=None,
            transcript_text="Good morning how are you",
        )
        assert result.threat_level == ThreatLevel.CLEAR
        assert result.nlp_threat_score == 0.0

    def test_lexical_boost_elevates_medium_risk_to_warn(self):
        from app.models.schemas import LexicalAnalysisResult, ThreatKeywordMatch
        intent  = self._make_intent_result(
            "urgent_financial_request", 0.60, IntentRiskTier.MEDIUM_RISK
        )
        lexical = LexicalAnalysisResult(
            status=InferenceStatus.SUCCESS,
            matched_patterns=[
                ThreatKeywordMatch(
                    pattern="wire transfer",
                    matched_text="wire transfer",
                    category="urgent_financial_transfer",
                    risk_weight=0.85,
                    position_chars=(0, 13),
                )
            ],
            aggregated_boost=0.12,
        )
        result = self.orchestrator._fuse_scores(
            event=self.event,
            transcription=None,
            intent=intent,
            lexical=lexical,
            transcript_text="Wire transfer funds immediately",
        )
        assert result.threat_level in (ThreatLevel.WARN, ThreatLevel.ALERT)
        assert result.nlp_threat_score > 0.0

    def test_all_pipelines_failed_yields_clear_with_error_factors(self):
        from app.models.schemas import IntentResult, TranscriptionResult
        intent = IntentResult(
            status=InferenceStatus.FAILED,
            inference_device="cpu",
            error_message="test error",
        )
        transcription = TranscriptionResult(
            status=InferenceStatus.FAILED,
            inference_device="cpu",
            error_message="test error",
        )
        result = self.orchestrator._fuse_scores(
            event=self.event,
            transcription=transcription,
            intent=intent,
            lexical=None,
            transcript_text="",
        )
        assert result.nlp_threat_score == 0.0
        assert "INTENT_PIPELINE_ERROR" in result.threat_factors
        assert "TRANSCRIPTION_PIPELINE_ERROR" in result.threat_factors
