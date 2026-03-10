"""
SentinelAI - Email Analysis Orchestrator
==========================================
Wires all analysis modules into a single synchronous analysis pipeline:

  AuthHeaderParser      → score_auth   (SPF/DKIM/DMARC composite)
  HeaderAnomalyScanner  → score_header (structural anomaly density)
  DomainAnalyser        → score_domain (entropy + look-alike signals)
  URLAnalyser           → score_url    (URL threat signals)
  BodyHeuristicScanner  → score_body   (urgency, credential requests, HTML tricks)

Composite Score Weighting:
  score_auth   × 0.20  (inverted: 0.0=all pass, 1.0=all fail)
  score_header × 0.15
  score_domain × 0.25  (highest-risk domain across all extracted domains)
  score_url    × 0.25  (highest-risk URL)
  score_body   × 0.15

The composite score is forwarded to the Risk Fusion Engine as model_type=NLP_INTENT
because it captures the semantic / social-engineering quality of the communication
channel most similar to what DeBERTa-v3 would score for live voice/chat.

Risk Thresholds (aligned with Phase 3 FusionWeightConfig):
  CRITICAL  ≥ 0.85
  HIGH      ≥ 0.70
  MEDIUM    ≥ 0.50
  LOW       ≥ 0.30
  MINIMAL   <  0.30
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import DomainAnalyser
from sentinel_ai.services.email_analysis.core.heuristics.header_body_scanner import (
    BodyHeuristicScanner,
    HeaderAnomalyScanner,
)
from sentinel_ai.services.email_analysis.core.heuristics.url_analyser import URLAnalyser
from sentinel_ai.services.email_analysis.core.parsers.auth_parser import AuthHeaderParser
from sentinel_ai.services.email_analysis.schemas.analysis import (
    DomainRiskRecord,
    EmailAnalysisRequest,
    EmailAnalysisResult,
    FusionScorePayload,
    RiskLevel,
    ThreatIndicator,
    URLRiskRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composite Score Weights
# ---------------------------------------------------------------------------
_W_AUTH   = 0.20
_W_HEADER = 0.15
_W_DOMAIN = 0.25
_W_URL    = 0.25
_W_BODY   = 0.15

# Risk level thresholds
_T_CRITICAL = 0.85
_T_HIGH     = 0.70
_T_MEDIUM   = 0.50
_T_LOW      = 0.30


def _classify(score: float) -> RiskLevel:
    if score >= _T_CRITICAL: return RiskLevel.CRITICAL
    if score >= _T_HIGH:     return RiskLevel.HIGH
    if score >= _T_MEDIUM:   return RiskLevel.MEDIUM
    if score >= _T_LOW:      return RiskLevel.LOW
    return RiskLevel.MINIMAL


def _score_capped_sum(weights: list[float]) -> float:
    """Sums a list of severity weights, capped at 1.0."""
    return min(1.0, sum(weights))


class EmailAnalysisOrchestrator:
    """
    Orchestrates the full email phishing analysis pipeline.
    Stateless — one shared instance is created at application startup.
    All analysis is CPU-bound and synchronous; the FastAPI endpoint runs
    this inside asyncio.run_in_executor() to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        self._auth_parser       = AuthHeaderParser()
        self._header_scanner    = HeaderAnomalyScanner()
        self._domain_analyser   = DomainAnalyser()
        self._url_analyser      = URLAnalyser()
        self._body_scanner      = BodyHeuristicScanner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, request: EmailAnalysisRequest) -> EmailAnalysisResult:
        """
        Runs the full analysis pipeline for a single email submission.

        This method is synchronous (CPU-bound). Wrap in run_in_executor()
        when calling from an async FastAPI handler.

        Args:
            request: Validated EmailAnalysisRequest from the API endpoint.

        Returns:
            EmailAnalysisResult with all sub-scores and a composite score.
        """
        t_start = time.monotonic()

        # --- Phase 1: Authentication ---
        auth_headers = None
        score_auth   = 0.0

        if request.raw_headers:
            try:
                auth_headers = self._auth_parser.parse(request.raw_headers)
                # Invert auth_score: 1.0=all pass → 0.0 risk; 0.0=all fail → 1.0 risk
                score_auth = round(1.0 - auth_headers.auth_score, 4)
            except Exception:
                logger.error("Auth header parsing failed", exc_info=True)
                score_auth = 0.5  # Unknown = moderate risk

        # --- Phase 2: Header Anomalies ---
        header_anomalies = []
        score_header     = 0.0

        if request.raw_headers:
            try:
                header_anomalies = self._header_scanner.scan(request.raw_headers)
                score_header = _score_capped_sum(
                    [h.severity_weight for h in header_anomalies]
                )
                # Add auth-indicator anomalies if auth parsing ran
                if auth_headers and auth_headers.auth_indicators:
                    score_header = min(1.0, score_header + 0.10 * len(auth_headers.auth_indicators))
            except Exception:
                logger.error("Header anomaly scan failed", exc_info=True)

        # --- Phase 3: Domain Analysis ---
        all_domains       = self._extract_domains(request)
        domain_records: list[DomainRiskRecord] = []
        score_domain      = 0.0

        for domain in all_domains:
            try:
                record = self._domain_analyser.analyse(domain)
                domain_records.append(record)
            except Exception:
                logger.debug("Domain analysis failed", extra={"domain": domain}, exc_info=True)

        highest_risk_domain = max(domain_records, key=lambda d: d.risk_score, default=None)
        if highest_risk_domain:
            score_domain = highest_risk_domain.risk_score

        # --- Phase 4: URL Analysis ---
        # Merge caller-supplied URLs with server-extracted URLs
        extracted_urls = self._url_analyser.extract_urls(request.body_text, request.body_html)
        all_urls       = list(dict.fromkeys(request.urls + extracted_urls))[:500]

        url_records: list[URLRiskRecord] = []
        score_url = 0.0

        for url in all_urls:
            try:
                record = self._url_analyser.analyse_url(url)
                url_records.append(record)
            except Exception:
                logger.debug("URL analysis failed", extra={"url": url[:80]}, exc_info=True)

        # Link text mismatch check (HTML-only)
        link_text_mismatches = []
        if request.body_html:
            pairs = self._url_analyser.extract_link_text_pairs(request.body_html)
            link_text_mismatches = self._url_analyser.detect_link_text_mismatch(pairs)

        highest_risk_url = max(url_records, key=lambda u: u.risk_score, default=None)
        if highest_risk_url:
            score_url = highest_risk_url.risk_score

        # Boost URL score if link-text mismatches were found
        if link_text_mismatches:
            score_url = min(1.0, score_url + 0.15 * len(link_text_mismatches))

        # --- Phase 5: Body Heuristics ---
        body_heuristics = []
        score_body      = 0.0

        if request.body_text or request.body_html:
            try:
                body_heuristics = self._body_scanner.scan(
                    body_text       = request.body_text,
                    body_html       = request.body_html,
                    has_attachments = len(request.attachments_meta) > 0,
                    has_urls        = len(all_urls) > 0,
                )
                score_body = _score_capped_sum(
                    [b.severity_weight for b in body_heuristics]
                )
            except Exception:
                logger.error("Body heuristic scan failed", exc_info=True)

        # --- Phase 6: Composite Score ---
        composite = (
            score_auth   * _W_AUTH
            + score_header * _W_HEADER
            + score_domain * _W_DOMAIN
            + score_url    * _W_URL
            + score_body   * _W_BODY
        )
        composite_score = round(min(1.0, max(0.0, composite)), 4)
        risk_level = _classify(composite_score)

        # --- Aggregate all unique indicators ---
        all_indicators: set[ThreatIndicator] = set()

        if auth_headers:
            all_indicators.update(auth_headers.auth_indicators)
        for ha in header_anomalies:
            all_indicators.add(ha.indicator)
        for dr in domain_records:
            all_indicators.update(dr.indicators)
        for ur in url_records:
            all_indicators.update(ur.indicators)
        for bh in body_heuristics:
            all_indicators.add(bh.indicator)
        if link_text_mismatches:
            all_indicators.add(ThreatIndicator.BODY_LINK_TEXT_MISMATCH)

        # --- Threat Summary ---
        threat_summary = self._build_threat_summary(
            risk_level       = risk_level,
            composite_score  = composite_score,
            all_indicators   = list(all_indicators),
            highest_domain   = highest_risk_domain,
            highest_url      = highest_risk_url,
        )

        processing_ms = int((time.monotonic() - t_start) * 1000)

        result = EmailAnalysisResult(
            request_id          = request.request_id,
            session_id          = request.session_id,
            user_id             = request.user_id,
            organization_id     = request.organization_id,
            auth_headers        = auth_headers,
            domains_analyzed    = domain_records,
            highest_risk_domain = highest_risk_domain,
            urls_analyzed       = url_records,
            highest_risk_url    = highest_risk_url,
            header_anomalies    = header_anomalies,
            body_heuristics     = body_heuristics,
            score_auth          = round(score_auth, 4),
            score_domain        = round(score_domain, 4),
            score_url           = round(score_url, 4),
            score_header        = round(score_header, 4),
            score_body          = round(score_body, 4),
            composite_score     = composite_score,
            risk_level          = risk_level,
            urls_extracted      = len(all_urls),
            domains_extracted   = len(domain_records),
            processing_time_ms  = processing_ms,
            all_indicators      = sorted(all_indicators, key=lambda i: i.value),
            threat_summary      = threat_summary,
        )

        logger.info(
            "Email analysis complete",
            extra={
                "request_id":      request.request_id,
                "session_id":      request.session_id,
                "composite_score": composite_score,
                "risk_level":      risk_level.value,
                "score_auth":      score_auth,
                "score_domain":    score_domain,
                "score_url":       score_url,
                "score_header":    score_header,
                "score_body":      score_body,
                "indicators":      len(all_indicators),
                "processing_ms":   processing_ms,
            },
        )
        return result

    def build_fusion_payload(
        self,
        result: EmailAnalysisResult,
        expected_models: Optional[list[str]] = None,
    ) -> FusionScorePayload:
        """
        Constructs the FusionScorePayload forwarded to the Risk Fusion Engine.
        Maps to model_type='nlp_intent' in the fusion weight matrix.

        Args:
            result:          Completed EmailAnalysisResult.
            expected_models: List of model_type strings for the session
                             (e.g. ["deepfake_voice", "nlp_intent"]).

        Returns:
            FusionScorePayload ready for Kafka publication or REST POST.
        """
        return FusionScorePayload(
            session_id         = result.session_id or result.analysis_id,
            user_id            = result.user_id,
            organization_id    = result.organization_id,
            model_type         = "nlp_intent",
            model_version      = "email-heuristics-v1",
            confidence_score   = result.composite_score,
            processing_time_ms = result.processing_time_ms,
            expected_models    = expected_models or ["nlp_intent"],
            source_channel     = "email",
            model_metadata     = {
                "analysis_id":    result.analysis_id,
                "risk_level":     result.risk_level.value,
                "score_auth":     result.score_auth,
                "score_domain":   result.score_domain,
                "score_url":      result.score_url,
                "score_header":   result.score_header,
                "score_body":     result.score_body,
                "indicators":     [i.value for i in result.all_indicators],
                "highest_risk_domain": (
                    result.highest_risk_domain.domain
                    if result.highest_risk_domain else None
                ),
                "highest_risk_url": (
                    result.highest_risk_url.raw_url[:200]
                    if result.highest_risk_url else None
                ),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_domains(self, request: EmailAnalysisRequest) -> list[str]:
        """
        Extracts all unique domain names from headers, body text, and URLs.
        Returns a deduplicated list limited to 200 domains.
        """
        import re as _re
        domains: list[str] = []
        seen: set[str] = set()

        def _add(d: str) -> None:
            d = d.lower().strip().rstrip(".")
            if d and d not in seen and len(d) > 3:
                seen.add(d)
                domains.append(d)

        # From headers: extract From/Reply-To/Return-Path domains
        if request.raw_headers:
            for m in _re.finditer(
                r'@([\w\.\-]+\.[a-z]{2,})', request.raw_headers, _re.I
            ):
                _add(m.group(1))

        # From caller-supplied URLs
        for url in request.urls:
            try:
                import urllib.parse as _up
                parsed = _up.urlparse(url)
                if parsed.hostname:
                    _add(parsed.hostname)
            except Exception:
                pass

        # From extracted URLs in body
        for url in self._url_analyser.extract_urls(request.body_text, request.body_html):
            try:
                import urllib.parse as _up
                parsed = _up.urlparse(url)
                if parsed.hostname:
                    _add(parsed.hostname)
            except Exception:
                pass

        return domains[:200]

    def _build_threat_summary(
        self,
        risk_level:      RiskLevel,
        composite_score: float,
        all_indicators:  list[ThreatIndicator],
        highest_domain:  Optional[DomainRiskRecord],
        highest_url:     Optional[URLRiskRecord],
    ) -> str:
        """Builds a concise human-readable threat summary for the result."""
        parts: list[str] = [
            f"[{risk_level.value.upper()}] Composite score: {composite_score:.3f}.",
        ]

        if highest_domain and highest_domain.risk_score > 0.30:
            la = f" (look-alike of {highest_domain.lookalike_target})" \
                if highest_domain.is_lookalike else ""
            parts.append(
                f"Highest-risk domain: {highest_domain.domain}"
                f" (score={highest_domain.risk_score:.3f}, entropy={highest_domain.entropy:.2f}){la}."
            )

        if highest_url and highest_url.risk_score > 0.30:
            parts.append(
                f"Highest-risk URL: {highest_url.raw_url[:80]}..."
                f" (score={highest_url.risk_score:.3f})."
            )

        if all_indicators:
            sample = [i.value for i in all_indicators[:5]]
            parts.append(f"Signals: {', '.join(sample)}" + (
                f" (+{len(all_indicators)-5} more)" if len(all_indicators) > 5 else ""
            ) + ".")

        return " ".join(parts)
