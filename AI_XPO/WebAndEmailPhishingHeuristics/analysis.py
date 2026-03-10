"""
SentinelAI - Email & Web Analysis Schemas
==========================================
All Pydantic v2 contracts for the email/web phishing analysis service.

Inbound (REST API):
  EmailAnalysisRequest  — raw email headers + body + URLs submitted for analysis

Intermediate:
  ParsedAuthHeaders     — structured SPF/DKIM/DMARC results
  DomainRiskRecord      — entropy + look-alike + typosquat signals for one domain
  URLRiskRecord         — per-URL threat signals (redirect chain, homograph, etc.)
  HeaderAnomalyRecord   — individual header-level red flag

Outbound (to Risk Fusion Engine):
  EmailAnalysisResult   — full analysis result (also returned to API caller)
  FusionScorePayload    — minimal ThreatScoreEvent-compatible structure for Kafka

Score taxonomy (all 0.0–1.0):
  auth_score            — SPF/DKIM/DMARC composite (1.0 = all pass, 0.0 = all fail/missing)
  domain_risk_score     — highest domain entropy/look-alike score across all domains
  url_risk_score        — highest URL threat signal across all extracted URLs
  header_anomaly_score  — density of suspicious header patterns
  body_risk_score       — NLP-free heuristic signals in body text
  composite_score       — weighted aggregate forwarded to Risk Fusion Engine
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SPFResult(str, enum.Enum):
    PASS         = "pass"
    FAIL         = "fail"
    SOFTFAIL     = "softfail"
    NEUTRAL      = "neutral"
    NONE         = "none"
    TEMPERROR    = "temperror"
    PERMERROR    = "permerror"
    MISSING      = "missing"


class DKIMResult(str, enum.Enum):
    PASS    = "pass"
    FAIL    = "fail"
    NEUTRAL = "neutral"
    NONE    = "none"
    POLICY  = "policy"
    MISSING = "missing"


class DMARCResult(str, enum.Enum):
    PASS    = "pass"
    FAIL    = "fail"
    NONE    = "none"
    MISSING = "missing"


class DMARCPolicy(str, enum.Enum):
    NONE        = "none"
    QUARANTINE  = "quarantine"
    REJECT      = "reject"
    MISSING     = "missing"


class ThreatIndicator(str, enum.Enum):
    """Enumerated threat signal labels attached to domain/URL findings."""
    # Domain signals
    HIGH_ENTROPY            = "high_entropy"
    LOOKALIKE_DOMAIN        = "lookalike_domain"
    TYPOSQUATTING           = "typosquatting"
    HOMOGRAPH_ATTACK        = "homograph_attack"
    SUBDOMAIN_ABUSE         = "subdomain_abuse"
    NEWLY_REGISTERED        = "newly_registered"
    PUNYCODE_DOMAIN         = "punycode_domain"
    TLD_MISMATCH            = "tld_mismatch"
    # URL signals
    URL_REDIRECT_CHAIN      = "url_redirect_chain"
    URL_IP_ADDRESS          = "url_ip_address"
    URL_EXCESSIVE_SUBDOMAINS= "url_excessive_subdomains"
    URL_CREDENTIAL_KEYWORDS = "url_credential_keywords"
    URL_ENCODED_PAYLOAD     = "url_encoded_payload"
    URL_SHORTENER           = "url_shortener"
    URL_SUSPICIOUS_PORT     = "url_suspicious_port"
    URL_HOMOGRAPH           = "url_homograph"
    # Header signals
    HEADER_FROM_DISPLAY_SPOOF   = "header_from_display_spoof"
    HEADER_REPLY_TO_DIVERGE     = "header_reply_to_diverge"
    HEADER_RECEIVED_ANOMALY     = "header_received_anomaly"
    HEADER_FORGED_DATE          = "header_forged_date"
    HEADER_MISSING_MESSAGE_ID   = "header_missing_message_id"
    HEADER_AUTH_RESULTS_SPOOF   = "header_auth_results_spoof"
    HEADER_X_MAILER_SUSPICIOUS  = "header_x_mailer_suspicious"
    # Body signals
    BODY_URGENCY_LANGUAGE       = "body_urgency_language"
    BODY_CREDENTIAL_REQUEST     = "body_credential_request"
    BODY_SENSITIVE_DATA_REQUEST = "body_sensitive_data_request"
    BODY_LINK_TEXT_MISMATCH     = "body_link_text_mismatch"
    BODY_EXCESSIVE_IMAGES       = "body_excessive_images"
    BODY_HTML_OBFUSCATION       = "body_html_obfuscation"


class RiskLevel(str, enum.Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    MINIMAL  = "minimal"


# ---------------------------------------------------------------------------
# Inbound Request
# ---------------------------------------------------------------------------

class EmailAnalysisRequest(BaseModel):
    """
    Raw email submission for phishing / social-engineering analysis.

    The caller may submit any subset of headers, body, and URLs.
    At least one of raw_headers, body_text, or urls must be provided.

    Fields:
        session_id:       Optional correlation ID (from ingestion pipeline)
        user_id:          Platform user receiving the email
        organization_id:  Tenant isolation key
        raw_headers:      Full RFC 2822 header block as a single string
        body_text:        Plain-text body (HTML is stripped server-side)
        body_html:        Optional raw HTML body for HTML-specific analysis
        urls:             Pre-extracted URL list (supplemented by server extraction)
        sender_ip:        IP address of the sending MTA (from Received headers)
        attachments_meta: List of attachment metadata dicts (name, mime_type, size)
    """
    request_id:        str  = Field(default_factory=lambda: str(uuid4()))
    session_id:        Optional[str] = None
    user_id:           str  = Field(..., min_length=1)
    organization_id:   str  = Field(..., min_length=1)

    raw_headers:       Optional[str] = Field(
        None,
        max_length=524_288,  # 512 KB max header block
        description="Full RFC 2822 email header block",
    )
    body_text:         Optional[str] = Field(
        None,
        max_length=5_242_880,  # 5 MB plain-text limit
        description="Plain-text body content",
    )
    body_html:         Optional[str] = Field(
        None,
        max_length=5_242_880,
        description="Raw HTML body for HTML-specific heuristics",
    )
    urls:              list[str] = Field(
        default_factory=list,
        max_length=500,
        description="Pre-extracted URLs (supplemented by server-side extraction)",
    )
    sender_ip:         Optional[str] = Field(
        None,
        description="Connecting MTA IP (first untrusted hop from Received chain)",
    )
    attachments_meta:  list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=50,
    )

    @model_validator(mode="after")
    def _require_at_least_one_input(self) -> "EmailAnalysisRequest":
        if not any([self.raw_headers, self.body_text, self.body_html, self.urls]):
            raise ValueError(
                "At least one of raw_headers, body_text, body_html, or urls must be provided"
            )
        return self

    @field_validator("urls", mode="before")
    @classmethod
    def _deduplicate_urls(cls, v: list) -> list:
        seen: set = set()
        deduped = []
        for url in v:
            if url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped[:500]


# ---------------------------------------------------------------------------
# Authentication Header Results
# ---------------------------------------------------------------------------

class SPFRecord(BaseModel):
    """Parsed SPF authentication result."""
    result:       SPFResult
    domain:       Optional[str] = None
    smtp_mailfrom: Optional[str] = None
    raw_header:   Optional[str] = None


class DKIMRecord(BaseModel):
    """Parsed DKIM authentication result — one per DKIM-Signature."""
    result:       DKIMResult
    domain:       Optional[str]  = None
    selector:     Optional[str]  = None
    header_b:     Optional[str]  = None  # Truncated signature value
    raw_header:   Optional[str]  = None


class DMARCRecord(BaseModel):
    """Parsed DMARC authentication result."""
    result:       DMARCResult
    policy:       DMARCPolicy    = DMARCPolicy.MISSING
    from_domain:  Optional[str]  = None
    pct:          Optional[int]  = Field(None, ge=0, le=100,
                                        description="DMARC policy application percentage")
    raw_header:   Optional[str]  = None


class ParsedAuthHeaders(BaseModel):
    """Aggregated SPF/DKIM/DMARC authentication results for a single email."""
    spf:            SPFRecord
    dkim:           list[DKIMRecord]  = Field(default_factory=list)
    dmarc:          DMARCRecord
    auth_score:     float             = Field(..., ge=0.0, le=1.0,
        description="Composite authentication score: 1.0=all pass, 0.0=all fail/missing")
    auth_indicators: list[ThreatIndicator] = Field(default_factory=list)
    summary:        str               = ""


# ---------------------------------------------------------------------------
# Domain Risk Record
# ---------------------------------------------------------------------------

class DomainRiskRecord(BaseModel):
    """
    Per-domain threat signal bundle.
    Produced for every unique domain extracted from headers, body, and URLs.
    """
    domain:              str
    entropy:             float = Field(..., ge=0.0,
        description="Shannon entropy of the domain label (bits/char)")
    is_lookalike:        bool  = False
    lookalike_target:    Optional[str] = Field(None,
        description="Brand/domain this is a look-alike of (e.g. 'paypal.com')")
    lookalike_algorithm: Optional[str] = Field(None,
        description="Detection algorithm: 'levenshtein'|'homograph'|'keyboard'|'soundex'")
    edit_distance:       Optional[int] = Field(None, ge=0)
    is_punycode:         bool  = False
    decoded_punycode:    Optional[str] = None
    has_mixed_scripts:   bool  = False   # e.g. Cyrillic 'а' in Latin domain
    tld:                 Optional[str]  = None
    registered_domain:  Optional[str]  = None
    subdomain_depth:     int   = Field(0, ge=0)
    risk_score:          float = Field(..., ge=0.0, le=1.0)
    indicators:          list[ThreatIndicator] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# URL Risk Record
# ---------------------------------------------------------------------------

class URLRiskRecord(BaseModel):
    """Per-URL threat signal bundle."""
    raw_url:                 str
    normalized_url:          Optional[str]  = None
    scheme:                  Optional[str]  = None
    host:                    Optional[str]  = None
    path:                    Optional[str]  = None
    is_ip_address:           bool  = False
    is_punycode:             bool  = False
    is_shortener:            bool  = False
    shortener_service:       Optional[str]  = None
    redirect_chain_depth:    int   = Field(0, ge=0)
    has_credential_keywords: bool  = False
    credential_keywords_found: list[str] = Field(default_factory=list)
    has_encoded_payload:     bool  = False
    excessive_subdomains:    bool  = False
    suspicious_port:         Optional[int]  = None
    domain_record:           Optional[DomainRiskRecord] = None
    risk_score:              float = Field(..., ge=0.0, le=1.0)
    indicators:              list[ThreatIndicator] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Header Anomaly Record
# ---------------------------------------------------------------------------

class HeaderAnomalyRecord(BaseModel):
    """A single anomalous email header finding."""
    header_name:    str
    header_value:   str = Field(..., max_length=2048)
    indicator:      ThreatIndicator
    description:    str
    severity_weight: float = Field(..., ge=0.0, le=1.0,
        description="Contribution weight of this anomaly to header_anomaly_score")


# ---------------------------------------------------------------------------
# Body Heuristic Record
# ---------------------------------------------------------------------------

class BodyHeuristicRecord(BaseModel):
    """Body-level phishing signal."""
    indicator:          ThreatIndicator
    description:        str
    matched_pattern:    Optional[str]  = Field(None,
        description="Sanitized snippet showing what triggered this signal (no sensitive data)")
    severity_weight:    float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Full Analysis Result
# ---------------------------------------------------------------------------

class EmailAnalysisResult(BaseModel):
    """
    Complete email/web phishing analysis result.
    Returned directly to the API caller AND forwarded to the Risk Fusion Engine.
    """
    analysis_id:        str     = Field(default_factory=lambda: str(uuid4()))
    request_id:         str
    session_id:         Optional[str]
    user_id:            str
    organization_id:    str
    analyzed_at:        datetime = Field(default_factory=datetime.utcnow)

    # --- Authentication ---
    auth_headers:       Optional[ParsedAuthHeaders] = None

    # --- Domain Analysis ---
    domains_analyzed:   list[DomainRiskRecord]      = Field(default_factory=list)
    highest_risk_domain: Optional[DomainRiskRecord] = None

    # --- URL Analysis ---
    urls_analyzed:      list[URLRiskRecord]          = Field(default_factory=list)
    highest_risk_url:   Optional[URLRiskRecord]      = None

    # --- Header Anomalies ---
    header_anomalies:   list[HeaderAnomalyRecord]   = Field(default_factory=list)

    # --- Body Heuristics ---
    body_heuristics:    list[BodyHeuristicRecord]   = Field(default_factory=list)

    # --- Composite Scoring ---
    score_auth:         float = Field(0.0, ge=0.0, le=1.0,
        description="Inverted auth score: 0.0=all auth pass, 1.0=all auth fail")
    score_domain:       float = Field(0.0, ge=0.0, le=1.0)
    score_url:          float = Field(0.0, ge=0.0, le=1.0)
    score_header:       float = Field(0.0, ge=0.0, le=1.0)
    score_body:         float = Field(0.0, ge=0.0, le=1.0)
    composite_score:    float = Field(0.0, ge=0.0, le=1.0,
        description="Weighted composite — forwarded to Risk Fusion Engine as NLP_INTENT score")
    risk_level:         RiskLevel = RiskLevel.MINIMAL

    # --- Processing Metadata ---
    urls_extracted:     int   = 0
    domains_extracted:  int   = 0
    processing_time_ms: int   = 0
    all_indicators:     list[ThreatIndicator] = Field(default_factory=list)
    threat_summary:     str   = ""


# ---------------------------------------------------------------------------
# Risk Fusion Engine Payload
# ---------------------------------------------------------------------------

class FusionScorePayload(BaseModel):
    """
    Minimal ThreatScoreEvent-compatible payload published to Kafka
    (or POSTed to the Risk Fusion Engine REST fallback) after analysis.
    Maps to ModelType.NLP_INTENT in the fusion weight matrix.
    """
    model_config = {"frozen": True}

    event_id:          str   = Field(default_factory=lambda: str(uuid4()))
    session_id:        str
    user_id:           str
    organization_id:   str
    model_type:        str   = "nlp_intent"
    model_version:     str   = "email-heuristics-v1"
    confidence_score:  float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int
    expected_models:   list[str] = Field(default_factory=list)
    source_channel:    str   = "email"
    model_metadata:    dict[str, Any] = Field(default_factory=dict)
    published_at:      datetime = Field(default_factory=datetime.utcnow)
