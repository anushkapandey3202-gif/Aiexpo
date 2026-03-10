"""
SentinelAI - Email Authentication Header Parser
=================================================
Parses SPF, DKIM, and DMARC authentication results from raw email headers.

RFC references:
  SPF:   RFC 7208  — parsed from Authentication-Results / Received-SPF
  DKIM:  RFC 6376  — parsed from Authentication-Results (multiple sigs supported)
  DMARC: RFC 7489  — parsed from Authentication-Results

Authentication-Results header format (RFC 7001):
  Authentication-Results: mx.example.com;
    spf=pass smtp.mailfrom=sender@example.com;
    dkim=pass header.d=example.com header.s=selector1;
    dmarc=pass (policy=reject) header.from=example.com

Composite auth_score algorithm:
  Each of SPF, DKIM, DMARC contributes a sub-score (0.0–1.0).
  SPF:   pass=1.0, softfail=0.3, neutral=0.5, fail/permerror=0.0, missing=0.0
  DKIM:  pass=1.0, neutral=0.5, fail=0.0, missing=0.0
         If multiple DKIM signatures: max(sub-scores) is used
  DMARC: pass=1.0, fail=0.0, none=0.5, missing=0.0
         DMARC policy modifiers applied: quarantine=−0.1, reject=+0.1 on pass
  Composite = (spf_sub × 0.35) + (dkim_sub × 0.35) + (dmarc_sub × 0.30)
  auth_risk_score = 1.0 − composite  (high risk when auth fails)

Security hardening:
  - Checks for Authentication-Results header spoofing:
    The topmost Authentication-Results header should be stamped by the
    receiving MTA. If the 'authserv-id' does not match any Received header
    host, it may indicate header injection by the sender.
  - Detects multiple conflicting Authentication-Results headers.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from sentinel_ai.services.email_analysis.schemas.analysis import (
    DKIMRecord,
    DKIMResult,
    DMARCPolicy,
    DMARCRecord,
    DMARCResult,
    ParsedAuthHeaders,
    SPFRecord,
    SPFResult,
    ThreatIndicator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# SPF result from Authentication-Results
_RE_AUTH_SPF = re.compile(
    r"spf\s*=\s*(pass|fail|softfail|neutral|none|temperror|permerror)"
    r"(?:[^;]*smtp\.mailfrom\s*=\s*([^\s;]+))?",
    re.IGNORECASE,
)

# Received-SPF header (RFC 7208 §9.1) — fallback when Auth-Results absent
_RE_RECEIVED_SPF = re.compile(
    r"^Received-SPF:\s*(pass|fail|softfail|neutral|none|temperror|permerror)"
    r"(?:.*?smtp\.mailfrom\s*=\s*([^\s;)]+))?",
    re.IGNORECASE | re.MULTILINE,
)

# DKIM result from Authentication-Results (multiple may be present)
_RE_AUTH_DKIM = re.compile(
    r"dkim\s*=\s*(pass|fail|neutral|none|policy)"
    r"(?:[^;]*header\.d\s*=\s*([^\s;]+))?"
    r"(?:[^;]*header\.s\s*=\s*([^\s;]+))?",
    re.IGNORECASE,
)

# DMARC result from Authentication-Results
_RE_AUTH_DMARC = re.compile(
    r"dmarc\s*=\s*(pass|fail|none)"
    r"(?:[^;]*\(policy\s*=\s*(none|quarantine|reject)\))?"
    r"(?:[^;]*header\.from\s*=\s*([^\s;]+))?",
    re.IGNORECASE,
)

# Extract authserv-id (first token after "Authentication-Results:")
_RE_AUTHSERV_ID = re.compile(
    r"^Authentication-Results:\s*([^\s;]+)",
    re.IGNORECASE | re.MULTILINE,
)

# Received header "from" host extraction
_RE_RECEIVED_HOST = re.compile(
    r"^Received:\s+from\s+(\S+)",
    re.IGNORECASE | re.MULTILINE,
)

# From header display name vs. email extraction
_RE_FROM_DISPLAY = re.compile(
    r'^From:\s*(.+?)\s*<([^>]+)>',
    re.IGNORECASE | re.MULTILINE,
)
_RE_FROM_SIMPLE = re.compile(
    r'^From:\s*([^\s<>@]+@[^\s<>]+)',
    re.IGNORECASE | re.MULTILINE,
)

# Reply-To extraction
_RE_REPLY_TO = re.compile(
    r'^Reply-To:\s*.*?([^\s<>@]+@[^\s<>]+)',
    re.IGNORECASE | re.MULTILINE,
)

# Message-ID presence
_RE_MESSAGE_ID = re.compile(r'^Message-ID:', re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# SPF Sub-scores
# ---------------------------------------------------------------------------
_SPF_SUBSCORES: dict[str, float] = {
    SPFResult.PASS.value:      1.0,
    SPFResult.NEUTRAL.value:   0.5,
    SPFResult.SOFTFAIL.value:  0.3,
    SPFResult.NONE.value:      0.0,
    SPFResult.TEMPERROR.value: 0.1,
    SPFResult.PERMERROR.value: 0.0,
    SPFResult.FAIL.value:      0.0,
    SPFResult.MISSING.value:   0.0,
}

_DKIM_SUBSCORES: dict[str, float] = {
    DKIMResult.PASS.value:    1.0,
    DKIMResult.NEUTRAL.value: 0.5,
    DKIMResult.NONE.value:    0.0,
    DKIMResult.FAIL.value:    0.0,
    DKIMResult.POLICY.value:  0.2,
    DKIMResult.MISSING.value: 0.0,
}

_DMARC_SUBSCORES: dict[str, float] = {
    DMARCResult.PASS.value:    1.0,
    DMARCResult.NONE.value:    0.5,
    DMARCResult.FAIL.value:    0.0,
    DMARCResult.MISSING.value: 0.0,
}


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

class AuthHeaderParser:
    """
    Parses SPF, DKIM, and DMARC authentication results from raw email headers.

    Stateless — one instance is shared across all requests.
    """

    def parse(self, raw_headers: str) -> ParsedAuthHeaders:
        """
        Parses all authentication-related headers from a raw RFC 2822 header block.

        Args:
            raw_headers: Full email header block as a UTF-8 string.

        Returns:
            ParsedAuthHeaders with individual records and a composite auth_score.
        """
        indicators: list[ThreatIndicator] = []

        # Unfold multi-line header values (RFC 2822 §2.2.3)
        unfolded = re.sub(r'\r?\n[ \t]+', ' ', raw_headers)

        spf_record    = self._parse_spf(unfolded)
        dkim_records  = self._parse_dkim(unfolded)
        dmarc_record  = self._parse_dmarc(unfolded)

        # Header-level security checks
        spoof_indicator = self._check_auth_results_spoof(unfolded)
        if spoof_indicator:
            indicators.append(spoof_indicator)

        reply_to_indicator = self._check_reply_to_divergence(unfolded)
        if reply_to_indicator:
            indicators.append(reply_to_indicator)

        if not _RE_MESSAGE_ID.search(unfolded):
            indicators.append(ThreatIndicator.HEADER_MISSING_MESSAGE_ID)

        # Composite auth score
        auth_score, summary = self._compute_auth_score(
            spf_record, dkim_records, dmarc_record
        )

        return ParsedAuthHeaders(
            spf             = spf_record,
            dkim            = dkim_records,
            dmarc           = dmarc_record,
            auth_score      = auth_score,
            auth_indicators = indicators,
            summary         = summary,
        )

    # ------------------------------------------------------------------
    # SPF
    # ------------------------------------------------------------------

    def _parse_spf(self, headers: str) -> SPFRecord:
        """Extracts SPF result from Authentication-Results, then Received-SPF fallback."""

        # Primary: Authentication-Results
        for auth_block in self._extract_auth_results_blocks(headers):
            m = _RE_AUTH_SPF.search(auth_block)
            if m:
                return SPFRecord(
                    result       = SPFResult(m.group(1).lower()),
                    smtp_mailfrom= m.group(2),
                    raw_header   = auth_block[:256],
                )

        # Fallback: Received-SPF
        m = _RE_RECEIVED_SPF.search(headers)
        if m:
            return SPFRecord(
                result       = SPFResult(m.group(1).lower()),
                smtp_mailfrom= m.group(2),
                raw_header   = m.group(0)[:256],
            )

        return SPFRecord(result=SPFResult.MISSING)

    # ------------------------------------------------------------------
    # DKIM
    # ------------------------------------------------------------------

    def _parse_dkim(self, headers: str) -> list[DKIMRecord]:
        """
        Extracts all DKIM results. Multiple DKIM signatures are common for
        forwarded mail (original sender + forwarder both sign).
        """
        records: list[DKIMRecord] = []
        for auth_block in self._extract_auth_results_blocks(headers):
            for m in _RE_AUTH_DKIM.finditer(auth_block):
                records.append(DKIMRecord(
                    result    = DKIMResult(m.group(1).lower()),
                    domain    = m.group(2),
                    selector  = m.group(3),
                    raw_header= auth_block[:256],
                ))

        if not records:
            records.append(DKIMRecord(result=DKIMResult.MISSING))

        return records

    # ------------------------------------------------------------------
    # DMARC
    # ------------------------------------------------------------------

    def _parse_dmarc(self, headers: str) -> DMARCRecord:
        """Extracts DMARC alignment result from Authentication-Results."""
        for auth_block in self._extract_auth_results_blocks(headers):
            m = _RE_AUTH_DMARC.search(auth_block)
            if m:
                policy_str = m.group(2).lower() if m.group(2) else "missing"
                try:
                    policy = DMARCPolicy(policy_str)
                except ValueError:
                    policy = DMARCPolicy.MISSING

                return DMARCRecord(
                    result     = DMARCResult(m.group(1).lower()),
                    policy     = policy,
                    from_domain= m.group(3),
                    raw_header = auth_block[:256],
                )

        return DMARCRecord(result=DMARCResult.MISSING, policy=DMARCPolicy.MISSING)

    # ------------------------------------------------------------------
    # Composite Auth Score
    # ------------------------------------------------------------------

    def _compute_auth_score(
        self,
        spf:    SPFRecord,
        dkims:  list[DKIMRecord],
        dmarc:  DMARCRecord,
    ) -> tuple[float, str]:
        """
        Computes the 0.0–1.0 composite authentication score.
        Higher score = stronger authentication = LOWER risk.
        """
        spf_sub   = _SPF_SUBSCORES.get(spf.result.value, 0.0)
        dkim_sub  = max((_DKIM_SUBSCORES.get(d.result.value, 0.0) for d in dkims), default=0.0)
        dmarc_sub = _DMARC_SUBSCORES.get(dmarc.result.value, 0.0)

        # DMARC policy modifier
        if dmarc.result == DMARCResult.PASS:
            if dmarc.policy == DMARCPolicy.REJECT:
                dmarc_sub = min(1.0, dmarc_sub + 0.1)
            elif dmarc.policy == DMARCPolicy.QUARANTINE:
                dmarc_sub = max(0.0, dmarc_sub - 0.05)

        composite = (spf_sub * 0.35) + (dkim_sub * 0.35) + (dmarc_sub * 0.30)

        parts = [
            f"SPF:{spf.result.value}({spf_sub:.2f})",
            f"DKIM:{dkims[0].result.value}({dkim_sub:.2f})" if dkims else "DKIM:missing(0)",
            f"DMARC:{dmarc.result.value}({dmarc_sub:.2f})",
            f"composite:{composite:.3f}",
        ]
        summary = " | ".join(parts)
        return round(composite, 4), summary

    # ------------------------------------------------------------------
    # Security Checks
    # ------------------------------------------------------------------

    def _check_auth_results_spoof(
        self, headers: str
    ) -> Optional[ThreatIndicator]:
        """
        Detects potential Authentication-Results header spoofing.
        If the authserv-id doesn't appear in any Received header, a malicious
        sender may have injected a forged authentication result.
        """
        authserv_ids = _RE_AUTHSERV_ID.findall(headers)
        received_hosts = set(_RE_RECEIVED_HOST.findall(headers))

        if len(authserv_ids) > 3:
            # Unusual number of Authentication-Results headers
            logger.debug(
                "Multiple Authentication-Results headers detected",
                extra={"count": len(authserv_ids)},
            )
            return ThreatIndicator.HEADER_AUTH_RESULTS_SPOOF

        if authserv_ids and received_hosts:
            # Check if the topmost authserv-id appears in Received chain
            topmost_id = authserv_ids[0].lower()
            if not any(topmost_id in h.lower() for h in received_hosts):
                logger.debug(
                    "Authentication-Results authserv-id not in Received chain",
                    extra={"authserv_id": topmost_id},
                )
                return ThreatIndicator.HEADER_AUTH_RESULTS_SPOOF

        return None

    def _check_reply_to_divergence(
        self, headers: str
    ) -> Optional[ThreatIndicator]:
        """
        Detects diverging From and Reply-To domains — a common phishing technique
        where the display name shows a trusted brand but replies go to an attacker domain.
        """
        from_match = _RE_FROM_SIMPLE.search(headers) or _RE_FROM_DISPLAY.search(headers)
        reply_match = _RE_REPLY_TO.search(headers)

        if not (from_match and reply_match):
            return None

        from_email = (
            from_match.group(2) if from_match.lastindex and from_match.lastindex >= 2
            else from_match.group(1)
        )
        reply_email = reply_match.group(1)

        def _extract_domain(email: str) -> str:
            return email.split("@")[-1].lower().strip(">").strip()

        from_domain  = _extract_domain(from_email)
        reply_domain = _extract_domain(reply_email)

        if from_domain and reply_domain and from_domain != reply_domain:
            logger.debug(
                "Reply-To domain diverges from From domain",
                extra={"from": from_domain, "reply_to": reply_domain},
            )
            return ThreatIndicator.HEADER_REPLY_TO_DIVERGE

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_auth_results_blocks(self, headers: str) -> list[str]:
        """
        Extracts all Authentication-Results header values from the header block.
        Returns a list of raw header value strings (header name stripped).
        """
        blocks: list[str] = []
        for m in re.finditer(
            r'^Authentication-Results:\s*(.+?)(?=\n\S|\Z)',
            headers,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        ):
            blocks.append(m.group(1).strip())
        return blocks
