"""
SentinelAI - Header Anomaly & Body Heuristics Engine
=====================================================
Two cooperating analysers for email structural signals:

HeaderAnomalyScanner:
  Detects suspicious patterns in raw email headers beyond SPF/DKIM/DMARC.
  Checks performed:
    1. From display-name ≠ actual email address (brand spoofing in display name)
    2. Received header anomalies (loop, non-routable IPs, excessive hops)
    3. Forged/future Date header
    4. Missing or malformed Message-ID
    5. Suspicious X-Mailer / User-Agent (known spam tools)
    6. X-Originating-IP: private/loopback address inconsistency

BodyHeuristicScanner:
  Applies rule-based heuristics to plain-text and HTML email bodies.
  These are NLP-free — no model inference, just pattern matching.
  Checks:
    1. Urgency language density (account suspended, verify immediately, etc.)
    2. Credential/sensitive data requests (OTP, SSN, account number, etc.)
    3. HTML obfuscation indicators (CSS display:none, zero-size font, etc.)
    4. Attachment + link combination (a common malware delivery pattern)
    5. Excessive image-to-text ratio (image-only phishing)
    6. Call-to-action imperative phrasing density

Both scanners return a list of HeaderAnomalyRecord / BodyHeuristicRecord
objects, each with a severity_weight. The calling orchestrator sums the
weights (capped at 1.0) to produce score_header and score_body.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from sentinel_ai.services.email_analysis.schemas.analysis import (
    BodyHeuristicRecord,
    HeaderAnomalyRecord,
    ThreatIndicator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header Patterns
# ---------------------------------------------------------------------------

_RE_FROM_DISPLAY = re.compile(
    r'^From:\s*"?([^"<\n]+?)"?\s*<([^>]+)>',
    re.IGNORECASE | re.MULTILINE,
)
_RE_DATE_HEADER = re.compile(
    r'^Date:\s*(.+)',
    re.IGNORECASE | re.MULTILINE,
)
_RE_MESSAGE_ID = re.compile(r'^Message-ID:\s*<[^>]+>', re.IGNORECASE | re.MULTILINE)
_RE_RECEIVED = re.compile(r'^Received:', re.IGNORECASE | re.MULTILINE)
_RE_RECEIVED_FROM_IP = re.compile(
    r'^Received:.*?\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]',
    re.IGNORECASE | re.MULTILINE,
)
_RE_X_MAILER = re.compile(r'^X-Mailer:\s*(.+)', re.IGNORECASE | re.MULTILINE)
_RE_X_ORIG_IP = re.compile(
    r'^X-Originating-IP:\s*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
    re.IGNORECASE | re.MULTILINE,
)

# Known suspicious mailer strings (bulk senders, spam tools)
_SUSPICIOUS_MAILERS: frozenset[str] = frozenset({
    "phpmailer", "swiftmailer", "sendgrid", "mailchimp",
    "massmailer", "bulk", "blaster", "anonymous",
    "email marketing", "campaignmonitor",
})

# Private / loopback IP ranges
_RE_PRIVATE_IP = re.compile(
    r'^(10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+'
    r'|192\.168\.\d+\.\d+|127\.0\.0\.1|::1)$'
)


# ---------------------------------------------------------------------------
# Body Patterns
# ---------------------------------------------------------------------------

# Urgency language (weighted trigger groups)
_URGENCY_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r'\b(immediately|urgent|as soon as possible|right away|without delay)\b', re.I), 0.12),
    (re.compile(r'\b(account (has been|will be) (suspended|terminated|disabled|blocked))\b', re.I), 0.20),
    (re.compile(r'\b(verify your account|confirm your (identity|account|email))\b', re.I), 0.18),
    (re.compile(r'\b(limited time|expires? (in|within)|last (chance|warning))\b', re.I), 0.10),
    (re.compile(r'\b(unusual (activity|sign.in|login)|suspicious (activity|access))\b', re.I), 0.15),
    (re.compile(r'\b(take action now|click (here|below|the link) (to|now))\b', re.I), 0.12),
    (re.compile(r'\b(your (account|password) (will|has|must)|security alert)\b', re.I), 0.14),
]

# Credential / sensitive data request patterns
_CREDENTIAL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r'\b(enter your (password|pin|otp|one.time (password|code)))\b', re.I), 0.30),
    (re.compile(r'\b(provide your (social security|SSN|bank account|routing number|credit card))\b', re.I), 0.35),
    (re.compile(r'\b(update your (payment|billing|credit card|bank) (information|details|info))\b', re.I), 0.25),
    (re.compile(r'\b(reply with your (password|pin|token|code|secret))\b', re.I), 0.40),
    (re.compile(r'\b(your (OTP|one.time (code|password)) is\b)', re.I), 0.10),  # Legitimate — lower weight
]

# HTML obfuscation
_HTML_OBFUSCATION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r'display\s*:\s*none', re.I), 0.15),
    (re.compile(r'font-size\s*:\s*0', re.I), 0.15),
    (re.compile(r'color\s*:\s*#fff(fff)?|color\s*:\s*white', re.I), 0.10),
    (re.compile(r'visibility\s*:\s*hidden', re.I), 0.10),
    (re.compile(r'opacity\s*:\s*0', re.I), 0.10),
    (re.compile(r'height\s*:\s*0|width\s*:\s*0', re.I), 0.08),
    # Zero-width space / soft hyphen injection
    (re.compile(r'[\u200b\u200c\u200d\uFEFF\u00AD]'), 0.20),
    # Comment injection in words (e.g. pay<!--x-->pal)
    (re.compile(r'\w+<!--[^>]*-->\w+'), 0.25),
]

# Image-to-text ratio (HTML emails with only images and minimal text)
_RE_IMG_TAG = re.compile(r'<img\b', re.I)


class HeaderAnomalyScanner:
    """
    Scans raw email headers for structural anomalies beyond SPF/DKIM/DMARC.
    Returns a list of HeaderAnomalyRecord objects.
    """

    def scan(self, raw_headers: str) -> list[HeaderAnomalyRecord]:
        """
        Runs all header anomaly checks against the raw header block.

        Args:
            raw_headers: Full RFC 2822 header block.

        Returns:
            List of HeaderAnomalyRecord, one per detected anomaly.
        """
        unfolded = re.sub(r'\r?\n[ \t]+', ' ', raw_headers)
        records: list[HeaderAnomalyRecord] = []

        records.extend(self._check_from_display_spoof(unfolded))
        records.extend(self._check_received_anomalies(unfolded))
        records.extend(self._check_date_header(unfolded))
        records.extend(self._check_message_id(unfolded))
        records.extend(self._check_x_mailer(unfolded))
        records.extend(self._check_x_originating_ip(unfolded))

        return records

    def _check_from_display_spoof(self, headers: str) -> list[HeaderAnomalyRecord]:
        """
        Detects From display-name spoofing: the display name contains a
        known brand or domain name while the actual email address is from
        a different domain.
        """
        records: list[HeaderAnomalyRecord] = []
        m = _RE_FROM_DISPLAY.search(headers)
        if not m:
            return records

        display_name = m.group(1).strip().lower()
        email_addr   = m.group(2).strip().lower()
        email_domain = email_addr.split("@")[-1].rstrip(">")

        from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import BRAND_CORPUS
        for brand in BRAND_CORPUS:
            brand_name = brand.split(".")[0]  # e.g. 'paypal'
            if brand_name in display_name and brand not in email_domain:
                records.append(HeaderAnomalyRecord(
                    header_name     = "From",
                    header_value    = m.group(0)[:256],
                    indicator       = ThreatIndicator.HEADER_FROM_DISPLAY_SPOOF,
                    description     = (
                        f"Display name contains brand '{brand_name}' but "
                        f"actual sending domain is '{email_domain}'"
                    ),
                    severity_weight = 0.35,
                ))
                break

        return records

    def _check_received_anomalies(self, headers: str) -> list[HeaderAnomalyRecord]:
        """Checks for excessive Received hops (relay loops) and private-IP origins."""
        records: list[HeaderAnomalyRecord] = []

        hop_count = len(_RE_RECEIVED.findall(headers))
        if hop_count > 12:
            records.append(HeaderAnomalyRecord(
                header_name     = "Received",
                header_value    = f"{hop_count} hops detected",
                indicator       = ThreatIndicator.HEADER_RECEIVED_ANOMALY,
                description     = f"Unusual number of Received hops ({hop_count}) — possible relay loop",
                severity_weight = 0.15,
            ))

        for m in _RE_RECEIVED_FROM_IP.finditer(headers):
            ip = m.group(1)
            if _RE_PRIVATE_IP.match(ip):
                records.append(HeaderAnomalyRecord(
                    header_name     = "Received",
                    header_value    = ip,
                    indicator       = ThreatIndicator.HEADER_RECEIVED_ANOMALY,
                    description     = f"Received header contains private/loopback IP {ip} — possible header injection",
                    severity_weight = 0.20,
                ))
                break

        return records

    def _check_date_header(self, headers: str) -> list[HeaderAnomalyRecord]:
        """
        Flags Date headers set far in the future (forgery) or more than
        7 days in the past (stale / replay).
        """
        records: list[HeaderAnomalyRecord] = []
        m = _RE_DATE_HEADER.search(headers)
        if not m:
            return records

        date_str = m.group(1).strip()
        try:
            email_dt = parsedate_to_datetime(date_str)
            now = datetime.now(timezone.utc)
            # Normalize to UTC
            if email_dt.tzinfo is None:
                from datetime import timezone as tz
                email_dt = email_dt.replace(tzinfo=tz.utc)
            delta_hours = (email_dt - now).total_seconds() / 3600

            if delta_hours > 24:
                records.append(HeaderAnomalyRecord(
                    header_name     = "Date",
                    header_value    = date_str[:128],
                    indicator       = ThreatIndicator.HEADER_FORGED_DATE,
                    description     = f"Date header is {delta_hours:.1f} hours in the future — possible forgery",
                    severity_weight = 0.25,
                ))
            elif delta_hours < -168:  # > 7 days old
                records.append(HeaderAnomalyRecord(
                    header_name     = "Date",
                    header_value    = date_str[:128],
                    indicator       = ThreatIndicator.HEADER_FORGED_DATE,
                    description     = f"Date header is {abs(delta_hours):.0f} hours old — possible replay",
                    severity_weight = 0.10,
                ))
        except Exception:
            records.append(HeaderAnomalyRecord(
                header_name     = "Date",
                header_value    = date_str[:128],
                indicator       = ThreatIndicator.HEADER_FORGED_DATE,
                description     = "Date header is malformed or unparseable",
                severity_weight = 0.15,
            ))

        return records

    def _check_message_id(self, headers: str) -> list[HeaderAnomalyRecord]:
        """Flags emails with missing or malformed Message-ID."""
        records: list[HeaderAnomalyRecord] = []
        if not _RE_MESSAGE_ID.search(headers):
            records.append(HeaderAnomalyRecord(
                header_name     = "Message-ID",
                header_value    = "(absent)",
                indicator       = ThreatIndicator.HEADER_MISSING_MESSAGE_ID,
                description     = "Message-ID header is absent — non-standard, common in spam/phishing",
                severity_weight = 0.12,
            ))
        return records

    def _check_x_mailer(self, headers: str) -> list[HeaderAnomalyRecord]:
        """Flags known bulk/spam X-Mailer values."""
        records: list[HeaderAnomalyRecord] = []
        m = _RE_X_MAILER.search(headers)
        if not m:
            return records

        mailer = m.group(1).strip().lower()
        for suspicious in _SUSPICIOUS_MAILERS:
            if suspicious in mailer:
                records.append(HeaderAnomalyRecord(
                    header_name     = "X-Mailer",
                    header_value    = m.group(1)[:128],
                    indicator       = ThreatIndicator.HEADER_X_MAILER_SUSPICIOUS,
                    description     = f"X-Mailer contains suspicious identifier: '{suspicious}'",
                    severity_weight = 0.08,
                ))
                break

        return records

    def _check_x_originating_ip(self, headers: str) -> list[HeaderAnomalyRecord]:
        """
        Flags X-Originating-IP headers containing private/loopback IPs,
        which may indicate header injection or misconfigured forwarders.
        """
        records: list[HeaderAnomalyRecord] = []
        m = _RE_X_ORIG_IP.search(headers)
        if m:
            ip = m.group(1)
            if _RE_PRIVATE_IP.match(ip):
                records.append(HeaderAnomalyRecord(
                    header_name     = "X-Originating-IP",
                    header_value    = ip,
                    indicator       = ThreatIndicator.HEADER_RECEIVED_ANOMALY,
                    description     = f"X-Originating-IP is a private/loopback address ({ip})",
                    severity_weight = 0.15,
                ))
        return records


class BodyHeuristicScanner:
    """
    Applies rule-based heuristics to email body content.
    No ML model inference — pure pattern matching.
    Returns a list of BodyHeuristicRecord objects.
    """

    def scan(
        self,
        body_text:         Optional[str],
        body_html:         Optional[str],
        has_attachments:   bool = False,
        has_urls:          bool = False,
    ) -> list[BodyHeuristicRecord]:
        """
        Runs all body heuristic checks.

        Args:
            body_text:       Plain-text body content.
            body_html:       Raw HTML body content.
            has_attachments: True if the email contains attachments.
            has_urls:        True if URLs were extracted from the body.

        Returns:
            List of BodyHeuristicRecord, one per detected signal.
        """
        records: list[BodyHeuristicRecord] = []
        content = body_text or ""

        records.extend(self._check_urgency(content))
        records.extend(self._check_credential_requests(content))

        if body_html:
            records.extend(self._check_html_obfuscation(body_html))
            records.extend(self._check_image_ratio(body_html, content))

        if has_attachments and has_urls:
            records.append(BodyHeuristicRecord(
                indicator       = ThreatIndicator.BODY_LINK_TEXT_MISMATCH,
                description     = "Email contains both URLs and attachments — common malware delivery pattern",
                matched_pattern = None,
                severity_weight = 0.15,
            ))

        return records

    def _check_urgency(self, text: str) -> list[BodyHeuristicRecord]:
        """Detects urgency language patterns and computes cumulative weight."""
        records: list[BodyHeuristicRecord] = []
        total_weight = 0.0

        for pattern, weight in _URGENCY_PATTERNS:
            m = pattern.search(text)
            if m:
                total_weight += weight
                if total_weight <= 0.60:  # Cap to avoid duplication noise
                    records.append(BodyHeuristicRecord(
                        indicator       = ThreatIndicator.BODY_URGENCY_LANGUAGE,
                        description     = f"Urgency language detected: '{m.group(0)[:80]}'",
                        matched_pattern = m.group(0)[:80],
                        severity_weight = weight,
                    ))

        return records

    def _check_credential_requests(self, text: str) -> list[BodyHeuristicRecord]:
        """Detects requests for credentials or sensitive personal data."""
        records: list[BodyHeuristicRecord] = []

        for pattern, weight in _CREDENTIAL_PATTERNS:
            m = pattern.search(text)
            if m:
                records.append(BodyHeuristicRecord(
                    indicator       = ThreatIndicator.BODY_CREDENTIAL_REQUEST,
                    description     = f"Credential/sensitive data request: '{m.group(0)[:80]}'",
                    matched_pattern = m.group(0)[:80],
                    severity_weight = weight,
                ))

        return records

    def _check_html_obfuscation(self, html: str) -> list[BodyHeuristicRecord]:
        """Detects common HTML obfuscation techniques used in phishing emails."""
        records: list[BodyHeuristicRecord] = []

        for pattern, weight in _HTML_OBFUSCATION_PATTERNS:
            m = pattern.search(html)
            if m:
                records.append(BodyHeuristicRecord(
                    indicator       = ThreatIndicator.BODY_HTML_OBFUSCATION,
                    description     = f"HTML obfuscation pattern detected: '{m.group(0)[:60]}'",
                    matched_pattern = m.group(0)[:60],
                    severity_weight = weight,
                ))

        return records

    def _check_image_ratio(self, html: str, text: str) -> list[BodyHeuristicRecord]:
        """
        Flags emails with many images but minimal text.
        Image-only phishing evades text-based filters.
        """
        records: list[BodyHeuristicRecord] = []
        img_count  = len(_RE_IMG_TAG.findall(html))
        text_words = len(text.split()) if text else 0

        if img_count >= 3 and text_words < 30:
            records.append(BodyHeuristicRecord(
                indicator       = ThreatIndicator.BODY_EXCESSIVE_IMAGES,
                description     = (
                    f"Email contains {img_count} images but only {text_words} words — "
                    "possible image-only phishing"
                ),
                matched_pattern = None,
                severity_weight = 0.18,
            ))

        return records
