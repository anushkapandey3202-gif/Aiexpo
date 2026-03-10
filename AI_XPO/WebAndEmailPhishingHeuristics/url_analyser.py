"""
SentinelAI - URL Risk Analyser
================================
Analyses individual URLs for phishing and malicious redirect signals.

Detection techniques:
  1. URL extraction from text/HTML (regex-based; supplement with caller-supplied URLs)
  2. IP-address host detection  — URLs using raw IPs bypass brand-name checks
  3. URL shortener detection    — expanded-URL analysis is outside scope here;
                                  shortener presence alone is a risk signal
  4. Credential keyword scan    — path/query contains login, password, verify, etc.
  5. Encoded payload detection  — excessive percent-encoding or base64 in path/query
  6. Excessive subdomain check  — subdomain count ≥ threshold is evasion indicator
  7. Suspicious port detection  — non-standard ports on HTTP/HTTPS URLs
  8. Homograph in host          — delegated to DomainAnalyser
  9. Link text / href mismatch  — HTML anchor text claims different domain than href

URL Risk Score weighting:
  is_ip_address           +0.30
  credential_keywords     +0.25 (for each keyword group hit, up to 0.25 total)
  is_shortener            +0.15
  encoded_payload         +0.20
  excessive_subdomains    +0.10
  suspicious_port         +0.15
  domain_risk (inherited) ×0.40 (blend domain score into URL score)
"""
from __future__ import annotations

import base64
import logging
import re
import urllib.parse
from typing import Optional

from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import DomainAnalyser
from sentinel_ai.services.email_analysis.schemas.analysis import (
    ThreatIndicator,
    URLRiskRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL Extraction Regex
# ---------------------------------------------------------------------------
_RE_URL_EXTRACT = re.compile(
    r'https?://(?:[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+)',
    re.IGNORECASE,
)

# HTML href extraction
_RE_HREF = re.compile(
    r'href\s*=\s*["\']?(https?://[^\s"\'<>]+)',
    re.IGNORECASE,
)

# HTML anchor text extraction
_RE_ANCHOR = re.compile(
    r'<a[^>]+href\s*=\s*["\']?(https?://[^\s"\'<>]+)["\']?[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

# IP address as hostname
_RE_IP_HOST = re.compile(
    r'^(?:\d{1,3}\.){3}\d{1,3}$'
)

# Excessive percent-encoding indicator (≥4 consecutive encoded chars)
_RE_ENCODED_BLOCK = re.compile(r'(?:%[0-9A-Fa-f]{2}){4,}')

# Potential base64 payload in URL (20+ base64 chars)
_RE_BASE64_PAYLOAD = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

# ---------------------------------------------------------------------------
# Known URL Shorteners
# ---------------------------------------------------------------------------
URL_SHORTENERS: frozenset[str] = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "buff.ly",
    "ift.tt", "dlvr.it", "lnkd.in", "fb.me", "youtu.be", "amzn.to",
    "short.link", "rb.gy", "cutt.ly", "tiny.cc", "is.gd", "v.gd",
    "bl.ink", "shorturl.at", "clck.ru", "trib.al", "soo.gd",
})

# ---------------------------------------------------------------------------
# Credential & Phishing Keywords
# ---------------------------------------------------------------------------
# Grouped by category — finding any 2+ groups is more significant than 1
CREDENTIAL_KEYWORD_GROUPS: list[frozenset[str]] = [
    frozenset({"login", "signin", "sign-in", "logon", "log-in"}),
    frozenset({"password", "passwd", "pwd", "credentials", "credential"}),
    frozenset({"verify", "verification", "confirm", "validate", "authentication"}),
    frozenset({"account", "billing", "payment", "invoice", "banking"}),
    frozenset({"secure", "security", "update", "upgrade", "alert", "warning"}),
    frozenset({"webmail", "outlook", "office365", "o365", "gmail", "icloud"}),
]

# Suspicious non-standard ports (for HTTP/HTTPS)
SUSPICIOUS_PORTS: frozenset[int] = frozenset({
    8080, 8443, 8888, 9090, 9443, 4443, 4444,
    1337, 31337, 6660, 6661, 6662, 6663, 6664, 6665, 6666, 6667,
})


# ---------------------------------------------------------------------------
# URL Analyser
# ---------------------------------------------------------------------------

class URLAnalyser:
    """
    Extracts and analyses URLs from email text and HTML content.
    Stateless — safe for concurrent use.
    """

    def __init__(self) -> None:
        self._domain_analyser = DomainAnalyser()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_urls(self, text: Optional[str], html: Optional[str]) -> list[str]:
        """
        Extracts all unique URLs from plain text and/or HTML content.
        Returns deduplicated list preserving first-seen order.
        """
        found: list[str] = []
        seen:  set[str]  = set()

        def _add(url: str) -> None:
            url = url.strip().rstrip(".,;)")  # Strip trailing punctuation
            if url and url not in seen:
                seen.add(url)
                found.append(url)

        if html:
            for m in _RE_HREF.finditer(html):
                _add(m.group(1))
        if text:
            for m in _RE_URL_EXTRACT.finditer(text):
                _add(m.group(0))

        return found

    def extract_link_text_pairs(self, html: str) -> list[tuple[str, str]]:
        """
        Extracts (href_url, anchor_text) pairs from HTML content.
        Used to detect link-text/domain mismatch attacks.
        """
        pairs: list[tuple[str, str]] = []
        for m in _RE_ANCHOR.finditer(html):
            href = m.group(1).strip()
            text = re.sub(r'<[^>]+>', '', m.group(2)).strip()  # strip nested tags
            if href and text:
                pairs.append((href, text))
        return pairs

    # ------------------------------------------------------------------
    # Single-URL Analysis
    # ------------------------------------------------------------------

    def analyse_url(self, raw_url: str) -> URLRiskRecord:
        """
        Full risk analysis for a single URL string.

        Args:
            raw_url: The raw URL string (may be malformed or suspicious).

        Returns:
            URLRiskRecord with all detected threat signals and a risk_score.
        """
        indicators: list[ThreatIndicator] = []
        risk_score  = 0.0

        # Parse URL
        try:
            parsed = urllib.parse.urlparse(raw_url)
        except Exception:
            return URLRiskRecord(
                raw_url    = raw_url,
                risk_score = 0.5,
                indicators = [ThreatIndicator.URL_ENCODED_PAYLOAD],
            )

        host    = parsed.hostname or ""
        scheme  = parsed.scheme.lower() if parsed.scheme else ""
        path    = parsed.path or ""
        query   = parsed.query or ""
        port    = parsed.port

        # --- IP address as host ---
        is_ip = bool(_RE_IP_HOST.match(host))
        if is_ip:
            indicators.append(ThreatIndicator.URL_IP_ADDRESS)
            risk_score += 0.30

        # --- URL shortener ---
        is_shortener    = False
        shortener_svc: Optional[str] = None
        clean_host      = host.lstrip("www.")
        if clean_host in URL_SHORTENERS:
            is_shortener  = True
            shortener_svc = clean_host
            indicators.append(ThreatIndicator.URL_SHORTENER)
            risk_score += 0.15

        # --- Credential keywords ---
        full_path_query = (path + "?" + query).lower()
        matched_groups  = 0
        all_keywords:   list[str] = []
        for group in CREDENTIAL_KEYWORD_GROUPS:
            hit_words = [kw for kw in group if kw in full_path_query]
            if hit_words:
                all_keywords.extend(hit_words)
                matched_groups += 1

        if matched_groups >= 1:
            indicators.append(ThreatIndicator.URL_CREDENTIAL_KEYWORDS)
            # Scale: 1 group = 0.10, 2 = 0.18, 3+ = 0.25
            risk_score += min(0.25, 0.10 + (matched_groups - 1) * 0.075)

        # --- Encoded payload ---
        has_encoded = False
        if _RE_ENCODED_BLOCK.search(path + query):
            has_encoded = True
            indicators.append(ThreatIndicator.URL_ENCODED_PAYLOAD)
            risk_score += 0.20
        elif _RE_BASE64_PAYLOAD.search(path + query):
            # Possible base64 payload (heuristic — high FP rate, lower weight)
            has_encoded = True
            indicators.append(ThreatIndicator.URL_ENCODED_PAYLOAD)
            risk_score += 0.10

        # --- Excessive subdomains ---
        host_parts = host.split(".")
        has_excessive_subdomains = len(host_parts) >= 5
        if has_excessive_subdomains:
            indicators.append(ThreatIndicator.URL_EXCESSIVE_SUBDOMAINS)
            risk_score += 0.10

        # --- Suspicious port ---
        suspicious_port: Optional[int] = None
        if port and port in SUSPICIOUS_PORTS:
            suspicious_port = port
            indicators.append(ThreatIndicator.URL_SUSPICIOUS_PORT)
            risk_score += 0.15
        elif port and scheme == "http" and port == 80:
            pass  # Standard
        elif port and scheme == "https" and port == 443:
            pass  # Standard
        elif port and port not in (80, 443, None):
            indicators.append(ThreatIndicator.URL_SUSPICIOUS_PORT)
            risk_score += 0.08

        # --- Domain analysis ---
        domain_record = None
        if host and not is_ip:
            domain_record = self._domain_analyser.analyse(host)
            # Blend domain risk into URL score
            risk_score += domain_record.risk_score * 0.40

        # --- Normalise URL ---
        try:
            normalised_url = urllib.parse.urlunparse((
                scheme, host,
                urllib.parse.quote(path, safe="/:@!$&'()*+,;=-._~"),
                parsed.params, query, ""
            ))
        except Exception:
            normalised_url = raw_url

        return URLRiskRecord(
            raw_url                  = raw_url,
            normalized_url           = normalised_url,
            scheme                   = scheme,
            host                     = host,
            path                     = path,
            is_ip_address            = is_ip,
            is_punycode              = domain_record.is_punycode if domain_record else False,
            is_shortener             = is_shortener,
            shortener_service        = shortener_svc,
            redirect_chain_depth     = 0,  # Populated externally if redirect following is enabled
            has_credential_keywords  = len(all_keywords) > 0,
            credential_keywords_found= all_keywords[:20],
            has_encoded_payload      = has_encoded,
            excessive_subdomains     = has_excessive_subdomains,
            suspicious_port          = suspicious_port,
            domain_record            = domain_record,
            risk_score               = round(min(1.0, max(0.0, risk_score)), 4),
            indicators               = list(dict.fromkeys(indicators)),
        )

    def detect_link_text_mismatch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str, str]]:
        """
        Detects anchor tag mismatches where the visible text shows a trusted
        brand domain but the href points to a different domain.

        Args:
            pairs: List of (href_url, anchor_text) tuples from extract_link_text_pairs().

        Returns:
            List of (href_domain, claimed_domain_in_text, anchor_text) for mismatches.
        """
        mismatches: list[tuple[str, str, str]] = []
        domain_pattern = re.compile(
            r'https?://(?:www\.)?([A-Za-z0-9\-\.]+\.[A-Za-z]{2,})',
            re.IGNORECASE,
        )

        for href, text in pairs:
            href_match = domain_pattern.search(href)
            text_match = domain_pattern.search(text)

            if not (href_match and text_match):
                continue

            href_domain = href_match.group(1).lower()
            text_domain = text_match.group(1).lower()

            if href_domain != text_domain:
                # Check if text_domain is in brand corpus (spoofing a known brand)
                from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import BRAND_CORPUS
                if text_domain in BRAND_CORPUS or any(
                    text_domain in brand for brand in BRAND_CORPUS
                ):
                    mismatches.append((href_domain, text_domain, text[:200]))

        return mismatches
