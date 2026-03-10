"""
SentinelAI - Email Analysis Test Suite
========================================
Tests are organized into five classes matching the five analysis layers:

  TestAuthHeaderParser      — SPF/DKIM/DMARC parsing correctness
  TestDomainAnalyser         — Entropy, look-alike, homograph, typosquat
  TestURLAnalyser            — Extraction, IP detection, shorteners, keywords
  TestHeaderBodyScanner      — Structural anomalies, urgency, obfuscation
  TestOrchestrator           — End-to-end composite scoring

Run with:
  pytest services/email_analysis/tests/test_email_analysis.py -v
"""
from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import pytest

from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import (
    DomainAnalyser,
    BRAND_CORPUS,
)
from sentinel_ai.services.email_analysis.core.heuristics.header_body_scanner import (
    BodyHeuristicScanner,
    HeaderAnomalyScanner,
)
from sentinel_ai.services.email_analysis.core.heuristics.url_analyser import URLAnalyser
from sentinel_ai.services.email_analysis.core.parsers.auth_parser import AuthHeaderParser
from sentinel_ai.services.email_analysis.core.orchestrator import EmailAnalysisOrchestrator
from sentinel_ai.services.email_analysis.schemas.analysis import (
    DMARCPolicy,
    DMARCResult,
    DKIMResult,
    SPFResult,
    ThreatIndicator,
    RiskLevel,
    EmailAnalysisRequest,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def auth_parser():
    return AuthHeaderParser()

@pytest.fixture(scope="module")
def domain_analyser():
    return DomainAnalyser()

@pytest.fixture(scope="module")
def url_analyser():
    return URLAnalyser()

@pytest.fixture(scope="module")
def header_scanner():
    return HeaderAnomalyScanner()

@pytest.fixture(scope="module")
def body_scanner():
    return BodyHeuristicScanner()

@pytest.fixture(scope="module")
def orchestrator():
    return EmailAnalysisOrchestrator()


# ===========================================================================
# Auth Header Parser Tests
# ===========================================================================

class TestAuthHeaderParser:

    ALL_PASS_HEADERS = """
Authentication-Results: mx.example.com;
  spf=pass smtp.mailfrom=sender@paypal.com;
  dkim=pass header.d=paypal.com header.s=selector1;
  dmarc=pass (policy=reject) header.from=paypal.com
Received: from mail.paypal.com ([192.0.2.1]) by mx.example.com
From: PayPal <service@paypal.com>
Message-ID: <abc123@paypal.com>
Date: Thu, 01 Jan 2026 12:00:00 +0000
""".strip()

    ALL_FAIL_HEADERS = """
Authentication-Results: mx.example.com;
  spf=fail smtp.mailfrom=attacker@evil.com;
  dkim=fail header.d=evil.com;
  dmarc=fail header.from=evil.com
Received: from evil.com ([10.0.0.1]) by mx.example.com
From: PayPal Security <noreply@evil.com>
""".strip()

    SOFTFAIL_HEADERS = """
Authentication-Results: mx.example.com;
  spf=softfail smtp.mailfrom=maybe@suspicious.com;
  dkim=none;
  dmarc=none header.from=suspicious.com
Message-ID: <xyz@suspicious.com>
Date: Wed, 15 Jan 2026 10:00:00 +0000
""".strip()

    def test_all_pass_spf(self, auth_parser):
        result = auth_parser.parse(self.ALL_PASS_HEADERS)
        assert result.spf.result == SPFResult.PASS
        assert result.spf.smtp_mailfrom == "sender@paypal.com"

    def test_all_pass_dkim(self, auth_parser):
        result = auth_parser.parse(self.ALL_PASS_HEADERS)
        assert len(result.dkim) == 1
        assert result.dkim[0].result == DKIMResult.PASS
        assert result.dkim[0].domain == "paypal.com"

    def test_all_pass_dmarc(self, auth_parser):
        result = auth_parser.parse(self.ALL_PASS_HEADERS)
        assert result.dmarc.result == DMARCResult.PASS
        assert result.dmarc.policy == DMARCPolicy.REJECT

    def test_all_pass_auth_score_high(self, auth_parser):
        result = auth_parser.parse(self.ALL_PASS_HEADERS)
        # All pass + reject policy → score near 1.0
        assert result.auth_score >= 0.95, f"Expected ≥0.95, got {result.auth_score}"

    def test_all_fail_auth_score_zero(self, auth_parser):
        result = auth_parser.parse(self.ALL_FAIL_HEADERS)
        assert result.auth_score == 0.0, f"Expected 0.0, got {result.auth_score}"

    def test_all_fail_spf(self, auth_parser):
        result = auth_parser.parse(self.ALL_FAIL_HEADERS)
        assert result.spf.result == SPFResult.FAIL

    def test_softfail_intermediate_score(self, auth_parser):
        result = auth_parser.parse(self.SOFTFAIL_HEADERS)
        assert 0.0 < result.auth_score < 0.90

    def test_missing_message_id_flagged(self, auth_parser):
        result = auth_parser.parse(self.ALL_FAIL_HEADERS)
        assert ThreatIndicator.HEADER_MISSING_MESSAGE_ID in result.auth_indicators

    def test_reply_to_divergence_detected(self, auth_parser):
        headers_with_divergent_reply = self.ALL_PASS_HEADERS + \
            "\nReply-To: attacker@evil-domain.com"
        result = auth_parser.parse(headers_with_divergent_reply)
        assert ThreatIndicator.HEADER_REPLY_TO_DIVERGE in result.auth_indicators

    def test_missing_auth_results_returns_missing(self, auth_parser):
        minimal_headers = "From: someone@example.com\nDate: Thu, 01 Jan 2026 12:00:00 +0000"
        result = auth_parser.parse(minimal_headers)
        assert result.spf.result  == SPFResult.MISSING
        assert result.dmarc.result == DMARCResult.MISSING

    def test_received_spf_fallback(self, auth_parser):
        headers_with_received_spf = (
            "Received-SPF: pass (mx.example.com: domain of user@legit.com "
            "designates 1.2.3.4 as permitted sender) smtp.mailfrom=user@legit.com\n"
            "From: User <user@legit.com>\nMessage-ID: <abc@legit.com>"
        )
        result = auth_parser.parse(headers_with_received_spf)
        assert result.spf.result == SPFResult.PASS


# ===========================================================================
# Domain Analyser Tests
# ===========================================================================

class TestDomainAnalyser:

    def test_legitimate_domain_low_risk(self, domain_analyser):
        record = domain_analyser.analyse("paypal.com")
        # Legitimate brand domain — should NOT trigger look-alike (it IS the brand)
        assert not record.is_lookalike
        assert record.risk_score < 0.30

    def test_paypa1_flagged_as_lookalike(self, domain_analyser):
        """Classic numeric substitution: 'l' → '1'"""
        record = domain_analyser.analyse("paypa1.com")
        assert record.is_lookalike, "paypa1.com should be flagged as a PayPal look-alike"
        assert record.lookalike_target is not None
        assert "paypal" in record.lookalike_target
        assert ThreatIndicator.LOOKALIKE_DOMAIN in record.indicators

    def test_paypol_flagged_as_lookalike(self, domain_analyser):
        """Single-character substitution: 'a' → 'o'"""
        record = domain_analyser.analyse("paypol.com")
        assert record.is_lookalike
        assert record.risk_score > 0.40

    def test_high_entropy_dga_domain(self, domain_analyser):
        """DGA-like domain with high Shannon entropy"""
        record = domain_analyser.analyse("xj8kp2mq9vrtc.com")
        assert record.entropy > 3.2, f"Expected entropy >3.2, got {record.entropy}"
        assert ThreatIndicator.HIGH_ENTROPY in record.indicators

    def test_low_entropy_legitimate_domain(self, domain_analyser):
        """Simple dictionary-word domain has low entropy"""
        record = domain_analyser.analyse("google.com")
        assert record.entropy < 3.5, f"Expected entropy <3.5, got {record.entropy}"

    def test_entropy_calculation_correctness(self, domain_analyser):
        """Verify Shannon entropy formula for known input"""
        # 'aab': p(a)=2/3, p(b)=1/3
        # H = -(2/3 * log2(2/3) + 1/3 * log2(1/3))
        expected = -(2/3 * math.log2(2/3) + 1/3 * math.log2(1/3))
        actual = domain_analyser._shannon_entropy("aab")
        assert abs(actual - expected) < 1e-9, f"Entropy mismatch: {actual} vs {expected}"

    def test_entropy_single_char_is_zero(self, domain_analyser):
        assert domain_analyser._shannon_entropy("aaaaaaa") == 0.0

    def test_entropy_all_unique_chars_max(self, domain_analyser):
        # All unique chars → maximum entropy for that length
        s = "abcde"
        h = domain_analyser._shannon_entropy(s)
        expected = math.log2(len(s))
        assert abs(h - expected) < 1e-9

    def test_punycode_domain_flagged(self, domain_analyser):
        record = domain_analyser.analyse("xn--pypl-vpae.com")
        assert record.is_punycode
        assert ThreatIndicator.PUNYCODE_DOMAIN in record.indicators

    def test_high_risk_tld_flagged(self, domain_analyser):
        record = domain_analyser.analyse("legit-bank.tk")
        assert ThreatIndicator.TLD_MISMATCH in record.indicators

    def test_excessive_subdomains_flagged(self, domain_analyser):
        record = domain_analyser.analyse("login.secure.verify.paypal.phishing.com")
        assert ThreatIndicator.SUBDOMAIN_ABUSE in record.indicators

    def test_levenshtein_exact_match_returns_zero(self, domain_analyser):
        assert domain_analyser._levenshtein("abc", "abc") == 0

    def test_levenshtein_single_insert(self, domain_analyser):
        assert domain_analyser._levenshtein("abc", "abcd") == 1

    def test_levenshtein_single_delete(self, domain_analyser):
        assert domain_analyser._levenshtein("abcd", "abc") == 1

    def test_levenshtein_single_substitution(self, domain_analyser):
        assert domain_analyser._levenshtein("abc", "axc") == 1

    def test_keyboard_typosquat_detected(self, domain_analyser):
        # 'a' → 's' are adjacent on QWERTY
        is_typo = domain_analyser._is_keyboard_typosquat("psypal", "paypal")
        assert is_typo

    def test_keyboard_typosquat_non_adjacent_not_detected(self, domain_analyser):
        # 'a' -> 'b' are NOT adjacent on QWERTY (b is bottom-row, far from a)
        is_typo = domain_analyser._is_keyboard_typosquat("pbypal", "paypal")
        assert not is_typo

    def test_levenshtein_google_goggle_is_one(self, domain_analyser):
        # google -> goggle: one substitution at index 2; distance = 1
        assert domain_analyser._levenshtein("google", "goggle") == 1

    def test_risk_score_bounded(self, domain_analyser):
        for domain in ["evil.tk", "paypa1.com", "xj8kp2.com", "google.com", "a.b.c.d.e.f.com"]:
            record = domain_analyser.analyse(domain)
            assert 0.0 <= record.risk_score <= 1.0, f"Risk score out of bounds for {domain}"


# ===========================================================================
# URL Analyser Tests
# ===========================================================================

class TestURLAnalyser:

    def test_ip_address_url_flagged(self, url_analyser):
        record = url_analyser.analyse_url("http://192.168.1.1/login.php")
        assert record.is_ip_address
        assert ThreatIndicator.URL_IP_ADDRESS in record.indicators
        assert record.risk_score >= 0.30

    def test_url_shortener_detected(self, url_analyser):
        record = url_analyser.analyse_url("https://bit.ly/2xYZ123")
        assert record.is_shortener
        assert record.shortener_service == "bit.ly"
        assert ThreatIndicator.URL_SHORTENER in record.indicators

    def test_credential_keywords_detected(self, url_analyser):
        record = url_analyser.analyse_url(
            "https://attacker.com/login/verify/account?password=reset"
        )
        assert record.has_credential_keywords
        assert ThreatIndicator.URL_CREDENTIAL_KEYWORDS in record.indicators
        assert len(record.credential_keywords_found) > 0

    def test_encoded_payload_detected(self, url_analyser):
        # 4+ consecutive percent-encoded chars
        record = url_analyser.analyse_url(
            "https://attacker.com/redir?url=%68%74%74%70%73%3A%2F%2F"
        )
        assert record.has_encoded_payload
        assert ThreatIndicator.URL_ENCODED_PAYLOAD in record.indicators

    def test_excessive_subdomains_flagged(self, url_analyser):
        record = url_analyser.analyse_url(
            "https://login.secure.verify.banking.attacker.com/update"
        )
        assert record.excessive_subdomains
        assert ThreatIndicator.URL_EXCESSIVE_SUBDOMAINS in record.indicators

    def test_suspicious_port_flagged(self, url_analyser):
        record = url_analyser.analyse_url("https://example.com:8080/sensitive")
        assert record.suspicious_port == 8080
        assert ThreatIndicator.URL_SUSPICIOUS_PORT in record.indicators

    def test_clean_https_url_low_risk(self, url_analyser):
        record = url_analyser.analyse_url("https://www.google.com/search?q=weather")
        assert record.risk_score < 0.30, f"Expected <0.30, got {record.risk_score}"

    def test_url_extraction_from_text(self, url_analyser):
        text = "Please visit http://attacker.com/phish and https://evil.tk/steal for details."
        urls = url_analyser.extract_urls(text, None)
        assert "http://attacker.com/phish" in urls
        assert "https://evil.tk/steal" in urls

    def test_url_extraction_from_html_href(self, url_analyser):
        html = '<a href="https://paypa1.com/login">Click here to verify</a>'
        urls = url_analyser.extract_urls(None, html)
        assert "https://paypa1.com/login" in urls

    def test_link_text_mismatch_detected(self, url_analyser):
        html = '<a href="https://evil-attacker.com/steal">https://www.paypal.com/signin</a>'
        pairs = url_analyser.extract_link_text_pairs(html)
        mismatches = url_analyser.detect_link_text_mismatch(pairs)
        assert len(mismatches) >= 1, "Expected link-text mismatch to be detected"

    def test_url_extraction_deduplication(self, url_analyser):
        text = "http://dup.com http://dup.com http://dup.com"
        urls = url_analyser.extract_urls(text, None)
        assert urls.count("http://dup.com") == 1

    def test_risk_score_bounded(self, url_analyser):
        for url in [
            "http://192.168.1.1/login?password=abc",
            "https://bit.ly/xyz",
            "https://www.google.com",
            "https://xn--pypl.com:9090/verify/account/%68%74",
        ]:
            record = url_analyser.analyse_url(url)
            assert 0.0 <= record.risk_score <= 1.0, f"Risk out of bounds for {url}"


# ===========================================================================
# Header & Body Scanner Tests
# ===========================================================================

class TestHeaderBodyScanner:

    DISPLAY_SPOOF_HEADERS = (
        "From: PayPal Customer Service <billing@totally-evil.com>\n"
        "Date: Thu, 01 Jan 2026 12:00:00 +0000\n"
        "Message-ID: <abc@evil.com>"
    )

    FORGED_DATE_HEADERS = (
        "From: sender@example.com\n"
        "Date: Thu, 01 Jan 2099 12:00:00 +0000\n"  # Far future
        "Message-ID: <abc@example.com>"
    )

    MISSING_MESSAGE_ID = (
        "From: sender@example.com\n"
        "Date: Thu, 01 Jan 2026 12:00:00 +0000\n"
        "Subject: Hello"
    )

    def test_from_display_spoof_detected(self, header_scanner):
        records = header_scanner.scan(self.DISPLAY_SPOOF_HEADERS)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.HEADER_FROM_DISPLAY_SPOOF in indicators

    def test_forged_future_date_detected(self, header_scanner):
        records = header_scanner.scan(self.FORGED_DATE_HEADERS)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.HEADER_FORGED_DATE in indicators

    def test_missing_message_id_detected(self, header_scanner):
        records = header_scanner.scan(self.MISSING_MESSAGE_ID)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.HEADER_MISSING_MESSAGE_ID in indicators

    def test_clean_headers_no_anomalies(self, header_scanner):
        clean = (
            "From: legit@example.com\n"
            "Date: Thu, 01 Jan 2026 12:00:00 +0000\n"
            "Message-ID: <unique-id@example.com>"
        )
        records = header_scanner.scan(clean)
        assert len(records) == 0, f"Expected no anomalies, got: {[r.indicator for r in records]}"

    def test_urgency_language_detected(self, body_scanner):
        body = "Your account has been suspended. Verify your account immediately or it will be terminated."
        records = body_scanner.scan(body, None)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.BODY_URGENCY_LANGUAGE in indicators

    def test_credential_request_detected(self, body_scanner):
        body = "Please reply with your password and provide your social security number for verification."
        records = body_scanner.scan(body, None)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.BODY_CREDENTIAL_REQUEST in indicators

    def test_html_obfuscation_detected(self, body_scanner):
        html = '<p style="display:none">hidden text</p><p>Click here</p>'
        records = body_scanner.scan(None, html)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.BODY_HTML_OBFUSCATION in indicators

    def test_image_heavy_email_flagged(self, body_scanner):
        html = "<img src='a.png'><img src='b.png'><img src='c.png'><img src='d.png'>"
        text = "Click"  # Minimal text
        records = body_scanner.scan(text, html)
        indicators = [r.indicator for r in records]
        assert ThreatIndicator.BODY_EXCESSIVE_IMAGES in indicators

    def test_clean_body_no_heuristics(self, body_scanner):
        body = "Hello, please find the quarterly report attached. Let us know if you have questions."
        records = body_scanner.scan(body, None)
        assert len(records) == 0, f"Expected no heuristics, got: {[r.indicator for r in records]}"

    def test_severity_weights_bounded(self, body_scanner, header_scanner):
        for scanner, text, html in [
            (body_scanner, "verify immediately suspend account password SSN", None),
        ]:
            records = scanner.scan(text, html) if hasattr(scanner, 'scan') and scanner == body_scanner else scanner.scan(text)
            for r in records:
                assert 0.0 <= r.severity_weight <= 1.0


# ===========================================================================
# Orchestrator (End-to-End) Tests
# ===========================================================================

class TestOrchestrator:

    CLEAN_REQUEST = EmailAnalysisRequest(
        user_id         = "user-001",
        organization_id = "org-001",
        raw_headers     = (
            "Authentication-Results: mx.example.com;\n"
            "  spf=pass smtp.mailfrom=user@legitimate.com;\n"
            "  dkim=pass header.d=legitimate.com;\n"
            "  dmarc=pass (policy=reject) header.from=legitimate.com\n"
            "From: Legitimate Sender <user@legitimate.com>\n"
            "Message-ID: <abc123@legitimate.com>\n"
            "Date: Thu, 01 Jan 2026 12:00:00 +0000\n"
            "Received: from mail.legitimate.com ([1.2.3.4]) by mx.example.com"
        ),
        body_text       = "Please find the monthly invoice attached. Contact us if you need anything.",
        urls            = ["https://www.legitimate.com/invoice/12345"],
    )

    PHISHING_REQUEST = EmailAnalysisRequest(
        user_id         = "user-001",
        organization_id = "org-001",
        raw_headers     = (
            "Authentication-Results: mx.attacker.com;\n"
            "  spf=fail smtp.mailfrom=spoofed@evil.com;\n"
            "  dkim=fail;\n"
            "  dmarc=fail header.from=paypa1.com\n"
            "From: PayPal Security <security@paypa1.com>\n"
            "Reply-To: attacker@evil-domain.com\n"
            "Date: Thu, 01 Jan 2099 00:00:00 +0000\n"
        ),
        body_text       = (
            "Your PayPal account has been suspended! Verify your account immediately "
            "or it will be permanently terminated. Click the link to enter your password "
            "and provide your social security number to unlock your account."
        ),
        body_html       = (
            '<p style="display:none">tracker</p>'
            '<a href="https://paypa1-secure.tk/verify/login?password=reset">'
            "https://www.paypal.com/signin</a>"
        ),
        urls            = ["https://paypa1-secure.tk/verify/login/account/confirm?pwd=update"],
    )

    def test_clean_email_minimal_risk(self, orchestrator):
        result = orchestrator.analyse(self.CLEAN_REQUEST)
        assert result.composite_score < 0.40, (
            f"Expected composite <0.40 for clean email, got {result.composite_score}"
        )
        assert result.risk_level in (RiskLevel.MINIMAL, RiskLevel.LOW)

    def test_phishing_email_high_risk(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert result.composite_score >= 0.65, (
            f"Expected composite ≥0.65 for phishing email, got {result.composite_score}"
        )
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_phishing_email_has_auth_failure_score(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert result.score_auth > 0.0, "SPF/DKIM/DMARC failures should produce non-zero auth risk"

    def test_phishing_email_domain_risk_nonzero(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert result.score_domain > 0.30, (
            f"paypa1.com should produce domain_risk > 0.30, got {result.score_domain}"
        )

    def test_phishing_email_url_risk_nonzero(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert result.score_url > 0.20

    def test_phishing_email_body_risk_nonzero(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert result.score_body > 0.20

    def test_composite_score_bounded(self, orchestrator):
        for req in [self.CLEAN_REQUEST, self.PHISHING_REQUEST]:
            result = orchestrator.analyse(req)
            assert 0.0 <= result.composite_score <= 1.0

    def test_all_indicators_populated(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert len(result.all_indicators) > 0

    def test_threat_summary_non_empty(self, orchestrator):
        result = orchestrator.analyse(self.PHISHING_REQUEST)
        assert len(result.threat_summary) > 20

    def test_processing_time_recorded(self, orchestrator):
        result = orchestrator.analyse(self.CLEAN_REQUEST)
        assert result.processing_time_ms >= 0

    def test_analysis_id_unique(self, orchestrator):
        r1 = orchestrator.analyse(self.CLEAN_REQUEST)
        r2 = orchestrator.analyse(self.CLEAN_REQUEST)
        assert r1.analysis_id != r2.analysis_id

    def test_fusion_payload_confidence_matches_composite(self, orchestrator):
        result  = orchestrator.analyse(self.PHISHING_REQUEST)
        payload = orchestrator.build_fusion_payload(result)
        assert payload.confidence_score == result.composite_score

    def test_fusion_payload_model_type(self, orchestrator):
        result  = orchestrator.analyse(self.CLEAN_REQUEST)
        payload = orchestrator.build_fusion_payload(result)
        assert payload.model_type == "nlp_intent"

    def test_urls_only_request(self, orchestrator):
        req = EmailAnalysisRequest(
            user_id         = "user-001",
            organization_id = "org-001",
            urls            = [
                "http://192.168.1.100/login.php",
                "https://bit.ly/evil",
                "https://paypa1-secure.tk/verify",
            ],
        )
        result = orchestrator.analyse(req)
        assert result.composite_score > 0.0
        assert len(result.urls_analyzed) == 3

    def test_headers_only_request(self, orchestrator):
        req = EmailAnalysisRequest(
            user_id         = "user-001",
            organization_id = "org-001",
            raw_headers     = (
                "Authentication-Results: mx.example.com;\n"
                "  spf=fail;\n  dkim=fail;\n  dmarc=fail\n"
                "From: Fake <spoof@evil.com>"
            ),
        )
        result = orchestrator.analyse(req)
        assert result.score_auth > 0.0


# ===========================================================================
# Schema Validation Tests
# ===========================================================================

class TestSchemaValidation:

    def test_request_requires_at_least_one_input(self):
        with pytest.raises(Exception):
            EmailAnalysisRequest(user_id="u1", organization_id="o1")

    def test_request_accepts_urls_only(self):
        req = EmailAnalysisRequest(
            user_id="u1", organization_id="o1",
            urls=["https://example.com"]
        )
        assert len(req.urls) == 1

    def test_url_deduplication_in_request(self):
        req = EmailAnalysisRequest(
            user_id="u1", organization_id="o1",
            urls=["https://dup.com", "https://dup.com", "https://dup.com"]
        )
        assert len(req.urls) == 1

    def test_request_id_auto_generated(self):
        req = EmailAnalysisRequest(
            user_id="u1", organization_id="o1",
            urls=["https://example.com"]
        )
        assert len(req.request_id) == 36  # UUID format


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
