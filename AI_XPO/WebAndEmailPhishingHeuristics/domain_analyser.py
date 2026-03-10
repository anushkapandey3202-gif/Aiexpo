"""
SentinelAI - Domain Name Risk Analyser
========================================
Calculates entropy and look-alike risk scores for domain names.

Detection techniques (applied in order of computational cost):

1. Shannon Entropy:
   Measures randomness of the second-level domain label.
   High entropy (>3.5 bits/char) indicates DGA or randomly generated domains.
   Formula: H = -Σ p(c) × log₂(p(c))  for each unique character c.

2. Look-Alike Detection (Levenshtein):
   Computes normalised edit distance between the candidate domain and every
   domain in the protected brand corpus.
   Score = 1 − (edit_distance / max(len(a), len(b)))
   Threshold: if similarity ≥ LOOKALIKE_THRESHOLD (0.80) AND candidate ≠ target → flag.

3. Homograph Attack Detection:
   Detects Unicode characters that are visually identical to ASCII characters.
   Maps confusable Unicode codepoints to their ASCII equivalents using the
   Unicode Consortium's confusables.txt data (abbreviated corpus below).
   After homograph normalisation, if the result matches a brand domain → flag.

4. Keyboard Proximity Typosquatting:
   Checks for single-character substitutions where the substitute key is
   adjacent on a QWERTY keyboard layout.
   e.g. 'paypal.com' → 'pwypal.com' (a→w adjacent on QWERTY)

5. Punycode / IDN Detection:
   Domains beginning with 'xn--' are decoded and both the raw and decoded
   forms are analysed. Mixed-script labels (Latin + Cyrillic) are flagged.

6. TLD Risk:
   Certain TLDs are statistically overrepresented in phishing campaigns.
   Domains using high-risk TLDs receive a static risk bump.

Protected Brand Corpus:
   The BRAND_CORPUS set below is a representative sample. In production,
   replace or augment from a database / external threat-intel feed.
   Corpus entries should be the registered domain (SLD + TLD), lowercase.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Optional

from sentinel_ai.services.email_analysis.schemas.analysis import (
    DomainRiskRecord,
    ThreatIndicator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENTROPY_HIGH_THRESHOLD   = 3.8   # bits/char — flag as high-entropy
ENTROPY_MEDIUM_THRESHOLD = 3.2
LOOKALIKE_THRESHOLD      = 0.80  # Normalised similarity to flag as look-alike
SUBDOMAIN_DEPTH_WARNING  = 4     # ≥4 subdomain levels is suspicious

# TLDs statistically overrepresented in phishing (Spamhaus / URLhaus data)
HIGH_RISK_TLDS: frozenset[str] = frozenset({
    ".tk", ".ml", ".ga", ".cf", ".gq",        # Free Freenom TLDs
    ".xyz", ".top", ".club", ".work", ".click",
    ".loan", ".online", ".site", ".website",
    ".win", ".bid", ".trade", ".accountant",
})

# ---------------------------------------------------------------------------
# Protected Brand Corpus (SLD+TLD, lowercase)
# Extend this in production from a threat-intel database.
# ---------------------------------------------------------------------------
BRAND_CORPUS: frozenset[str] = frozenset({
    # Financial
    "paypal.com", "paypal.me", "chase.com", "bankofamerica.com",
    "wellsfargo.com", "citibank.com", "barclays.co.uk", "hsbc.com",
    "americanexpress.com", "capitalone.com", "usbank.com",
    # Tech
    "google.com", "gmail.com", "microsoft.com", "apple.com",
    "amazon.com", "facebook.com", "meta.com", "twitter.com", "x.com",
    "linkedin.com", "instagram.com", "dropbox.com", "icloud.com",
    "outlook.com", "live.com", "hotmail.com", "yahoo.com",
    # Cloud / infra
    "aws.amazon.com", "azure.microsoft.com", "console.cloud.google.com",
    "github.com", "gitlab.com", "atlassian.com", "slack.com",
    # E-commerce
    "ebay.com", "etsy.com", "shopify.com", "stripe.com",
    # Crypto
    "coinbase.com", "binance.com", "kraken.com", "blockchain.com",
    # Government / critical
    "irs.gov", "ssa.gov", "usps.com", "fedex.com", "ups.com", "dhl.com",
})

# ---------------------------------------------------------------------------
# Homograph Confusable Map (Unicode → ASCII approximations)
# Source: https://www.unicode.org/reports/tr39/  (abbreviated)
# ---------------------------------------------------------------------------
CONFUSABLE_MAP: dict[str, str] = {
    # Cyrillic look-alikes
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x',
    'у': 'y', 'і': 'i', 'і': 'i', 'ѕ': 's', 'ԁ': 'd',
    # Greek look-alikes
    'ο': 'o', 'ρ': 'p', 'ν': 'v', 'υ': 'u', 'ω': 'w',
    # Common Unicode substitutions
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't',
    'ǀ': 'l', 'ɩ': 'i', 'ı': 'i',
    # Full-width Latin
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},
}

# ---------------------------------------------------------------------------
# QWERTY Keyboard Adjacency Map (for typosquat detection)
# ---------------------------------------------------------------------------
QWERTY_ADJACENCY: dict[str, set[str]] = {
    'q': {'w', 'a', 's'},     'w': {'q', 'e', 'a', 's', 'd'},
    'e': {'w', 'r', 's', 'd', 'f'}, 'r': {'e', 't', 'd', 'f', 'g'},
    't': {'r', 'y', 'f', 'g', 'h'}, 'y': {'t', 'u', 'g', 'h', 'j'},
    'u': {'y', 'i', 'h', 'j', 'k'}, 'i': {'u', 'o', 'j', 'k', 'l'},
    'o': {'i', 'p', 'k', 'l'},       'p': {'o', 'l'},
    'a': {'q', 'w', 's', 'z'},       's': {'a', 'w', 'e', 'd', 'z', 'x'},
    'd': {'s', 'e', 'r', 'f', 'x', 'c'}, 'f': {'d', 'r', 't', 'g', 'c', 'v'},
    'g': {'f', 't', 'y', 'h', 'v', 'b'}, 'h': {'g', 'y', 'u', 'j', 'b', 'n'},
    'j': {'h', 'u', 'i', 'k', 'n', 'm'}, 'k': {'j', 'i', 'o', 'l', 'm'},
    'l': {'k', 'o', 'p'},
    'z': {'a', 's', 'x'},             'x': {'z', 's', 'd', 'c'},
    'c': {'x', 'd', 'f', 'v'},        'v': {'c', 'f', 'g', 'b'},
    'b': {'v', 'g', 'h', 'n'},        'n': {'b', 'h', 'j', 'm'},
    'm': {'n', 'j', 'k'},
    '0': {'o', '9'},                   '1': {'l', '2'},
}


# ---------------------------------------------------------------------------
# Domain Analyser
# ---------------------------------------------------------------------------

class DomainAnalyser:
    """
    Computes entropy and look-alike risk scores for domain names.
    Stateless — safe for concurrent use across asyncio tasks.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, domain: str) -> DomainRiskRecord:
        """
        Full risk analysis for a single domain name.

        Args:
            domain: Fully-qualified domain name (e.g. 'login.paypa1.com').

        Returns:
            DomainRiskRecord with entropy, look-alike flags, and composite risk_score.
        """
        domain = domain.lower().strip().rstrip(".")

        indicators:         list[ThreatIndicator] = []
        is_lookalike        = False
        lookalike_target:   Optional[str]  = None
        lookalike_algorithm: Optional[str] = None
        edit_distance:      Optional[int]  = None
        is_punycode         = False
        decoded_punycode:   Optional[str]  = None
        has_mixed_scripts   = False

        # --- Structural decomposition ---
        tld, registered, subdomain_depth = self._decompose_domain(domain)

        # --- Punycode / IDN ---
        if "xn--" in domain:
            is_punycode = True
            indicators.append(ThreatIndicator.PUNYCODE_DOMAIN)
            try:
                decoded_punycode = domain.encode("ascii").decode("idna")
            except (UnicodeError, UnicodeDecodeError):
                decoded_punycode = None

        # --- Homograph normalisation ---
        normalised = self._apply_confusable_map(registered or domain.split(".")[0])

        # --- Mixed script detection ---
        if self._has_mixed_scripts(registered or ""):
            has_mixed_scripts = True
            indicators.append(ThreatIndicator.HOMOGRAPH_ATTACK)

        # --- Shannon entropy on the SLD ---
        sld = registered.replace("." + tld, "") if registered and tld else domain.split(".")[0]
        entropy = self._shannon_entropy(sld)

        if entropy >= ENTROPY_HIGH_THRESHOLD:
            indicators.append(ThreatIndicator.HIGH_ENTROPY)

        # --- TLD risk ---
        if tld and ("." + tld) in HIGH_RISK_TLDS:
            indicators.append(ThreatIndicator.TLD_MISMATCH)

        # --- Subdomain abuse ---
        if subdomain_depth >= SUBDOMAIN_DEPTH_WARNING:
            indicators.append(ThreatIndicator.SUBDOMAIN_ABUSE)

        # --- Look-alike detection ---
        candidate = registered or domain
        la_target, la_algo, la_dist = self._detect_lookalike(candidate, normalised)
        if la_target:
            is_lookalike        = True
            lookalike_target    = la_target
            lookalike_algorithm = la_algo
            edit_distance       = la_dist
            indicators.append(ThreatIndicator.LOOKALIKE_DOMAIN)
            if la_algo == "keyboard":
                indicators.append(ThreatIndicator.TYPOSQUATTING)
            elif la_algo == "homograph":
                indicators.append(ThreatIndicator.HOMOGRAPH_ATTACK)

        # --- Composite risk score ---
        risk_score = self._compute_risk_score(
            entropy          = entropy,
            is_lookalike     = is_lookalike,
            is_punycode      = is_punycode,
            has_mixed_scripts= has_mixed_scripts,
            subdomain_depth  = subdomain_depth,
            tld              = tld or "",
            indicator_count  = len(set(indicators)),
        )

        return DomainRiskRecord(
            domain               = domain,
            entropy              = round(entropy, 4),
            is_lookalike         = is_lookalike,
            lookalike_target     = lookalike_target,
            lookalike_algorithm  = lookalike_algorithm,
            edit_distance        = edit_distance,
            is_punycode          = is_punycode,
            decoded_punycode     = decoded_punycode,
            has_mixed_scripts    = has_mixed_scripts,
            tld                  = tld,
            registered_domain    = registered,
            subdomain_depth      = subdomain_depth,
            risk_score           = round(risk_score, 4),
            indicators           = list(dict.fromkeys(indicators)),  # preserve order, dedup
        )

    # ------------------------------------------------------------------
    # Shannon Entropy
    # ------------------------------------------------------------------

    def _shannon_entropy(self, label: str) -> float:
        """
        Computes Shannon entropy of a domain label in bits per character.
        Empty or single-character labels return 0.0.
        """
        if not label or len(label) < 2:
            return 0.0

        freq: dict[str, int] = {}
        for ch in label:
            freq[ch] = freq.get(ch, 0) + 1

        length = len(label)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)

        return entropy

    # ------------------------------------------------------------------
    # Look-alike Detection
    # ------------------------------------------------------------------

    def _detect_lookalike(
        self, candidate: str, normalised: str
    ) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Tests the candidate domain against the brand corpus using three techniques.
        Returns (target, algorithm, edit_distance) or (None, None, None).

        Checks in order of specificity:
          1. Direct/normalised match (homograph)
          2. Levenshtein similarity (general look-alike)
          3. Keyboard proximity typosquat
        """
        # Strip TLD for SLD-only comparison
        candidate_sld = candidate.split(".")[0]
        normalised_sld = normalised.split(".")[0]

        for brand in BRAND_CORPUS:
            brand_sld = brand.split(".")[0]

            if candidate_sld == brand_sld:
                continue  # Exact match — legitimate

            # 1. Homograph: normalised form matches brand SLD
            if normalised_sld == brand_sld and normalised_sld != candidate_sld:
                return brand, "homograph", 0

            # 2. Levenshtein similarity
            dist = self._levenshtein(candidate_sld, brand_sld)
            max_len = max(len(candidate_sld), len(brand_sld), 1)
            similarity = 1.0 - (dist / max_len)

            if similarity >= LOOKALIKE_THRESHOLD:
                return brand, "levenshtein", dist

            # 3. Keyboard proximity (single-char substitution on QWERTY)
            if self._is_keyboard_typosquat(candidate_sld, brand_sld):
                return brand, "keyboard", 1

        return None, None, None

    def _levenshtein(self, a: str, b: str) -> int:
        """Iterative Levenshtein distance (Wagner-Fischer algorithm). O(m×n) time."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        # Optimised: only keep two rows
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)

        for i, ca in enumerate(a, 1):
            curr[0] = i
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev, curr = curr, prev

        return prev[len(b)]

    def _is_keyboard_typosquat(self, candidate: str, target: str) -> bool:
        """
        Returns True if candidate differs from target by exactly one character
        substitution where the substitute key is QWERTY-adjacent to the original.
        """
        if len(candidate) != len(target):
            return False

        diffs = [
            (tc, cc)
            for tc, cc in zip(target, candidate)
            if tc != cc
        ]

        if len(diffs) != 1:
            return False

        target_char, candidate_char = diffs[0]
        adjacent = QWERTY_ADJACENCY.get(target_char, set())
        return candidate_char in adjacent

    # ------------------------------------------------------------------
    # Homograph / Confusable Normalisation
    # ------------------------------------------------------------------

    def _apply_confusable_map(self, label: str) -> str:
        """
        Replaces Unicode confusable characters with their ASCII equivalents.
        Used to detect homograph attacks (e.g. Cyrillic 'а' → ASCII 'a').
        """
        return "".join(CONFUSABLE_MAP.get(ch, ch) for ch in label)

    def _has_mixed_scripts(self, label: str) -> bool:
        """
        Returns True if the label mixes characters from different Unicode scripts.
        A domain using both Latin and Cyrillic characters is a homograph attack signal.
        """
        scripts: set[str] = set()
        for ch in label:
            if ch in ('-', '.'):
                continue
            try:
                script = unicodedata.name(ch, "").split()[0]
            except Exception:
                continue
            if script in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "CJK"):
                scripts.add(script)

        return len(scripts) > 1

    # ------------------------------------------------------------------
    # Domain Decomposition
    # ------------------------------------------------------------------

    def _decompose_domain(self, domain: str) -> tuple[Optional[str], Optional[str], int]:
        """
        Decomposes a domain into (tld, registered_domain, subdomain_depth).
        Uses a simplified heuristic (last label = TLD, second-to-last = SLD).
        Production: replace with tldextract library for full PSL support.
        """
        parts = domain.split(".")
        if len(parts) < 2:
            return None, None, 0

        tld = parts[-1]
        # Handle compound TLDs (co.uk, com.au, etc.)
        if len(parts) >= 3 and parts[-2] in ("co", "com", "net", "org", "gov", "edu"):
            tld = f"{parts[-2]}.{parts[-1]}"
            registered = f"{parts[-3]}.{tld}" if len(parts) >= 3 else None
            subdomain_depth = len(parts) - 3
        else:
            registered = f"{parts[-2]}.{tld}"
            subdomain_depth = len(parts) - 2

        return tld, registered, max(0, subdomain_depth)

    # ------------------------------------------------------------------
    # Risk Score Computation
    # ------------------------------------------------------------------

    def _compute_risk_score(
        self,
        entropy:          float,
        is_lookalike:     bool,
        is_punycode:      bool,
        has_mixed_scripts: bool,
        subdomain_depth:  int,
        tld:              str,
        indicator_count:  int,
    ) -> float:
        """
        Computes a composite domain risk score from 0.0 to 1.0.

        Components:
          - Entropy contribution (max 0.25)
          - Look-alike / homograph (major signal — 0.45 max)
          - Punycode + mixed script (0.15)
          - Subdomain depth (0.10)
          - High-risk TLD (0.10)
          - Indicator density bump (up to 0.05)
        """
        score = 0.0

        # Entropy (scaled: 0 at 0.0 bits, max 0.25 at ENTROPY_HIGH_THRESHOLD)
        entropy_contribution = min(0.25, (entropy / ENTROPY_HIGH_THRESHOLD) * 0.25)
        score += entropy_contribution

        # Look-alike / homograph (dominant signal)
        if is_lookalike:
            score += 0.45
        elif has_mixed_scripts:
            score += 0.30

        # Punycode
        if is_punycode:
            score += 0.10
        if has_mixed_scripts and not is_lookalike:
            score += 0.05

        # Subdomain depth (0.02 per level above warning, max 0.10)
        if subdomain_depth >= SUBDOMAIN_DEPTH_WARNING:
            score += min(0.10, (subdomain_depth - SUBDOMAIN_DEPTH_WARNING + 1) * 0.025)

        # High-risk TLD
        if ("." + tld) in HIGH_RISK_TLDS:
            score += 0.10

        # Indicator density bump (small boost for many co-occurring signals)
        if indicator_count >= 3:
            score += 0.05

        return min(1.0, max(0.0, score))
