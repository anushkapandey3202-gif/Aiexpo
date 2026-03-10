"""
SentinelAI - AES-256-GCM Field-Level Encryption
================================================
Provides authenticated encryption (AEAD) for sensitive database columns.

Design decisions:
- AES-256-GCM chosen for authenticated encryption — prevents silent corruption.
- 96-bit random nonce per encryption operation prevents nonce reuse attacks.
- Ciphertext envelope: [version(4B)] + [nonce(12B)] + [ciphertext+GCM_tag(16B)]
- Key version prefix supports zero-downtime key rotation.
- SHA-256 hashing for encrypted fields that require indexed lookups (e.g. email).
- Module-level singleton FieldEncryptor is thread-safe (AESGCM is stateless).
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import struct
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# GCM nonce — 96 bits (12 bytes) is the NIST-recommended size
_NONCE_SIZE: int = 12
# Key version stored as big-endian uint32 (4 bytes)
_KEY_VERSION_SIZE: int = 4
# Minimum valid ciphertext length = version + nonce + GCM tag (no plaintext)
_MIN_CIPHERTEXT_LEN: int = _KEY_VERSION_SIZE + _NONCE_SIZE + 16


class EncryptionError(Exception):
    """Raised when an encryption operation fails."""


class DecryptionError(Exception):
    """Raised when decryption or GCM authentication fails."""


class FieldEncryptor:
    """
    AES-256-GCM symmetric encryptor for individual database field values.

    Ciphertext envelope layout (binary):
    ┌──────────────────┬──────────────────┬─────────────────────────────────┐
    │ Key Version (4B) │   Nonce (12B)    │  Ciphertext + GCM Auth Tag (≥16B)│
    │ big-endian uint32│ random per call  │  ciphertext || 16-byte tag       │
    └──────────────────┴──────────────────┴─────────────────────────────────┘

    Thread safety: AESGCM has no mutable state; safe for concurrent calls.
    """

    def __init__(self, master_key_b64: str, key_version: int) -> None:
        """
        Args:
            master_key_b64: Base64-encoded 32-byte AES-256 key.
            key_version:    Integer version tag written into every ciphertext.
        """
        raw_key = base64.b64decode(master_key_b64)
        if len(raw_key) != 32:
            raise ValueError(
                f"AES-256 requires exactly 32 bytes; decoded key is {len(raw_key)} bytes"
            )
        self._aesgcm: AESGCM = AESGCM(raw_key)
        self._key_version: int = key_version
        # Zero the local variable — raw_key reference is still held by AESGCM
        del raw_key

    def encrypt(
        self,
        plaintext: str | bytes,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Encrypts plaintext with AES-256-GCM.

        Args:
            plaintext:       UTF-8 string or bytes to encrypt.
            associated_data: Optional AAD bound to the ciphertext (e.g., row ID).
                             Must be supplied identically on decryption.

        Returns:
            Binary envelope: version(4B) + nonce(12B) + ciphertext+tag(≥16B).

        Raises:
            EncryptionError: On any underlying crypto failure.
        """
        try:
            if isinstance(plaintext, str):
                plaintext = plaintext.encode("utf-8")

            nonce: bytes = os.urandom(_NONCE_SIZE)
            ciphertext: bytes = self._aesgcm.encrypt(nonce, plaintext, associated_data)
            version_prefix: bytes = struct.pack(">I", self._key_version)

            return version_prefix + nonce + ciphertext

        except Exception as exc:
            logger.error(
                "AES-256-GCM encryption failed",
                extra={"key_version": self._key_version},
                exc_info=False,  # Don't log plaintext in stack trace
            )
            raise EncryptionError("Field encryption failed") from exc

    def decrypt(
        self,
        ciphertext_blob: bytes,
        associated_data: Optional[bytes] = None,
    ) -> str:
        """
        Decrypts and authenticates a ciphertext envelope.

        Args:
            ciphertext_blob: Binary envelope from the database column.
            associated_data: Must match the AAD provided at encryption time.

        Returns:
            Decrypted plaintext as a UTF-8 string.

        Raises:
            DecryptionError: On GCM tag mismatch, truncated blob, or decode failure.
        """
        if len(ciphertext_blob) < _MIN_CIPHERTEXT_LEN:
            raise DecryptionError(
                f"Ciphertext blob is {len(ciphertext_blob)}B — minimum valid length is {_MIN_CIPHERTEXT_LEN}B"
            )

        try:
            stored_version: int = struct.unpack(">I", ciphertext_blob[:_KEY_VERSION_SIZE])[0]
            nonce: bytes = ciphertext_blob[_KEY_VERSION_SIZE : _KEY_VERSION_SIZE + _NONCE_SIZE]
            ciphertext: bytes = ciphertext_blob[_KEY_VERSION_SIZE + _NONCE_SIZE :]

            if stored_version != self._key_version:
                # Log for alerting — operator must re-encrypt during key rotation
                logger.warning(
                    "Key version mismatch: field was encrypted with a different key version",
                    extra={
                        "stored_version": stored_version,
                        "current_version": self._key_version,
                    },
                )
                # Future: implement a key ring lookup here for graceful rotation

            plaintext: bytes = self._aesgcm.decrypt(nonce, ciphertext, associated_data)
            return plaintext.decode("utf-8")

        except DecryptionError:
            raise
        except Exception as exc:
            # InvalidTag from GCM means data integrity was violated
            logger.error(
                "AES-256-GCM decryption failed — possible data corruption or AAD mismatch",
                extra={"key_version": self._key_version},
                exc_info=False,
            )
            raise DecryptionError(
                "Decryption failed: GCM authentication tag mismatch or corrupted ciphertext"
            ) from exc

    @staticmethod
    def hash_for_lookup(value: str) -> str:
        """
        Produces a deterministic, non-reversible SHA-256 hex digest for indexed queries
        on encrypted fields (e.g., find user by email without decrypting all rows).

        Normalizes input (lowercase + strip) before hashing.
        Returns a 64-character hex string.
        """
        normalized = value.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialized)
# ---------------------------------------------------------------------------

_encryptor: Optional[FieldEncryptor] = None


def _get_encryptor() -> FieldEncryptor:
    """
    Lazily initializes and returns the module-level FieldEncryptor singleton.
    Thread-safe for read-heavy workloads (initialization race is benign).
    """
    global _encryptor
    if _encryptor is None:
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()
        _encryptor = FieldEncryptor(
            master_key_b64=cfg.AES_MASTER_KEY.get_secret_value(),
            key_version=cfg.AES_KEY_VERSION,
        )
        logger.info(
            "FieldEncryptor initialized",
            extra={"key_version": cfg.AES_KEY_VERSION},
        )
    return _encryptor


# ---------------------------------------------------------------------------
# Public convenience functions — import these in ORM models
# ---------------------------------------------------------------------------

def encrypt_field(value: str, associated_data: Optional[bytes] = None) -> bytes:
    """Encrypts a string field value. Returns AES-256-GCM ciphertext blob."""
    return _get_encryptor().encrypt(value, associated_data)


def decrypt_field(blob: bytes, associated_data: Optional[bytes] = None) -> str:
    """Decrypts an AES-256-GCM ciphertext blob. Returns plaintext string."""
    return _get_encryptor().decrypt(blob, associated_data)


def hash_field(value: str) -> str:
    """Returns a SHA-256 hex digest for indexed lookup of an encrypted field."""
    return FieldEncryptor.hash_for_lookup(value)
