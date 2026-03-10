"""
SentinelAI - Authentication Log ORM Model
==========================================
Immutable, append-only audit log for all authentication and session events.

Security design:
- Records are NEVER updated after insertion (audit integrity guarantee).
- IP addresses are AES-256-GCM encrypted at rest.
- ip_address_hash (SHA-256) allows threat correlation queries across events
  without decrypting individual records.
- risk_score is populated by a separate risk-assessment service and backfilled
  asynchronously — defaults to 0.0 on initial write.
- Covers the full authentication lifecycle: from login attempt through
  MFA challenges, token refresh, password reset, and account lockout events.
"""
from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel_ai.database.base import Base
from sentinel_ai.database.encryption import decrypt_field, encrypt_field, hash_field

if TYPE_CHECKING:
    from sentinel_ai.database.models.user import User


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AuthEventType(str, enum.Enum):
    """Classification of authentication lifecycle events."""

    # Session events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    SESSION_EXPIRED = "session_expired"

    # Token lifecycle
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKE = "token_revoke"
    TOKEN_INVALID = "token_invalid"

    # MFA events
    MFA_CHALLENGE_SENT = "mfa_challenge_sent"
    MFA_CHALLENGE_SUCCESS = "mfa_challenge_success"
    MFA_CHALLENGE_FAILURE = "mfa_challenge_failure"
    MFA_BACKUP_CODE_USED = "mfa_backup_code_used"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Credential management
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET_REQUEST = "password_reset_request"
    PASSWORD_RESET_COMPLETE = "password_reset_complete"
    EMAIL_VERIFICATION_SENT = "email_verification_sent"
    EMAIL_VERIFICATION_COMPLETE = "email_verification_complete"

    # Account state changes
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    ACCOUNT_DEACTIVATED = "account_deactivated"

    # API keys
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"

    # Threat signals
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CREDENTIAL_STUFFING_DETECTED = "credential_stuffing_detected"


class MFAMethod(str, enum.Enum):
    """MFA mechanism used in an authentication challenge."""

    TOTP = "totp"
    SMS = "sms"
    EMAIL_OTP = "email_otp"
    HARDWARE_KEY = "hardware_key"
    BACKUP_CODE = "backup_code"
    PUSH_NOTIFICATION = "push_notification"


class AuthFailureReason(str, enum.Enum):
    """Standardized failure codes for failed authentication events."""

    INVALID_PASSWORD = "invalid_password"
    INVALID_MFA_CODE = "invalid_mfa_code"
    EXPIRED_MFA_CODE = "expired_mfa_code"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_INACTIVE = "account_inactive"
    ACCOUNT_NOT_VERIFIED = "account_not_verified"
    INVALID_TOKEN = "invalid_token"
    EXPIRED_TOKEN = "expired_token"
    UNKNOWN_USER = "unknown_user"
    RATE_LIMITED = "rate_limited"
    IP_BLOCKED = "ip_blocked"
    GEO_RESTRICTED = "geo_restricted"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AuthenticationLog(Base):
    """
    Append-only authentication event audit record.

    Every significant auth event in SentinelAI produces exactly one record.
    Records must never be modified after creation — the application layer
    must enforce this by never issuing UPDATE statements against this table.

    Consider enabling PostgreSQL Row Security Policy (RLS) to enforce
    append-only at the database level for the application DB role.
    """

    __tablename__ = "authentication_logs"

    # user_id is nullable: pre-auth failures (e.g., unknown username) have no user
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    event_type: Mapped[AuthEventType] = mapped_column(
        Enum(AuthEventType, name="auth_event_type", create_type=True),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # Network / Device Identifiers (all sensitive fields encrypted)
    # ------------------------------------------------------------------
    ip_address_encrypted: Mapped[Optional[bytes]] = mapped_column(
        BYTEA,
        nullable=True,
        comment="AES-256-GCM encrypted client IP address",
    )
    ip_address_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256(ip_address) — enables threat correlation without decryption",
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="HTTP User-Agent header (truncated at 512 chars)",
    )
    device_fingerprint_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256 of client device fingerprint for cross-session correlation",
    )

    # ------------------------------------------------------------------
    # Session / Request Correlation
    # ------------------------------------------------------------------
    session_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        index=True,
        comment="JWT jti claim or server-side session ID",
    )
    request_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="Distributed trace request ID (X-Request-ID header)",
    )

    # ------------------------------------------------------------------
    # MFA
    # ------------------------------------------------------------------
    mfa_method: Mapped[Optional[MFAMethod]] = mapped_column(
        Enum(MFAMethod, name="mfa_method", create_type=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Risk Assessment
    # ------------------------------------------------------------------
    risk_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        comment="ML risk score 0.0 (benign) to 1.0 (critical threat) — backfilled async",
    )
    is_suspicious: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="True if risk_score exceeds the suspicious threshold",
    )

    # ------------------------------------------------------------------
    # Geolocation (non-PII: country, city, ASN only — no precise coordinates)
    # ------------------------------------------------------------------
    geolocation: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment='E.g. {"country": "US", "city": "New York", "asn": "AS15169"}',
    )

    # ------------------------------------------------------------------
    # Event Details
    # ------------------------------------------------------------------
    failure_reason: Mapped[Optional[AuthFailureReason]] = mapped_column(
        Enum(AuthFailureReason, name="auth_failure_reason", create_type=True),
        nullable=True,
        comment="Structured failure code for failed auth events",
    )
    failure_detail: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="Human-readable failure detail (never expose to end-user)",
    )
    event_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Extended event data: TLS version, cipher suite, device OS, etc.",
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user: Mapped[Optional["User"]] = relationship("User", back_populates="auth_logs")

    # ------------------------------------------------------------------
    # Composite Indexes
    # ------------------------------------------------------------------
    __table_args__ = (
        Index("ix_auth_logs_user_event_type", "user_id", "event_type"),
        Index("ix_auth_logs_ip_hash_event", "ip_address_hash", "event_type"),
        Index("ix_auth_logs_suspicious_risk", "is_suspicious", "risk_score"),
        Index("ix_auth_logs_created_at", "created_at"),
        Index("ix_auth_logs_session", "session_id"),
        {
            "comment": "Append-only authentication and session event audit log",
        },
    )

    # ------------------------------------------------------------------
    # Encrypted Property Accessors
    # ------------------------------------------------------------------

    @property
    def ip_address(self) -> Optional[str]:
        """Decrypts and returns the client IP address."""
        if not self.ip_address_encrypted:
            return None
        return decrypt_field(bytes(self.ip_address_encrypted))

    @ip_address.setter
    def ip_address(self, value: str) -> None:
        """Encrypts the IP and stores its hash for correlation queries."""
        self.ip_address_encrypted = encrypt_field(value)
        self.ip_address_hash = hash_field(value)

    def __repr__(self) -> str:
        return (
            f"<AuthLog id={self.id!s:.8} user={self.user_id!s:.8 if self.user_id else 'anon'} "
            f"event={self.event_type} risk={self.risk_score:.3f} suspicious={self.is_suspicious}>"
        )
