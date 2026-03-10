"""
SentinelAI - User ORM Model
============================
Core user entity for the SentinelAI platform.

Security design:
- Email and last-login IP are AES-256-GCM encrypted at rest (BYTEA columns).
- An email_hash (SHA-256) column allows unique-constraint enforcement and
  indexed lookups without ever storing plaintext email in the database.
- MFA secret and backup codes are also encrypted at rest.
- Soft-delete pattern preserves audit trail while hiding deleted accounts.
- account lockout is tracked via failed_login_attempts + locked_until.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel_ai.database.base import Base
from sentinel_ai.database.encryption import decrypt_field, encrypt_field, hash_field

if TYPE_CHECKING:
    from sentinel_ai.database.models.auth_log import AuthenticationLog
    from sentinel_ai.database.models.rbac import UserRole
    from sentinel_ai.database.models.threat_alert import ThreatAlert


class User(Base):
    """
    Platform user account.

    Columns ending in `_encrypted` contain AES-256-GCM ciphertext blobs.
    Access these only through their corresponding Python property (e.g., `.email`).
    Columns ending in `_hash` are SHA-256 digests of their cleartext counterpart,
    used for indexed equality lookups (WHERE email_hash = SHA256(input)).
    """

    __tablename__ = "users"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique, URL-safe username (alphanumeric + underscore)",
    )
    display_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Optional human-readable display name",
    )

    # Email stored encrypted; SHA-256 hash for unique constraint + lookup
    email_encrypted: Mapped[bytes] = mapped_column(
        BYTEA,
        nullable=False,
        comment="AES-256-GCM encrypted email address",
    )
    email_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="SHA-256(normalize(email)) — enables lookups without decryption",
    )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    hashed_password: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="bcrypt hash of the user's password (cost factor ≥ 12)",
    )
    password_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Timestamp of last password change — drives password-age policy",
    )

    # Multi-Factor Authentication
    mfa_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    mfa_secret_encrypted: Mapped[Optional[bytes]] = mapped_column(
        BYTEA,
        nullable=True,
        comment="AES-256-GCM encrypted TOTP shared secret (RFC 6238)",
    )
    mfa_backup_codes_encrypted: Mapped[Optional[bytes]] = mapped_column(
        BYTEA,
        nullable=True,
        comment="AES-256-GCM encrypted JSON array of one-time backup codes",
    )

    # ------------------------------------------------------------------
    # Account State
    # ------------------------------------------------------------------
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_system_account: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Service accounts — exempt from interactive auth requirements",
    )

    # Brute-force lockout
    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Consecutive failed login attempts since last success",
    )
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Account locked until this UTC timestamp (NULL = not locked)",
    )

    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_login_ip_encrypted: Mapped[Optional[bytes]] = mapped_column(
        BYTEA,
        nullable=True,
        comment="AES-256-GCM encrypted IP address of last successful login",
    )

    # ------------------------------------------------------------------
    # Soft Delete
    # ------------------------------------------------------------------
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Soft-delete timestamp — NULL means not deleted",
    )
    deleted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="ID of admin who performed the soft delete",
    )

    # ------------------------------------------------------------------
    # Extended Metadata (non-PII only)
    # ------------------------------------------------------------------
    profile_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Non-sensitive preferences: timezone, locale, UI settings",
    )
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Tenant isolation key — NULL for super-admin accounts",
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        foreign_keys="UserRole.user_id",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    auth_logs: Mapped[List["AuthenticationLog"]] = relationship(
        "AuthenticationLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    threat_alerts_targeted: Mapped[List["ThreatAlert"]] = relationship(
        "ThreatAlert",
        foreign_keys="ThreatAlert.target_user_id",
        back_populates="target_user",
        lazy="dynamic",
    )

    # ------------------------------------------------------------------
    # Table-level constraints and indexes
    # ------------------------------------------------------------------
    __table_args__ = (
        Index("ix_users_org_active", "organization_id", "is_active"),
        Index("ix_users_org_username", "organization_id", "username"),
        Index("ix_users_deleted_at", "deleted_at"),
        {
            "comment": "Platform user accounts — PII encrypted via AES-256-GCM",
        },
    )

    # ------------------------------------------------------------------
    # Encrypted Property Accessors
    # ------------------------------------------------------------------

    @property
    def email(self) -> str:
        """Decrypts and returns the user's email address."""
        if not self.email_encrypted:
            raise ValueError("No email stored for this user")
        return decrypt_field(bytes(self.email_encrypted))

    @email.setter
    def email(self, value: str) -> None:
        """Encrypts the email and atomically updates the lookup hash."""
        normalized = value.lower().strip()
        self.email_encrypted = encrypt_field(normalized)
        self.email_hash = hash_field(normalized)

    @property
    def last_login_ip(self) -> Optional[str]:
        """Decrypts and returns the last login IP, or None if never logged in."""
        if not self.last_login_ip_encrypted:
            return None
        return decrypt_field(bytes(self.last_login_ip_encrypted))

    @last_login_ip.setter
    def last_login_ip(self, value: str) -> None:
        self.last_login_ip_encrypted = encrypt_field(value)

    # ------------------------------------------------------------------
    # Derived Properties
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """True if the account is currently under a brute-force lockout."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until

    @property
    def is_deleted(self) -> bool:
        """True if the user has been soft-deleted."""
        return self.deleted_at is not None

    def __repr__(self) -> str:
        return (
            f"<User id={self.id!s:.8} username={self.username!r} "
            f"active={self.is_active} org={self.organization_id!s:.8 if self.organization_id else 'global'}>"
        )
