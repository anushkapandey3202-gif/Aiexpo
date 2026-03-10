"""
SentinelAI - RBAC ORM Models
==============================
Implements a full Role-Based Access Control system with:

  Permission   — atomic (resource, action) pairs (e.g., 'threat_alerts:read')
  Role         — named group of permissions, optionally scoped to an organization
  RolePermission — M2M junction granting a permission to a role (audited)
  UserRole     — M2M junction assigning a role to a user with optional expiry
                 and full revocation audit trail

Design notes:
- System roles and permissions are flagged is_system_* and cannot be deleted.
- UserRole supports time-bounded access (expires_at) for least-privilege patterns.
- All M2M junctions record the granting actor and timestamp for compliance audits.
- Organization-scoped roles allow per-tenant custom RBAC without schema duplication.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel_ai.database.base import Base

if TYPE_CHECKING:
    from sentinel_ai.database.models.user import User


# ---------------------------------------------------------------------------
# Permission
# ---------------------------------------------------------------------------

class Permission(Base):
    """
    Atomic permission record: a single (resource, action) pair.

    Naming convention: '{resource}:{action}'
    Examples:
      - 'threat_alerts:read'
      - 'users:admin'
      - 'voiceprints:write'
      - 'ml_models:execute'
    """

    __tablename__ = "permissions"

    name: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
        index=True,
        comment="Canonical permission key, e.g. 'threat_alerts:read'",
    )
    resource: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Protected resource name, e.g. 'threat_alerts'",
    )
    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="Action on the resource: read | write | delete | admin | execute",
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_system_permission: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="System permissions are seeded at startup and cannot be deleted",
    )

    role_permissions: Mapped[List["RolePermission"]] = relationship(
        "RolePermission",
        back_populates="permission",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("resource", "action", name="uq_permissions_resource_action"),
        {
            "comment": "Atomic resource-action permission definitions for RBAC",
        },
    )

    def __repr__(self) -> str:
        return f"<Permission {self.name!r} system={self.is_system_permission}>"


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

class Role(Base):
    """
    Named role grouping a set of permissions.

    Roles can be:
    - Global system roles (is_system_role=True): SUPER_ADMIN, ANALYST, VIEWER, etc.
    - Organization-scoped custom roles (organization_id set): for tenant-defined RBAC.

    System roles are seeded at deployment and cannot be modified or deleted.
    """

    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Machine-readable role name, e.g. 'SUPER_ADMIN', 'SOC_ANALYST'",
    )
    display_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Human-readable label shown in the UI",
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_system_role: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Seeded at startup — cannot be modified or deleted",
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Auto-assigned to new users on registration",
    )
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="NULL = global role; set for tenant-scoped custom roles",
    )

    role_permissions: Mapped[List["RolePermission"]] = relationship(
        "RolePermission",
        back_populates="role",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="role",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Role names must be unique per organization (NULL org = global namespace)
        UniqueConstraint("organization_id", "name", name="uq_roles_org_name"),
        Index("ix_roles_org_system", "organization_id", "is_system_role"),
        {
            "comment": "Named roles grouping permissions for platform RBAC",
        },
    )

    @property
    def permissions(self) -> List[Permission]:
        """Returns the list of Permission objects attached to this role."""
        return [rp.permission for rp in self.role_permissions if rp.permission is not None]

    def __repr__(self) -> str:
        return (
            f"<Role {self.name!r} system={self.is_system_role} "
            f"org={self.organization_id!s:.8 if self.organization_id else 'global'}>"
        )


# ---------------------------------------------------------------------------
# RolePermission (M2M junction)
# ---------------------------------------------------------------------------

class RolePermission(Base):
    """
    Many-to-many junction table: Role <-> Permission.
    Adds an audit trail recording who granted the permission to the role.
    """

    __tablename__ = "role_permissions"

    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    permission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    granted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User ID of the admin who added this permission to the role",
    )

    role: Mapped["Role"] = relationship("Role", back_populates="role_permissions")
    permission: Mapped["Permission"] = relationship("Permission", back_populates="role_permissions")
    granting_user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[granted_by])

    __table_args__ = (
        UniqueConstraint(
            "role_id", "permission_id", name="uq_role_permissions_role_perm"
        ),
        {
            "comment": "Role-permission assignments with granting-actor audit trail",
        },
    )

    def __repr__(self) -> str:
        return f"<RolePermission role={self.role_id!s:.8} perm={self.permission_id!s:.8}>"


# ---------------------------------------------------------------------------
# UserRole (M2M junction)
# ---------------------------------------------------------------------------

class UserRole(Base):
    """
    Many-to-many junction table: User <-> Role.

    Supports:
    - Permanent role assignments (expires_at = NULL)
    - Time-bounded access grants (expires_at set — for temporary elevation)
    - Soft-revocation with revoked_at + revoked_by audit trail
    - Only assignments where is_active=True AND expires_at > NOW() are effective
    """

    __tablename__ = "user_roles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    granted_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Admin who granted this role to the user",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="NULL = permanent; set for time-bounded privilege elevation",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="Admin who revoked this role assignment",
    )
    revocation_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Audit note explaining why the role was revoked",
    )

    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="user_roles",
    )
    role: Mapped["Role"] = relationship("Role", back_populates="user_roles")
    granting_admin: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[granted_by]
    )
    revoking_admin: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[revoked_by]
    )

    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_user_roles_user_role"),
        Index("ix_user_roles_user_active", "user_id", "is_active"),
        Index("ix_user_roles_expires", "expires_at"),
        {
            "comment": "User-role assignments with time-bound and revocation support",
        },
    )

    @property
    def is_expired(self) -> bool:
        """True if this assignment has passed its expiry time."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_effective(self) -> bool:
        """True if the assignment is both active and within its validity window."""
        return self.is_active and not self.is_expired

    def revoke(self, revoked_by_user_id: uuid.UUID, reason: Optional[str] = None) -> None:
        """
        Soft-revokes this role assignment.
        Idempotent: calling on an already-revoked assignment is a no-op.
        """
        if not self.is_active:
            return
        self.is_active = False
        self.revoked_at = datetime.now(timezone.utc)
        self.revoked_by = revoked_by_user_id
        self.revocation_reason = reason

    def __repr__(self) -> str:
        return (
            f"<UserRole user={self.user_id!s:.8} role={self.role_id!s:.8} "
            f"active={self.is_active} expires={self.expires_at}>"
        )
