"""
SentinelAI - SQLAlchemy Async Engine & Declarative Base
========================================================
Provides:
  - Shared DeclarativeBase with UUID PK and audit timestamp columns
  - Async engine factory with production-grade pool configuration
  - Constraint naming convention for deterministic Alembic migrations
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, MetaData, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import AsyncAdaptedQueuePool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constraint Naming Convention
# Alembic autogenerate requires deterministic names to detect diffs reliably.
# ---------------------------------------------------------------------------
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(AsyncAttrs, DeclarativeBase):
    """
    Shared declarative base for all SentinelAI ORM models.

    Every concrete model inheriting from Base automatically gets:
    - `id`         — UUIDv4 primary key (DB-generated fallback via gen_random_uuid())
    - `created_at` — Timezone-aware timestamp set on INSERT
    - `updated_at` — Timezone-aware timestamp updated on every UPDATE

    AsyncAttrs enables `await obj.awaitable_attrs.relationship_name` for lazy
    relationship loading in async contexts without triggering implicit I/O.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        comment="UUIDv4 primary key",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("CURRENT_TIMESTAMP"),
        comment="Record creation timestamp (UTC)",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=text("CURRENT_TIMESTAMP"),
        comment="Last modification timestamp (UTC)",
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes mapped columns to a plain dict.
        Converts UUID → str and datetime → ISO-8601 string.
        Excludes encrypted binary blobs (_encrypted suffix columns).
        """
        result: dict[str, Any] = {}
        for col in self.__table__.columns:
            if col.name.endswith("_encrypted"):
                continue  # Never serialize raw ciphertext
            value = getattr(self, col.name)
            if isinstance(value, uuid.UUID):
                value = str(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            result[col.name] = value
        return result


# ---------------------------------------------------------------------------
# Async Engine Factory
# ---------------------------------------------------------------------------

def create_engine() -> AsyncEngine:
    """
    Creates and returns a configured SQLAlchemy async engine for PostgreSQL.

    Pool settings are loaded from Settings to ensure they're environment-driven.
    pool_pre_ping=True validates stale connections before checkout — critical
    for long-running services where PostgreSQL may close idle connections.
    """
    from sentinel_ai.config.settings import get_settings

    cfg = get_settings()

    ssl_required = cfg.POSTGRES_SSL_MODE not in ("disable", "allow")

    engine = create_async_engine(
        cfg.async_database_url,
        echo=cfg.DEBUG,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=cfg.POSTGRES_POOL_SIZE,
        max_overflow=cfg.POSTGRES_MAX_OVERFLOW,
        pool_timeout=cfg.POSTGRES_POOL_TIMEOUT,
        pool_recycle=cfg.POSTGRES_POOL_RECYCLE,
        pool_pre_ping=True,
        connect_args={
            "ssl": ssl_required,
            "server_settings": {
                "application_name": "sentinelai",
                # Hard cap per-query execution time at the DB level
                "statement_timeout": "30000",  # 30 seconds
                # Locks held for longer than 10s are released to prevent deadlocks
                "lock_timeout": "10000",
            },
        },
    )

    logger.info(
        "PostgreSQL async engine initialized",
        extra={
            "host": cfg.POSTGRES_HOST,
            "port": cfg.POSTGRES_PORT,
            "db": cfg.POSTGRES_DB,
            "pool_size": cfg.POSTGRES_POOL_SIZE,
            "ssl": ssl_required,
        },
    )
    return engine
