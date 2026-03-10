"""
SentinelAI - Alembic Async Migration Environment
==================================================
Configures Alembic to work with the SQLAlchemy 2.0 async engine (asyncpg).
Supports both offline (SQL script) and online (direct DB) migration modes.

Features:
- Pulls DB URL from application Settings (never hard-coded)
- Imports all ORM models before autogenerate to ensure complete schema diff
- Compares server defaults and column types for accurate migration detection
- Async-capable via asyncio.run() wrapping run_sync()

Usage:
  # Generate a new migration
  alembic revision --autogenerate -m "add threat_alerts_external_ticket"

  # Apply pending migrations
  alembic upgrade head

  # Rollback one migration
  alembic downgrade -1
"""
import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Import all ORM models so that Base.metadata is fully populated for autogenerate.
# Alembic inspects Base.metadata — any model NOT imported here will be invisible.
# ---------------------------------------------------------------------------
from sentinel_ai.database.base import Base  # noqa: F401 — registers metadata
from sentinel_ai.database.models.auth_log import AuthenticationLog  # noqa: F401
from sentinel_ai.database.models.rbac import (  # noqa: F401
    Permission,
    Role,
    RolePermission,
    UserRole,
)
from sentinel_ai.database.models.threat_alert import ThreatAlert  # noqa: F401
from sentinel_ai.database.models.user import User  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alembic config object — gives access to alembic.ini values
# ---------------------------------------------------------------------------
config = context.config

# Configure Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Inject DB URL from application settings (overrides alembic.ini sqlalchemy.url)
# ---------------------------------------------------------------------------
from sentinel_ai.config.settings import get_settings as _get_settings  # noqa: E402

_settings = _get_settings()
config.set_main_option("sqlalchemy.url", _settings.sync_database_url)

# Target metadata for autogenerate
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline migration mode — generates SQL script without a live DB connection
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Emits migration SQL to stdout or a file without connecting to the database.
    Useful for review, DBAs, and environments where direct DB access is restricted.

    Run with: alembic upgrade head --sql > migration.sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()
        logger.info("Offline migrations emitted to SQL")


# ---------------------------------------------------------------------------
# Online migration mode — applies migrations directly against the database
# ---------------------------------------------------------------------------

def _run_migrations_sync(connection: Connection) -> None:
    """Synchronous migration runner — called inside run_sync() from async context."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Render model-level server defaults for accurate diff detection
        render_as_batch=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_migrations_async() -> None:
    """
    Creates an async engine and runs migrations via run_sync().
    NullPool is intentional — migration connections must not be pooled.
    """
    section = config.get_section(config.config_ini_section, {})
    # Override with async DSN (asyncpg) for the engine
    section["sqlalchemy.url"] = _settings.async_database_url

    connectable = async_engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    logger.info(
        "Running Alembic migrations",
        extra={
            "host": _settings.POSTGRES_HOST,
            "db": _settings.POSTGRES_DB,
            "environment": _settings.ENVIRONMENT,
        },
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_run_migrations_sync)

    await connectable.dispose()
    logger.info("Alembic migrations completed successfully")


def run_migrations_online() -> None:
    """Entry point for online migration mode — wraps async execution."""
    asyncio.run(_run_migrations_async())


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
