"""
SentinelAI - Async Session Factory & FastAPI Dependency
========================================================
Provides:
  - Module-level async engine singleton (created once at import time)
  - AsyncSessionFactory for programmatic session management
  - get_db_session() context manager for service-layer use
  - get_db() async generator for FastAPI Depends() injection
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from sentinel_ai.database.base import create_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine Singleton
# Created once at module import — reused across the entire application lifetime.
# ---------------------------------------------------------------------------
_engine = create_engine()

# ---------------------------------------------------------------------------
# Session Factory
# expire_on_commit=False is essential for async: after commit() the ORM objects
# remain accessible without triggering an implicit synchronous refresh.
# ---------------------------------------------------------------------------
AsyncSessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# Context Manager (service layer / background tasks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager wrapping a database session with automatic
    commit-on-success and rollback-on-exception semantics.

    Usage (service layer or background tasks):
        async with get_db_session() as session:
            user = await session.get(User, user_id)
            session.add(user)
            # commit happens automatically on __aexit__

    Raises:
        Any exception propagated from the session body — rollback is guaranteed.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception(
                "Database session rolled back due to unhandled exception"
            )
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.
    Session is committed on successful response and rolled back on exception.

    Usage:
        @router.post("/alerts")
        async def create_alert(
            payload: AlertCreateSchema,
            db: AsyncSession = Depends(get_db),
        ) -> AlertSchema:
            ...
    """
    async with get_db_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Startup / Shutdown Hooks
# ---------------------------------------------------------------------------

async def dispose_engine() -> None:
    """
    Gracefully closes all pooled connections.
    Call from the FastAPI lifespan shutdown handler.
    """
    await _engine.dispose()
    logger.info("PostgreSQL connection pool disposed cleanly")
