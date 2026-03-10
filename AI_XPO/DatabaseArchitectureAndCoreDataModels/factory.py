"""
SentinelAI - Voiceprint Repository Factory
============================================
Provides a singleton VoiceprintRepository instance selected at runtime
based on the VECTOR_STORE_BACKEND environment variable.

Pattern: Application-level singleton with FastAPI dependency injection.
  - `startup_voiceprint_repository()` — call in FastAPI lifespan startup
  - `shutdown_voiceprint_repository()` — call in FastAPI lifespan shutdown
  - `get_voiceprint_repository()` — FastAPI Depends() injectable

Switching backends (Pinecone ↔ Milvus) requires only an env-var change:
  VECTOR_STORE_BACKEND=pinecone   →  PineconeVoiceprintRepository
  VECTOR_STORE_BACKEND=milvus     →  MilvusVoiceprintRepository
"""
from __future__ import annotations

import logging
from typing import Optional

from sentinel_ai.database.vector_store.base import VoiceprintRepository

logger = logging.getLogger(__name__)

# Module-level singleton — initialized during application startup
_repository: Optional[VoiceprintRepository] = None


# ---------------------------------------------------------------------------
# Startup / Shutdown (call from FastAPI lifespan)
# ---------------------------------------------------------------------------

async def startup_voiceprint_repository() -> None:
    """
    Initializes the VoiceprintRepository singleton.
    Call once from the FastAPI lifespan `startup` phase.

    Example (main.py):
        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from sentinel_ai.database.vector_store.factory import (
            startup_voiceprint_repository,
            shutdown_voiceprint_repository,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await startup_voiceprint_repository()
            yield
            await shutdown_voiceprint_repository()

        app = FastAPI(lifespan=lifespan)
    """
    global _repository
    if _repository is not None:
        logger.warning("VoiceprintRepository already initialized — skipping")
        return

    _repository = await _create_repository()
    logger.info(
        "VoiceprintRepository startup complete",
        extra={"backend": _repository.__class__.__name__},
    )


async def shutdown_voiceprint_repository() -> None:
    """
    Gracefully closes the VoiceprintRepository singleton.
    Call from the FastAPI lifespan `shutdown` phase.
    """
    global _repository
    if _repository is None:
        return
    await _repository.close()
    _repository = None
    logger.info("VoiceprintRepository shut down cleanly")


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------

async def get_voiceprint_repository() -> VoiceprintRepository:
    """
    FastAPI dependency that returns the initialized VoiceprintRepository.
    Raises RuntimeError if called before startup_voiceprint_repository().

    Usage:
        @router.post("/voiceprints/enroll")
        async def enroll_voiceprint(
            payload: EnrollmentRequest,
            repo: VoiceprintRepository = Depends(get_voiceprint_repository),
        ) -> EnrollmentResponse:
            vector_id = await repo.upsert(VoiceprintRecord(...))
            ...
    """
    if _repository is None:
        raise RuntimeError(
            "VoiceprintRepository is not initialized. "
            "Ensure startup_voiceprint_repository() is called in the application lifespan."
        )
    return _repository


# ---------------------------------------------------------------------------
# Internal Factory
# ---------------------------------------------------------------------------

async def _create_repository() -> VoiceprintRepository:
    """Instantiates and initializes the correct backend based on settings."""
    from sentinel_ai.config.settings import get_settings

    cfg = get_settings()
    backend = cfg.VECTOR_STORE_BACKEND

    if backend == "pinecone":
        from sentinel_ai.database.vector_store.pinecone_client import (
            PineconeVoiceprintRepository,
        )
        repo: VoiceprintRepository = PineconeVoiceprintRepository()

    elif backend == "milvus":
        from sentinel_ai.database.vector_store.milvus_client import (
            MilvusVoiceprintRepository,
        )
        repo = MilvusVoiceprintRepository()

    else:
        raise ValueError(
            f"Unsupported VECTOR_STORE_BACKEND: '{backend}'. "
            "Valid options: 'pinecone', 'milvus'."
        )

    logger.info(
        "Initializing VoiceprintRepository backend",
        extra={"backend": backend, "class": repo.__class__.__name__},
    )
    await repo.initialize()
    return repo
