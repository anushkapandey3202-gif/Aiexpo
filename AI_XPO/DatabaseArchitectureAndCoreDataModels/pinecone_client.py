"""
SentinelAI - Pinecone Voiceprint Repository
=============================================
Production Pinecone implementation of VoiceprintRepository.

Architecture:
- Serverless Pinecone index (AWS-hosted) with one index per environment.
- Per-organization data isolation enforced via Pinecone namespaces:
    namespace = '{PINECONE_NAMESPACE_PREFIX}:{organization_id}'
  This avoids index fan-out while ensuring org-level isolation at the
  Pinecone query layer.
- Vector ID format: '{organization_id}:{user_id}' (deterministic upserts).
- Pinecone Python SDK v3+ is synchronous; all calls are offloaded to a
  dedicated ThreadPoolExecutor to preserve async event loop responsiveness.
- Index creation is idempotent — safe to call on every startup.

Thread safety:
  AESGCM is stateless; Pinecone Index objects are stateless HTTP clients.
  The ThreadPoolExecutor handles all blocking network I/O.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Optional

from pinecone import Pinecone, ServerlessSpec

from sentinel_ai.config.settings import get_settings
from sentinel_ai.database.vector_store.base import (
    SimilarityMatch,
    VoiceprintDimensionError,
    VoiceprintNotFoundError,
    VoiceprintRecord,
    VoiceprintRepository,
    VoiceprintRepositoryError,
)

logger = logging.getLogger(__name__)

# Dedicated thread pool — avoids contending with FastAPI's default executor
_EXECUTOR = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="pinecone-io",
)


class PineconeVoiceprintRepository(VoiceprintRepository):
    """
    Pinecone-backed voiceprint repository for production cloud deployments.

    Initialization:
        repo = PineconeVoiceprintRepository()
        await repo.initialize()          # idempotent — safe on every startup
        ...
        await repo.close()               # no-op for Pinecone (stateless HTTP)
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._client: Pinecone = Pinecone(
            api_key=cfg.PINECONE_API_KEY.get_secret_value()
        )
        self._index_name: str = cfg.PINECONE_INDEX_NAME
        self._dimension: int = cfg.PINECONE_DIMENSION
        self._metric: str = cfg.PINECONE_METRIC
        self._namespace_prefix: str = cfg.PINECONE_NAMESPACE_PREFIX
        # Index client — set after initialize()
        self._index: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Creates the Pinecone serverless index if absent, then loads the client."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_EXECUTOR, self._create_index_if_not_exists)
        self._index = await loop.run_in_executor(
            _EXECUTOR,
            self._client.Index,
            self._index_name,
        )
        logger.info(
            "PineconeVoiceprintRepository initialized",
            extra={
                "index": self._index_name,
                "dimension": self._dimension,
                "metric": self._metric,
                "environment": self._cfg.PINECONE_ENVIRONMENT,
            },
        )

    def _create_index_if_not_exists(self) -> None:
        """Blocking: creates the serverless index if it doesn't already exist."""
        existing_names = [idx.name for idx in self._client.list_indexes()]
        if self._index_name in existing_names:
            logger.debug(
                "Pinecone index already exists — skipping creation",
                extra={"index": self._index_name},
            )
            return

        logger.info(
            "Creating Pinecone serverless index",
            extra={"index": self._index_name, "dimension": self._dimension},
        )
        self._client.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._metric,
            spec=ServerlessSpec(
                cloud="aws",
                region=self._cfg.AWS_REGION,
            ),
        )
        logger.info(
            "Pinecone index created successfully",
            extra={"index": self._index_name},
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _namespace(self, organization_id: str) -> str:
        """Constructs an org-scoped namespace string."""
        return f"{self._namespace_prefix}:{organization_id}"

    def _vector_id(self, user_id: str, organization_id: str) -> str:
        """Deterministic vector ID for upsert idempotency."""
        return f"{organization_id}:{user_id}"

    def _assert_initialized(self) -> None:
        if self._index is None:
            raise VoiceprintRepositoryError(
                "Repository not initialized — call await repository.initialize() at startup"
            )

    def _validate_dimension(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise VoiceprintDimensionError(
                f"Expected embedding dimension {self._dimension}, got {len(embedding)}"
            )

    # ------------------------------------------------------------------
    # VoiceprintRepository Implementation
    # ------------------------------------------------------------------

    async def upsert(self, record: VoiceprintRecord) -> str:
        """
        Upserts a voiceprint into the org-scoped namespace.
        Existing vectors for the same (org, user) pair are atomically replaced.
        """
        self._assert_initialized()
        self._validate_dimension(record.embedding)

        vector_id = self._vector_id(record.user_id, record.organization_id)
        namespace = self._namespace(record.organization_id)

        # Pinecone metadata values must be scalar (str/int/float/bool)
        metadata: dict[str, Any] = {
            "user_id": record.user_id,
            "organization_id": record.organization_id,
            "speaker_label": record.speaker_label or "",
            "model_version": record.model_version,
            "audio_duration_seconds": float(record.audio_duration_seconds or 0.0),
            "sample_rate": int(record.sample_rate or 0),
        }
        # Merge caller-supplied metadata (scalar values only)
        for k, v in record.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v

        upsert_fn = partial(
            self._index.upsert,
            vectors=[
                {
                    "id": vector_id,
                    "values": record.embedding,
                    "metadata": metadata,
                }
            ],
            namespace=namespace,
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(_EXECUTOR, upsert_fn)
            logger.info(
                "Voiceprint upserted to Pinecone",
                extra={
                    "vector_id": vector_id,
                    "namespace": namespace,
                    "upserted_count": response.upserted_count,
                },
            )
            return vector_id
        except Exception as exc:
            logger.error(
                "Pinecone upsert failed",
                extra={"vector_id": vector_id, "namespace": namespace},
                exc_info=True,
            )
            raise VoiceprintRepositoryError(
                f"Failed to upsert voiceprint for user '{record.user_id}': {exc}"
            ) from exc

    async def query_similar(
        self,
        embedding: list[float],
        organization_id: str,
        top_k: int = 5,
        score_threshold: float = 0.80,
    ) -> list[SimilarityMatch]:
        """
        Cosine nearest-neighbour search within the org's Pinecone namespace.
        Returns matches sorted by score descending, filtered by threshold.
        """
        self._assert_initialized()
        self._validate_dimension(embedding)

        if not 1 <= top_k <= 100:
            raise ValueError(f"top_k must be 1–100, got {top_k}")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(f"score_threshold must be 0.0–1.0, got {score_threshold}")

        namespace = self._namespace(organization_id)
        query_fn = partial(
            self._index.query,
            vector=embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(_EXECUTOR, query_fn)
        except Exception as exc:
            logger.error(
                "Pinecone similarity query failed",
                extra={"namespace": namespace, "top_k": top_k},
                exc_info=True,
            )
            raise VoiceprintRepositoryError(f"Pinecone query failed: {exc}") from exc

        matches: list[SimilarityMatch] = []
        for match in response.matches:
            if match.score < score_threshold:
                continue
            meta = match.metadata or {}
            matches.append(
                SimilarityMatch(
                    user_id=meta.get("user_id", ""),
                    organization_id=meta.get("organization_id", organization_id),
                    score=float(match.score),
                    speaker_label=meta.get("speaker_label") or None,
                    metadata=meta,
                )
            )

        logger.debug(
            "Pinecone similarity query completed",
            extra={
                "namespace": namespace,
                "total_results": len(response.matches),
                "results_above_threshold": len(matches),
                "threshold": score_threshold,
            },
        )
        return matches  # Pinecone returns results sorted by score desc already

    async def fetch(
        self, user_id: str, organization_id: str
    ) -> Optional[VoiceprintRecord]:
        """Fetches a specific voiceprint by (user_id, org_id) pair."""
        self._assert_initialized()

        vector_id = self._vector_id(user_id, organization_id)
        namespace = self._namespace(organization_id)

        fetch_fn = partial(
            self._index.fetch,
            ids=[vector_id],
            namespace=namespace,
        )
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(_EXECUTOR, fetch_fn)
        except Exception as exc:
            raise VoiceprintRepositoryError(
                f"Pinecone fetch failed for user '{user_id}': {exc}"
            ) from exc

        if vector_id not in response.vectors:
            return None

        vec = response.vectors[vector_id]
        meta = vec.metadata or {}
        return VoiceprintRecord(
            user_id=meta.get("user_id", user_id),
            organization_id=meta.get("organization_id", organization_id),
            embedding=list(vec.values),
            speaker_label=meta.get("speaker_label") or None,
            model_version=meta.get("model_version", "unknown"),
            audio_duration_seconds=meta.get("audio_duration_seconds"),
            sample_rate=int(meta.get("sample_rate", 0)) or None,
            metadata=meta,
        )

    async def delete(self, user_id: str, organization_id: str) -> bool:
        """Permanently deletes a voiceprint from Pinecone."""
        self._assert_initialized()

        vector_id = self._vector_id(user_id, organization_id)
        namespace = self._namespace(organization_id)

        delete_fn = partial(
            self._index.delete,
            ids=[vector_id],
            namespace=namespace,
        )
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(_EXECUTOR, delete_fn)
            logger.info(
                "Voiceprint deleted from Pinecone",
                extra={"vector_id": vector_id, "namespace": namespace},
            )
            # Pinecone delete is idempotent — no return count in v3 delete API
            return True
        except Exception as exc:
            logger.error(
                "Pinecone delete failed",
                extra={"vector_id": vector_id},
                exc_info=True,
            )
            raise VoiceprintRepositoryError(
                f"Failed to delete voiceprint for user '{user_id}': {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """
        Validates Pinecone connectivity by describing the index.
        Returns True if the index is in READY state.
        """
        loop = asyncio.get_event_loop()
        try:
            describe_fn = partial(
                self._client.describe_index,
                self._index_name,
            )
            description = await loop.run_in_executor(_EXECUTOR, describe_fn)
            ready: bool = description is not None and description.status.ready
            logger.debug(
                "Pinecone health check",
                extra={"index": self._index_name, "ready": ready},
            )
            return ready
        except Exception:
            logger.error("Pinecone health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        """No-op: Pinecone SDK uses stateless HTTPS — no connections to close."""
        logger.info("PineconeVoiceprintRepository closed (no persistent connections)")
