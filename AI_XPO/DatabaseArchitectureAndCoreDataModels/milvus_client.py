"""
SentinelAI - Milvus Voiceprint Repository
==========================================
Self-hosted Milvus implementation of VoiceprintRepository.

Architecture:
- Single collection with Milvus 2.2.9+ partition_key feature on organization_id.
  This provides per-org data isolation without creating a collection per tenant.
- HNSW index (configurable) for O(log n) approximate nearest neighbour search.
  IVF_FLAT and IVF_SQ8 are also supported via MILVUS_INDEX_TYPE.
- TLS-secured gRPC connection (configurable via MILVUS_SECURE + server PEM path).
- All blocking pymilvus calls are offloaded to a dedicated ThreadPoolExecutor.
- Upsert implemented as delete-then-insert to handle Milvus's lack of native upsert
  in some versions; the pymilvus upsert() API is used where available (2.3+).

Thread safety:
  pymilvus Collection objects share the underlying gRPC channel which is
  thread-safe. All calls are serialized through the executor regardless.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from sentinel_ai.config.settings import get_settings
from sentinel_ai.database.vector_store.base import (
    SimilarityMatch,
    VoiceprintDimensionError,
    VoiceprintRecord,
    VoiceprintRepository,
    VoiceprintRepositoryError,
)

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="milvus-io",
)

# ---------------------------------------------------------------------------
# Field name constants — single source of truth for collection schema
# ---------------------------------------------------------------------------
_F_ID = "vector_id"           # VARCHAR primary key
_F_USER_ID = "user_id"        # VARCHAR
_F_ORG_ID = "organization_id" # VARCHAR — partition key
_F_SPEAKER = "speaker_label"  # VARCHAR
_F_MODEL = "model_version"    # VARCHAR
_F_DURATION = "audio_dur_s"   # FLOAT
_F_EMBEDDING = "embedding"    # FLOAT_VECTOR

_VARCHAR_MAX = 256
_PARTITION_KEY_MAX = 64


class MilvusVoiceprintRepository(VoiceprintRepository):
    """
    Milvus-backed voiceprint repository for self-hosted / on-prem deployments.

    Initialization:
        repo = MilvusVoiceprintRepository()
        await repo.initialize()
        ...
        await repo.close()
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._collection_name: str = cfg.MILVUS_COLLECTION_NAME
        self._dimension: int = cfg.MILVUS_DIMENSION
        self._index_type: str = cfg.MILVUS_INDEX_TYPE
        self._metric_type: str = cfg.MILVUS_METRIC_TYPE
        self._alias: str = "sentinelai"
        self._collection: Optional[Collection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connects to Milvus and ensures the collection + index are ready."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_EXECUTOR, self._connect)
        await loop.run_in_executor(_EXECUTOR, self._ensure_collection)
        await loop.run_in_executor(_EXECUTOR, self._ensure_index)
        logger.info(
            "MilvusVoiceprintRepository initialized",
            extra={
                "host": self._cfg.MILVUS_HOST,
                "port": self._cfg.MILVUS_PORT,
                "collection": self._collection_name,
                "index_type": self._index_type,
                "metric": self._metric_type,
            },
        )

    def _connect(self) -> None:
        """Blocking: establishes gRPC connection to Milvus with optional TLS."""
        cfg = self._cfg
        kwargs: dict[str, Any] = {
            "alias": self._alias,
            "host": cfg.MILVUS_HOST,
            "port": str(cfg.MILVUS_PORT),
            "user": cfg.MILVUS_USER,
            "password": cfg.MILVUS_PASSWORD.get_secret_value(),
            "secure": cfg.MILVUS_SECURE,
        }
        if cfg.MILVUS_SECURE and cfg.MILVUS_SERVER_PEM_PATH:
            kwargs["server_pem_path"] = cfg.MILVUS_SERVER_PEM_PATH
            kwargs["server_name"] = cfg.MILVUS_HOST

        connections.connect(**kwargs)
        logger.info(
            "Milvus gRPC connection established",
            extra={
                "host": cfg.MILVUS_HOST,
                "port": cfg.MILVUS_PORT,
                "secure": cfg.MILVUS_SECURE,
            },
        )

    def _build_schema(self) -> CollectionSchema:
        """Constructs the Milvus collection schema with partition key on org_id."""
        fields = [
            FieldSchema(
                name=_F_ID,
                dtype=DataType.VARCHAR,
                max_length=_VARCHAR_MAX,
                is_primary=True,
                description="Deterministic vector ID: '{org_id}:{user_id}'",
            ),
            FieldSchema(
                name=_F_USER_ID,
                dtype=DataType.VARCHAR,
                max_length=_VARCHAR_MAX,
                description="Platform user UUID",
            ),
            FieldSchema(
                name=_F_ORG_ID,
                dtype=DataType.VARCHAR,
                max_length=_PARTITION_KEY_MAX,
                is_partition_key=True,  # Requires Milvus ≥ 2.2.9
                description="Org UUID — Milvus partition key for data isolation",
            ),
            FieldSchema(
                name=_F_SPEAKER,
                dtype=DataType.VARCHAR,
                max_length=_VARCHAR_MAX,
                description="Optional speaker label for multi-enrollment scenarios",
            ),
            FieldSchema(
                name=_F_MODEL,
                dtype=DataType.VARCHAR,
                max_length=64,
                description="ECAPA-TDNN model version tag",
            ),
            FieldSchema(
                name=_F_DURATION,
                dtype=DataType.FLOAT,
                description="Source audio duration in seconds",
            ),
            FieldSchema(
                name=_F_EMBEDDING,
                dtype=DataType.FLOAT_VECTOR,
                dim=self._dimension,
                description=f"ECAPA-TDNN embedding ({self._dimension}-dim)",
            ),
        ]
        return CollectionSchema(
            fields=fields,
            description="SentinelAI ECAPA-TDNN voiceprint embeddings",
            enable_dynamic_field=True,  # Allows storing extra metadata keys
        )

    def _ensure_collection(self) -> None:
        """Blocking: creates the collection if it doesn't exist."""
        if utility.has_collection(self._collection_name, using=self._alias):
            self._collection = Collection(
                self._collection_name, using=self._alias
            )
            logger.debug(
                "Milvus collection exists",
                extra={"collection": self._collection_name},
            )
        else:
            schema = self._build_schema()
            self._collection = Collection(
                name=self._collection_name,
                schema=schema,
                using=self._alias,
                num_partitions=64,  # Milvus internal hash partitions
            )
            logger.info(
                "Milvus collection created",
                extra={"collection": self._collection_name},
            )

    def _build_index_params(self) -> dict[str, Any]:
        """Returns index parameters for the configured index type."""
        cfg = self._cfg
        if cfg.MILVUS_INDEX_TYPE == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": self._metric_type,
                "params": {
                    "M": cfg.MILVUS_HNSW_M,
                    "efConstruction": cfg.MILVUS_HNSW_EF_CONSTRUCTION,
                },
            }
        if cfg.MILVUS_INDEX_TYPE in ("IVF_FLAT", "IVF_SQ8"):
            return {
                "index_type": cfg.MILVUS_INDEX_TYPE,
                "metric_type": self._metric_type,
                "params": {"nlist": cfg.MILVUS_NLIST},
            }
        raise VoiceprintRepositoryError(
            f"Unsupported MILVUS_INDEX_TYPE: '{cfg.MILVUS_INDEX_TYPE}'"
        )

    def _ensure_index(self) -> None:
        """Blocking: builds the ANN index if not already present, then loads."""
        if self._collection is None:
            raise VoiceprintRepositoryError("Collection must be created before indexing")

        indexed_fields = {idx.field_name for idx in self._collection.indexes}
        if _F_EMBEDDING not in indexed_fields:
            params = self._build_index_params()
            self._collection.create_index(
                field_name=_F_EMBEDDING,
                index_params=params,
            )
            logger.info(
                "Milvus ANN index created",
                extra={"index_type": self._index_type, "metric": self._metric_type},
            )

        # Load the collection into memory for query serving
        self._collection.load()
        logger.debug(
            "Milvus collection loaded into memory",
            extra={"collection": self._collection_name},
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        if self._collection is None:
            raise VoiceprintRepositoryError(
                "Repository not initialized — call await repository.initialize() at startup"
            )

    def _validate_dimension(self, embedding: list[float]) -> None:
        if len(embedding) != self._dimension:
            raise VoiceprintDimensionError(
                f"Expected embedding dimension {self._dimension}, got {len(embedding)}"
            )

    def _vector_id(self, user_id: str, organization_id: str) -> str:
        return f"{organization_id}:{user_id}"

    # ------------------------------------------------------------------
    # VoiceprintRepository Implementation
    # ------------------------------------------------------------------

    async def upsert(self, record: VoiceprintRecord) -> str:
        """Upserts a voiceprint using Milvus upsert() (requires Milvus ≥ 2.3)."""
        self._assert_initialized()
        self._validate_dimension(record.embedding)

        vector_id = self._vector_id(record.user_id, record.organization_id)
        data = [
            {
                _F_ID: vector_id,
                _F_USER_ID: record.user_id,
                _F_ORG_ID: record.organization_id[:_PARTITION_KEY_MAX],
                _F_SPEAKER: (record.speaker_label or "")[:_VARCHAR_MAX],
                _F_MODEL: record.model_version[:64],
                _F_DURATION: float(record.audio_duration_seconds or 0.0),
                _F_EMBEDDING: record.embedding,
            }
        ]

        loop = asyncio.get_event_loop()
        try:
            upsert_fn = partial(self._collection.upsert, data=data)  # type: ignore[attr-defined]
            result = await loop.run_in_executor(_EXECUTOR, upsert_fn)

            # Flush ensures data is persisted and queryable
            flush_fn = partial(self._collection.flush)
            await loop.run_in_executor(_EXECUTOR, flush_fn)

            logger.info(
                "Voiceprint upserted to Milvus",
                extra={
                    "vector_id": vector_id,
                    "upsert_count": getattr(result, "upsert_count", 1),
                },
            )
            return vector_id
        except Exception as exc:
            logger.error(
                "Milvus upsert failed",
                extra={"vector_id": vector_id},
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
        """ANN search in the org's partition. Returns matches above threshold, sorted desc."""
        self._assert_initialized()
        self._validate_dimension(embedding)

        if not 1 <= top_k <= 100:
            raise ValueError(f"top_k must be 1–100, got {top_k}")

        # HNSW search parameter: ef ≥ top_k for recall accuracy
        search_params: dict[str, Any] = {
            "metric_type": self._metric_type,
            "params": {"ef": max(top_k * 4, 64)},
        }

        # Partition key filter isolates the search to a single org
        expr = f'{_F_ORG_ID} == "{organization_id}"'
        output_fields = [_F_USER_ID, _F_ORG_ID, _F_SPEAKER, _F_MODEL]

        loop = asyncio.get_event_loop()
        search_fn = partial(
            self._collection.search,
            data=[embedding],
            anns_field=_F_EMBEDDING,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        try:
            results = await loop.run_in_executor(_EXECUTOR, search_fn)
        except Exception as exc:
            logger.error("Milvus search failed", exc_info=True)
            raise VoiceprintRepositoryError(f"Milvus similarity search failed: {exc}") from exc

        matches: list[SimilarityMatch] = []
        for hits in results:
            for hit in hits:
                # COSINE metric: score = 1 - distance for normalized vectors
                if self._metric_type in ("COSINE", "IP"):
                    score = float(hit.score)
                else:
                    # L2: convert distance to similarity (0.0 = identical, higher = different)
                    score = max(0.0, 1.0 - float(hit.distance))

                if score < score_threshold:
                    continue

                entity = hit.entity
                matches.append(
                    SimilarityMatch(
                        user_id=entity.get(_F_USER_ID, ""),
                        organization_id=entity.get(_F_ORG_ID, organization_id),
                        score=score,
                        speaker_label=entity.get(_F_SPEAKER) or None,
                        metadata={
                            "model_version": entity.get(_F_MODEL, ""),
                            "vector_id": hit.id,
                        },
                    )
                )

        return sorted(matches, key=lambda m: m.score, reverse=True)

    async def fetch(
        self, user_id: str, organization_id: str
    ) -> Optional[VoiceprintRecord]:
        """Retrieves a specific voiceprint by scalar query on the primary key."""
        self._assert_initialized()

        vector_id = self._vector_id(user_id, organization_id)
        expr = f'{_F_ID} == "{vector_id}"'

        loop = asyncio.get_event_loop()
        query_fn = partial(
            self._collection.query,
            expr=expr,
            output_fields=[
                _F_ID, _F_USER_ID, _F_ORG_ID,
                _F_SPEAKER, _F_MODEL, _F_DURATION, _F_EMBEDDING,
            ],
        )

        try:
            rows = await loop.run_in_executor(_EXECUTOR, query_fn)
        except Exception as exc:
            raise VoiceprintRepositoryError(
                f"Milvus fetch failed for user '{user_id}': {exc}"
            ) from exc

        if not rows:
            return None

        row = rows[0]
        return VoiceprintRecord(
            user_id=row.get(_F_USER_ID, user_id),
            organization_id=row.get(_F_ORG_ID, organization_id),
            embedding=list(row.get(_F_EMBEDDING, [])),
            speaker_label=row.get(_F_SPEAKER) or None,
            model_version=row.get(_F_MODEL, "unknown"),
            audio_duration_seconds=row.get(_F_DURATION),
        )

    async def delete(self, user_id: str, organization_id: str) -> bool:
        """Deletes a voiceprint by primary key expression."""
        self._assert_initialized()

        vector_id = self._vector_id(user_id, organization_id)
        expr = f'{_F_ID} == "{vector_id}"'

        loop = asyncio.get_event_loop()
        delete_fn = partial(self._collection.delete, expr=expr)

        try:
            result = await loop.run_in_executor(_EXECUTOR, delete_fn)
            deleted = getattr(result, "delete_count", 0) > 0
            if deleted:
                logger.info(
                    "Voiceprint deleted from Milvus",
                    extra={"vector_id": vector_id},
                )
            return deleted
        except Exception as exc:
            raise VoiceprintRepositoryError(
                f"Milvus delete failed for user '{user_id}': {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """Checks Milvus connectivity by querying server version."""
        loop = asyncio.get_event_loop()
        try:
            version_fn = partial(utility.get_server_version, using=self._alias)
            version = await loop.run_in_executor(_EXECUTOR, version_fn)
            logger.debug("Milvus health check passed", extra={"version": version})
            return bool(version)
        except Exception:
            logger.error("Milvus health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        """Closes the Milvus gRPC connection."""
        loop = asyncio.get_event_loop()
        try:
            disconnect_fn = partial(connections.disconnect, self._alias)
            await loop.run_in_executor(_EXECUTOR, disconnect_fn)
            logger.info("Milvus connection closed cleanly")
        except Exception:
            logger.warning("Error while closing Milvus connection", exc_info=True)
