"""
SentinelAI - Abstract Voiceprint Vector Repository
====================================================
Defines the interface contract that all vector store backends must satisfy.

Domain model:
  VoiceprintRecord  — immutable data class for storage/retrieval
  SimilarityMatch   — immutable result of a nearest-neighbour query

Interface:
  VoiceprintRepository — abstract base class (Strategy pattern)
    Concrete implementations: PineconeVoiceprintRepository, MilvusVoiceprintRepository

All operations are async. Implementations must be safe for concurrent use
from an asyncio event loop (blocking SDK calls must be offloaded to an executor).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Domain Data Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceprintRecord:
    """
    Immutable voiceprint embedding record.

    Fields:
        user_id:                 Platform user identifier (UUID as string)
        organization_id:         Tenant/org isolation key
        embedding:               ECAPA-TDNN float vector (default dim=192)
        speaker_label:           Optional label (e.g., 'CEO_JohnDoe_v2')
        audio_duration_seconds:  Duration of source audio used for enrollment
        sample_rate:             Sample rate of source audio (Hz)
        model_version:           ECAPA-TDNN version tag for drift tracking
        metadata:                Arbitrary key-value pairs (str/int/float/bool only)
    """

    user_id: str
    organization_id: str
    embedding: list[float]
    speaker_label: Optional[str] = None
    audio_duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    model_version: str = "ecapa-tdnn-v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ValueError("user_id must not be empty")
        if not self.organization_id:
            raise ValueError("organization_id must not be empty")
        if not self.embedding:
            raise ValueError("embedding vector must not be empty")
        if len(self.embedding) < 64:
            raise ValueError(
                f"Embedding dimension {len(self.embedding)} seems too low — "
                "ECAPA-TDNN produces 192-dim vectors by default"
            )


@dataclass(frozen=True)
class SimilarityMatch:
    """
    Result of a cosine similarity search against the voiceprint index.

    Fields:
        user_id:         Matched user's platform ID
        organization_id: Matched user's org (always same as query org)
        score:           Cosine similarity 0.0–1.0 (higher = more similar)
        speaker_label:   Optional label stored with the matched embedding
        metadata:        Full metadata dict from the matched vector record
    """

    user_id: str
    organization_id: str
    score: float
    speaker_label: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class VoiceprintRepositoryError(Exception):
    """Base exception for all vector store operations."""


class VoiceprintNotFoundError(VoiceprintRepositoryError):
    """Raised when a requested voiceprint does not exist in the index."""


class VoiceprintDimensionError(VoiceprintRepositoryError):
    """Raised when an embedding vector has an unexpected dimension."""


# ---------------------------------------------------------------------------
# Abstract Repository (Strategy interface)
# ---------------------------------------------------------------------------

class VoiceprintRepository(abc.ABC):
    """
    Abstract interface for voiceprint vector storage.

    Implementors (PineconeVoiceprintRepository, MilvusVoiceprintRepository) must:
    1. Override all abstract methods with async implementations.
    2. Offload any blocking I/O to a ThreadPoolExecutor.
    3. Validate embedding dimensions before sending to the underlying store.
    4. Enforce per-organization data isolation via namespaces or partition keys.
    5. Be safe for concurrent use from multiple asyncio tasks.

    Lifecycle:
        repo = PineconeVoiceprintRepository()
        await repo.initialize()   # ← call once at application startup
        ...
        await repo.close()        # ← call on application shutdown
    """

    @abc.abstractmethod
    async def initialize(self) -> None:
        """
        Performs one-time setup: creates index/collection, builds indexes.
        Must be idempotent — safe to call if the index already exists.
        Call once from the application startup handler.
        """

    @abc.abstractmethod
    async def upsert(self, record: VoiceprintRecord) -> str:
        """
        Inserts or updates a voiceprint embedding.

        If a vector for (user_id, organization_id) already exists it is
        atomically replaced with the new embedding.

        Args:
            record: VoiceprintRecord containing the embedding and metadata.

        Returns:
            The vector ID string assigned by the backend.

        Raises:
            VoiceprintRepositoryError: On any backend failure.
            VoiceprintDimensionError:  If embedding dimension is unexpected.
        """

    @abc.abstractmethod
    async def query_similar(
        self,
        embedding: list[float],
        organization_id: str,
        top_k: int = 5,
        score_threshold: float = 0.80,
    ) -> list[SimilarityMatch]:
        """
        Nearest-neighbour similarity search within an organization's namespace.

        Args:
            embedding:        Query ECAPA-TDNN embedding vector.
            organization_id:  Restricts search to this org's voiceprints.
            top_k:            Maximum number of candidates to return (1–100).
            score_threshold:  Minimum cosine similarity to include in results.

        Returns:
            List of SimilarityMatch, sorted by score descending.
            Empty list if no matches exceed the threshold.

        Raises:
            VoiceprintRepositoryError: On backend failure.
        """

    @abc.abstractmethod
    async def fetch(
        self, user_id: str, organization_id: str
    ) -> Optional[VoiceprintRecord]:
        """
        Retrieves a specific voiceprint by user and org ID.

        Args:
            user_id:         Platform user UUID string.
            organization_id: Tenant/org UUID string.

        Returns:
            VoiceprintRecord if found, None if no voiceprint exists.

        Raises:
            VoiceprintRepositoryError: On backend failure.
        """

    @abc.abstractmethod
    async def delete(self, user_id: str, organization_id: str) -> bool:
        """
        Permanently deletes a voiceprint from the index.

        Args:
            user_id:         Platform user UUID string.
            organization_id: Tenant/org UUID string.

        Returns:
            True if the vector was deleted, False if it didn't exist.

        Raises:
            VoiceprintRepositoryError: On backend failure.
        """

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Verifies backend connectivity and readiness.

        Returns:
            True if the vector store is reachable and the collection/index
            is loaded and ready to serve queries.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Releases all backend resources and connections.
        Call from the application shutdown handler.
        """
