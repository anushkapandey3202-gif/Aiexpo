"""
SentinelAI ML Service — Pinecone Vector Store Client

Manages voiceprint embedding storage and cosine similarity retrieval
against the Pinecone index that stores enrolled speaker embeddings.

Production Features:
- Async wrapper around the synchronous Pinecone SDK using thread executor.
- Circuit breaker pattern to fail fast under Pinecone unavailability.
- Retry with exponential backoff for transient network errors.
- Structured result mapping to domain VoiceprintMatch schemas.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import List, Optional

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import VoiceprintMatch

logger = get_logger("vector_store.pinecone_client")


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast; not forwarding calls
    HALF_OPEN = "half_open"  # Testing if backend recovered


class CircuitBreaker:
    """
    Simple async circuit breaker for Pinecone calls.
    Transitions: CLOSED → OPEN after `failure_threshold` failures,
    OPEN → HALF_OPEN after `recovery_timeout_s`, then CLOSED on success.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        success_threshold: int = 2,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_s
        self._success_threshold = success_threshold
        self._failures = 0
        self._successes = 0
        self._last_failure_time: float = 0.0
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, coro):
        """Execute coroutine through the circuit breaker."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._successes = 0
                    logger.info("Circuit breaker → HALF_OPEN (testing recovery).")
                else:
                    raise RuntimeError(
                        f"Pinecone circuit breaker is OPEN. "
                        f"Retry after {self._recovery_timeout}s."
                    )

        try:
            result = await coro
            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._successes += 1
                    if self._successes >= self._success_threshold:
                        self._state = CircuitState.CLOSED
                        self._failures = 0
                        logger.info("Circuit breaker → CLOSED (backend recovered).")
                elif self._state == CircuitState.CLOSED:
                    self._failures = 0
            return result
        except Exception:
            async with self._lock:
                self._failures += 1
                self._last_failure_time = time.monotonic()
                if self._failures >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        "Circuit breaker → OPEN.",
                        extra={"failures": self._failures, "threshold": self._failure_threshold},
                    )
            raise


# ---------------------------------------------------------------------------
# Pinecone Client
# ---------------------------------------------------------------------------


class PineconeVoiceprintClient:
    """
    Async-safe Pinecone client for voiceprint embedding operations.

    Operations:
    - query()  : Nearest-neighbor cosine search for a given embedding.
    - upsert() : Enroll or update a speaker's voiceprint.
    - delete() : Remove a speaker's voiceprint (GDPR / right to erasure).
    - health_check() : Ping the index and return latency.
    """

    _MAX_RETRIES = 3
    _BASE_BACKOFF_S = 0.5

    def __init__(self) -> None:
        self._settings = get_settings()
        self._index = None
        self._circuit_breaker = CircuitBreaker()
        self._initialized = False

    async def initialize(self) -> None:
        """
        Connect to Pinecone and get index handle.
        Called once at service startup — not in __init__ to allow async context.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_initialize)
        self._initialized = True
        logger.info(
            "PineconeVoiceprintClient initialized.",
            extra={
                "index": self._settings.PINECONE_INDEX_NAME,
                "environment": self._settings.PINECONE_ENVIRONMENT,
            },
        )

    def _sync_initialize(self) -> None:
        """Synchronous Pinecone initialization in thread executor."""
        try:
            from pinecone import Pinecone

            pc = Pinecone(
                api_key=self._settings.PINECONE_API_KEY.get_secret_value()
            )
            self._index = pc.Index(self._settings.PINECONE_INDEX_NAME)
        except ImportError:
            logger.warning(
                "pinecone-client package not installed. "
                "Voiceprint lookups will return empty results.",
            )
            self._index = None
        except Exception as exc:
            logger.error(
                "Failed to initialize Pinecone client.",
                extra={"error": str(exc)},
                exc_info=True,
            )
            raise

    async def query(
        self,
        embedding: np.ndarray,
        speaker_id_filter: Optional[str] = None,
    ) -> List[VoiceprintMatch]:
        """
        Query top-K nearest voiceprints via cosine similarity.

        Args:
            embedding:         L2-normalized float32 numpy array, shape [192].
            speaker_id_filter: If provided, restrict search to this speaker's namespace.

        Returns:
            List of VoiceprintMatch sorted by cosine_score descending.
        """
        if not self._initialized or self._index is None:
            logger.warning("Pinecone not available; returning empty voiceprint matches.")
            return []

        embedding_list = embedding.tolist()
        metadata_filter = (
            {"speaker_id": {"$eq": speaker_id_filter}} if speaker_id_filter else None
        )

        async def _do_query():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._index.query(
                    vector=embedding_list,
                    top_k=self._settings.PINECONE_TOP_K,
                    namespace=self._settings.PINECONE_NAMESPACE,
                    include_metadata=True,
                    filter=metadata_filter,
                ),
            )

        for attempt in range(self._MAX_RETRIES):
            try:
                response = await self._circuit_breaker.call(_do_query())
                return self._parse_query_response(response)
            except RuntimeError as e:
                if "circuit breaker is OPEN" in str(e):
                    logger.warning("Pinecone circuit breaker open; bypassing voiceprint step.")
                    return []
                raise
            except Exception as exc:
                wait = self._BASE_BACKOFF_S * (2 ** attempt)
                logger.warning(
                    "Pinecone query failed; retrying.",
                    extra={"attempt": attempt + 1, "wait_s": wait, "error": str(exc)},
                )
                if attempt < self._MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
                else:
                    logger.error("Pinecone query exhausted retries.", exc_info=True)
                    return []

        return []

    def _parse_query_response(self, response) -> List[VoiceprintMatch]:
        """Map raw Pinecone query response to typed VoiceprintMatch objects."""
        matches: List[VoiceprintMatch] = []
        if response is None or not hasattr(response, "matches"):
            return matches

        for match in response.matches:
            score = float(match.score) if hasattr(match, "score") else 0.0
            metadata = dict(match.metadata) if hasattr(match, "metadata") and match.metadata else {}
            speaker_id = metadata.get("speaker_id", match.id)
            matches.append(
                VoiceprintMatch(
                    speaker_id=speaker_id,
                    cosine_score=min(max(score, 0.0), 1.0),
                    metadata=metadata,
                )
            )

        return sorted(matches, key=lambda m: m.cosine_score, reverse=True)

    async def upsert(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Enroll or update a speaker voiceprint.
        Returns True on success, False on failure (non-blocking).
        """
        if not self._initialized or self._index is None:
            return False

        vector_id = f"spk_{speaker_id}"
        upsert_metadata = {"speaker_id": speaker_id, **(metadata or {})}
        embedding_list = embedding.tolist()

        async def _do_upsert():
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._index.upsert(
                    vectors=[(vector_id, embedding_list, upsert_metadata)],
                    namespace=self._settings.PINECONE_NAMESPACE,
                ),
            )

        try:
            await self._circuit_breaker.call(_do_upsert())
            logger.info("Voiceprint upserted.", extra={"speaker_id": speaker_id})
            return True
        except Exception as exc:
            logger.error(
                "Voiceprint upsert failed.",
                extra={"speaker_id": speaker_id, "error": str(exc)},
            )
            return False

    async def delete(self, speaker_id: str) -> bool:
        """Delete a speaker's voiceprint (GDPR erasure support)."""
        if not self._initialized or self._index is None:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._index.delete(
                    ids=[f"spk_{speaker_id}"],
                    namespace=self._settings.PINECONE_NAMESPACE,
                ),
            )
            logger.info("Voiceprint deleted.", extra={"speaker_id": speaker_id})
            return True
        except Exception as exc:
            logger.error(
                "Voiceprint deletion failed.",
                extra={"speaker_id": speaker_id, "error": str(exc)},
            )
            return False

    async def health_check(self) -> dict:
        """Ping the Pinecone index and return status/latency."""
        if not self._initialized or self._index is None:
            return {"healthy": False, "detail": "Pinecone not initialized"}

        start = time.perf_counter()
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._index.describe_index_stats()
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "healthy": True,
                "latency_ms": round(latency_ms, 2),
                "circuit_state": self._circuit_breaker.state.value,
            }
        except Exception as exc:
            return {
                "healthy": False,
                "detail": str(exc),
                "circuit_state": self._circuit_breaker.state.value,
            }


# Module-level singleton
_pinecone_client: Optional[PineconeVoiceprintClient] = None


def get_pinecone_client() -> PineconeVoiceprintClient:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = PineconeVoiceprintClient()
    return _pinecone_client
