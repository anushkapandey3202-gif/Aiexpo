"""
SentinelAI - Fusion Session Store (Redis)
==========================================
Manages the lifecycle of ThreatScoreAggregation objects in Redis.

Each in-flight session aggregator is stored at:
    Key:   fusion:session:{session_id}
    Value: JSON-serialised ThreatScoreAggregation
    TTL:   FusionWeightConfig.session_ttl_seconds (default: 120s)

TTL-Expiry Worker:
  A background asyncio task scans for sessions whose TTL has passed without
  receiving all expected model scores. These are fused with available data
  (is_partial=True) and forwarded to the fusion pipeline.

  The expiry worker uses a Redis sorted set `fusion:expiry_index` where the
  score is the expiry Unix timestamp. A ZRANGEBYSCORE query efficiently finds
  all expired sessions without scanning all keys.

Concurrency:
  Redis operations are atomic at the command level. The Lua-based
  `compare_and_swap_status` script prevents double-fusion races
  (e.g., TTL expiry firing concurrently with the final score arriving).
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from redis.asyncio import Redis

from sentinel_ai.services.risk_fusion.schemas.fusion import (
    FusionStatus,
    ThreatScoreAggregation,
    FusionWeightConfig,
)

logger = logging.getLogger(__name__)

# Redis key prefixes
_SESSION_PREFIX = "fusion:session:"
_EXPIRY_INDEX   = "fusion:expiry_index"   # sorted set: member=session_id, score=expiry_ts

# Lua script: atomically transition session status only if current status matches expected
# Prevents double-fusion races between TTL expiry and final score arrival
_CAS_STATUS_SCRIPT = """
local key    = KEYS[1]
local expect = ARGV[1]
local newval = ARGV[2]
local raw    = redis.call("GET", key)
if raw == false then return 0 end
local obj = cjson.decode(raw)
if obj["status"] ~= expect then return 0 end
obj["status"] = newval
redis.call("SET", key, cjson.encode(obj))
return 1
"""


class FusionSessionStore:
    """
    Redis-backed store for in-flight ThreatScoreAggregation objects.

    One instance is shared across all Kafka consumer tasks.
    All public methods are async-safe.
    """

    def __init__(self, redis_client: Redis, config: Optional[FusionWeightConfig] = None) -> None:
        self._redis = redis_client
        self._config = config or FusionWeightConfig()
        self._ttl_seconds = self._config.session_ttl_seconds
        self._expiry_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_expiry_worker(self) -> None:
        """Starts the background TTL-expiry scan task."""
        self._expiry_task = asyncio.create_task(
            self._expiry_worker_loop(),
            name="fusion-session-expiry",
        )
        logger.info(
            "FusionSessionStore expiry worker started",
            extra={"scan_interval_seconds": 10, "session_ttl": self._ttl_seconds},
        )

    async def stop_expiry_worker(self) -> None:
        """Cancels the expiry worker gracefully."""
        if self._expiry_task and not self._expiry_task.done():
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
        logger.info("FusionSessionStore expiry worker stopped")

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    async def create(self, aggregation: ThreatScoreAggregation) -> None:
        """
        Persists a new ThreatScoreAggregation and registers it in the expiry index.
        Idempotent — if the session already exists, the existing record is preserved.
        """
        session_key = f"{_SESSION_PREFIX}{aggregation.session_id}"

        # Only SET if key doesn't exist (NX = Not eXists)
        set_result = await self._redis.set(
            session_key,
            aggregation.model_dump_json(),
            ex=self._ttl_seconds,
            nx=True,
        )

        if set_result is None:
            # Key already exists — session already being tracked
            logger.debug(
                "Fusion session already exists in store — skipping create",
                extra={"session_id": aggregation.session_id},
            )
            return

        # Register in expiry sorted set (score = expiry Unix timestamp)
        expiry_ts = time.time() + self._ttl_seconds
        await self._redis.zadd(
            _EXPIRY_INDEX,
            {aggregation.session_id: expiry_ts},
        )

        logger.debug(
            "Fusion session created in store",
            extra={
                "session_id": aggregation.session_id,
                "ttl_seconds": self._ttl_seconds,
                "expiry_ts": expiry_ts,
            },
        )

    async def get(self, session_id: str) -> Optional[ThreatScoreAggregation]:
        """
        Retrieves a session aggregator by session_id.
        Returns None if not found (expired or never created).
        """
        session_key = f"{_SESSION_PREFIX}{session_id}"
        raw = await self._redis.get(session_key)
        if raw is None:
            return None
        return ThreatScoreAggregation.model_validate_json(raw)

    async def update(self, aggregation: ThreatScoreAggregation) -> None:
        """
        Overwrites the stored session state, resetting the TTL.
        ONLY call this for ACCUMULATING sessions — completed sessions must not be modified.
        """
        if aggregation.status not in (
            FusionStatus.ACCUMULATING,
            FusionStatus.FUSING,
        ):
            logger.warning(
                "Attempted to update a terminal session state — ignored",
                extra={
                    "session_id": aggregation.session_id,
                    "status": aggregation.status,
                },
            )
            return

        session_key = f"{_SESSION_PREFIX}{aggregation.session_id}"
        await self._redis.setex(
            session_key,
            self._ttl_seconds,
            aggregation.model_dump_json(),
        )

    async def delete(self, session_id: str) -> None:
        """Removes a session from the store and the expiry index."""
        await self._redis.delete(f"{_SESSION_PREFIX}{session_id}")
        await self._redis.zrem(_EXPIRY_INDEX, session_id)

    # ------------------------------------------------------------------
    # Compare-and-Swap Status (atomic Lua script)
    # ------------------------------------------------------------------

    async def try_claim_for_fusion(self, session_id: str) -> bool:
        """
        Atomically transitions a session from ACCUMULATING → FUSING.
        Returns True if this caller successfully claimed the session for fusion.
        Returns False if another task already claimed it (race condition guard).

        This prevents double-fusion in scenarios where:
          - The TTL expiry worker fires at the same time as the final score arrives.
          - Multiple Kafka consumer replicas process the same partition.
        """
        session_key = f"{_SESSION_PREFIX}{session_id}"
        result = await self._redis.eval(
            _CAS_STATUS_SCRIPT,
            1,              # Number of KEYS
            session_key,    # KEYS[1]
            FusionStatus.ACCUMULATING.value,  # ARGV[1] — expected current status
            FusionStatus.FUSING.value,        # ARGV[2] — new status to set
        )
        claimed = bool(result)

        if claimed:
            logger.debug(
                "Session claimed for fusion (CAS ACCUMULATING→FUSING)",
                extra={"session_id": session_id},
            )
        else:
            logger.debug(
                "Session claim failed — already being fused or completed",
                extra={"session_id": session_id},
            )

        return claimed

    # ------------------------------------------------------------------
    # TTL Expiry Worker
    # ------------------------------------------------------------------

    async def _expiry_worker_loop(self) -> None:
        """
        Background task that scans the expiry sorted set every 10 seconds.
        Sessions whose expiry timestamp has passed are yielded to the fusion
        pipeline as partial fusions (is_partial=True).

        Uses a callback pattern: callers register an expiry_callback via
        set_expiry_callback() which is invoked with the expired session.
        """
        logger.info("Fusion session expiry worker loop started")
        while True:
            try:
                await asyncio.sleep(10)
                await self._process_expired_sessions()
            except asyncio.CancelledError:
                logger.info("Fusion expiry worker loop cancelled")
                return
            except Exception:
                logger.error(
                    "Unexpected error in fusion expiry worker",
                    exc_info=True,
                )

    async def _process_expired_sessions(self) -> None:
        """
        Finds all sessions whose expiry timestamp ≤ now and triggers
        partial fusion via the registered callback.
        """
        now_ts = time.time()
        expired_ids: list[str] = await self._redis.zrangebyscore(
            _EXPIRY_INDEX,
            min=0,
            max=now_ts,
        )

        if not expired_ids:
            return

        logger.info(
            "Processing expired fusion sessions",
            extra={"count": len(expired_ids)},
        )

        for session_id_bytes in expired_ids:
            session_id = (
                session_id_bytes.decode()
                if isinstance(session_id_bytes, bytes)
                else session_id_bytes
            )

            try:
                aggregation = await self.get(session_id)
                if aggregation is None:
                    # Already cleaned up
                    await self._redis.zrem(_EXPIRY_INDEX, session_id)
                    continue

                if aggregation.status != FusionStatus.ACCUMULATING:
                    # Already fused or in progress — remove from expiry index only
                    await self._redis.zrem(_EXPIRY_INDEX, session_id)
                    continue

                # Invoke the registered expiry callback
                if self._expiry_callback is not None:
                    await self._expiry_callback(aggregation)

                # Clean up expiry index entry (session key TTL handles the rest)
                await self._redis.zrem(_EXPIRY_INDEX, session_id)

            except Exception:
                logger.error(
                    "Error processing expired fusion session",
                    extra={"session_id": session_id},
                    exc_info=True,
                )

    def set_expiry_callback(self, callback) -> None:
        """
        Registers the async callable invoked when a session TTL expires.
        Signature: async def callback(aggregation: ThreatScoreAggregation) -> None
        """
        self._expiry_callback = callback

    # Initialize callback slot
    _expiry_callback = None
