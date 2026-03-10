"""
SentinelAI - WebSocket Rate Limiting Middleware
=================================================
Implements a sliding-window rate limiter backed by Redis for WebSocket connections.

Two independent limiters run per connection attempt:
  1. IP-based limiter   — limits connection attempts per source IP
  2. User-based limiter — limits concurrent sessions per authenticated user_id

Uses the Redis sliding window algorithm (sorted set with TTL-based cleanup):
  - ZADD score=now member=request_id
  - ZREMRANGEBYSCORE 0 (now - window_ms)
  - ZCARD to count active requests in window
  - EXPIRE to clean up empty keys

This approach is accurate under high concurrency and works across multiple
ingestion service instances (unlike in-process rate limiting).

Penalty Box:
  Users/IPs that exceed limits are added to a Redis penalty set with a
  PENALTY_TTL. While in the penalty box, all connection attempts are rejected
  with a RATE_LIMITED error and a Retry-After hint.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional, Tuple

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limit Configuration
# ---------------------------------------------------------------------------

# WebSocket connection attempts per IP per window
IP_LIMIT_CONNECTIONS: int = 20
IP_LIMIT_WINDOW_SECONDS: int = 60

# WebSocket connection attempts per user_id per window
USER_LIMIT_CONNECTIONS: int = 10
USER_LIMIT_WINDOW_SECONDS: int = 60

# Audio chunks per session per window (prevents chunk flooding)
CHUNK_LIMIT_PER_SESSION: int = 500
CHUNK_LIMIT_WINDOW_SECONDS: int = 60

# Penalty box TTL for repeated offenders
PENALTY_TTL_SECONDS: int = 300  # 5 minutes

# Redis key prefixes
_IP_RATE_PREFIX = "rate:ws:ip:"
_USER_RATE_PREFIX = "rate:ws:user:"
_CHUNK_RATE_PREFIX = "rate:chunk:session:"
_PENALTY_BOX_PREFIX = "penalty:ws:"


class RateLimitExceededError(Exception):
    """Raised when a rate limit window is breached."""

    def __init__(
        self,
        message: str,
        retry_after_ms: int,
        limit_type: str,
    ) -> None:
        super().__init__(message)
        self.retry_after_ms = retry_after_ms
        self.limit_type = limit_type


class WebSocketRateLimiter:
    """
    Redis-backed sliding window rate limiter for WebSocket connections.

    One instance is shared across all WebSocket handlers (stateless client wrapper).
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Connection Attempt Limiting (called pre-auth on upgrade request)
    # ------------------------------------------------------------------

    async def check_ip_connection_limit(self, ip_address: str) -> None:
        """
        Enforces the IP-based connection attempt rate limit.

        Args:
            ip_address: Client IP string (may be X-Forwarded-For resolved).

        Raises:
            RateLimitExceededError: If the IP has exceeded the connection limit.
        """
        await self._check_penalty_box(f"ip:{ip_address}", "ip")

        allowed, retry_after = await self._sliding_window_check(
            key=f"{_IP_RATE_PREFIX}{ip_address}",
            limit=IP_LIMIT_CONNECTIONS,
            window_seconds=IP_LIMIT_WINDOW_SECONDS,
        )
        if not allowed:
            # Add to penalty box for repeated attempts
            await self._add_to_penalty_box(f"ip:{ip_address}")
            raise RateLimitExceededError(
                f"Too many WebSocket connection attempts from IP {ip_address}. "
                f"Limit: {IP_LIMIT_CONNECTIONS} per {IP_LIMIT_WINDOW_SECONDS}s.",
                retry_after_ms=retry_after,
                limit_type="ip_connection",
            )

    async def check_user_connection_limit(self, user_id: str) -> None:
        """
        Enforces the per-user connection attempt rate limit (post-auth).

        Args:
            user_id: Authenticated user UUID string.

        Raises:
            RateLimitExceededError: If the user has exceeded their connection limit.
        """
        await self._check_penalty_box(f"user:{user_id}", "user")

        allowed, retry_after = await self._sliding_window_check(
            key=f"{_USER_RATE_PREFIX}{user_id}",
            limit=USER_LIMIT_CONNECTIONS,
            window_seconds=USER_LIMIT_WINDOW_SECONDS,
        )
        if not allowed:
            await self._add_to_penalty_box(f"user:{user_id}")
            raise RateLimitExceededError(
                f"User '{user_id}' has exceeded the WebSocket connection limit. "
                f"Limit: {USER_LIMIT_CONNECTIONS} per {USER_LIMIT_WINDOW_SECONDS}s.",
                retry_after_ms=retry_after,
                limit_type="user_connection",
            )

    # ------------------------------------------------------------------
    # Chunk Rate Limiting (called per audio chunk within a session)
    # ------------------------------------------------------------------

    async def check_chunk_limit(self, session_id: str) -> None:
        """
        Enforces the per-session audio chunk submission rate.
        Prevents chunk flooding attacks within an authenticated session.

        Args:
            session_id: Active WebSocket session UUID string.

        Raises:
            RateLimitExceededError: If chunk rate limit is exceeded.
        """
        allowed, retry_after = await self._sliding_window_check(
            key=f"{_CHUNK_RATE_PREFIX}{session_id}",
            limit=CHUNK_LIMIT_PER_SESSION,
            window_seconds=CHUNK_LIMIT_WINDOW_SECONDS,
        )
        if not allowed:
            raise RateLimitExceededError(
                f"Session '{session_id}' is submitting chunks too rapidly. "
                f"Limit: {CHUNK_LIMIT_PER_SESSION} chunks per {CHUNK_LIMIT_WINDOW_SECONDS}s.",
                retry_after_ms=retry_after,
                limit_type="chunk_rate",
            )

    # ------------------------------------------------------------------
    # Sliding Window Algorithm
    # ------------------------------------------------------------------

    async def _sliding_window_check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int]:
        """
        Redis sorted-set sliding window check.

        Returns:
            Tuple of (is_allowed: bool, retry_after_ms: int).
            retry_after_ms is 0 if allowed; positive if rate limited.
        """
        now_ms = int(time.time() * 1000)
        window_start_ms = now_ms - (window_seconds * 1000)
        member = str(uuid.uuid4())

        # Atomic pipeline: add current request, clean old entries, count window size
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.zadd(key, {member: now_ms})
            pipe.zremrangebyscore(key, 0, window_start_ms)
            pipe.zcard(key)
            pipe.expire(key, window_seconds + 1)  # +1 for clock skew
            results = await pipe.execute()

        current_count: int = results[2]

        if current_count > limit:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "key": key,
                    "count": current_count,
                    "limit": limit,
                    "window_seconds": window_seconds,
                },
            )
            # Remove the just-added member since the request is rejected
            await self._redis.zrem(key, member)
            retry_after_ms = window_seconds * 1000
            return False, retry_after_ms

        return True, 0

    # ------------------------------------------------------------------
    # Penalty Box
    # ------------------------------------------------------------------

    async def _check_penalty_box(self, identifier: str, limit_type: str) -> None:
        """
        Rejects immediately if the identifier is in the penalty box.
        Penalty box is populated for repeat rate limit offenders.
        """
        penalty_key = f"{_PENALTY_BOX_PREFIX}{identifier}"
        in_penalty = await self._redis.exists(penalty_key)
        if in_penalty:
            ttl = await self._redis.ttl(penalty_key)
            retry_after_ms = max(0, ttl * 1000)
            raise RateLimitExceededError(
                f"'{identifier}' is in the rate limit penalty box. "
                f"Retry in {ttl} seconds.",
                retry_after_ms=retry_after_ms,
                limit_type=f"{limit_type}_penalty",
            )

    async def _add_to_penalty_box(self, identifier: str) -> None:
        """Adds an identifier to the penalty box with PENALTY_TTL_SECONDS TTL."""
        penalty_key = f"{_PENALTY_BOX_PREFIX}{identifier}"
        await self._redis.setex(penalty_key, PENALTY_TTL_SECONDS, "1")
        logger.warning(
            "Identifier added to rate limit penalty box",
            extra={
                "identifier": identifier,
                "ttl_seconds": PENALTY_TTL_SECONDS,
            },
        )

    async def remove_from_penalty_box(self, identifier: str) -> None:
        """
        Administrative method to release an identifier from the penalty box.
        Used by admin API endpoints for manual intervention.
        """
        penalty_key = f"{_PENALTY_BOX_PREFIX}{identifier}"
        await self._redis.delete(penalty_key)
        logger.info(
            "Identifier removed from rate limit penalty box",
            extra={"identifier": identifier},
        )
