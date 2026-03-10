"""
SentinelAI - WebSocket JWT Authenticator
==========================================
Handles JWT authentication for WebSocket connections.

WebSocket-specific challenges (vs. HTTP bearer auth):
- HTTP headers are available during the HANDSHAKE only, not on individual frames.
- Browser WebSocket API cannot set custom headers — tokens must travel as a
  query parameter (?token=<JWT>) during the upgrade request.
- Token validation must happen BEFORE the upgrade completes where possible;
  if the upgrade completes first (standard FastAPI behaviour), the server
  must immediately send an ERROR frame and close the connection.

Security decisions:
- RS256 (asymmetric) is used — the public key is safe to distribute;
  private key never leaves the auth service.
- `jti` (JWT ID) claim is required and stored in Redis for revocation checks.
  A revoked JTI is rejected even if the token is not yet expired.
- `audio:ingest` scope is required for WebSocket ingestion access.
- Tokens are short-lived (15 min); the WebSocket session may outlive the token
  — the token is validated once at connect time, not re-checked per frame.
- The raw token string is never logged; only jti and sub are logged.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import jwt
from fastapi import WebSocket
from jwt import ExpiredSignatureError, InvalidSignatureError, PyJWTError
from redis.asyncio import Redis

from sentinel_ai.services.ingestion.schemas.audio import JWTClaims

logger = logging.getLogger(__name__)

# Scope required for WebSocket audio ingestion
REQUIRED_SCOPE = "audio:ingest"

# Redis key prefix for JTI revocation list
_REVOKED_JTI_PREFIX = "revoked_jti:"


class WebSocketAuthError(Exception):
    """Raised when WebSocket JWT authentication fails for any reason."""

    def __init__(self, message: str, is_expired: bool = False) -> None:
        super().__init__(message)
        self.is_expired = is_expired


class WebSocketAuthenticator:
    """
    Validates JWT bearer tokens for WebSocket upgrade requests.

    Responsibilities:
    1. Extracts token from WebSocket query parameters
    2. Verifies RS256 signature against the platform public key
    3. Validates standard claims (exp, iss, aud)
    4. Checks JTI against Redis revocation list
    5. Enforces required scopes (audio:ingest)

    Thread safety: Stateless after __init__; safe for concurrent WebSocket connections.
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client
        self._public_key: Optional[str] = None
        self._algorithm: Optional[str] = None
        self._issuer: Optional[str] = None
        self._audience: Optional[str] = None

    def _load_settings(self) -> None:
        """Lazy-loads settings and public key on first use."""
        if self._public_key is not None:
            return

        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()

        try:
            with open(cfg.JWT_PUBLIC_KEY_PATH, "r") as f:
                self._public_key = f.read()
        except OSError as exc:
            logger.critical(
                "Cannot read JWT public key — WebSocket auth will fail all connections",
                extra={"path": cfg.JWT_PUBLIC_KEY_PATH},
            )
            raise RuntimeError(
                f"JWT public key not readable at '{cfg.JWT_PUBLIC_KEY_PATH}': {exc}"
            ) from exc

        self._algorithm = cfg.JWT_ALGORITHM
        self._issuer = cfg.JWT_ISSUER
        self._audience = cfg.JWT_AUDIENCE

        logger.info(
            "JWT public key loaded",
            extra={
                "algorithm": self._algorithm,
                "issuer": self._issuer,
                "audience": self._audience,
            },
        )

    def _extract_token_from_websocket(self, websocket: WebSocket) -> str:
        """
        Extracts the bearer token from the WebSocket query parameters.

        The mobile client must include ?token=<JWT> in the upgrade URL.
        An Authorization header is also checked as a fallback (native clients).

        Raises:
            WebSocketAuthError: If no token is present in the request.
        """
        # Primary: query parameter (browser + mobile clients)
        token = websocket.query_params.get("token")
        if token:
            return token

        # Fallback: Authorization header (native mobile clients that support it)
        auth_header = websocket.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[len("Bearer "):]

        raise WebSocketAuthError(
            "No authentication token provided. "
            "Include ?token=<JWT> in the WebSocket URL or Authorization: Bearer <JWT> header."
        )

    async def _check_jti_not_revoked(self, jti: str) -> None:
        """
        Checks the Redis revocation list for the given JTI.
        Revoked JTIs are stored at: revoked_jti:{jti}

        Raises:
            WebSocketAuthError: If the JTI has been revoked (logout / token rotation).
        """
        revocation_key = f"{_REVOKED_JTI_PREFIX}{jti}"
        is_revoked = await self._redis.exists(revocation_key)
        if is_revoked:
            logger.warning(
                "Rejected revoked JWT JTI on WebSocket connect",
                extra={"jti": jti},
            )
            raise WebSocketAuthError("Token has been revoked. Please re-authenticate.")

    async def authenticate(self, websocket: WebSocket) -> JWTClaims:
        """
        Authenticates a WebSocket upgrade request.

        Full validation pipeline:
        1. Extract token from query param / header
        2. Decode and verify RS256 signature
        3. Validate exp, iss, aud
        4. Check JTI not in Redis revocation list
        5. Validate required scope

        Args:
            websocket: The incoming WebSocket connection (pre-accept).

        Returns:
            JWTClaims: Validated, typed claims from the token.

        Raises:
            WebSocketAuthError: On any authentication or authorization failure.
        """
        self._load_settings()

        raw_token = self._extract_token_from_websocket(websocket)

        try:
            raw_claims: dict = jwt.decode(
                raw_token,
                self._public_key,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["sub", "exp", "iat", "jti", "iss", "aud"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                    "verify_signature": True,
                },
            )
        except ExpiredSignatureError as exc:
            logger.info(
                "Rejected expired JWT on WebSocket connect",
                extra={"client": websocket.client.host if websocket.client else "unknown"},
            )
            raise WebSocketAuthError(
                "JWT token has expired. Please obtain a new token.", is_expired=True
            ) from exc
        except InvalidSignatureError as exc:
            logger.warning(
                "Rejected JWT with invalid signature",
                extra={"client": websocket.client.host if websocket.client else "unknown"},
            )
            raise WebSocketAuthError("JWT signature verification failed.") from exc
        except PyJWTError as exc:
            logger.warning(
                "JWT validation failed",
                extra={
                    "error": type(exc).__name__,
                    "client": websocket.client.host if websocket.client else "unknown",
                },
            )
            raise WebSocketAuthError(f"Invalid token: {exc}") from exc

        # Validate required custom claims are present
        for required_claim in ("sub", "org", "jti"):
            if required_claim not in raw_claims:
                raise WebSocketAuthError(
                    f"Token is missing required claim: '{required_claim}'"
                )

        jti: str = raw_claims["jti"]
        sub: str = raw_claims["sub"]

        # Redis revocation check (async)
        await self._check_jti_not_revoked(jti)

        # Scope enforcement
        scopes: list[str] = raw_claims.get("scopes", [])
        if REQUIRED_SCOPE not in scopes:
            logger.warning(
                "JWT missing required scope for audio ingestion",
                extra={
                    "sub": sub,
                    "jti": jti,
                    "scopes_present": scopes,
                    "required_scope": REQUIRED_SCOPE,
                },
            )
            raise WebSocketAuthError(
                f"Insufficient scope. '{REQUIRED_SCOPE}' scope is required for audio ingestion."
            )

        claims = JWTClaims(
            sub=sub,
            org=raw_claims.get("org", ""),
            jti=jti,
            iss=raw_claims["iss"],
            aud=raw_claims["aud"] if isinstance(raw_claims["aud"], str) else raw_claims["aud"][0],
            exp=raw_claims["exp"],
            iat=raw_claims["iat"],
            roles=raw_claims.get("roles", []),
            scopes=scopes,
        )

        logger.info(
            "WebSocket authentication successful",
            extra={
                "sub": sub,
                "org": claims.org,
                "jti": jti,
                "scopes": scopes,
                "token_remaining_seconds": claims.exp - int(time.time()),
            },
        )
        return claims

    async def revoke_jti(self, jti: str, ttl_seconds: int = 3600) -> None:
        """
        Adds a JTI to the Redis revocation list.
        TTL should be set to the remaining lifetime of the token.

        Called on user logout or explicit session termination.
        """
        revocation_key = f"{_REVOKED_JTI_PREFIX}{jti}"
        await self._redis.setex(revocation_key, ttl_seconds, "revoked")
        logger.info(
            "JWT JTI added to revocation list",
            extra={"jti": jti, "ttl_seconds": ttl_seconds},
        )
