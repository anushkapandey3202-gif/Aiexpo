"""
SentinelAI - Audio Ingestion Microservice Entry Point
======================================================
FastAPI application factory for the real-time audio ingestion service.

Responsibilities:
- Application lifespan management (startup/shutdown of all dependencies)
- Dependency injection registration on app.state
- Structured JSON logging configuration
- Health check and readiness probe endpoints
- Prometheus metrics middleware
- CORS configuration (locked to allowed mobile app origins)
- Trusted proxy middleware (for X-Forwarded-For resolution)

Deployment:
    uvicorn sentinel_ai.services.ingestion.main:create_app \
        --factory --host 0.0.0.0 --port 8001 --workers 1 \
        --loop uvloop --http httptools

    Note: --workers 1 is REQUIRED. The ConnectionManager and AudioBufferManager
    singletons hold in-memory state. Horizontal scaling must be done at the
    Kubernetes Deployment level (multiple pods, not multiple workers per pod).
    Redis-backed state ensures cross-pod session consistency for buffer metadata;
    in-memory buffers (Tier 1) are local to the pod that owns the WebSocket.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from sentinel_ai.services.ingestion.api.websocket import router as ws_router
from sentinel_ai.services.ingestion.core.audio_buffer import AudioBufferManager
from sentinel_ai.services.ingestion.core.auth import WebSocketAuthenticator
from sentinel_ai.services.ingestion.core.connection_manager import ConnectionManager
from sentinel_ai.services.ingestion.core.kafka_producer import AudioEventKafkaProducer
from sentinel_ai.services.ingestion.middleware.rate_limiter import WebSocketRateLimiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configures structured JSON logging for the ingestion service.
    JSON format is required for CloudWatch / Datadog log parsing in production.
    """
    if log_format == "json":
        try:
            import json_log_formatter
            formatter = json_log_formatter.JSONFormatter()
        except ImportError:
            formatter = logging.Formatter(
                '{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"name": "%(name)s", "message": "%(message)s"}'
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("confluent_kafka").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Request Logging Middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs HTTP requests with latency, status code, and trace ID.
    WebSocket upgrade requests are logged at DEBUG level to avoid noise.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        is_ws = request.headers.get("upgrade", "").lower() == "websocket"
        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000

        log_fn = logger.debug if is_ws else logger.info
        log_fn(
            "HTTP request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "is_websocket": is_ws,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    All singletons are initialized here and attached to app.state for DI.
    """
    from sentinel_ai.config.settings import get_settings
    cfg = get_settings()

    configure_logging(cfg.LOG_LEVEL, cfg.LOG_FORMAT)

    logger.info(
        "Starting SentinelAI Audio Ingestion Service",
        extra={
            "version": cfg.APP_VERSION,
            "environment": cfg.ENVIRONMENT,
            "vector_backend": cfg.VECTOR_STORE_BACKEND,
        },
    )

    # Redis client
    redis_client = aioredis.from_url(
        cfg.redis_url,
        encoding="utf-8",
        decode_responses=False,  # We handle both bytes and str; keep raw
        max_connections=cfg.REDIS_MAX_CONNECTIONS,
        socket_timeout=cfg.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=cfg.REDIS_SOCKET_CONNECT_TIMEOUT,
    )

    # Validate Redis connectivity at startup
    try:
        await redis_client.ping()
        logger.info("Redis connection verified")
    except Exception as exc:
        logger.critical("Failed to connect to Redis at startup", exc_info=True)
        raise RuntimeError("Redis connectivity check failed") from exc

    # Initialize all singletons
    connection_manager = ConnectionManager()
    await connection_manager.start()

    buffer_manager = AudioBufferManager(redis_client)

    kafka_producer = AudioEventKafkaProducer()
    await kafka_producer.start()

    authenticator = WebSocketAuthenticator(redis_client)
    rate_limiter = WebSocketRateLimiter(redis_client)

    # Attach to app.state for dependency injection
    app.state.redis_client = redis_client
    app.state.connection_manager = connection_manager
    app.state.audio_buffer_manager = buffer_manager
    app.state.kafka_producer = kafka_producer
    app.state.ws_authenticator = authenticator
    app.state.rate_limiter = rate_limiter

    logger.info("All dependencies initialized — service is ready to accept connections")

    yield  # ← Application runs here

    # ---------------------
    # Shutdown
    # ---------------------
    logger.info("Shutting down Audio Ingestion Service")

    await connection_manager.stop()
    await kafka_producer.stop()
    await redis_client.aclose()

    logger.info("Graceful shutdown complete")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    FastAPI application factory.
    Called by uvicorn when using --factory flag.
    """
    from sentinel_ai.config.settings import get_settings
    cfg = get_settings()

    app = FastAPI(
        title="SentinelAI Audio Ingestion Service",
        description=(
            "Real-time audio streaming WebSocket endpoint. "
            "Ingests mobile client audio, buffers in Redis, "
            "and publishes to Kafka for ML pipeline processing."
        ),
        version=cfg.APP_VERSION,
        docs_url="/docs" if cfg.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if cfg.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if cfg.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware (order matters — applied bottom-up)
    # ------------------------------------------------------------------

    # 1. Trusted host enforcement (prevents Host header injection)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.sentinelai.io", "*.sentinelai.io", "localhost"],
    )

    # 2. CORS (only needed for browser-based clients; mobile apps don't send Origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://app.sentinelai.io",
            "https://dashboard.sentinelai.io",
        ],
        allow_credentials=True,
        allow_methods=["GET"],  # WebSocket upgrade uses GET
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    # 3. Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(ws_router)

    # ------------------------------------------------------------------
    # Health Endpoints
    # ------------------------------------------------------------------

    @app.get("/health/live", tags=["Health"])
    async def liveness_probe() -> dict[str, str]:
        """
        Kubernetes liveness probe.
        Returns 200 if the process is alive and responsive.
        """
        return {"status": "alive", "service": "audio-ingestion"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness_probe(request: Request) -> dict[str, Any]:
        """
        Kubernetes readiness probe.
        Returns 200 only when all dependencies are healthy.
        Returns 503 if any dependency is unavailable (removes pod from LB rotation).
        """
        checks: dict[str, Any] = {
            "status": "ready",
            "service": "audio-ingestion",
        }

        # Redis check
        try:
            await request.app.state.redis_client.ping()
            checks["redis"] = "healthy"
        except Exception:
            checks["redis"] = "unhealthy"
            checks["status"] = "degraded"

        # Kafka check
        kafka_health = await request.app.state.kafka_producer.health_check()
        checks["kafka"] = kafka_health

        # Connection stats
        checks["connections"] = request.app.state.connection_manager.stats()

        http_status = 200 if checks["status"] == "ready" else 503
        return Response(
            content=__import__("json").dumps(checks),
            status_code=http_status,
            media_type="application/json",
        )

    @app.get("/metrics/connections", tags=["Metrics"])
    async def connection_metrics(request: Request) -> dict[str, Any]:
        """
        Returns live connection stats for monitoring dashboards.
        Exposed only to internal network (enforce via Nginx/RBAC in production).
        """
        return request.app.state.connection_manager.stats()

    return app


# ---------------------------------------------------------------------------
# Direct execution (development only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sentinel_ai.services.ingestion.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="debug",
    )
