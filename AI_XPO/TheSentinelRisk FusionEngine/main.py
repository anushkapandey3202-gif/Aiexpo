"""
SentinelAI - Risk Fusion Engine Microservice Entry Point
=========================================================
FastAPI application factory for the Risk Fusion Engine.

Component wiring (startup order matters):
  1. Redis client          → FusionSessionStore, AlertPublisher
  2. FusionWeightConfig    → ScoreFusionEngine
  3. FusionSessionStore    → FusionPipeline, expiry worker
  4. FusionPersistenceService
  5. AlertPublisher        → Kafka alert producer
  6. FusionPipeline        → wires all above + registers expiry callback
  7. ThreatScoreConsumer   → starts Kafka poll loop
  8. WebSocketAuthenticator→ alert WS endpoint

Shutdown order (reverse of startup):
  1. ThreatScoreConsumer.stop()   — drain in-flight events, commit offsets
  2. FusionSessionStore expiry worker stop
  3. AlertPublisher.stop()        — flush Kafka alert producer, close WS
  4. Redis client close

Deployment:
    uvicorn sentinel_ai.services.risk_fusion.main:create_app \
        --factory --host 0.0.0.0 --port 8002 --workers 1 \
        --loop uvloop --http httptools

    Single worker required: FusionSessionStore in-memory callback registry
    and AlertPublisher WebSocket registry are not cross-process-safe.
    Scale via K8s replicas — each pod owns a distinct set of Kafka partitions.
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

from sentinel_ai.services.risk_fusion.api.websocket import http_router, router as ws_router
from sentinel_ai.services.risk_fusion.core.alert_publisher import AlertPublisher
from sentinel_ai.services.risk_fusion.core.kafka_consumer import ThreatScoreConsumer
from sentinel_ai.services.risk_fusion.core.persistence import FusionPersistenceService
from sentinel_ai.services.risk_fusion.core.pipeline import FusionPipeline
from sentinel_ai.services.risk_fusion.core.score_aggregator import ScoreFusionEngine
from sentinel_ai.services.risk_fusion.core.session_store import FusionSessionStore
from sentinel_ai.services.risk_fusion.schemas.fusion import FusionWeightConfig
from sentinel_ai.services.ingestion.core.auth import WebSocketAuthenticator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    try:
        import json_log_formatter
        formatter = json_log_formatter.JSONFormatter()
    except ImportError:
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
        )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    for noisy in ("uvicorn.access", "confluent_kafka"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Request Logging Middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "HTTP request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": ms,
            },
        )
        response.headers["X-Request-ID"] = rid
        return response


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from sentinel_ai.config.settings import get_settings
    cfg = get_settings()

    configure_logging(cfg.LOG_LEVEL)

    logger.info(
        "Starting Risk Fusion Engine",
        extra={"version": cfg.APP_VERSION, "env": cfg.ENVIRONMENT},
    )

    # --- Redis ---
    redis_client = aioredis.from_url(
        cfg.redis_url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=cfg.REDIS_MAX_CONNECTIONS,
        socket_timeout=cfg.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=cfg.REDIS_SOCKET_CONNECT_TIMEOUT,
    )
    await redis_client.ping()
    logger.info("Redis connection verified")

    # --- Fusion Weight Config ---
    fusion_config = FusionWeightConfig(
        weight_deepfake_video = getattr(cfg, "FUSION_WEIGHT_DEEPFAKE_VIDEO", 0.40),
        weight_deepfake_voice = getattr(cfg, "FUSION_WEIGHT_DEEPFAKE_VOICE", 0.30),
        weight_nlp_intent     = getattr(cfg, "FUSION_WEIGHT_NLP_INTENT",     0.20),
        weight_voiceprint_sim = getattr(cfg, "FUSION_WEIGHT_VOICEPRINT_SIM", 0.10),
        persist_threshold     = getattr(cfg, "FUSION_PERSIST_THRESHOLD",     0.30),
        session_ttl_seconds   = getattr(cfg, "FUSION_SESSION_TTL_SECONDS",   120),
    )

    # --- Core Components ---
    session_store    = FusionSessionStore(redis_client, config=fusion_config)
    fusion_engine    = ScoreFusionEngine(config=fusion_config)
    persistence_svc  = FusionPersistenceService()
    alert_publisher  = AlertPublisher()
    authenticator    = WebSocketAuthenticator(redis_client)

    await alert_publisher.start()
    await session_store.start_expiry_worker()

    # --- Pipeline (wires everything together) ---
    pipeline = FusionPipeline(
        session_store   = session_store,
        fusion_engine   = fusion_engine,
        persistence_svc = persistence_svc,
        alert_publisher = alert_publisher,
    )

    # --- Kafka Consumer ---
    consumer = ThreatScoreConsumer(
        message_handler=pipeline.handle_score_event,
    )
    await consumer.start()

    # --- Register on app.state ---
    app.state.redis_client      = redis_client
    app.state.session_store     = session_store
    app.state.fusion_engine     = fusion_engine
    app.state.pipeline          = pipeline
    app.state.alert_publisher   = alert_publisher
    app.state.kafka_consumer    = consumer
    app.state.ws_authenticator  = authenticator
    app.state.fusion_config     = fusion_config

    logger.info("Risk Fusion Engine fully initialized — consuming threat_scores topic")

    yield  # ← Application running

    # --- Graceful Shutdown ---
    logger.info("Shutting down Risk Fusion Engine")
    await consumer.stop()
    await session_store.stop_expiry_worker()
    await alert_publisher.stop()
    await redis_client.aclose()
    logger.info("Risk Fusion Engine shutdown complete")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    from sentinel_ai.config.settings import get_settings
    cfg = get_settings()

    app = FastAPI(
        title       = "SentinelAI Risk Fusion Engine",
        description = (
            "Subscribes to the threat_scores Kafka topic, aggregates deepfake, "
            "voiceprint, and NLP intent scores per session, applies weighted fusion "
            "with booster coefficients, persists results to PostgreSQL, and dispatches "
            "real-time WebSocket alerts to mobile clients."
        ),
        version     = cfg.APP_VERSION,
        docs_url    = "/docs"      if cfg.ENVIRONMENT != "production" else None,
        redoc_url   = "/redoc"     if cfg.ENVIRONMENT != "production" else None,
        openapi_url = "/openapi.json" if cfg.ENVIRONMENT != "production" else None,
        lifespan    = lifespan,
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.sentinelai.io", "*.sentinelai.io", "localhost"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://app.sentinelai.io", "https://dashboard.sentinelai.io"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Org-Id"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(ws_router)
    app.include_router(http_router)

    # --- Health Endpoints ---

    @app.get("/health/live", tags=["Health"])
    async def liveness() -> dict:
        return {"status": "alive", "service": "risk-fusion"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness(request: Request) -> Response:
        checks: dict[str, Any] = {"status": "ready", "service": "risk-fusion"}

        try:
            await request.app.state.redis_client.ping()
            checks["redis"] = "healthy"
        except Exception:
            checks["redis"] = "unhealthy"
            checks["status"] = "degraded"

        kafka_health = request.app.state.kafka_consumer._running
        checks["kafka_consumer"] = "running" if kafka_health else "stopped"
        if not kafka_health:
            checks["status"] = "degraded"

        checks["alert_connections"] = request.app.state.alert_publisher.stats()
        checks["fusion_config"] = {
            "persist_threshold": request.app.state.fusion_config.persist_threshold,
            "session_ttl_seconds": request.app.state.fusion_config.session_ttl_seconds,
        }

        import json
        return Response(
            content    = json.dumps(checks),
            status_code= 200 if checks["status"] == "ready" else 503,
            media_type = "application/json",
        )

    @app.get("/health/fusion-config", tags=["Health"])
    async def fusion_config_endpoint(request: Request) -> dict:
        """Returns the active fusion weight configuration (non-secret, safe to expose)."""
        cfg: FusionWeightConfig = request.app.state.fusion_config
        return cfg.model_dump()

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sentinel_ai.services.risk_fusion.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="debug",
    )
