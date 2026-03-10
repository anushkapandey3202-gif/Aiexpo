"""
SentinelAI - Email & Web Phishing Analysis Service
====================================================
Standalone FastAPI microservice for email and web phishing heuristics.

This service is intentionally decoupled from the rest of the SentinelAI
platform and can be deployed independently. It communicates outward via:
  1. Kafka topic `sentinelai.threat.scores` — forwards FusionScorePayload
     to the Risk Fusion Engine for sessions being actively monitored.
  2. REST POST `/fusion/ingest` fallback — used if Kafka is unavailable.
  3. Synchronous REST response — always returned to the API caller.

Startup wiring order:
  1. Redis client           — rate limiting
  2. EmailAnalysisOrchestrator — pre-warm brand corpus + compile regexes
  3. Kafka producer (optional) — for Risk Fusion Engine forwarding
  4. Mount API router

Workers:
  This service is CPU-bound (regex, entropy, Levenshtein).
  Use multiple uvicorn workers: --workers $(nproc)
  Each worker gets its own EmailAnalysisOrchestrator instance (stateless).
  Redis is shared across workers (separate connections per worker).

Run:
  uvicorn sentinel_ai.services.email_analysis.main:create_app \\
    --factory --host 0.0.0.0 --port 8003 --workers 4 \\
    --loop uvloop --http httptools
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from sentinel_ai.services.email_analysis.api.router import router
from sentinel_ai.services.email_analysis.core.orchestrator import EmailAnalysisOrchestrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    try:
        import json_log_formatter
        formatter = json_log_formatter.JSONFormatter()
    except ImportError:
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","msg":"%(message)s"}'
        )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


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
            "HTTP",
            extra={
                "request_id": rid,
                "method":     request.method,
                "path":       request.url.path,
                "status":     response.status_code,
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

    _configure_logging(getattr(cfg, "LOG_LEVEL", "INFO"))
    logger.info("Starting Email Analysis Service", extra={
        "version": getattr(cfg, "APP_VERSION", "1.0.0"),
        "env":     getattr(cfg, "ENVIRONMENT", "production"),
    })

    # --- Redis (optional — rate limiting degrades gracefully without it) ---
    redis_client: Optional[Any] = None
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(
            cfg.redis_url,
            encoding         = "utf-8",
            decode_responses = False,
            max_connections  = getattr(cfg, "REDIS_MAX_CONNECTIONS", 20),
        )
        await redis_client.ping()
        logger.info("Redis connection verified")
    except Exception:
        logger.warning("Redis unavailable — rate limiting disabled", exc_info=True)

    # --- Email Analysis Orchestrator ---
    orchestrator = EmailAnalysisOrchestrator()
    logger.info("EmailAnalysisOrchestrator initialized")

    # --- Kafka producer (optional — falls back to REST POST) ---
    kafka_producer: Optional[Any] = None
    try:
        from confluent_kafka import Producer
        kafka_cfg = {
            "bootstrap.servers": getattr(cfg, "KAFKA_BOOTSTRAP_SERVERS", ""),
            "acks":              "1",
            "compression.type":  "snappy",
            "security.protocol": getattr(cfg, "KAFKA_SECURITY_PROTOCOL", "SASL_SSL"),
            "ssl.ca.location":   getattr(cfg, "KAFKA_SSL_CA_LOCATION",
                                         "/etc/ssl/certs/ca-certificates.crt"),
            "sasl.mechanism":    getattr(cfg, "KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
            "sasl.username":     getattr(cfg, "KAFKA_SASL_USERNAME", ""),
            "sasl.password": (
                getattr(cfg, "KAFKA_SASL_PASSWORD", None).get_secret_value()
                if hasattr(getattr(cfg, "KAFKA_SASL_PASSWORD", None), "get_secret_value")
                else getattr(cfg, "KAFKA_SASL_PASSWORD", "")
            ),
        }
        if kafka_cfg["bootstrap.servers"]:
            kafka_producer = Producer(kafka_cfg)
            logger.info("Kafka producer initialized for Risk Fusion Engine forwarding")
    except Exception:
        logger.warning("Kafka producer unavailable — will use REST fallback", exc_info=True)

    # --- Attach to app.state ---
    app.state.redis_client          = redis_client
    app.state.email_orchestrator    = orchestrator
    app.state.email_kafka_producer  = kafka_producer
    app.state.fusion_rest_url       = getattr(cfg, "FUSION_ENGINE_REST_URL", None)

    logger.info("Email Analysis Service ready")
    yield

    # --- Graceful Shutdown ---
    logger.info("Shutting down Email Analysis Service")
    if kafka_producer:
        kafka_producer.flush(10)
    if redis_client:
        await redis_client.aclose()
    logger.info("Email Analysis Service shutdown complete")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    try:
        from sentinel_ai.config.settings import get_settings
        cfg = get_settings()
        env = getattr(cfg, "ENVIRONMENT", "production")
        version = getattr(cfg, "APP_VERSION", "1.0.0")
    except Exception:
        env = "production"
        version = "1.0.0"

    app = FastAPI(
        title       = "SentinelAI — Email & Web Phishing Analysis",
        description = (
            "Standalone phishing heuristics service. Parses SPF/DKIM/DMARC, "
            "calculates domain name entropy (Shannon bits/char), detects look-alike "
            "domains via Levenshtein + homograph analysis, scans URLs for credential "
            "keywords and encoded payloads, and produces a composite risk score "
            "structured for the SentinelAI Risk Fusion Engine."
        ),
        version     = version,
        docs_url    = "/docs"         if env != "production" else None,
        redoc_url   = "/redoc"        if env != "production" else None,
        openapi_url = "/openapi.json" if env != "production" else None,
        lifespan    = lifespan,
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["api.sentinelai.io", "*.sentinelai.io", "localhost"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["https://app.sentinelai.io", "https://dashboard.sentinelai.io"],
        allow_credentials = True,
        allow_methods     = ["GET", "POST"],
        allow_headers     = [
            "Authorization", "Content-Type",
            "X-Request-ID", "X-Org-Id", "X-Expected-Models",
        ],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(router)

    @app.get("/health/live", tags=["Health"])
    async def liveness() -> dict:
        return {"status": "alive", "service": "email-analysis"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness(request: Request) -> Response:
        import json
        checks: dict = {"status": "ready", "service": "email-analysis"}
        try:
            rc = getattr(request.app.state, "redis_client", None)
            if rc:
                await rc.ping()
                checks["redis"] = "healthy"
            else:
                checks["redis"] = "disabled"
        except Exception:
            checks["redis"]  = "unhealthy"
            checks["status"] = "degraded"

        checks["orchestrator"]    = "ready"
        checks["kafka_producer"]  = (
            "ready" if getattr(request.app.state, "email_kafka_producer", None) else "disabled"
        )
        checks["fusion_rest_url"] = getattr(request.app.state, "fusion_rest_url", None) or "not_configured"

        return Response(
            content     = json.dumps(checks),
            status_code = 200 if checks["status"] == "ready" else 503,
            media_type  = "application/json",
        )

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "sentinel_ai.services.email_analysis.main:create_app",
        factory   = True,
        host      = "0.0.0.0",
        port      = 8003,
        workers   = 4,
        log_level = "info",
    )
