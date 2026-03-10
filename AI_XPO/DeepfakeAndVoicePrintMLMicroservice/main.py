"""
SentinelAI ML Service — FastAPI Application Entry Point

Manages the full service lifecycle:
- Model loading (ECAPA-TDNN, RawNet3) at startup.
- Pinecone client initialization.
- Kafka producer + consumer startup.
- Health, readiness, and Prometheus metrics endpoints.
- Graceful shutdown with in-flight task draining.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.inference.ecapa_tdnn import get_ecapa_inference
from app.inference.rawnet3 import get_rawnet3_inference
from app.kafka.consumer import AudioKafkaConsumer
from app.kafka.producer import get_kafka_producer
from app.models.schemas import ComponentHealth, HealthResponse
from app.services.inference_orchestrator import InferenceOrchestrator
from app.vector_store.pinecone_client import get_pinecone_client

configure_logging()
logger = get_logger("main")
settings = get_settings()

# ---------------------------------------------------------------------------
# Module-level references (populated during lifespan)
# ---------------------------------------------------------------------------
_consumer: AudioKafkaConsumer | None = None
_consumer_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Manages startup and shutdown of all long-lived service components.
    FastAPI's lifespan replaces deprecated @app.on_event decorators.
    """
    global _consumer, _consumer_task

    logger.info(
        "SentinelAI ML Service starting.",
        extra={"version": settings.SERVICE_VERSION, "environment": settings.ENVIRONMENT.value},
    )

    # ── 1. Load ML models (CPU/GPU) ────────────────────────────────────────
    logger.info("Loading ML models...")
    loop = asyncio.get_event_loop()

    ecapa = get_ecapa_inference()
    rawnet3 = get_rawnet3_inference()

    # Model loading is synchronous (disk I/O + CUDA init) — run in executor
    await loop.run_in_executor(None, ecapa.load)
    await loop.run_in_executor(None, rawnet3.load)
    logger.info("ML models loaded successfully.")

    # ── 2. Initialize Pinecone ─────────────────────────────────────────────
    pinecone_client = get_pinecone_client()
    try:
        await pinecone_client.initialize()
    except Exception as exc:
        logger.error(
            "Pinecone initialization failed; voiceprint pipeline will be degraded.",
            extra={"error": str(exc)},
        )

    # ── 3. Start Kafka producer ────────────────────────────────────────────
    producer = get_kafka_producer()
    await producer.start()

    # ── 4. Start Kafka consumer as background task ─────────────────────────
    orchestrator = InferenceOrchestrator()
    _consumer = AudioKafkaConsumer(
        orchestrator=orchestrator,
        producer_publish_fn=producer.publish_threat_score,
    )
    _consumer_task = asyncio.create_task(
        _consumer.start(),
        name="kafka-audio-consumer",
    )
    logger.info("Kafka consumer task started.")

    # ── Service ready ──────────────────────────────────────────────────────
    logger.info("SentinelAI ML Service is READY.")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("SentinelAI ML Service shutting down...")

    if _consumer:
        await _consumer.stop()

    if _consumer_task and not _consumer_task.done():
        try:
            await asyncio.wait_for(_consumer_task, timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("Consumer task did not stop within 15s; cancelling.")
            _consumer_task.cancel()

    await producer.stop()
    logger.info("SentinelAI ML Service shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title="SentinelAI ML Inference Service",
        description=(
            "Real-time deepfake detection and voiceprint authentication "
            "via Kafka-driven async ML inference pipelines."
        ),
        version=settings.SERVICE_VERSION,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["Authorization", settings.CORRELATION_ID_HEADER],
    )

    # ── Correlation ID middleware ──────────────────────────────────────────
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        from app.core.logging import set_correlation_id
        import uuid
        cid = request.headers.get(settings.CORRELATION_ID_HEADER, str(uuid.uuid4()))
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers[settings.CORRELATION_ID_HEADER] = cid
        return response

    # ── Health / Readiness ─────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["Observability"])
    async def health_check():
        """
        Liveness probe — returns 200 if the service process is alive.
        Does NOT check downstream dependencies.
        """
        return HealthResponse(
            status="healthy",
            service=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            environment=settings.ENVIRONMENT.value,
        )

    @app.get("/ready", response_model=HealthResponse, tags=["Observability"])
    async def readiness_check():
        """
        Readiness probe — checks all critical dependencies.
        Returns 503 if any critical component is unhealthy.
        """
        components: list[ComponentHealth] = []
        all_healthy = True

        # ECAPA-TDNN
        ecapa = get_ecapa_inference()
        ecapa_healthy = ecapa.is_loaded
        components.append(ComponentHealth(
            name="ecapa_tdnn",
            healthy=ecapa_healthy,
            detail="Loaded" if ecapa_healthy else "Not loaded",
        ))

        # RawNet3
        rawnet3 = get_rawnet3_inference()
        rawnet3_healthy = rawnet3.is_loaded
        components.append(ComponentHealth(
            name="rawnet3",
            healthy=rawnet3_healthy,
            detail="Loaded" if rawnet3_healthy else "Not loaded",
        ))

        # Pinecone
        pinecone_status = await get_pinecone_client().health_check()
        components.append(ComponentHealth(
            name="pinecone",
            healthy=pinecone_status.get("healthy", False),
            latency_ms=pinecone_status.get("latency_ms"),
            detail=pinecone_status.get("detail"),
        ))

        # Kafka consumer task
        consumer_alive = (
            _consumer_task is not None
            and not _consumer_task.done()
        )
        components.append(ComponentHealth(
            name="kafka_consumer",
            healthy=consumer_alive,
            detail="Running" if consumer_alive else "Not running",
        ))

        all_healthy = all(c.healthy for c in components)
        status_str = "healthy" if all_healthy else "degraded"

        response = HealthResponse(
            status=status_str,
            service=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            environment=settings.ENVIRONMENT.value,
            components=components,
        )

        if not all_healthy:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(mode="json"),
            )
        return response

    @app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
    async def prometheus_metrics():
        """
        Prometheus scrape endpoint.
        Exposes: inference latency histograms, threat level counters,
        Kafka lag gauge, GPU memory gauge.
        """
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            return PlainTextResponse(
                generate_latest().decode("utf-8"),
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            return PlainTextResponse(
                "# prometheus_client not installed\n",
                media_type="text/plain",
            )

    @app.get("/info", tags=["Observability"])
    async def service_info():
        """Returns service metadata and device info."""
        from app.inference.device_manager import get_device_manager
        dm = get_device_manager()
        return {
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "device": dm.device_name,
            "memory_stats": dm.get_memory_stats(),
            "models": {
                "ecapa_tdnn": get_ecapa_inference().model_version,
                "rawnet3": get_rawnet3_inference().model_version,
            },
        }

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_config=None,  # Disable uvicorn's default logging; we use our own
        access_log=False,
        loop="uvloop",
    )
