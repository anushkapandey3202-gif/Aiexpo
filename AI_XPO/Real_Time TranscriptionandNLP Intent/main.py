"""
SentinelAI NLP Service — FastAPI Application Entry Point

Manages full NLP service lifecycle:
  - Whisper model loading (faster-whisper or openai-whisper fallback).
  - DeBERTa-v3 intent classifier loading (fine-tuned or zero-shot).
  - Kafka consumer + producer startup.
  - Health / readiness / metrics endpoints.
  - Graceful shutdown with in-flight task draining.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.inference.intent_classifier import get_intent_engine, get_intent_model_manager
from app.inference.whisper_engine import get_whisper_engine, get_whisper_model_manager
from app.kafka.consumer import NLPKafkaConsumer
from app.kafka.producer import get_nlp_producer
from app.models.schemas import ComponentHealth, HealthResponse
from app.services.nlp_orchestrator import NLPOrchestrator

configure_logging()
logger   = get_logger("main")
settings = get_settings()

_consumer:      NLPKafkaConsumer | None = None
_consumer_task: asyncio.Task | None    = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _consumer, _consumer_task

    logger.info(
        "SentinelAI NLP Service starting.",
        extra={"version": settings.SERVICE_VERSION, "environment": settings.ENVIRONMENT.value},
    )

    loop = asyncio.get_event_loop()

    # ── 1. Load Whisper ────────────────────────────────────────────────────
    logger.info("Loading Whisper transcription model...")
    whisper_mgr = get_whisper_model_manager()
    await loop.run_in_executor(None, whisper_mgr.load)
    logger.info(
        "Whisper ready.",
        extra={
            "model_size": settings.WHISPER_MODEL_SIZE.value,
            "backend":    whisper_mgr.backend,
            "device":     whisper_mgr.device,
        },
    )

    # ── 2. Load DeBERTa-v3 Intent Classifier ──────────────────────────────
    logger.info("Loading DeBERTa-v3 intent classifier...")
    intent_mgr = get_intent_model_manager()
    await loop.run_in_executor(None, intent_mgr.load)
    logger.info(
        "Intent classifier ready.",
        extra={"mode": intent_mgr.mode.value, "device": intent_mgr.device},
    )

    # ── 3. Start Kafka producer ────────────────────────────────────────────
    producer = get_nlp_producer()
    await producer.start()

    # ── 4. Start Kafka consumer ────────────────────────────────────────────
    orchestrator = NLPOrchestrator()
    _consumer    = NLPKafkaConsumer(
        orchestrator=orchestrator,
        producer_publish_fn=producer.publish_nlp_threat_score,
    )
    _consumer_task = asyncio.create_task(
        _consumer.start(),
        name="kafka-nlp-consumer",
    )
    logger.info("SentinelAI NLP Service is READY.")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("SentinelAI NLP Service shutting down...")
    if _consumer:
        await _consumer.stop()
    if _consumer_task and not _consumer_task.done():
        try:
            await asyncio.wait_for(_consumer_task, timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning("NLP consumer task did not stop in 20s; cancelling.")
            _consumer_task.cancel()
    await producer.stop()
    logger.info("SentinelAI NLP Service shutdown complete.")


# ── Application Factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="SentinelAI NLP Inference Service",
        description=(
            "Real-time speech-to-text transcription (Whisper) and "
            "intent classification (DeBERTa-v3) for social engineering detection."
        ),
        version=settings.SERVICE_VERSION,
        docs_url="/docs"       if not settings.is_production else None,
        redoc_url="/redoc"     if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["Authorization", settings.CORRELATION_ID_HEADER],
    )

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        import uuid
        from app.core.logging import set_correlation_id
        cid = request.headers.get(settings.CORRELATION_ID_HEADER, str(uuid.uuid4()))
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers[settings.CORRELATION_ID_HEADER] = cid
        return response

    # ── Health ─────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Observability"])
    async def health():
        return HealthResponse(
            status="healthy",
            service=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
            environment=settings.ENVIRONMENT.value,
        )

    # ── Readiness ──────────────────────────────────────────────────────────
    @app.get("/ready", response_model=HealthResponse, tags=["Observability"])
    async def readiness():
        components: list[ComponentHealth] = []

        # Whisper
        wm = get_whisper_model_manager()
        components.append(ComponentHealth(
            name="whisper",
            healthy=wm.is_loaded,
            detail=f"{wm.backend}/{wm.device}" if wm.is_loaded else "Not loaded",
        ))

        # Intent classifier
        im = get_intent_model_manager()
        components.append(ComponentHealth(
            name="intent_classifier",
            healthy=im.is_loaded,
            detail=f"{im.mode.value}/{im.device}" if im.is_loaded else "Not loaded",
        ))

        # Kafka consumer
        consumer_alive = (
            _consumer_task is not None and not _consumer_task.done()
        )
        components.append(ComponentHealth(
            name="kafka_consumer",
            healthy=consumer_alive,
            detail="Running" if consumer_alive else "Not running",
        ))

        all_healthy = all(c.healthy for c in components)
        response    = HealthResponse(
            status="healthy" if all_healthy else "degraded",
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

    # ── Prometheus Metrics ─────────────────────────────────────────────────
    @app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
    async def metrics():
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            return PlainTextResponse(
                generate_latest().decode("utf-8"),
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            return PlainTextResponse("# prometheus_client not installed\n")

    # ── Service Info ───────────────────────────────────────────────────────
    @app.get("/info", tags=["Observability"])
    async def info():
        wm = get_whisper_model_manager()
        im = get_intent_model_manager()
        return {
            "service":     settings.SERVICE_NAME,
            "version":     settings.SERVICE_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "models": {
                "whisper": {
                    "backend":      wm.backend,
                    "model_size":   settings.WHISPER_MODEL_SIZE.value,
                    "compute_type": settings.WHISPER_COMPUTE_TYPE.value,
                    "device":       wm.device,
                },
                "intent_classifier": {
                    "model_name": settings.INTENT_MODEL_NAME,
                    "mode":       im.mode.value,
                    "device":     im.device,
                    "labels":     len(settings.INTENT_LABELS),
                },
            },
            "intent_taxonomy": {
                "total_labels":      len(settings.INTENT_LABELS),
                "high_risk_labels":  settings.HIGH_RISK_INTENT_LABELS,
                "medium_risk_labels": settings.MEDIUM_RISK_INTENT_LABELS,
            },
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_config=None,
        access_log=False,
        loop="uvloop",
    )
