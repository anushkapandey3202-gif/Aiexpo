"""
SentinelAI - Email Analysis REST API Router
=============================================
Exposes two primary endpoints:

  POST /email/analyse        — submit a raw email for full phishing analysis
  POST /email/analyse/batch  — submit up to 50 emails in a single request

Both endpoints are async-safe: the CPU-bound analysis pipeline runs inside
asyncio.run_in_executor() so the FastAPI event loop is never blocked.

After analysis, if a session_id is present and the composite score exceeds
the configured forward_threshold, a FusionScorePayload is published to Kafka
(or POSTed to the Risk Fusion Engine REST fallback if Kafka is unavailable).

Rate limiting is enforced per org_id via Redis sliding window counters.
Requests from orgs exceeding the limit receive HTTP 429.

Additional utility endpoints:
  GET /email/health          — liveness + analyser readiness probe
  GET /email/brands          — lists the active brand corpus (for client UI)
  POST /email/url/analyse    — analyse a single URL without a full email
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from sentinel_ai.services.email_analysis.core.orchestrator import EmailAnalysisOrchestrator
from sentinel_ai.services.email_analysis.schemas.analysis import (
    EmailAnalysisRequest,
    EmailAnalysisResult,
    FusionScorePayload,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/email", tags=["Email & Web Phishing Analysis"])

# Thread pool dedicated to CPU-bound analysis work
# Workers = CPU count (analysis is pure Python, no GIL-releasing C extensions)
_ANALYSIS_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="email-analysis",
)

# Rate limit: requests per org per minute
_RATE_LIMIT_REQUESTS = 300
_RATE_LIMIT_WINDOW_S = 60

# Forward composite scores above this threshold to Risk Fusion Engine
_FORWARD_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Dependency Injectors
# ---------------------------------------------------------------------------

def _get_orchestrator(request: Request) -> EmailAnalysisOrchestrator:
    return request.app.state.email_orchestrator


def _get_kafka_producer(request: Request) -> Optional[Any]:
    return getattr(request.app.state, "email_kafka_producer", None)


async def _enforce_rate_limit(request: Request) -> None:
    """
    Enforces per-org rate limiting using Redis sliding window.
    Raises HTTP 429 if the org exceeds _RATE_LIMIT_REQUESTS per minute.
    Skips silently if Redis is unavailable (fail-open).
    """
    redis = getattr(request.app.state, "redis_client", None)
    if redis is None:
        return

    # Org ID resolved from JWT middleware in production — use header fallback here
    org_id = request.headers.get("X-Org-Id", "unknown")
    key    = f"ratelimit:email:{org_id}"
    now_ms = int(time.time() * 1000)
    window_start_ms = now_ms - (_RATE_LIMIT_WINDOW_S * 1000)

    try:
        pipe = redis.pipeline()
        pipe.zadd(key, {str(now_ms): now_ms})
        pipe.zremrangebyscore(key, 0, window_start_ms)
        pipe.zcard(key)
        pipe.expire(key, _RATE_LIMIT_WINDOW_S + 5)
        results = await pipe.execute()
        count = results[2]

        if count > _RATE_LIMIT_REQUESTS:
            logger.warning(
                "Email analysis rate limit exceeded",
                extra={"org_id": org_id, "count": count},
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":       "rate_limit_exceeded",
                    "message":     f"Limit of {_RATE_LIMIT_REQUESTS} requests per minute exceeded",
                    "retry_after": _RATE_LIMIT_WINDOW_S,
                },
                headers={"Retry-After": str(_RATE_LIMIT_WINDOW_S)},
            )
    except HTTPException:
        raise
    except Exception:
        logger.debug("Rate-limit Redis check failed — failing open", exc_info=True)


# ---------------------------------------------------------------------------
# Primary Analysis Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/analyse",
    response_model=EmailAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Analyse a single email for phishing & social engineering signals",
    description=(
        "Accepts raw email headers, body text, HTML, and URLs. "
        "Returns SPF/DKIM/DMARC results, domain entropy scores, URL risk records, "
        "header anomalies, body heuristics, and a composite risk score. "
        "If session_id is provided and the score exceeds the forward threshold, "
        "the result is forwarded to the Risk Fusion Engine via Kafka."
    ),
)
async def analyse_email(
    payload:      EmailAnalysisRequest,
    request:      Request,
    orchestrator: EmailAnalysisOrchestrator = Depends(_get_orchestrator),
    _rate:        None = Depends(_enforce_rate_limit),
) -> EmailAnalysisResult:
    """
    Full email phishing analysis endpoint.

    The analysis pipeline is CPU-bound; it runs in a dedicated ThreadPoolExecutor
    to avoid blocking the FastAPI event loop.
    """
    loop = asyncio.get_event_loop()

    try:
        result: EmailAnalysisResult = await loop.run_in_executor(
            _ANALYSIS_EXECUTOR,
            orchestrator.analyse,
            payload,
        )
    except Exception as exc:
        logger.error(
            "Email analysis pipeline error",
            extra={"request_id": payload.request_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "analysis_failed", "message": str(exc)},
        )

    # Forward to Risk Fusion Engine if session is linked and score is significant
    if payload.session_id and result.composite_score >= _FORWARD_THRESHOLD:
        asyncio.create_task(
            _forward_to_fusion(
                result       = result,
                orchestrator = orchestrator,
                producer     = _get_kafka_producer(request),
                fusion_url   = getattr(request.app.state, "fusion_rest_url", None),
                expected_models = request.headers.get(
                    "X-Expected-Models", "nlp_intent"
                ).split(","),
            ),
            name=f"fusion-forward-{result.analysis_id}",
        )

    return result


# ---------------------------------------------------------------------------
# Batch Analysis Endpoint
# ---------------------------------------------------------------------------

class BatchAnalysisRequest(EmailAnalysisRequest.__bases__[0]):  # type: ignore[misc]
    """Wrapper for batch submission — accepts a list of EmailAnalysisRequest objects."""
    from pydantic import BaseModel, Field as _Field
    items: list[EmailAnalysisRequest] = _Field(..., min_length=1, max_length=50)


@router.post(
    "/analyse/batch",
    status_code=status.HTTP_200_OK,
    summary="Analyse up to 50 emails concurrently",
)
async def analyse_batch(
    request:      Request,
    orchestrator: EmailAnalysisOrchestrator = Depends(_get_orchestrator),
    _rate:        None = Depends(_enforce_rate_limit),
) -> dict:
    """
    Accepts a JSON body with an `items` array of up to 50 EmailAnalysisRequest objects.
    Returns an array of EmailAnalysisResult objects in the same order.
    Analyses are run concurrently (bounded by _ANALYSIS_EXECUTOR worker count).
    """
    try:
        body = await request.json()
        from pydantic import BaseModel, Field
        class _Batch(BaseModel):
            items: list[EmailAnalysisRequest] = Field(..., min_length=1, max_length=50)
        batch = _Batch.model_validate(body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_batch", "message": str(exc)},
        )

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(_ANALYSIS_EXECUTOR, orchestrator.analyse, item)
        for item in batch.items
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    response_items = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            response_items.append({
                "request_id": batch.items[i].request_id,
                "error":      str(res),
                "status":     "failed",
            })
        else:
            response_items.append(res.model_dump())

    return {
        "total":     len(batch.items),
        "succeeded": sum(1 for r in results if not isinstance(r, Exception)),
        "failed":    sum(1 for r in results if isinstance(r, Exception)),
        "results":   response_items,
    }


# ---------------------------------------------------------------------------
# Single URL Analysis Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/url/analyse",
    status_code=status.HTTP_200_OK,
    summary="Analyse a single URL for phishing signals",
)
async def analyse_url(
    request: Request,
    orchestrator: EmailAnalysisOrchestrator = Depends(_get_orchestrator),
) -> dict:
    """Lightweight single-URL analysis without the full email pipeline."""
    try:
        body = await request.json()
        raw_url: str = body.get("url", "").strip()
        if not raw_url:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "missing_url", "message": "'url' field is required"},
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    loop   = asyncio.get_event_loop()
    record = await loop.run_in_executor(
        _ANALYSIS_EXECUTOR,
        orchestrator._url_analyser.analyse_url,
        raw_url,
    )
    return record.model_dump()


# ---------------------------------------------------------------------------
# Utility Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", tags=["Health"])
async def email_analysis_health(request: Request) -> dict:
    """Liveness + readiness probe for the email analysis service."""
    redis_ok = False
    try:
        redis = getattr(request.app.state, "redis_client", None)
        if redis:
            await redis.ping()
            redis_ok = True
    except Exception:
        pass

    return {
        "status":              "ready",
        "service":             "email-analysis",
        "redis":               "healthy" if redis_ok else "unavailable",
        "executor_threads":    _ANALYSIS_EXECUTOR._max_workers,
        "brand_corpus_size":   _get_corpus_size(),
        "rate_limit":          f"{_RATE_LIMIT_REQUESTS} req/min per org",
        "forward_threshold":   _FORWARD_THRESHOLD,
    }


@router.get("/brands", tags=["Configuration"])
async def list_brand_corpus() -> dict:
    """Returns the active brand protection corpus (domain list only, no internals)."""
    from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import BRAND_CORPUS
    sorted_brands = sorted(BRAND_CORPUS)
    return {
        "count":   len(sorted_brands),
        "brands":  sorted_brands,
    }


# ---------------------------------------------------------------------------
# Risk Fusion Engine Forwarding
# ---------------------------------------------------------------------------

async def _forward_to_fusion(
    result:          EmailAnalysisResult,
    orchestrator:    EmailAnalysisOrchestrator,
    producer:        Optional[Any],
    fusion_url:      Optional[str],
    expected_models: list[str],
) -> None:
    """
    Forwards a FusionScorePayload to the Risk Fusion Engine.
    Tries Kafka first; falls back to REST POST if Kafka producer is unavailable.
    Runs as a fire-and-forget asyncio.Task — errors are logged but not re-raised.
    """
    try:
        fusion_payload: FusionScorePayload = orchestrator.build_fusion_payload(
            result, expected_models=expected_models
        )

        if producer is not None:
            await _publish_to_kafka(producer, fusion_payload)
            logger.debug(
                "Fusion payload published to Kafka",
                extra={
                    "analysis_id":    result.analysis_id,
                    "session_id":     result.session_id,
                    "composite_score": result.composite_score,
                },
            )
        elif fusion_url:
            await _post_to_fusion_rest(fusion_url, fusion_payload)
        else:
            logger.warning(
                "No Kafka producer or REST URL configured — fusion payload dropped",
                extra={"analysis_id": result.analysis_id},
            )
    except Exception:
        logger.error(
            "Failed to forward analysis result to Risk Fusion Engine",
            extra={"analysis_id": result.analysis_id},
            exc_info=True,
        )


async def _publish_to_kafka(producer: Any, payload: FusionScorePayload) -> None:
    """Publishes a FusionScorePayload to the `sentinelai.threat.scores` Kafka topic."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    _exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="email-kafka")
    loop  = asyncio.get_event_loop()

    await loop.run_in_executor(
        _exec,
        partial(
            producer.produce,
            topic   = "sentinelai.threat.scores",
            value   = payload.model_dump_json().encode("utf-8"),
            key     = payload.organization_id.encode("utf-8"),
            headers = [
                ("sentinel-model-type",  b"nlp_intent"),
                ("sentinel-source",      b"email-analysis"),
                ("sentinel-org-id",      payload.organization_id.encode()),
            ],
        ),
    )
    await loop.run_in_executor(_exec, partial(producer.poll, 0))


async def _post_to_fusion_rest(fusion_url: str, payload: FusionScorePayload) -> None:
    """POSTs a FusionScorePayload to the Risk Fusion Engine REST fallback endpoint."""
    import httpx
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"{fusion_url}/fusion/ingest",
            content      = payload.model_dump_json(),
            headers      = {"Content-Type": "application/json"},
        )
        response.raise_for_status()


def _get_corpus_size() -> int:
    try:
        from sentinel_ai.services.email_analysis.core.heuristics.domain_analyser import BRAND_CORPUS
        return len(BRAND_CORPUS)
    except Exception:
        return 0
