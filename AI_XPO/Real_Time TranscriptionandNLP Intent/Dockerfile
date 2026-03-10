# =============================================================================
# SentinelAI NLP Service — Dockerfile
# Multi-stage build with CTranslate2 + HuggingFace transformers support.
# =============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt


# ── Runtime ───────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="sentinel-nlp-service" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    HF_HOME="/models/cache/huggingface" \
    TRANSFORMERS_CACHE="/models/cache/huggingface" \
    # Disable HF Hub telemetry
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN groupadd --gid 10001 sentinel \
    && useradd --uid 10001 --gid sentinel --shell /bin/bash --create-home sentinel

RUN mkdir -p /models/whisper /models/intent_classifier /models/cache/huggingface /run/secrets \
    && chown -R sentinel:sentinel /models /run/secrets

WORKDIR /app
COPY --chown=sentinel:sentinel . .

USER sentinel

EXPOSE 8002 9091

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--no-access-log"]
