# =============================================================================
# SentinelAI ML Service — Production Dockerfile
# Multi-stage build: isolates build deps from runtime image.
# Base: nvidia/cuda for GPU inference; falls back to CPU automatically.
# =============================================================================

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System deps required for torch, torchaudio, and numpy compilation
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


# ── Stage 2: CUDA runtime image ───────────────────────────────────────────────
# Use nvidia/cuda for GPU nodes; Python base for CPU-only nodes.
# Switch base image via --build-arg in your CI/CD pipeline.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime

LABEL maintainer="SentinelAI Platform Team" \
      org.opencontainers.image.title="sentinel-ml-service" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.description="Deepfake & Voiceprint ML inference microservice"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # Torch thread management for containerized environments
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TORCH_HOME=/models/cache

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for runtime security
RUN groupadd --gid 10001 sentinel \
    && useradd --uid 10001 --gid sentinel --shell /bin/bash --create-home sentinel

# Model directory (PVC mount point in Kubernetes)
RUN mkdir -p /models/cache /models/ecapa_tdnn /models/rawnet3 \
    && chown -R sentinel:sentinel /models

WORKDIR /app
COPY --chown=sentinel:sentinel . .

# JWT public key mount point (Kubernetes secret)
RUN mkdir -p /run/secrets && chown sentinel:sentinel /run/secrets

USER sentinel

# Expose FastAPI and Prometheus ports
EXPOSE 8001 9090

# Liveness probe executable
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use uvloop for best async performance
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--no-access-log"]
