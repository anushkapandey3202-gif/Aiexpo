# =============================================================================
# SentinelAI — ML Inference Service
# Supports: Whisper, ECAPA-TDNN, DeBERTa-v3, ViT deepfake detection
# GPU-accelerated with CUDA 12.1 base; gracefully falls back to CPU
# =============================================================================

# ── Stage 1: Model cache warmer ───────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS model-prep

WORKDIR /model-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir huggingface_hub[cli]==0.21.4

# Pre-download model weights at build time (baked into layer cache)
# Override MODEL_CACHE_DIR at build time if using a shared NFS/S3 mount
ARG HF_TOKEN=""
ARG MODEL_CACHE_DIR="/model-cache/weights"
ENV HF_HOME=${MODEL_CACHE_DIR} \
    TRANSFORMERS_CACHE=${MODEL_CACHE_DIR} \
    HF_HUB_TOKEN=${HF_TOKEN}

RUN mkdir -p ${MODEL_CACHE_DIR}

# Download DeBERTa-v3-base (NLP intent classification)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/deberta-v3-base', \
    cache_dir='${MODEL_CACHE_DIR}', \
    ignore_patterns=['*.msgpack','*.h5','flax_*'])"

# Download ViT-base (video frame deepfake detection)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('google/vit-base-patch16-224', \
    cache_dir='${MODEL_CACHE_DIR}', \
    ignore_patterns=['*.msgpack','*.h5','flax_*'])"


# ── Stage 2: CUDA runtime base ────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu-base

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    ffmpeg \
    libpq5 \
    libssl3 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python


# ── Stage 3: Python dependency builder ───────────────────────────────────────
FROM gpu-base AS py-builder

WORKDIR /build

RUN pip install --upgrade pip wheel setuptools

COPY requirements/ml.txt ./requirements.txt

# Install PyTorch with CUDA 12.1 index, then the rest
RUN pip install --no-cache-dir \
    torch==2.2.1+cu121 \
    torchaudio==2.2.1+cu121 \
    torchvision==0.17.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 4: Production inference image ───────────────────────────────────────
FROM gpu-base AS production

LABEL maintainer="sentinelai-platform@yourorg.com" \
      service="sentinelai-ml-inference" \
      version="1.0.0"

# Non-root user
RUN groupadd --gid 1001 sentinel && \
    useradd --uid 1001 --gid sentinel --shell /bin/bash --create-home sentinel

# Copy Python packages
COPY --from=py-builder /install /usr/local

# Copy model weights from cache stage
ARG MODEL_CACHE_DIR="/model-cache/weights"
COPY --from=model-prep /model-cache/weights /opt/sentinelai/models
RUN chmod -R 550 /opt/sentinelai/models

WORKDIR /app

COPY --chown=sentinel:sentinel ./services/ml_inference ./

# Runtime directories
RUN mkdir -p /app/tmp /app/logs /app/audio_scratch && \
    chown -R sentinel:sentinel /app/tmp /app/logs /app/audio_scratch && \
    chmod 750 /app/tmp /app/logs /app/audio_scratch

USER sentinel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    PORT=8001 \
    TRANSFORMERS_CACHE=/opt/sentinelai/models \
    HF_HOME=/opt/sentinelai/models \
    # Torch performance tuning
    TORCH_HOME=/opt/sentinelai/models \
    OMP_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES=0 \
    TOKENIZERS_PARALLELISM=false \
    # Whisper config
    WHISPER_MODEL_SIZE=base.en \
    # ECAPA-TDNN embedding dim
    VOICE_EMBEDDING_DIM=192 \
    # Inference batch settings
    MAX_AUDIO_SECONDS=30 \
    MAX_VIDEO_FRAMES=64

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health/live || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["sh", "-c", \
     "uvicorn main:app \
      --host 0.0.0.0 \
      --port $PORT \
      --workers 1 \
      --loop uvloop \
      --http httptools \
      --timeout-keep-alive 120 \
      --log-level info \
      --no-server-header"]
