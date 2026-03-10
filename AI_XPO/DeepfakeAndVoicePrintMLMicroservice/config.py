"""
SentinelAI ML Service — Core Configuration
Manages all environment-driven settings via pydantic-settings.
All secrets are injected at runtime; never hardcoded.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DevicePreference(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service Identity ---
    SERVICE_NAME: str = "sentinel-ml-service"
    SERVICE_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.PRODUCTION
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # --- FastAPI ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_WORKERS: int = 1  # Single worker; Kafka consumer manages concurrency
    ALLOWED_ORIGINS: List[str] = ["*"]

    # --- JWT (RS256 public key for validation only) ---
    JWT_PUBLIC_KEY_PATH: str = "/run/secrets/jwt_public_key.pem"
    JWT_ALGORITHM: str = "RS256"
    JWT_AUDIENCE: str = "sentinel-services"
    JWT_ISSUER: str = "sentinel-auth-service"

    # --- Kafka ---
    KAFKA_BOOTSTRAP_SERVERS: List[str] = Field(
        default=["kafka:9092"],
        description="Kafka broker addresses",
    )
    KAFKA_AUDIO_TOPIC: str = "raw_audio_events"
    KAFKA_THREAT_TOPIC: str = "threat_scores"
    KAFKA_CONSUMER_GROUP: str = "ml-inference-group"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    KAFKA_MAX_POLL_RECORDS: int = 10
    KAFKA_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_HEARTBEAT_INTERVAL_MS: int = 10000
    KAFKA_ENABLE_AUTO_COMMIT: bool = False
    KAFKA_SASL_MECHANISM: Optional[str] = None
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[SecretStr] = None
    KAFKA_SSL_CA_LOCATION: Optional[str] = None

    # --- Pinecone ---
    PINECONE_API_KEY: SecretStr = Field(..., description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "sentinel-voiceprints"
    PINECONE_NAMESPACE: str = "voiceprint-v1"
    PINECONE_TOP_K: int = 5
    PINECONE_SCORE_THRESHOLD: float = 0.88
    PINECONE_EMBEDDING_DIM: int = 192  # ECAPA-TDNN output dim

    # --- Redis ---
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_RESULT_TTL_SECONDS: int = 3600
    REDIS_DEDUP_TTL_SECONDS: int = 300

    # --- Model Paths (mounted via Kubernetes PVC or S3 sync init-container) ---
    ECAPA_TDNN_MODEL_PATH: str = "/models/ecapa_tdnn/ecapa_tdnn_v1.pt"
    RAWNET3_MODEL_PATH: str = "/models/rawnet3/rawnet3_v1.pt"
    MODEL_CACHE_DIR: str = "/models/cache"

    # --- Inference ---
    DEVICE_PREFERENCE: DevicePreference = DevicePreference.AUTO
    INFERENCE_BATCH_SIZE: int = 4
    INFERENCE_TIMEOUT_SECONDS: float = 10.0
    MAX_CONCURRENT_INFERENCES: int = 8
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_MAX_DURATION_SECONDS: float = 30.0
    AUDIO_MIN_DURATION_SECONDS: float = 1.0

    # --- Threat Scoring ---
    DEEPFAKE_THRESHOLD_WARN: float = 0.55
    DEEPFAKE_THRESHOLD_ALERT: float = 0.80
    VOICEPRINT_MISMATCH_THRESHOLD: float = 0.30
    COMBINED_THREAT_WEIGHT_DEEPFAKE: float = 0.60
    COMBINED_THREAT_WEIGHT_VOICE: float = 0.40

    # --- Observability ---
    PROMETHEUS_PORT: int = 9090
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[AnyHttpUrl] = None
    OTEL_SERVICE_NAME: str = "sentinel-ml-service"
    CORRELATION_ID_HEADER: str = "X-Correlation-ID"

    # --- AWS (for S3 model artifact fetching) ---
    AWS_REGION: str = "us-east-1"
    S3_MODEL_BUCKET: Optional[str] = None
    S3_MODEL_PREFIX: str = "models/"

    @field_validator("KAFKA_BOOTSTRAP_SERVERS", mode="before")
    @classmethod
    def parse_kafka_servers(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}")
        return upper

    @property
    def kafka_bootstrap_servers_str(self) -> str:
        return ",".join(self.KAFKA_BOOTSTRAP_SERVERS)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — safe across async contexts."""
    return Settings()
