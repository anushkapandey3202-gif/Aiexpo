"""
SentinelAI - Core Application Settings
=======================================
Aggregates all environment-based configuration via Pydantic Settings v2.
Each subsystem has its own settings mixin for SOLID/SRP compliance.
Never import raw os.environ — always consume settings from this module.
"""
from __future__ import annotations

import base64
import logging
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-setting Mixins
# ---------------------------------------------------------------------------

class DatabaseSettings(BaseSettings):
    """PostgreSQL async connection pool configuration."""

    POSTGRES_HOST: str = Field(..., description="PostgreSQL host")
    POSTGRES_PORT: int = Field(5432, ge=1024, le=65535)
    POSTGRES_DB: str = Field(..., description="Database name")
    POSTGRES_USER: str = Field(..., description="Database user")
    POSTGRES_PASSWORD: SecretStr = Field(..., description="Database password")
    POSTGRES_POOL_SIZE: int = Field(20, ge=5, le=100, description="Connection pool size")
    POSTGRES_MAX_OVERFLOW: int = Field(40, ge=0, le=200, description="Max overflow beyond pool_size")
    POSTGRES_POOL_TIMEOUT: int = Field(30, ge=5, le=120, description="Seconds to wait for a connection")
    POSTGRES_POOL_RECYCLE: int = Field(3600, description="Recycle connections after N seconds")
    POSTGRES_SSL_MODE: Literal[
        "disable", "allow", "prefer", "require", "verify-ca", "verify-full"
    ] = "require"

    @computed_field
    @property
    def async_database_url(self) -> str:
        """asyncpg DSN — used by SQLAlchemy async engine at runtime."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """psycopg2 DSN — used exclusively by Alembic migrations."""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


class RedisSettings(BaseSettings):
    """Redis state/cache configuration."""

    REDIS_HOST: str = Field("localhost")
    REDIS_PORT: int = Field(6379, ge=1, le=65535)
    REDIS_PASSWORD: Optional[SecretStr] = None
    REDIS_DB: int = Field(0, ge=0, le=15)
    REDIS_SSL: bool = True
    REDIS_MAX_CONNECTIONS: int = Field(50, ge=10)
    REDIS_SOCKET_TIMEOUT: int = Field(5, ge=1)
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(5, ge=1)

    @computed_field
    @property
    def redis_url(self) -> str:
        scheme = "rediss" if self.REDIS_SSL else "redis"
        password_part = (
            f":{self.REDIS_PASSWORD.get_secret_value()}@" if self.REDIS_PASSWORD else ""
        )
        return f"{scheme}://{password_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


class PineconeSettings(BaseSettings):
    """Pinecone serverless vector store configuration."""

    PINECONE_API_KEY: SecretStr = Field(..., description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = Field(..., description="Pinecone cloud region (e.g. us-east1-gcp)")
    PINECONE_INDEX_NAME: str = Field("sentinelai-voiceprints")
    PINECONE_DIMENSION: int = Field(192, description="ECAPA-TDNN output dimension")
    PINECONE_METRIC: Literal["cosine", "euclidean", "dotproduct"] = "cosine"
    PINECONE_REPLICAS: int = Field(2, ge=1, le=10)
    PINECONE_SHARDS: int = Field(1, ge=1, le=20)
    PINECONE_NAMESPACE_PREFIX: str = Field(
        "org", description="Namespace prefix for per-org isolation: '{prefix}:{org_id}'"
    )


class MilvusSettings(BaseSettings):
    """Milvus self-hosted vector store configuration."""

    MILVUS_HOST: str = Field("localhost")
    MILVUS_PORT: int = Field(19530)
    MILVUS_USER: str = Field("root")
    MILVUS_PASSWORD: SecretStr = Field(...)
    MILVUS_SECURE: bool = True
    MILVUS_SERVER_PEM_PATH: Optional[str] = Field(
        None, description="Path to TLS server cert (required when MILVUS_SECURE=True)"
    )
    MILVUS_COLLECTION_NAME: str = Field("sentinelai_voiceprints")
    MILVUS_DIMENSION: int = Field(192)
    MILVUS_INDEX_TYPE: Literal["IVF_FLAT", "IVF_SQ8", "HNSW"] = "HNSW"
    MILVUS_METRIC_TYPE: Literal["COSINE", "L2", "IP"] = "COSINE"
    MILVUS_NLIST: int = Field(1024, description="IVF cluster count (ignored for HNSW)")
    MILVUS_HNSW_M: int = Field(16, description="HNSW M param — higher = better recall, more RAM")
    MILVUS_HNSW_EF_CONSTRUCTION: int = Field(200, description="Build-time search breadth")


class EncryptionSettings(BaseSettings):
    """AES-256-GCM field-level encryption configuration."""

    # Must be exactly 32 bytes when base64-decoded
    AES_MASTER_KEY: SecretStr = Field(
        ..., description="Base64-encoded 32-byte AES-256 master key"
    )
    AES_KEY_VERSION: int = Field(1, ge=1, description="Current key version for rotation tracking")
    AES_KEY_ROTATION_DAYS: int = Field(90, ge=30, le=365)

    @model_validator(mode="after")
    def _validate_aes_key_length(self) -> "EncryptionSettings":
        try:
            key_bytes = base64.b64decode(self.AES_MASTER_KEY.get_secret_value())
        except Exception as exc:
            raise ValueError("AES_MASTER_KEY must be a valid base64 string") from exc
        if len(key_bytes) != 32:
            raise ValueError(
                f"AES_MASTER_KEY must decode to exactly 32 bytes (AES-256), got {len(key_bytes)}"
            )
        return self


class JWTSettings(BaseSettings):
    """JWT asymmetric signing configuration (RS256/ES256)."""

    JWT_ALGORITHM: Literal["RS256", "ES256"] = "RS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(15, ge=5, le=60)
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(7, ge=1, le=30)
    JWT_PRIVATE_KEY_PATH: str = Field(
        "/run/secrets/jwt_private.pem",
        description="Path to PEM-encoded RSA/EC private key (Docker secret or K8s secret mount)",
    )
    JWT_PUBLIC_KEY_PATH: str = Field("/run/secrets/jwt_public.pem")
    JWT_ISSUER: str = Field("sentinelai.io")
    JWT_AUDIENCE: str = Field("sentinelai-api")


class AWSSettings(BaseSettings):
    """AWS infrastructure configuration."""

    AWS_REGION: str = Field("us-east-1")
    AWS_S3_EVIDENCE_BUCKET: str = Field(..., description="S3 bucket for raw threat evidence")
    AWS_S3_EVIDENCE_KMS_KEY_ID: str = Field(..., description="KMS CMK ARN for S3 SSE-KMS")
    AWS_SAGEMAKER_ENDPOINT_DEEPFAKE: Optional[str] = Field(
        None, description="SageMaker real-time inference endpoint for ViT deepfake model"
    )
    AWS_SAGEMAKER_ENDPOINT_VOICE: Optional[str] = Field(
        None, description="SageMaker endpoint for ECAPA-TDNN voice biometrics"
    )


# ---------------------------------------------------------------------------
# Master Settings
# ---------------------------------------------------------------------------

class Settings(
    DatabaseSettings,
    RedisSettings,
    PineconeSettings,
    MilvusSettings,
    EncryptionSettings,
    JWTSettings,
    AWSSettings,
):
    """
    Single source of truth for all SentinelAI runtime configuration.
    Loaded once at startup; consumed via get_settings() everywhere.
    """

    APP_NAME: str = "SentinelAI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "production"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # Vector store backend selection (single env var to switch implementations)
    VECTOR_STORE_BACKEND: Literal["pinecone", "milvus"] = "pinecone"

    # ML Detection Thresholds
    DEEPFAKE_CONFIDENCE_THRESHOLD: float = Field(
        0.85, ge=0.0, le=1.0,
        description="ViT confidence above which a deepfake alert is triggered"
    )
    VOICE_SIMILARITY_THRESHOLD: float = Field(
        0.92, ge=0.0, le=1.0,
        description="ECAPA-TDNN cosine similarity above which a voice clone is flagged"
    )
    SOCIAL_ENGINEERING_THRESHOLD: float = Field(
        0.80, ge=0.0, le=1.0,
        description="DeBERTa-v3 intent score above which a social engineering alert fires"
    )

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(100, ge=10)
    RATE_LIMIT_BURST_SIZE: int = Field(20, ge=5)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the application-wide Settings singleton.
    Cached via lru_cache — safe to call from anywhere without performance penalty.
    """
    _settings = Settings()  # type: ignore[call-arg]
    logger.info(
        "SentinelAI settings loaded",
        extra={
            "environment": _settings.ENVIRONMENT,
            "vector_backend": _settings.VECTOR_STORE_BACKEND,
            "app_version": _settings.APP_VERSION,
        },
    )
    return _settings


# Module-level convenience alias
settings: Settings = get_settings()
