"""
SentinelAI NLP Service — Core Configuration
All runtime configuration sourced from environment variables.
Secrets are never hardcoded; injected via Kubernetes Secrets or Vault.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
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


class WhisperModelSize(str, Enum):
    """
    Available Whisper model sizes.
    Production recommendation: 'large-v3' on GPU, 'medium' on CPU.
    Distil-Whisper variants for 6× faster inference with ~1% WER delta.
    """
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    DISTIL_MEDIUM = "distil-medium.en"
    DISTIL_LARGE_V3 = "distil-large-v3"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Service Identity ──────────────────────────────────────────────────────
    SERVICE_NAME: str = "sentinel-nlp-service"
    SERVICE_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.PRODUCTION
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # ── FastAPI ───────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8002
    API_WORKERS: int = 1
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_PUBLIC_KEY_PATH: str = "/run/secrets/jwt_public_key.pem"
    JWT_ALGORITHM: str = "RS256"
    JWT_AUDIENCE: str = "sentinel-services"
    JWT_ISSUER: str = "sentinel-auth-service"

    # ── Kafka ─────────────────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: List[str] = Field(default=["kafka:9092"])
    KAFKA_AUDIO_TOPIC: str = "raw_audio_events"
    KAFKA_THREAT_TOPIC: str = "threat_scores"
    KAFKA_TRANSCRIPT_TOPIC: str = "transcription_events"
    KAFKA_CONSUMER_GROUP: str = "nlp-inference-group"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    KAFKA_MAX_POLL_RECORDS: int = 5
    KAFKA_SESSION_TIMEOUT_MS: int = 45000
    KAFKA_HEARTBEAT_INTERVAL_MS: int = 15000
    KAFKA_ENABLE_AUTO_COMMIT: bool = False
    KAFKA_SASL_MECHANISM: Optional[str] = None
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[SecretStr] = None
    KAFKA_SSL_CA_LOCATION: Optional[str] = None

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/1"
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_TRANSCRIPT_TTL_SECONDS: int = 7200
    REDIS_DEDUP_TTL_SECONDS: int = 300

    # ── Whisper ───────────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: WhisperModelSize = WhisperModelSize.LARGE_V3
    WHISPER_MODEL_PATH: str = "/models/whisper"
    WHISPER_LANGUAGE: Optional[str] = "en"
    WHISPER_TASK: str = "transcribe"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_BEST_OF: int = 5
    WHISPER_TEMPERATURE: float = 0.0
    WHISPER_CONDITION_ON_PREV_TOKENS: bool = False
    WHISPER_VAD_FILTER: bool = True
    WHISPER_VAD_THRESHOLD: float = 0.5
    WHISPER_MIN_SILENCE_DURATION_MS: int = 500
    WHISPER_WORD_TIMESTAMPS: bool = True
    WHISPER_MAX_SEGMENT_DURATION_S: float = 30.0
    WHISPER_BATCH_SIZE: int = 1
    WHISPER_COMPUTE_TYPE: str = "float16"  # float16 (GPU) | int8 (CPU)
    WHISPER_NUM_WORKERS: int = 1

    # ── DeBERTa-v3 Intent Classifier ─────────────────────────────────────────
    INTENT_MODEL_NAME: str = "microsoft/deberta-v3-base"
    INTENT_MODEL_PATH: str = "/models/intent_classifier"
    INTENT_TOKENIZER_MAX_LENGTH: int = 512
    INTENT_INFERENCE_BATCH_SIZE: int = 8
    INTENT_CONFIDENCE_THRESHOLD: float = 0.65
    INTENT_TOP_K_LABELS: int = 3
    INTENT_USE_ONNX: bool = False
    INTENT_NUM_LABELS: int = 10

    # Intent label map: index → label name
    INTENT_LABEL_MAP: Dict[int, str] = Field(
        default={
            0: "BENIGN",
            1: "OTP_HARVESTING",
            2: "FINANCIAL_FRAUD",
            3: "IMPERSONATION",
            4: "PHISHING_LINK",
            5: "CREDENTIAL_THEFT",
            6: "VISHING",
            7: "ACCOUNT_TAKEOVER",
            8: "SOCIAL_ENGINEERING",
            9: "URGENCY_MANIPULATION",
        }
    )

    THREAT_INTENT_LABELS: List[str] = Field(
        default=[
            "OTP_HARVESTING",
            "FINANCIAL_FRAUD",
            "IMPERSONATION",
            "PHISHING_LINK",
            "CREDENTIAL_THEFT",
            "VISHING",
            "ACCOUNT_TAKEOVER",
            "SOCIAL_ENGINEERING",
            "URGENCY_MANIPULATION",
        ]
    )

    # ── Inference Engine ─────────────────────────────────────────────────────
    DEVICE_PREFERENCE: DevicePreference = DevicePreference.AUTO
    MAX_CONCURRENT_INFERENCES: int = 4
    INFERENCE_TIMEOUT_SECONDS: float = 30.0
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_MAX_DURATION_SECONDS: float = 300.0
    AUDIO_MIN_DURATION_SECONDS: float = 0.5
    TRANSCRIPT_MIN_WORDS: int = 3

    # ── PII Redaction ─────────────────────────────────────────────────────────
    REDACT_PII_IN_LOGS: bool = True
    REDACT_PII_IN_TOPIC: bool = False
    PII_REDACTION_PLACEHOLDER: str = "[REDACTED]"

    # ── AWS ───────────────────────────────────────────────────────────────────
    AWS_REGION: str = "us-east-1"
    S3_MODEL_BUCKET: Optional[str] = None
    S3_AUDIO_BUCKET: Optional[str] = None

    # ── Observability ─────────────────────────────────────────────────────────
    PROMETHEUS_PORT: int = 9091
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[AnyHttpUrl] = None
    OTEL_SERVICE_NAME: str = "sentinel-nlp-service"
    CORRELATION_ID_HEADER: str = "X-Correlation-ID"

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

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        if not (0.0 <= self.INTENT_CONFIDENCE_THRESHOLD <= 1.0):
            raise ValueError("INTENT_CONFIDENCE_THRESHOLD must be in [0.0, 1.0]")
        if not (0.0 <= self.WHISPER_VAD_THRESHOLD <= 1.0):
            raise ValueError("WHISPER_VAD_THRESHOLD must be in [0.0, 1.0]")
        return self

    @property
    def kafka_bootstrap_servers_str(self) -> str:
        return ",".join(self.KAFKA_BOOTSTRAP_SERVERS)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION

    @property
    def threat_label_set(self) -> frozenset:
        return frozenset(self.THREAT_INTENT_LABELS)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton — safe for async contexts."""
    return Settings()
