"""
SentinelAI ML Service — Structured Logging
Produces JSON-formatted logs compatible with ELK / CloudWatch / Datadog.
Propagates correlation IDs and inference trace metadata on every record.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from app.core.config import get_settings

# --- Context Variables (propagated across async task boundaries) ---
_correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id", default=""
)
_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_user_id_var: ContextVar[str] = ContextVar("user_id", default="")
_pipeline_var: ContextVar[str] = ContextVar("pipeline", default="")


def set_correlation_id(correlation_id: str) -> None:
    _correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    val = _correlation_id_var.get()
    return val if val else str(uuid.uuid4())


def set_request_context(
    session_id: str = "",
    user_id: str = "",
    pipeline: str = "",
) -> None:
    _session_id_var.set(session_id)
    _user_id_var.set(user_id)
    _pipeline_var.set(pipeline)


def clear_request_context() -> None:
    _correlation_id_var.set("")
    _session_id_var.set("")
    _user_id_var.set("")
    _pipeline_var.set("")


class SentinelJSONFormatter(logging.Formatter):
    """
    Formats every log record as a single-line JSON object.
    Enriches records with service metadata and current request context.
    """

    _settings = None

    def _get_settings(self):
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    def format(self, record: logging.LogRecord) -> str:
        settings = self._get_settings()

        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%f"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "environment": settings.ENVIRONMENT.value,
            # Request context
            "correlation_id": _correlation_id_var.get() or "N/A",
            "session_id": _session_id_var.get() or "N/A",
            "user_id": _user_id_var.get() or "N/A",
            "pipeline": _pipeline_var.get() or "N/A",
            # Source location
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Attach exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Attach any extra fields passed via extra={} in log calls
        extra_keys = set(record.__dict__.keys()) - {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key in extra_keys:
            log_data[key] = getattr(record, key)

        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Context manager for timing and logging inference performance."""

    def __init__(self, logger: logging.Logger, operation: str, **metadata: Any):
        self._logger = logger
        self._operation = operation
        self._metadata = metadata
        self._start: float = 0.0

    def __enter__(self) -> "PerformanceLogger":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        level = logging.ERROR if exc_type else logging.INFO
        self._logger.log(
            level,
            "Operation completed" if not exc_type else "Operation failed",
            extra={
                "operation": self._operation,
                "duration_ms": round(elapsed_ms, 3),
                "success": exc_type is None,
                **self._metadata,
            },
        )
        return False  # Do not suppress exceptions


def configure_logging() -> None:
    """
    Bootstraps the root logger and silences noisy third-party loggers.
    Should be called exactly once at application startup.
    """
    settings = get_settings()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SentinelJSONFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Suppress verbose third-party loggers in production
    if settings.is_production:
        for noisy_logger in [
            "aiokafka",
            "kafka",
            "uvicorn.access",
            "httpx",
            "httpcore",
            "pinecone",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Factory for module-level loggers — consistent naming convention."""
    return logging.getLogger(f"sentinel.{name}")
