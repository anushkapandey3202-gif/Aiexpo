"""
SentinelAI ML Service — Inference Device Manager
Manages GPU/CPU device selection, memory monitoring, and
a concurrency semaphore to prevent GPU OOM during burst load.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import torch

from app.core.config import DevicePreference, get_settings
from app.core.logging import get_logger

logger = get_logger("inference.device_manager")


class DeviceManager:
    """
    Singleton managing PyTorch device selection and inference concurrency.

    Responsibilities:
    - Resolve the optimal compute device at startup (CUDA > MPS > CPU).
    - Expose an asyncio.Semaphore to cap concurrent GPU/CPU inference tasks,
      preventing memory exhaustion under burst load.
    - Provide memory telemetry for observability hooks.
    """

    _instance: Optional["DeviceManager"] = None

    def __new__(cls) -> "DeviceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        self._device: torch.device = self._resolve_device(
            settings.DEVICE_PREFERENCE
        )
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCES)
        self._initialized = True
        logger.info(
            "DeviceManager initialized",
            extra={
                "device": str(self._device),
                "max_concurrent_inferences": settings.MAX_CONCURRENT_INFERENCES,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
        )

    @staticmethod
    def _resolve_device(preference: DevicePreference) -> torch.device:
        """Resolve device based on preference and hardware availability."""
        if preference == DevicePreference.CUDA:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")

        if preference == DevicePreference.MPS:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("MPS requested but unavailable; falling back to CPU.")
            return torch.device("cpu")

        if preference == DevicePreference.CPU:
            return torch.device("cpu")

        # AUTO: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_name(self) -> str:
        return str(self._device)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore

    def get_memory_stats(self) -> dict:
        """Return GPU memory stats (empty dict on CPU/MPS)."""
        if self._device.type != "cuda":
            return {"device": self.device_name, "memory_tracking": "N/A"}
        allocated = torch.cuda.memory_allocated(self._device) / 1e6
        reserved = torch.cuda.memory_reserved(self._device) / 1e6
        total = torch.cuda.get_device_properties(self._device).total_memory / 1e6
        return {
            "device": self.device_name,
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "total_mb": round(total, 2),
            "utilization_pct": round((allocated / total) * 100, 2) if total > 0 else 0,
        }

    def clear_cache(self) -> None:
        """Flush GPU memory cache — call between large batch inferences."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA memory cache cleared.")


# Module-level singleton accessor
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager
