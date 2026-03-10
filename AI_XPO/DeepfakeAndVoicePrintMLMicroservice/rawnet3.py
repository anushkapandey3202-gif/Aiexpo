"""
SentinelAI ML Service — RawNet3 Anti-Spoofing Inference Wrapper

RawNet3 is a raw-waveform end-to-end anti-spoofing model that operates
directly on PCM audio without handcrafted features, using sinc-based
filters and residual blocks with GRU aggregation.

Production Architecture:
- SincConv filters learn frequency-selective preprocessing from raw audio.
- ResBlock stack extracts hierarchical temporal features.
- GRU aggregates temporal context for utterance-level spoof detection.
- Output: scalar spoof probability in [0.0, 1.0].

NOTE: Architecture mirrors RawNet3's published topology. Swap the
_sync_infer method's model call with a SpeechBrain-loaded checkpoint
or your fine-tuned .pt file without changing any surrounding async logic.
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import get_settings
from app.core.logging import get_logger
from app.inference.device_manager import get_device_manager

logger = get_logger("inference.rawnet3")


# ---------------------------------------------------------------------------
# SincConv — Learns bandpass sinc filters from raw waveform
# ---------------------------------------------------------------------------


class SincConv(nn.Module):
    """
    Parameterized sinc-function convolution layer.
    Learns low/high cutoff frequencies for each filter,
    replacing handcrafted MFCC/mel features with data-driven equivalents.
    """

    def __init__(
        self,
        num_filters: int = 128,
        kernel_size: int = 1024,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._kernel_size = kernel_size + (1 if kernel_size % 2 == 0 else 0)
        self._num_filters = num_filters

        # Mel-spaced initialization for low and band frequencies
        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel_points = torch.linspace(
            self._hz2mel(low_hz), self._hz2mel(high_hz), num_filters + 1
        )
        hz_points = self._mel2hz(mel_points)
        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))
        self._min_low_hz = min_low_hz
        self._min_band_hz = min_band_hz

        # Hamming window for filter kernel
        n_lin = torch.linspace(0.0, (self._kernel_size - 1) / 2.0, steps=self._kernel_size // 2)
        self.register_buffer("_window", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self._kernel_size))
        n = (self._kernel_size - 1) / 2.0
        self.register_buffer("_n", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    @staticmethod
    def _hz2mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel2hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, 1, T]
        low = self._min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self._min_band_hz + torch.abs(self.band_hz_),
            self._min_low_hz,
            self._sample_rate / 2.0,
        )
        band = (high - low)[:, 0]
        f_times_t = self._n * low  # [num_filters, kernel_half]

        low_pass_l = 2.0 * low * torch.sinc(2 * low * self._n / self._sample_rate)
        low_pass_h = 2.0 * high * torch.sinc(2 * high * self._n / self._sample_rate)

        band_pass = low_pass_h - low_pass_l
        band_pass = band_pass / (2.0 * band.unsqueeze(1))
        band_pass = band_pass * self._window

        filters = torch.cat(
            [band_pass, torch.flip(band_pass, dims=[1])], dim=1
        ).view(self._num_filters, 1, self._kernel_size)

        return F.conv1d(
            waveform,
            filters,
            stride=1,
            padding=self._kernel_size // 2,
            groups=1,
        )


# ---------------------------------------------------------------------------
# ResBlock — Residual block with batch norm and leaky ReLU
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block operating over the channel/time representation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.maxpool = nn.MaxPool1d(kernel_size=3)

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.activation(out + residual[:, :, : out.shape[2]])
        return self.maxpool(out)


# ---------------------------------------------------------------------------
# RawNet3 — End-to-end anti-spoofing model
# ---------------------------------------------------------------------------


class RawNet3Model(nn.Module):
    """
    RawNet3 anti-spoofing model.
    Input:  [batch, 1, time_samples] — raw 16kHz PCM waveform
    Output: [batch, 2]               — [genuine_logit, spoof_logit]
    """

    def __init__(
        self,
        sinc_filters: int = 128,
        sinc_kernel: int = 1024,
        res_channels: list[int] | None = None,
        gru_hidden: int = 1024,
        gru_layers: int = 3,
        num_classes: int = 2,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        res_channels = res_channels or [128, 256, 256, 256, 256, 256]

        self.sinc_conv = SincConv(
            num_filters=sinc_filters,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
        )
        self.bn_sinc = nn.BatchNorm1d(sinc_filters)
        self.ln_sinc = nn.LeakyReLU(0.3)

        blocks = []
        in_ch = sinc_filters
        for out_ch in res_channels:
            blocks.append(ResBlock(in_ch, out_ch))
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)

        self.gru = nn.GRU(
            input_size=res_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(gru_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x = self.ln_sinc(self.bn_sinc(self.sinc_conv(x)))  # [B, sinc_filters, T']
        x = self.res_blocks(x)                              # [B, res_channels[-1], T'']
        x = x.permute(0, 2, 1)                              # [B, T'', C] for GRU
        _, h_n = self.gru(x)                                # h_n: [layers, B, gru_hidden]
        x = self.fc(h_n[-1])                                # [B, 2]
        return x


# ---------------------------------------------------------------------------
# Audio Preprocessor for RawNet3 (minimal — raw waveform)
# ---------------------------------------------------------------------------


class RawAudioPreprocessor:
    """
    Minimal preprocessing for RawNet3: decode PCM bytes to float waveform tensor.
    RawNet3 consumes raw waveforms; no mel features required.
    """

    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0) -> None:
        self._sample_rate = sample_rate
        # RawNet3 typically uses fixed-length segments (4s = 64,000 samples at 16kHz)
        self._max_samples = int(sample_rate * max_length_seconds)

    def process(self, audio_bytes: bytes, device: torch.device) -> torch.Tensor:
        """
        Returns: [1, 1, max_samples] — batched, mono, truncated/padded waveform.
        """
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0

        # Truncate or zero-pad to fixed length
        if len(audio_np) > self._max_samples:
            audio_np = audio_np[: self._max_samples]
        else:
            pad = self._max_samples - len(audio_np)
            audio_np = np.pad(audio_np, (0, pad), mode="constant")

        waveform = torch.from_numpy(audio_np).to(device)
        return waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, T]


# ---------------------------------------------------------------------------
# Public Inference Wrapper
# ---------------------------------------------------------------------------


class RawNet3Inference:
    """
    Manages RawNet3 lifecycle: model loading, warm-up, async anti-spoofing inference.
    Returns a spoof probability scalar for each audio segment.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._device_manager = get_device_manager()
        self._device = self._device_manager.device
        self._model: Optional[RawNet3Model] = None
        self._preprocessor = RawAudioPreprocessor(
            sample_rate=self._settings.AUDIO_SAMPLE_RATE
        )
        self._model_version = "rawnet3-v1"

    def load(self) -> None:
        """Load RawNet3 weights. Called once at service startup."""
        model_path = Path(self._settings.RAWNET3_MODEL_PATH)
        self._model = RawNet3Model(sample_rate=self._settings.AUDIO_SAMPLE_RATE)

        if model_path.exists():
            logger.info("Loading RawNet3 weights from disk.", extra={"path": str(model_path)})
            state_dict = torch.load(model_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
        else:
            logger.warning(
                "RawNet3 weights not found — running with untrained weights. "
                "Spoof probability output will be random. Dev/test only.",
                extra={"path": str(model_path)},
            )

        self._model.to(self._device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

        self._warmup()
        logger.info(
            "RawNet3 ready.",
            extra={"device": self._device_manager.device_name, "version": self._model_version},
        )

    def _warmup(self) -> None:
        """Single forward pass to prime CUDA kernels."""
        dummy = torch.randn(1, 1, 64000, device=self._device)
        with torch.no_grad():
            _ = self._model(dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        logger.debug("RawNet3 warm-up pass complete.")

    async def predict(self, audio_bytes: bytes) -> Tuple[float, float, float]:
        """
        Async spoof detection inference.

        Returns:
            (spoof_probability, genuine_probability, latency_ms)
        """
        if self._model is None:
            raise RuntimeError("RawNet3Inference.load() must be called before inference.")

        loop = asyncio.get_event_loop()
        async with self._device_manager.semaphore:
            start = time.perf_counter()
            spoof_prob, genuine_prob = await loop.run_in_executor(
                None, self._sync_predict, audio_bytes
            )
            latency_ms = (time.perf_counter() - start) * 1000

        return spoof_prob, genuine_prob, latency_ms

    def _sync_predict(self, audio_bytes: bytes) -> Tuple[float, float]:
        """Synchronous RawNet3 forward pass in thread executor."""
        waveform = self._preprocessor.process(audio_bytes, self._device)

        with torch.no_grad():
            logits = self._model(waveform)  # [1, 2]: [genuine, spoof]

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        genuine_prob = float(probs[0])
        spoof_prob = float(probs[1])
        return spoof_prob, genuine_prob

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Module-level singleton
_rawnet3_instance: Optional[RawNet3Inference] = None


def get_rawnet3_inference() -> RawNet3Inference:
    global _rawnet3_instance
    if _rawnet3_instance is None:
        _rawnet3_instance = RawNet3Inference()
    return _rawnet3_instance
