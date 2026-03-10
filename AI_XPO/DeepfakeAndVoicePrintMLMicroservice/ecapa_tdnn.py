"""
SentinelAI ML Service — ECAPA-TDNN Voiceprint Inference Wrapper

ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
Time Delay Neural Network) extracts 192-dim speaker embeddings from raw audio.

Production Architecture:
- Model weights loaded from disk at startup and pinned to device.
- Audio preprocessed to 16kHz mono, normalized, and windowed.
- Inference executed under the DeviceManager semaphore.
- Embeddings are L2-normalized before Pinecone cosine search.

NOTE: The model architecture below is a structural placeholder that
mirrors ECAPA-TDNN's actual layer topology. In production, replace
the weight-loading block with SpeechBrain's pretrained ECAPA-TDNN
or load a fine-tuned checkpoint from your S3 model store.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.config import get_settings
from app.core.logging import PerformanceLogger, get_logger
from app.inference.device_manager import get_device_manager

logger = get_logger("inference.ecapa_tdnn")


# ---------------------------------------------------------------------------
# ECAPA-TDNN Architecture (structural — matches real topology for hot-swapping)
# ---------------------------------------------------------------------------


class SEModule(nn.Module):
    """Squeeze-and-Excitation block used in ECAPA-TDNN channels."""

    def __init__(self, channels: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(2)


class Res2NetBlock(nn.Module):
    """Res2Net block with scale-wise feature hierarchy."""

    def __init__(self, in_channels: int, out_channels: int, scale: int = 8,
                 dilation: int = 1) -> None:
        super().__init__()
        self.scale = scale
        width = out_channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=3, dilation=dilation,
                      padding=dilation)
            for _ in range(scale - 1)
        ])
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEModule(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pointwise(x)
        splits = torch.chunk(x, self.scale, dim=1)
        out = [splits[0]]
        y = splits[0]
        for i, conv in enumerate(self.convs):
            y = conv(y + splits[i + 1]) if i > 0 else conv(splits[i + 1])
            out.append(y)
        x = torch.cat(out, dim=1)
        x = self.bn(x)
        x = self.se(x)
        return x + residual if residual.shape == x.shape else x


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling — computes weighted mean and std
    over the time dimension using a learned attention mechanism.
    Aggregates frame-level features into a single utterance embedding.
    """

    def __init__(self, channels: int, attention_dim: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        mean = x.mean(dim=2, keepdim=True).expand_as(x)
        std = x.std(dim=2, keepdim=True).expand_as(x)
        attn_input = torch.cat([x, mean, std], dim=1)
        attn_weights = self.attention(attn_input)  # [B, C, T]
        weighted_mean = (attn_weights * x).sum(dim=2)
        weighted_std = (attn_weights * (x - weighted_mean.unsqueeze(2)) ** 2).sum(dim=2).sqrt()
        return torch.cat([weighted_mean, weighted_std], dim=1)  # [B, 2C]


class ECAPATDNNModel(nn.Module):
    """
    ECAPA-TDNN speaker encoder.
    Input:  [batch, 1, time_samples]  — 16kHz mono PCM
    Output: [batch, 192]              — L2-normalized speaker embedding
    """

    def __init__(
        self,
        input_dim: int = 80,          # Mel filterbank features
        channels: int = 512,
        embedding_dim: int = 192,
        scale: int = 8,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, channels, kernel_size=5, padding=2)
        self.bn0 = nn.BatchNorm1d(channels)
        self.blocks = nn.ModuleList([
            Res2NetBlock(channels, channels, scale=scale, dilation=2),
            Res2NetBlock(channels, channels, scale=scale, dilation=3),
            Res2NetBlock(channels, channels, scale=scale, dilation=4),
        ])
        self.mfa = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.bn_mfa = nn.BatchNorm1d(channels * 3)
        self.pooling = AttentiveStatisticsPooling(channels * 3)
        self.fc = nn.Linear(channels * 6, embedding_dim)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, mel_bins, T]
        x = F.relu(self.bn0(self.input_proj(x)))
        frame_feats = []
        for block in self.blocks:
            x = block(x)
            frame_feats.append(x)
        x = torch.cat(frame_feats, dim=1)
        x = F.relu(self.bn_mfa(self.mfa(x)))
        x = self.pooling(x)
        x = self.bn_out(self.fc(x))
        return F.normalize(x, p=2, dim=1)  # L2-normalize


# ---------------------------------------------------------------------------
# Audio Preprocessor
# ---------------------------------------------------------------------------


class AudioPreprocessor:
    """
    Converts raw PCM audio bytes → mel-filterbank feature tensor.
    Uses torchaudio for GPU-compatible, differentiable transforms.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80) -> None:
        try:
            import torchaudio
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                hop_length=160,
                n_mels=n_mels,
                f_min=20.0,
                f_max=7600.0,
            )
            self._amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=80
            )
            self._torchaudio = torchaudio
        except ImportError:
            logger.warning(
                "torchaudio not available; falling back to random feature simulation."
            )
            self._mel_transform = None
            self._amplitude_to_db = None
            self._torchaudio = None

        self._sample_rate = sample_rate
        self._n_mels = n_mels

    def process(self, audio_bytes: bytes, device: torch.device) -> torch.Tensor:
        """
        Process raw audio bytes into a mel-spectrogram tensor.
        Returns: [1, n_mels, time_frames]
        """
        if self._torchaudio is None:
            # Graceful degradation — generate plausible-dim random features
            # for integration testing without torchaudio installed
            n_frames = max(100, len(audio_bytes) // 320)
            return torch.randn(1, self._n_mels, n_frames, device=device)

        # Convert PCM bytes to float32 waveform tensor
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0  # Normalize to [-1.0, 1.0]
        waveform = torch.from_numpy(audio_np).unsqueeze(0).to(device)  # [1, T]

        mel = self._mel_transform.to(device)(waveform)   # [1, n_mels, T]
        mel_db = self._amplitude_to_db.to(device)(mel)   # [1, n_mels, T]

        # Per-utterance CMVN (Cepstral Mean and Variance Normalization)
        mean = mel_db.mean(dim=2, keepdim=True)
        std = mel_db.std(dim=2, keepdim=True).clamp(min=1e-8)
        return (mel_db - mean) / std


# ---------------------------------------------------------------------------
# Public Inference Wrapper
# ---------------------------------------------------------------------------


class ECAPATDNNInference:
    """
    Manages the full lifecycle of ECAPA-TDNN inference:
    model loading, warm-up, and async embedding extraction.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._device_manager = get_device_manager()
        self._device = self._device_manager.device
        self._model: Optional[ECAPATDNNModel] = None
        self._preprocessor = AudioPreprocessor(
            sample_rate=self._settings.AUDIO_SAMPLE_RATE
        )
        self._model_version = "ecapa-tdnn-v1"

    def load(self) -> None:
        """
        Load model weights. Called once at service startup.
        In production, loads from the path in settings (S3-synced PVC mount).
        """
        model_path = Path(self._settings.ECAPA_TDNN_MODEL_PATH)
        self._model = ECAPATDNNModel(embedding_dim=self._settings.PINECONE_EMBEDDING_DIM)

        if model_path.exists():
            logger.info("Loading ECAPA-TDNN weights from disk.", extra={"path": str(model_path)})
            state_dict = torch.load(model_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
        else:
            logger.warning(
                "ECAPA-TDNN weights not found — running with random weights. "
                "This is only acceptable in development/testing.",
                extra={"path": str(model_path)},
            )

        self._model.to(self._device)
        self._model.eval()

        # Disable gradient computation globally for this model
        for param in self._model.parameters():
            param.requires_grad_(False)

        self._warmup()
        logger.info(
            "ECAPA-TDNN ready.",
            extra={"device": self._device_manager.device_name, "version": self._model_version},
        )

    def _warmup(self) -> None:
        """Single forward pass to initialize CUDA kernels and avoid cold-start latency."""
        dummy = torch.randn(1, 80, 200, device=self._device)
        with torch.no_grad():
            _ = self._model(dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        logger.debug("ECAPA-TDNN warm-up pass complete.")

    async def extract_embedding(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        """
        Asynchronously extract a 192-dim L2-normalized speaker embedding.

        Args:
            audio_bytes: Raw 16kHz mono PCM-16 audio bytes.

        Returns:
            (embedding_array, latency_ms): numpy float32 array of shape [192,]
        """
        if self._model is None:
            raise RuntimeError("ECAPATDNNInference.load() must be called before inference.")

        loop = asyncio.get_event_loop()
        async with self._device_manager.semaphore:
            start = time.perf_counter()
            embedding = await loop.run_in_executor(
                None, self._sync_extract, audio_bytes
            )
            latency_ms = (time.perf_counter() - start) * 1000

        return embedding, latency_ms

    def _sync_extract(self, audio_bytes: bytes) -> np.ndarray:
        """Synchronous inference — executed in thread pool executor."""
        mel_features = self._preprocessor.process(audio_bytes, self._device)
        mel_features = mel_features.unsqueeze(0)  # [1, 1, n_mels, T] → batch dim

        with torch.no_grad():
            # ECAPATDNNModel expects [B, n_mels, T]; preprocessor returns [1, n_mels, T]
            embedding = self._model(mel_features.squeeze(1))  # [1, 192]

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Module-level singleton
_ecapa_instance: Optional[ECAPATDNNInference] = None


def get_ecapa_inference() -> ECAPATDNNInference:
    global _ecapa_instance
    if _ecapa_instance is None:
        _ecapa_instance = ECAPATDNNInference()
    return _ecapa_instance
