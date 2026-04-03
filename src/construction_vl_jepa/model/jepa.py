"""VL-JEPA model: Joint Embedding Predictive Architecture for industrial time-series.

Core architecture:
  - Context Encoder (x-encoder): encodes the context window → s_x
  - Target Encoder (y-encoder): EMA copy of context encoder, encodes target window → s_y
  - Predictor: takes s_x and predicts ŝ_y
  - Loss: L2(ŝ_y, s_y) in latent space (no gradient through target encoder)

The target encoder is updated via exponential moving average (EMA) of the context
encoder, following the JEPA / BYOL / DINO lineage. This prevents representation collapse.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from construction_vl_jepa.config import ModelConfig
from construction_vl_jepa.model.encoders import SensorPatchEncoder
from construction_vl_jepa.model.predictor import JEPAPredictor


class VLJEPA(nn.Module):
    """Full VL-JEPA model for sensor time-series anomaly detection.

    During training: forward() returns the JEPA loss.
    During inference: encode() + predict() to compute anomaly scores.
    """

    def __init__(self, cfg: ModelConfig, n_sensors: int, sample_rate_hz: float = 1.0):
        super().__init__()
        self.cfg = cfg

        patch_size = 60  # sensor_patch_size default, in samples
        n_context_patches = int(cfg.context_window * sample_rate_hz) // patch_size
        n_target_patches = int(cfg.target_window * sample_rate_hz) // patch_size

        # Context encoder (gradient-trained)
        self.context_encoder = SensorPatchEncoder(
            patch_size=patch_size,
            n_sensors=n_sensors,
            latent_dim=cfg.latent_dim,
            depth=cfg.encoder_depth,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )

        # Target encoder (EMA of context encoder — no gradient)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor network
        self.predictor = JEPAPredictor(
            latent_dim=cfg.latent_dim,
            depth=cfg.predictor_depth,
            n_heads=cfg.n_heads,
            n_target_tokens=n_target_patches + 1,  # +1 for CLS
            dropout=cfg.dropout,
        )

        # Masking ratio for JEPA-style training
        self.masking_ratio = cfg.masking_ratio

    def forward(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            context: (batch, n_ctx_patches, patch_size, n_sensors)
            target:  (batch, n_tgt_patches, patch_size, n_sensors)

        Returns:
            Dict with 'loss', 'prediction_error' (detached, for monitoring).
        """
        # Encode context (with gradient)
        context_latent = self.context_encoder(context)  # (B, N_ctx+1, D)

        # Encode target (no gradient — EMA encoder)
        with torch.no_grad():
            target_latent = self.target_encoder(target)  # (B, N_tgt+1, D)

        # Predict target latent from context latent
        predicted_target = self.predictor(context_latent)  # (B, N_tgt+1, D)

        # L2 loss in latent space
        loss = nn.functional.mse_loss(predicted_target, target_latent)

        # Per-sample prediction error for monitoring
        with torch.no_grad():
            per_sample_error = (
                (predicted_target - target_latent).pow(2).mean(dim=(1, 2))
            )

        return {
            "loss": loss,
            "prediction_error": per_sample_error,
        }

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sensor window into latent space (inference).

        Args:
            x: (batch, n_patches, patch_size, n_sensors)

        Returns:
            Latent representation (batch, n_tokens, latent_dim).
        """
        return self.context_encoder(x)

    @torch.no_grad()
    def predict_and_score(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Inference: compute anomaly score as prediction error.

        Args:
            context: (batch, n_ctx_patches, patch_size, n_sensors)
            target:  (batch, n_tgt_patches, patch_size, n_sensors)

        Returns:
            Dict with 'anomaly_score' (per-sample L2 error) and 'context_latent'.
        """
        context_latent = self.context_encoder(context)
        target_latent = self.target_encoder(target)
        predicted_target = self.predictor(context_latent)

        anomaly_score = (predicted_target - target_latent).pow(2).mean(dim=(1, 2))

        # Also compute per-token errors for "contributing signals" analysis
        per_token_error = (predicted_target - target_latent).pow(2).mean(dim=-1)

        # CLS token embedding for vector DB storage
        cls_embedding = context_latent[:, 0, :]

        return {
            "anomaly_score": anomaly_score,
            "per_token_error": per_token_error,
            "cls_embedding": cls_embedding,
            "predicted_latent": predicted_target,
            "actual_latent": target_latent,
        }

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float) -> None:
        """Update target encoder via exponential moving average of context encoder.

        Called after each training step. τ typically anneals from 0.996 → 0.999.
        """
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_tgt.data.mul_(ema_decay).add_(param_ctx.data, alpha=1 - ema_decay)
