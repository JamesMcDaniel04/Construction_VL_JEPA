"""PyTorch Lightning module wrapping the VL-JEPA model.

Handles training loop, EMA scheduling, learning rate warmup, and logging.
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from construction_vl_jepa.config import ModelConfig, TrainingConfig
from construction_vl_jepa.model.jepa import VLJEPA


class JEPALightningModule(pl.LightningModule):
    """Lightning wrapper for JEPA training."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        n_sensors: int,
        sample_rate_hz: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg

        self.model = VLJEPA(model_cfg, n_sensors=n_sensors, sample_rate_hz=sample_rate_hz)

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> dict:
        return self.model(context, target)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        result = self.model(batch["context"], batch["target"])
        loss = result["loss"]

        # Update target encoder EMA
        ema_decay = self._get_ema_decay()
        self.model.update_target_encoder(ema_decay)

        # Logging
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/prediction_error_mean", result["prediction_error"].mean())
        self.log("train/ema_decay", ema_decay)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        result = self.model(batch["context"], batch["target"])
        self.log("val/loss", result["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/prediction_error_mean", result["prediction_error"].mean(), on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
        )

        # Warmup + cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.training_cfg.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_cfg.total_epochs - self.training_cfg.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.training_cfg.warmup_epochs],
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _get_ema_decay(self) -> float:
        """Anneal EMA decay from start to end over training."""
        progress = self.current_epoch / max(self.training_cfg.total_epochs, 1)
        return (
            self.training_cfg.ema_decay_end
            - (self.training_cfg.ema_decay_end - self.training_cfg.ema_decay_start)
            * (1 - progress)
        )
