"""Training entry point for the VL-JEPA model.

Usage:
    pfil-train                           # uses default config
    pfil-train --config configs/custom.yaml
    pfil-train --config configs/default.yaml --data-path data/sensors.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split

from construction_vl_jepa.config import load_config
from construction_vl_jepa.data.dataset import SensorWindowDataset, load_csv_dataset
from construction_vl_jepa.training.lightning_module import JEPALightningModule
from construction_vl_jepa.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Train VL-JEPA model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data-path", type=str, required=True, help="Path to sensor CSV")
    parser.add_argument(
        "--sensor-columns", type=str, nargs="+", required=True, help="Sensor column names"
    )
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Samples per second")
    parser.add_argument("--failure-timestamps", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs (None=auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("config_loaded", config_path=args.config)

    # Load dataset
    dataset = load_csv_dataset(
        csv_path=args.data_path,
        sensor_columns=args.sensor_columns,
        timestamp_column=args.timestamp_column,
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        sample_rate_hz=args.sample_rate,
        failure_timestamps=args.failure_timestamps,
    )

    log.info("dataset_loaded", n_samples=len(dataset), n_sensors=dataset.n_sensors)

    # Train/val split
    val_size = int(len(dataset) * cfg.data.val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    lightning_module = JEPALightningModule(
        model_cfg=cfg.model,
        training_cfg=cfg.training,
        n_sensors=cfg.data.max_sensors,
        sample_rate_hz=args.sample_rate,
    )

    # Callbacks
    output_dir = Path(args.output_dir)
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="jepa-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=20,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Loggers
    loggers = [
        TensorBoardLogger(save_dir=output_dir / "tb_logs", name="jepa"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.total_epochs,
        accelerator="auto",
        devices=args.gpus or "auto",
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=False,
    )

    log.info("training_start", total_epochs=cfg.training.total_epochs)
    trainer.fit(lightning_module, train_loader, val_loader)
    log.info("training_complete", best_model=trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
