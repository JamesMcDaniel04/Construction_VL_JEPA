"""Inference server that loads a trained JEPA model and scores incoming data.

Can run as a standalone service or be embedded in the streaming pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from construction_vl_jepa.config import load_config
from construction_vl_jepa.inference.anomaly_detector import AnomalyDetector
from construction_vl_jepa.inference.pattern_matcher import PatternMatcher
from construction_vl_jepa.model.jepa import VLJEPA
from construction_vl_jepa.training.lightning_module import JEPALightningModule
from construction_vl_jepa.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> VLJEPA:
    """Load a trained JEPA model from a Lightning checkpoint."""
    lightning_module = JEPALightningModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    return lightning_module.model


class InferenceServer:
    """Inference server wrapping the anomaly detector and pattern matcher."""

    def __init__(
        self,
        model: VLJEPA,
        config_path: str | None = None,
        device: str = "cpu",
    ):
        self.cfg = load_config(config_path)
        self.device = device

        self.detector = AnomalyDetector(model, self.cfg.inference, device=device)
        self.pattern_matcher = PatternMatcher(
            self.cfg.inference,
            qdrant_url=self.cfg.database.qdrant_url,
            collection=self.cfg.database.qdrant_collection,
        )

        log.info("inference_server_ready", device=device)

    def process_window(
        self,
        machine_id: str,
        context: np.ndarray,
        target: np.ndarray,
        sensor_names: list[str] | None = None,
    ) -> dict:
        """Process a single context→target window and return full analysis.

        Args:
            machine_id: Machine identifier.
            context: Context window array (n_ctx_patches, patch_size, n_sensors).
            target: Target window array (n_tgt_patches, patch_size, n_sensors).
            sensor_names: Optional sensor names for contributing signal analysis.

        Returns:
            Dict with anomaly score, pattern matches, health status, and details.
        """
        # Convert to tensors and add batch dimension
        ctx_tensor = torch.from_numpy(context).float().unsqueeze(0)
        tgt_tensor = torch.from_numpy(target).float().unsqueeze(0)

        # Score the window
        anomaly_result = self.detector.score_window(
            machine_id, ctx_tensor, tgt_tensor, sensor_names
        )

        # Get the CLS embedding for pattern matching
        with torch.no_grad():
            ctx_on_device = ctx_tensor.to(self.device)
            latent = self.detector.model.encode(ctx_on_device)
            cls_embedding = latent[0, 0].cpu().numpy()

        # Pattern matching
        matches = []
        if anomaly_result.is_anomaly:
            matches = self.pattern_matcher.find_similar(cls_embedding)

        # Store embedding for future pattern matching
        from datetime import datetime

        self.pattern_matcher.store_embedding(
            embedding=cls_embedding,
            machine_id=machine_id,
            window_start=datetime.now(),
            window_end=datetime.now(),
            metadata={
                "anomaly_score": anomaly_result.score,
                "is_anomaly": anomaly_result.is_anomaly,
            },
        )

        # Health status
        health = self.detector.get_machine_health(machine_id)

        return {
            "machine_id": machine_id,
            "anomaly_score": anomaly_result.score,
            "normalized_score": anomaly_result.normalized_score,
            "threshold": anomaly_result.threshold,
            "is_anomaly": anomaly_result.is_anomaly,
            "is_trend_anomaly": anomaly_result.is_trend_anomaly,
            "contributing_signals": anomaly_result.contributing_signals,
            "pattern_matches": [
                {
                    "pattern_id": m.pattern_id,
                    "similarity": m.similarity,
                    "machine_id": m.machine_id,
                    "event_type": m.event_type,
                    "lead_time_hours": m.lead_time_hours,
                }
                for m in matches
            ],
            "health": health,
        }


def main():
    """CLI entry point for starting the inference server."""
    setup_logging()

    parser = argparse.ArgumentParser(description="PFIL Inference Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    server = InferenceServer(model, config_path=args.config, device=args.device)

    log.info("inference_server_started", checkpoint=args.checkpoint)
    # In production, this would be connected to Kafka/Bytewax stream
    # For now, just confirms the server is ready
    print(f"Inference server ready. Model loaded from {args.checkpoint}")


if __name__ == "__main__":
    main()
