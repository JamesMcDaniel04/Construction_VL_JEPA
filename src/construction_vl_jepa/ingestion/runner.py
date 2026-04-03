"""Ingestion pipeline runner — connects Kafka consumer to stream processor to inference.

This is the main real-time pipeline loop:
  Kafka → Stream Processor → Inference Server → Anomaly DB + Alert Router
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from construction_vl_jepa.config import load_config
from construction_vl_jepa.ingestion.kafka_consumer import SensorKafkaConsumer
from construction_vl_jepa.ingestion.stream_processor import StreamProcessor
from construction_vl_jepa.inference.server import InferenceServer, load_model_from_checkpoint
from construction_vl_jepa.utils.logging import get_logger, setup_logging

log = get_logger(__name__)

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False
    log.info("shutdown_requested")


def main():
    setup_logging()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    parser = argparse.ArgumentParser(description="PFIL Ingestion Pipeline")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Initialize components
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    inference_server = InferenceServer(model, config_path=args.config, device=args.device)

    stream_processor = StreamProcessor(
        model_cfg=cfg.model,
        data_cfg=cfg.data,
        sample_rate_hz=args.sample_rate,
    )

    consumer = SensorKafkaConsumer(cfg.kafka)
    consumer.connect()

    log.info("pipeline_started")

    try:
        while _running:
            # Consume batch from Kafka
            readings = consumer.consume_batch(max_messages=500, timeout=1.0)

            if not readings:
                continue

            # Process through stream processor
            ready_windows = stream_processor.process_readings(readings)

            # Run inference on ready windows
            for machine_id, context_patches, target_patches in ready_windows:
                try:
                    result = inference_server.process_window(
                        machine_id=machine_id,
                        context=context_patches,
                        target=target_patches,
                    )

                    if result["is_anomaly"]:
                        log.warning(
                            "anomaly_detected",
                            machine_id=machine_id,
                            score=result["anomaly_score"],
                            normalized=result["normalized_score"],
                            n_matches=len(result["pattern_matches"]),
                        )
                except Exception as e:
                    log.error("inference_error", machine_id=machine_id, error=str(e))

    finally:
        consumer.close()
        log.info("pipeline_stopped")


if __name__ == "__main__":
    main()
