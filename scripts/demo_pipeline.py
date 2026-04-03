"""End-to-end demo: generate data → train model → run inference → detect anomalies.

This script demonstrates the full PFIL pipeline without external dependencies
(no Kafka, no database, no Qdrant). Everything runs in-memory.

Usage:
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --epochs 10 --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from construction_vl_jepa.config import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
from construction_vl_jepa.data.dataset import SensorWindowDataset
from construction_vl_jepa.inference.anomaly_detector import AnomalyDetector
from construction_vl_jepa.inference.pattern_matcher import PatternMatcher
from construction_vl_jepa.model.jepa import VLJEPA


def main():
    parser = argparse.ArgumentParser(description="PFIL End-to-End Demo")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use smaller model for quick demo")
    args = parser.parse_args()

    print("=" * 70)
    print("Pre-Failure Intelligence Layer — End-to-End Demo")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Generate synthetic data
    # -----------------------------------------------------------------------
    print("\n[1/5] Generating synthetic sensor data...")
    from scripts.generate_synthetic_data import generate_dataset

    df, failures = generate_dataset(n_days=7, sample_rate_hz=0.1, n_sensors=8, n_failures=2)
    sensor_columns = [c for c in df.columns if c != "timestamp"]
    print(f"  Generated {len(df):,} samples, {len(sensor_columns)} sensors, {len(failures)} failures")

    # -----------------------------------------------------------------------
    # 2. Configure model (small for demo)
    # -----------------------------------------------------------------------
    print("\n[2/5] Configuring model...")
    if args.quick:
        model_cfg = ModelConfig(
            latent_dim=64, context_window=600, target_window=200,
            predictor_depth=2, encoder_depth=4, n_heads=4, dropout=0.1,
        )
    else:
        model_cfg = ModelConfig(
            latent_dim=128, context_window=1200, target_window=400,
            predictor_depth=4, encoder_depth=6, n_heads=8, dropout=0.1,
        )

    data_cfg = DataConfig(sensor_patch_size=60, max_sensors=8)
    training_cfg = TrainingConfig(
        batch_size=16, learning_rate=3e-4, total_epochs=args.epochs,
        warmup_epochs=1, ema_decay_start=0.996, ema_decay_end=0.999,
    )

    sample_rate = 0.1
    print(f"  Latent dim: {model_cfg.latent_dim}, Encoder depth: {model_cfg.encoder_depth}")

    # -----------------------------------------------------------------------
    # 3. Build dataset and train
    # -----------------------------------------------------------------------
    print("\n[3/5] Training JEPA model...")

    # Create failure mask
    import pandas as pd

    failure_timestamps = [f["timestamp"] for f in failures]
    failure_mask = np.zeros(len(df), dtype=bool)
    for ts in failure_timestamps:
        ft = pd.Timestamp(ts)
        mask = (
            (df["timestamp"] >= ft - pd.Timedelta(hours=data_cfg.failure_exclusion_hours))
            & (df["timestamp"] <= ft + pd.Timedelta(hours=data_cfg.failure_exclusion_hours))
        )
        failure_mask |= mask.values

    dataset = SensorWindowDataset(
        data=df, sensor_ids=sensor_columns,
        model_cfg=model_cfg, data_cfg=data_cfg,
        sample_rate_hz=sample_rate, failure_mask=failure_mask,
    )
    print(f"  Dataset: {len(dataset)} valid windows")

    if len(dataset) == 0:
        print("  ERROR: No valid training windows. Increase data or reduce window sizes.")
        return

    loader = DataLoader(dataset, batch_size=training_cfg.batch_size, shuffle=True)

    model = VLJEPA(model_cfg, n_sensors=data_cfg.max_sensors, sample_rate_hz=sample_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_cfg.learning_rate, weight_decay=0.05
    )

    model.train()
    for epoch in range(training_cfg.total_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            result = model(batch["context"], batch["target"])
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # EMA update
            progress = epoch / max(training_cfg.total_epochs, 1)
            ema_decay = training_cfg.ema_decay_end - (
                training_cfg.ema_decay_end - training_cfg.ema_decay_start
            ) * (1 - progress)
            model.update_target_encoder(ema_decay)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{training_cfg.total_epochs} — loss: {avg_loss:.6f}")

    # -----------------------------------------------------------------------
    # 4. Run inference on test windows
    # -----------------------------------------------------------------------
    print("\n[4/5] Running inference...")
    inference_cfg = InferenceConfig(threshold_k=3.0)
    detector = AnomalyDetector(model, inference_cfg)
    pattern_matcher = PatternMatcher(inference_cfg)

    # Create a dataset that includes failure windows for testing
    test_dataset = SensorWindowDataset(
        data=df, sensor_ids=sensor_columns,
        model_cfg=model_cfg, data_cfg=data_cfg,
        sample_rate_hz=sample_rate, failure_mask=None,  # include everything
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    scores = []
    anomalies_found = 0

    for i, batch in enumerate(test_loader):
        result = detector.score_window(
            machine_id="CNC-042",
            context=batch["context"],
            target=batch["target"],
            sensor_names=sensor_columns,
        )
        scores.append(result.score)

        if result.is_anomaly:
            anomalies_found += 1

        # Store embedding for pattern matching
        with torch.no_grad():
            latent = model.encode(batch["context"])
            cls_emb = latent[0, 0].numpy()
        from datetime import datetime
        pattern_matcher.store_embedding(
            embedding=cls_emb,
            machine_id="CNC-042",
            window_start=datetime.now(),
            window_end=datetime.now(),
            metadata={"anomaly_score": result.score, "is_anomaly": result.is_anomaly},
        )

        if i >= 200:  # limit for demo
            break

    print(f"  Scored {len(scores)} windows")
    print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
    print(f"  Mean score: {np.mean(scores):.6f}")
    print(f"  Anomalies detected: {anomalies_found}")

    # -----------------------------------------------------------------------
    # 5. Pattern matching demo
    # -----------------------------------------------------------------------
    print("\n[5/5] Pattern matching demo...")
    if len(scores) > 10:
        # Query with the highest-scoring window's embedding
        highest_idx = np.argmax(scores)
        # Re-encode that window
        test_batch = test_dataset[highest_idx]
        with torch.no_grad():
            query_latent = model.encode(test_batch["context"].unsqueeze(0))
            query_emb = query_latent[0, 0].numpy()

        matches = pattern_matcher.find_similar(query_emb, top_k=3, score_threshold=0.5)
        print(f"  Found {len(matches)} similar historical states for highest-scoring window")
        for m in matches:
            print(f"    - similarity: {m.similarity:.3f}, score: {m.metadata.get('anomaly_score', 'N/A')}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Stored {len(pattern_matcher._memory_store)} latent states for pattern matching")
    print("=" * 70)


if __name__ == "__main__":
    main()
