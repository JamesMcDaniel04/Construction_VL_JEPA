"""Tests for the anomaly detection engine."""

import torch
import pytest

from construction_vl_jepa.config import InferenceConfig, ModelConfig
from construction_vl_jepa.inference.anomaly_detector import AnomalyDetector, MachineBaseline
from construction_vl_jepa.model.jepa import VLJEPA


@pytest.fixture
def detector():
    model_cfg = ModelConfig(
        latent_dim=32, context_window=120, target_window=60,
        predictor_depth=2, encoder_depth=2, n_heads=4,
    )
    model = VLJEPA(model_cfg, n_sensors=4, sample_rate_hz=1.0)
    inference_cfg = InferenceConfig(threshold_k=3.0, trend_window=10)
    return AnomalyDetector(model, inference_cfg)


class TestMachineBaseline:
    def test_update_computes_stats(self):
        baseline = MachineBaseline()
        for i in range(100):
            baseline.update(float(i))
        assert baseline.mean > 0
        assert baseline.std > 0

    def test_initial_state(self):
        baseline = MachineBaseline()
        assert baseline.mean == 0.0
        assert baseline.std == 1.0


class TestAnomalyDetector:
    def test_score_window_returns_result(self, detector):
        context = torch.randn(1, 2, 60, 4)
        target = torch.randn(1, 1, 60, 4)
        result = detector.score_window("machine-1", context, target)
        assert result.machine_id == "machine-1"
        assert result.score >= 0
        assert isinstance(result.is_anomaly, bool)

    def test_baseline_builds_over_time(self, detector):
        for i in range(50):
            context = torch.randn(1, 2, 60, 4)
            target = torch.randn(1, 1, 60, 4)
            detector.score_window("machine-1", context, target)

        baseline = detector.baselines["machine-1"]
        assert len(baseline.scores) == 50
        assert baseline.std > 0

    def test_health_status(self, detector):
        # Score some normal windows to build baseline
        for _ in range(20):
            context = torch.randn(1, 2, 60, 4)
            target = torch.randn(1, 1, 60, 4)
            detector.score_window("machine-1", context, target)

        health = detector.get_machine_health("machine-1")
        assert health["status"] in ("green", "yellow", "red")
        assert health["trend"] in ("stable", "rising")

    def test_unknown_machine_health(self, detector):
        health = detector.get_machine_health("unknown-machine")
        assert health["status"] == "unknown"
