"""Tests for the sensor window dataset."""

import numpy as np
import pytest

from construction_vl_jepa.config import DataConfig, ModelConfig
from construction_vl_jepa.data.dataset import SensorWindowDataset


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_steps = 5000
    n_sensors = 4
    return np.random.randn(n_steps, n_sensors).astype(np.float32)


@pytest.fixture
def small_cfg():
    return (
        ModelConfig(context_window=120, target_window=60),
        DataConfig(sensor_patch_size=60, max_sensors=4),
    )


class TestSensorWindowDataset:
    def test_basic_creation(self, sample_data, small_cfg):
        model_cfg, data_cfg = small_cfg
        ds = SensorWindowDataset(
            data=sample_data,
            sensor_ids=[f"s{i}" for i in range(4)],
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            sample_rate_hz=1.0,
        )
        assert len(ds) > 0

    def test_item_shapes(self, sample_data, small_cfg):
        model_cfg, data_cfg = small_cfg
        ds = SensorWindowDataset(
            data=sample_data,
            sensor_ids=[f"s{i}" for i in range(4)],
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            sample_rate_hz=1.0,
        )
        item = ds[0]
        # context: 120 steps / 60 patch_size = 2 patches
        assert item["context"].shape == (2, 60, 4)
        # target: 60 steps / 60 patch_size = 1 patch
        assert item["target"].shape == (1, 60, 4)

    def test_failure_mask_excludes_windows(self, sample_data, small_cfg):
        model_cfg, data_cfg = small_cfg
        # Mask out a large chunk
        mask = np.zeros(len(sample_data), dtype=bool)
        mask[1000:4000] = True

        ds_masked = SensorWindowDataset(
            data=sample_data,
            sensor_ids=[f"s{i}" for i in range(4)],
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            sample_rate_hz=1.0,
            failure_mask=mask,
        )
        ds_unmasked = SensorWindowDataset(
            data=sample_data,
            sensor_ids=[f"s{i}" for i in range(4)],
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            sample_rate_hz=1.0,
        )
        assert len(ds_masked) < len(ds_unmasked)

    def test_sensor_padding(self, small_cfg):
        model_cfg, data_cfg = small_cfg
        # Data with only 2 sensors, max_sensors=4
        data = np.random.randn(5000, 2).astype(np.float32)
        ds = SensorWindowDataset(
            data=data,
            sensor_ids=["s0", "s1"],
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            sample_rate_hz=1.0,
        )
        item = ds[0]
        assert item["context"].shape[-1] == 4  # padded to max_sensors

    def test_normalization(self, sample_data, small_cfg):
        model_cfg, data_cfg = small_cfg
        ds = SensorWindowDataset(
            data=sample_data,
            sensor_ids=[f"s{i}" for i in range(4)],
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            sample_rate_hz=1.0,
        )
        # Normalized data should be roughly zero-mean
        assert abs(ds.means.mean()) < 1.0  # rough check
