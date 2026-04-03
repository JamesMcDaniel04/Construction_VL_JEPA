"""PyTorch datasets for sensor time-series windows.

Handles loading, normalization, and windowing of sensor data for JEPA training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from construction_vl_jepa.config import DataConfig, ModelConfig


class SensorWindowDataset(Dataset):
    """Dataset that yields (context_window, target_window) pairs for JEPA training.

    Each sample is a pair of sensor reading windows from "normal" operating periods.
    Context window: [t0, t0 + context_len)
    Target window:  [t0 + context_len, t0 + context_len + target_len)
    """

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        sensor_ids: list[str],
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        sample_rate_hz: float = 1.0,
        failure_mask: np.ndarray | None = None,
    ):
        """
        Args:
            data: Sensor readings array of shape (n_timesteps, n_sensors).
            sensor_ids: List of sensor column names.
            model_cfg: Model configuration.
            data_cfg: Data configuration.
            sample_rate_hz: Samples per second in the data.
            failure_mask: Boolean array where True = exclude (near failure).
        """
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.sample_rate_hz = sample_rate_hz

        if isinstance(data, pd.DataFrame):
            data = data[sensor_ids].values

        # Pad or truncate sensor count to max_sensors
        n_sensors = data.shape[1]
        if n_sensors > data_cfg.max_sensors:
            data = data[:, : data_cfg.max_sensors]
        elif n_sensors < data_cfg.max_sensors:
            pad = np.zeros((data.shape[0], data_cfg.max_sensors - n_sensors))
            data = np.concatenate([data, pad], axis=1)

        # Per-sensor z-score normalization
        if data_cfg.normalization == "per_sensor_zscore":
            self.means = np.nanmean(data, axis=0, keepdims=True)
            self.stds = np.nanstd(data, axis=0, keepdims=True)
            self.stds[self.stds < 1e-8] = 1.0  # avoid div by zero
            data = (data - self.means) / self.stds
        else:
            self.means = np.zeros((1, data.shape[1]))
            self.stds = np.ones((1, data.shape[1]))

        # Replace NaNs with 0 after normalization
        data = np.nan_to_num(data, nan=0.0)

        self.data = data.astype(np.float32)
        self.n_sensors = self.data.shape[1]

        # Window sizes in samples
        self.context_len = int(model_cfg.context_window * sample_rate_hz)
        self.target_len = int(model_cfg.target_window * sample_rate_hz)
        self.total_len = self.context_len + self.target_len

        # Build valid start indices (exclude failure windows)
        max_start = len(self.data) - self.total_len
        if max_start <= 0:
            self.valid_indices = np.array([], dtype=np.int64)
        else:
            all_indices = np.arange(max_start)
            if failure_mask is not None:
                # Exclude any window that overlaps with a failure-masked region
                valid = []
                for i in all_indices:
                    if not failure_mask[i : i + self.total_len].any():
                        valid.append(i)
                self.valid_indices = np.array(valid, dtype=np.int64)
            else:
                self.valid_indices = all_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.valid_indices[idx]

        context = self.data[start : start + self.context_len]  # (context_len, n_sensors)
        target = self.data[
            start + self.context_len : start + self.total_len
        ]  # (target_len, n_sensors)

        # Reshape into patches: (n_patches, patch_size, n_sensors)
        patch_samples = int(self.data_cfg.sensor_patch_size * self.sample_rate_hz)
        context_patches = self._to_patches(context, patch_samples)
        target_patches = self._to_patches(target, patch_samples)

        return {
            "context": torch.from_numpy(context_patches),  # (n_ctx_patches, patch_size, n_sensors)
            "target": torch.from_numpy(target_patches),  # (n_tgt_patches, patch_size, n_sensors)
        }

    def _to_patches(self, arr: np.ndarray, patch_size: int) -> np.ndarray:
        """Split time-series into non-overlapping patches."""
        n_steps = arr.shape[0]
        n_patches = n_steps // patch_size
        trimmed = arr[: n_patches * patch_size]
        return trimmed.reshape(n_patches, patch_size, self.n_sensors)


def load_csv_dataset(
    csv_path: str | Path,
    sensor_columns: list[str],
    timestamp_column: str = "timestamp",
    model_cfg: ModelConfig | None = None,
    data_cfg: DataConfig | None = None,
    sample_rate_hz: float = 1.0,
    failure_timestamps: list[str] | None = None,
) -> SensorWindowDataset:
    """Load a sensor CSV and return a SensorWindowDataset.

    Args:
        csv_path: Path to the CSV file.
        sensor_columns: Column names for sensor readings.
        timestamp_column: Name of the timestamp column.
        model_cfg: Model config (uses defaults if None).
        data_cfg: Data config (uses defaults if None).
        sample_rate_hz: Sampling rate of the data.
        failure_timestamps: List of ISO timestamps marking known failures.
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if data_cfg is None:
        data_cfg = DataConfig()

    df = pd.read_csv(csv_path, parse_dates=[timestamp_column])
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    failure_mask = None
    if failure_timestamps:
        exclusion_samples = int(data_cfg.failure_exclusion_hours * 3600 * sample_rate_hz)
        failure_mask = np.zeros(len(df), dtype=bool)
        for ts in failure_timestamps:
            ft = pd.Timestamp(ts)
            mask = (df[timestamp_column] >= ft - pd.Timedelta(hours=data_cfg.failure_exclusion_hours)) & (
                df[timestamp_column] <= ft + pd.Timedelta(hours=data_cfg.failure_exclusion_hours)
            )
            failure_mask |= mask.values

    return SensorWindowDataset(
        data=df,
        sensor_ids=sensor_columns,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        sample_rate_hz=sample_rate_hz,
        failure_mask=failure_mask,
    )
