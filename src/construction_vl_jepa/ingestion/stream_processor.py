"""Stream processing for real-time feature computation and inference triggering.

Uses a windowed buffer to accumulate sensor readings and trigger
inference when a complete context+target window is available.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from construction_vl_jepa.config import DataConfig, ModelConfig
from construction_vl_jepa.data.schema import SensorReading
from construction_vl_jepa.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SensorBuffer:
    """Ring buffer for accumulating sensor readings for a single machine."""

    max_sensors: int
    context_samples: int
    target_samples: int
    sensor_index: dict[str, int] = field(default_factory=dict)
    buffer: np.ndarray | None = None
    write_pos: int = 0
    total_written: int = 0

    def __post_init__(self):
        total_samples = self.context_samples + self.target_samples
        # Allocate 2x for ring buffer overlap
        self.buffer = np.zeros((total_samples * 2, self.max_sensors), dtype=np.float32)

    def add_reading(self, sensor_id: str, value: float) -> None:
        """Add a single sensor reading to the buffer."""
        if sensor_id not in self.sensor_index:
            idx = len(self.sensor_index)
            if idx >= self.max_sensors:
                return  # ignore sensors beyond max
            self.sensor_index[sensor_id] = idx

        col = self.sensor_index[sensor_id]
        self.buffer[self.write_pos, col] = value
        self.write_pos = (self.write_pos + 1) % len(self.buffer)
        self.total_written += 1

    def has_full_window(self) -> bool:
        """Check if we have enough data for a context+target window."""
        return self.total_written >= (self.context_samples + self.target_samples)

    def get_window(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract the most recent context and target windows.

        Returns:
            (context, target) arrays each of shape (samples, max_sensors).
        """
        total = self.context_samples + self.target_samples
        end = self.write_pos
        start = (end - total) % len(self.buffer)

        if start < end:
            window = self.buffer[start:end].copy()
        else:
            window = np.concatenate([
                self.buffer[start:],
                self.buffer[:end],
            ])

        context = window[: self.context_samples]
        target = window[self.context_samples :]
        return context, target


class StreamProcessor:
    """Manages sensor buffers for multiple machines and triggers inference."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        sample_rate_hz: float = 1.0,
    ):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.sample_rate_hz = sample_rate_hz

        context_samples = int(model_cfg.context_window * sample_rate_hz)
        target_samples = int(model_cfg.target_window * sample_rate_hz)

        self.context_samples = context_samples
        self.target_samples = target_samples
        self.patch_size = int(data_cfg.sensor_patch_size * sample_rate_hz)

        self.buffers: dict[str, SensorBuffer] = {}
        self.last_inference: dict[str, float] = defaultdict(float)

    def _get_buffer(self, machine_id: str) -> SensorBuffer:
        if machine_id not in self.buffers:
            self.buffers[machine_id] = SensorBuffer(
                max_sensors=self.data_cfg.max_sensors,
                context_samples=self.context_samples,
                target_samples=self.target_samples,
            )
        return self.buffers[machine_id]

    def process_readings(
        self, readings: list[SensorReading]
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Process a batch of sensor readings and return ready windows.

        Args:
            readings: Batch of SensorReading objects.

        Returns:
            List of (machine_id, context_patches, target_patches) tuples
            for machines that have complete windows ready for inference.
        """
        # Group readings by machine
        by_machine: dict[str, list[SensorReading]] = defaultdict(list)
        for r in readings:
            by_machine[r.machine_id].append(r)

        ready_windows = []

        for machine_id, machine_readings in by_machine.items():
            buf = self._get_buffer(machine_id)

            # Add readings to buffer
            for r in sorted(machine_readings, key=lambda x: x.timestamp):
                if r.quality_flag != "bad":
                    buf.add_reading(r.sensor_id, r.value)

            # Check if we should trigger inference
            now = time.time()
            if (
                buf.has_full_window()
                and (now - self.last_inference[machine_id]) >= self.model_cfg.target_window
            ):
                context, target = buf.get_window()
                context_patches = self._to_patches(context)
                target_patches = self._to_patches(target)
                ready_windows.append((machine_id, context_patches, target_patches))
                self.last_inference[machine_id] = now

        return ready_windows

    def _to_patches(self, arr: np.ndarray) -> np.ndarray:
        """Convert raw samples to patches: (n_patches, patch_size, n_sensors)."""
        n_steps, n_sensors = arr.shape
        n_patches = n_steps // self.patch_size
        trimmed = arr[: n_patches * self.patch_size]
        return trimmed.reshape(n_patches, self.patch_size, n_sensors)
