"""Generate synthetic sensor data with injected anomalies for development and testing.

Creates realistic-looking sensor time-series for a CNC-like machine with:
  - Normal operating patterns with daily/shift cycles
  - Gradual degradation leading to failure events
  - Sudden transient anomalies
  - Sensor noise and occasional bad readings

Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic_sensors.csv
    python scripts/generate_synthetic_data.py --output data/synthetic_sensors.csv --days 30 --n-sensors 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_normal_sensor(
    n_steps: int, base: float, amplitude: float, noise_std: float, period: int = 86400
) -> np.ndarray:
    """Generate a normal sensor signal with daily cycle + noise."""
    t = np.arange(n_steps)
    cycle = amplitude * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, noise_std, n_steps)
    return base + cycle + noise


def inject_gradual_degradation(
    signal: np.ndarray, start: int, end: int, magnitude: float
) -> np.ndarray:
    """Inject a gradual ramp-up anomaly (bearing wear, etc.)."""
    signal = signal.copy()
    ramp_len = end - start
    ramp = np.linspace(0, magnitude, ramp_len)
    # Add increasing noise too
    extra_noise = np.random.normal(0, magnitude * 0.3, ramp_len) * np.linspace(0, 1, ramp_len)
    signal[start:end] += ramp + extra_noise
    return signal


def inject_sudden_anomaly(
    signal: np.ndarray, center: int, duration: int, magnitude: float
) -> np.ndarray:
    """Inject a sudden spike anomaly (tool break, power surge, etc.)."""
    signal = signal.copy()
    start = max(0, center - duration // 2)
    end = min(len(signal), center + duration // 2)
    signal[start:end] += magnitude * np.random.uniform(0.5, 1.5, end - start)
    return signal


def generate_dataset(
    n_days: int = 14,
    sample_rate_hz: float = 1.0,
    n_sensors: int = 8,
    n_failures: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    """Generate a complete synthetic sensor dataset with failure events.

    Returns:
        (dataframe, failure_events) where failure_events is a list of dicts
        with timestamp and description.
    """
    np.random.seed(seed)
    n_steps = int(n_days * 86400 * sample_rate_hz)

    # Sensor definitions (mimicking a CNC machine)
    sensor_configs = [
        {"name": "spindle_vibration_x", "base": 2.5, "amplitude": 0.3, "noise": 0.15},
        {"name": "spindle_vibration_y", "base": 2.3, "amplitude": 0.25, "noise": 0.12},
        {"name": "spindle_temperature", "base": 45.0, "amplitude": 5.0, "noise": 0.5},
        {"name": "coolant_temperature", "base": 22.0, "amplitude": 3.0, "noise": 0.3},
        {"name": "coolant_pressure", "base": 4.5, "amplitude": 0.2, "noise": 0.1},
        {"name": "feed_rate", "base": 200.0, "amplitude": 10.0, "noise": 2.0},
        {"name": "spindle_load", "base": 65.0, "amplitude": 8.0, "noise": 3.0},
        {"name": "power_consumption", "base": 15.0, "amplitude": 2.0, "noise": 0.8},
    ][:n_sensors]

    # Generate timestamps
    start_time = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start=start_time, periods=n_steps, freq=f"{int(1/sample_rate_hz)}s")

    # Generate normal signals
    data = {"timestamp": timestamps}
    signals = {}
    for cfg in sensor_configs:
        signals[cfg["name"]] = generate_normal_sensor(
            n_steps, cfg["base"], cfg["amplitude"], cfg["noise"]
        )

    # Inject failures
    failure_events = []
    failure_spacing = n_steps // (n_failures + 1)

    for i in range(n_failures):
        failure_center = failure_spacing * (i + 1)
        failure_ts = timestamps[min(failure_center, n_steps - 1)]

        # Gradual degradation in vibration (10-day ramp for bearing failure)
        degradation_start = max(0, failure_center - int(10 * 86400 * sample_rate_hz))
        degradation_end = failure_center

        # Vibration increases
        for vib_sensor in ["spindle_vibration_x", "spindle_vibration_y"]:
            if vib_sensor in signals:
                signals[vib_sensor] = inject_gradual_degradation(
                    signals[vib_sensor], degradation_start, degradation_end, magnitude=3.0
                )

        # Temperature creeps up
        if "spindle_temperature" in signals:
            signals["spindle_temperature"] = inject_gradual_degradation(
                signals["spindle_temperature"], degradation_start, degradation_end, magnitude=15.0
            )

        # Sudden spike at failure point
        for sensor_name in signals:
            signals[sensor_name] = inject_sudden_anomaly(
                signals[sensor_name],
                center=failure_center,
                duration=int(300 * sample_rate_hz),  # 5 min spike
                magnitude=signals[sensor_name][degradation_start] * 0.5,
            )

        failure_events.append({
            "timestamp": failure_ts.isoformat(),
            "event_type": "failure",
            "description": f"Bearing failure #{i+1} — spindle bearing degradation",
            "machine_id": "CNC-042",
        })

    # Also inject a transient anomaly (not a failure) for false positive testing
    transient_center = n_steps // 4
    if "coolant_pressure" in signals:
        signals["coolant_pressure"] = inject_sudden_anomaly(
            signals["coolant_pressure"],
            center=transient_center,
            duration=int(60 * sample_rate_hz),
            magnitude=2.0,
        )

    # Inject occasional bad readings (NaN)
    for sensor_name in signals:
        bad_indices = np.random.choice(n_steps, size=max(1, n_steps // 10000), replace=False)
        signals[sensor_name][bad_indices] = np.nan

    for name, values in signals.items():
        data[name] = values

    df = pd.DataFrame(data)
    return df, failure_events


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data")
    parser.add_argument("--output", type=str, default="data/synthetic_sensors.csv")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--n-sensors", type=int, default=8)
    parser.add_argument("--n-failures", type=int, default=2)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.days} days of synthetic data with {args.n_sensors} sensors...")
    df, failures = generate_dataset(
        n_days=args.days,
        sample_rate_hz=args.sample_rate,
        n_sensors=args.n_sensors,
        n_failures=args.n_failures,
        seed=args.seed,
    )

    # Save sensor data
    df.to_csv(output_path, index=False)
    print(f"Saved sensor data to {output_path} ({len(df):,} rows)")

    # Save failure events
    import json

    failures_path = output_path.with_suffix(".failures.json")
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"Saved {len(failures)} failure events to {failures_path}")

    # Print summary
    print(f"\nSensor columns: {[c for c in df.columns if c != 'timestamp']}")
    print(f"Failure timestamps: {[f['timestamp'] for f in failures]}")
    print(f"\nTo train: pfil-train --data-path {output_path} --sensor-columns {' '.join(c for c in df.columns if c != 'timestamp')}")


if __name__ == "__main__":
    main()
