"""Feature engineering for sensor data.

Computes rolling statistics and cross-sensor features used by both
training and real-time inference pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_stats(
    df: pd.DataFrame,
    sensor_columns: list[str],
    windows: list[int] = (60, 300, 3600, 86400),
) -> pd.DataFrame:
    """Compute rolling mean, std, skew, kurtosis for each sensor over multiple windows.

    Args:
        df: DataFrame with sensor columns and a DatetimeIndex or sorted by time.
        sensor_columns: Columns to compute features for.
        windows: Window sizes in number of rows (assumed 1 Hz sampling).

    Returns:
        DataFrame with feature columns appended.
    """
    features = {}
    for col in sensor_columns:
        series = df[col]
        for w in windows:
            prefix = f"{col}_w{w}"
            rolling = series.rolling(window=w, min_periods=1)
            features[f"{prefix}_mean"] = rolling.mean()
            features[f"{prefix}_std"] = rolling.std().fillna(0)
            features[f"{prefix}_skew"] = rolling.skew().fillna(0)
            features[f"{prefix}_kurt"] = rolling.kurt().fillna(0)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def cross_sensor_correlation(
    df: pd.DataFrame,
    sensor_columns: list[str],
    window: int = 300,
) -> pd.DataFrame:
    """Compute pairwise rolling correlation coefficients between sensors.

    Args:
        df: DataFrame with sensor columns.
        sensor_columns: Sensor columns to correlate.
        window: Rolling window size in rows.

    Returns:
        DataFrame with correlation feature columns.
    """
    features = {}
    for i, col_a in enumerate(sensor_columns):
        for col_b in sensor_columns[i + 1 :]:
            corr = df[col_a].rolling(window=window, min_periods=2).corr(df[col_b]).fillna(0)
            features[f"corr_{col_a}_{col_b}_w{window}"] = corr

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def temporal_features(
    df: pd.DataFrame,
    timestamp_column: str = "timestamp",
    last_maintenance: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add time-based features: hour of day, day of week, time since last maintenance.

    Args:
        df: DataFrame with a timestamp column.
        timestamp_column: Name of timestamp column.
        last_maintenance: Timestamp of most recent maintenance event.

    Returns:
        DataFrame with temporal feature columns.
    """
    ts = pd.to_datetime(df[timestamp_column])
    features = {
        "hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24),
        "dow_sin": np.sin(2 * np.pi * ts.dt.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * ts.dt.dayofweek / 7),
    }

    if last_maintenance is not None:
        features["hours_since_maintenance"] = (ts - last_maintenance).dt.total_seconds() / 3600
    else:
        features["hours_since_maintenance"] = 0.0

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def compute_all_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
    timestamp_column: str = "timestamp",
    last_maintenance: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = rolling_stats(df, sensor_columns)
    if len(sensor_columns) <= 20:  # skip cross-corr for very wide sensor sets
        df = cross_sensor_correlation(df, sensor_columns)
    df = temporal_features(df, timestamp_column, last_maintenance)
    return df
