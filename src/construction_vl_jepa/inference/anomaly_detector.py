"""Anomaly detection engine with adaptive thresholds and trend detection.

Implements per-machine statistical baselines, trend-based alerting, and
alert cooldown suppression.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import torch

from construction_vl_jepa.config import InferenceConfig
from construction_vl_jepa.model.jepa import VLJEPA


@dataclass
class MachineBaseline:
    """Per-machine running statistics for adaptive thresholding."""

    scores: deque = field(default_factory=lambda: deque(maxlen=10000))
    mean: float = 0.0
    std: float = 1.0
    last_alert_time: float = 0.0

    def update(self, score: float) -> None:
        self.scores.append(score)
        if len(self.scores) >= 10:
            arr = np.array(self.scores)
            self.mean = float(np.mean(arr))
            self.std = float(np.std(arr))
            if self.std < 1e-8:
                self.std = 1.0


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single window."""

    machine_id: str
    score: float
    normalized_score: float  # (score - μ) / σ
    is_anomaly: bool
    is_trend_anomaly: bool
    contributing_signals: list[dict]
    timestamp: float


class AnomalyDetector:
    """Real-time anomaly detection using JEPA prediction error."""

    def __init__(self, model: VLJEPA, cfg: InferenceConfig, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.cfg = cfg
        self.device = device

        # Per-machine baselines
        self.baselines: dict[str, MachineBaseline] = defaultdict(MachineBaseline)

        # Recent scores for trend detection
        self.recent_scores: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=cfg.trend_window)
        )

    def score_window(
        self,
        machine_id: str,
        context: torch.Tensor,
        target: torch.Tensor,
        sensor_names: list[str] | None = None,
    ) -> AnomalyResult:
        """Score a single context→target window pair.

        Args:
            machine_id: Identifier for the machine.
            context: (1, n_ctx_patches, patch_size, n_sensors)
            target:  (1, n_tgt_patches, patch_size, n_sensors)
            sensor_names: Optional names for contributing signal analysis.

        Returns:
            AnomalyResult with score, threshold check, and contributing signals.
        """
        context = context.to(self.device)
        target = target.to(self.device)

        result = self.model.predict_and_score(context, target)
        raw_score = float(result["anomaly_score"][0].cpu())

        # Update per-machine baseline
        baseline = self.baselines[machine_id]
        baseline.update(raw_score)

        # Normalized score (how many σ from mean)
        normalized = (raw_score - baseline.mean) / baseline.std

        # Track for trend detection
        self.recent_scores[machine_id].append(raw_score)

        # Threshold check
        is_anomaly = normalized > self.cfg.threshold_k

        # Trend check: sustained upward drift
        is_trend = self._check_trend(machine_id)

        # Contributing signals analysis
        contributing = self._analyze_contributions(
            result["per_token_error"][0].cpu().numpy(),
            sensor_names,
        )

        # Alert cooldown
        now = time.time()
        if is_anomaly and (now - baseline.last_alert_time) < self.cfg.alert_cooldown:
            # Suppress unless score is escalating
            if normalized <= self.cfg.threshold_k * 1.5:
                is_anomaly = False

        if is_anomaly:
            baseline.last_alert_time = now

        return AnomalyResult(
            machine_id=machine_id,
            score=raw_score,
            normalized_score=normalized,
            is_anomaly=is_anomaly or is_trend,
            is_trend_anomaly=is_trend,
            contributing_signals=contributing,
            timestamp=now,
        )

    def _check_trend(self, machine_id: str) -> bool:
        """Check for sustained upward drift in recent scores."""
        scores = list(self.recent_scores[machine_id])
        if len(scores) < self.cfg.trend_window // 2:
            return False

        arr = np.array(scores)
        # Simple linear regression slope
        x = np.arange(len(arr))
        slope = np.polyfit(x, arr, 1)[0]

        return slope > self.cfg.trend_slope_threshold

    def _analyze_contributions(
        self,
        per_token_error: np.ndarray,
        sensor_names: list[str] | None,
    ) -> list[dict]:
        """Identify which tokens/signals contributed most to the anomaly score."""
        # per_token_error: (n_tokens,) — error per predicted token
        if len(per_token_error) == 0:
            return []

        # Rank tokens by error
        ranked = np.argsort(per_token_error)[::-1]
        total_error = per_token_error.sum()

        contributions = []
        for rank, idx in enumerate(ranked[:5]):  # top 5 contributors
            error = float(per_token_error[idx])
            entry = {
                "token_index": int(idx),
                "error": error,
                "fraction": float(error / total_error) if total_error > 0 else 0,
                "rank": rank + 1,
            }
            if sensor_names and idx < len(sensor_names):
                entry["sensor"] = sensor_names[idx]
            contributions.append(entry)

        return contributions

    def get_machine_health(self, machine_id: str) -> dict:
        """Get current health status for a machine."""
        baseline = self.baselines.get(machine_id)
        if baseline is None or len(baseline.scores) == 0:
            return {"status": "unknown", "score": 0.0, "trend": "stable"}

        recent = list(self.recent_scores.get(machine_id, []))
        if not recent:
            return {"status": "unknown", "score": 0.0, "trend": "stable"}

        current = recent[-1]
        normalized = (current - baseline.mean) / baseline.std

        if normalized > self.cfg.threshold_k:
            status = "red"
        elif normalized > self.cfg.threshold_k * 0.7:
            status = "yellow"
        else:
            status = "green"

        is_trend = self._check_trend(machine_id)
        trend = "rising" if is_trend else "stable"

        return {
            "status": status,
            "score": current,
            "normalized_score": normalized,
            "trend": trend,
            "baseline_mean": baseline.mean,
            "baseline_std": baseline.std,
        }
