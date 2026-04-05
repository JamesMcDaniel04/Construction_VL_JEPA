"""Train calibrated issue-ranking and escalation policy checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional

from maintenance_triage_copilot.models.policy import (
    ESCALATION_FEATURE_NAMES,
    ISSUE_FEATURE_NAMES,
    CalibratedTriagePolicy,
)
from maintenance_triage_copilot.utils.logging import setup_logging

ESCALATION_LABELS = {
    "proceed_with_guided_inspection": 0.0,
    "escalate_for_visual_review": 0.55,
    "escalate_to_senior_technician": 1.0,
}


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _train_binary_classifier(
    examples: list[tuple[list[float], float]],
    feature_dim: int,
    *,
    epochs: int,
    lr: float,
) -> tuple[list[float], float]:
    if not examples:
        bootstrap = CalibratedTriagePolicy.bootstrap()
        return bootstrap.default_issue_weights, bootstrap.default_issue_bias

    features = torch.tensor([item[0] for item in examples], dtype=torch.float32)
    labels = torch.tensor([item[1] for item in examples], dtype=torch.float32).unsqueeze(1)
    weights = torch.zeros((feature_dim, 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr)

    for _ in range(epochs):
        logits = features @ weights + bias
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (
        weights.detach().squeeze(1).tolist(),
        float(bias.detach().item()),
    )


def _train_regressor(
    examples: list[tuple[list[float], float]],
    feature_dim: int,
    *,
    epochs: int,
    lr: float,
) -> tuple[list[float], float]:
    if not examples:
        bootstrap = CalibratedTriagePolicy.bootstrap()
        return bootstrap.escalation_weights, bootstrap.escalation_bias

    features = torch.tensor([item[0] for item in examples], dtype=torch.float32)
    labels = torch.tensor([item[1] for item in examples], dtype=torch.float32).unsqueeze(1)
    weights = torch.zeros((feature_dim, 1), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=lr)

    for _ in range(epochs):
        logits = features @ weights + bias
        predictions = torch.sigmoid(logits)
        loss = functional.mse_loss(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (
        weights.detach().squeeze(1).tolist(),
        float(bias.detach().item()),
    )


def train_policy_from_rows(
    rows: list[dict],
    *,
    epochs: int,
    lr: float,
) -> CalibratedTriagePolicy:
    feature_sets: dict[str, list[tuple[list[float], float]]] = defaultdict(list)
    default_examples: list[tuple[list[float], float]] = []
    escalation_examples: list[tuple[list[float], float]] = []

    for row in rows:
        correct_issue = row["correct_issue_class"]
        for candidate in row.get("candidates", []):
            vector = [float(candidate["features"].get(name, 0.0)) for name in ISSUE_FEATURE_NAMES]
            label = 1.0 if candidate["issue_class"] == correct_issue else 0.0
            feature_sets[candidate["issue_class"]].append((vector, label))
            default_examples.append((vector, label))

        escalation_vector = [
            float(row["escalation_features"].get(name, 0.0))
            for name in ESCALATION_FEATURE_NAMES
        ]
        escalation_examples.append(
            (escalation_vector, ESCALATION_LABELS[row["escalation_label"]])
        )

    issue_weights: dict[str, list[float]] = {}
    issue_bias: dict[str, float] = {}
    for issue_class, examples in feature_sets.items():
        weights, bias = _train_binary_classifier(
            examples,
            len(ISSUE_FEATURE_NAMES),
            epochs=epochs,
            lr=lr,
        )
        issue_weights[issue_class] = weights
        issue_bias[issue_class] = bias

    default_issue_weights, default_issue_bias = _train_binary_classifier(
        default_examples,
        len(ISSUE_FEATURE_NAMES),
        epochs=epochs,
        lr=lr,
    )
    escalation_weights, escalation_bias = _train_regressor(
        escalation_examples,
        len(ESCALATION_FEATURE_NAMES),
        epochs=epochs,
        lr=lr,
    )

    return CalibratedTriagePolicy(
        issue_weights=issue_weights,
        issue_bias=issue_bias,
        default_issue_weights=default_issue_weights,
        default_issue_bias=default_issue_bias,
        escalation_weights=escalation_weights,
        escalation_bias=escalation_bias,
        escalation_thresholds={
            "escalate_to_senior_technician": 0.8,
            "escalate_for_visual_review": 0.5,
        },
        metadata={
            "trained_examples": str(len(rows)),
            "epochs": str(epochs),
            "learning_rate": str(lr),
            "issue_feature_names": ",".join(ISSUE_FEATURE_NAMES),
            "escalation_feature_names": ",".join(ESCALATION_FEATURE_NAMES),
        },
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Train calibrated triage policy")
    parser.add_argument("--data", required=True, help="JSONL training file")
    parser.add_argument("--output", required=True, help="Output JSON policy checkpoint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-2)
    args = parser.parse_args()

    rows = _load_rows(Path(args.data))
    policy = train_policy_from_rows(rows, epochs=args.epochs, lr=args.lr)
    policy.save(args.output)
    print(f"Saved calibrated policy checkpoint to {args.output}")


if __name__ == "__main__":
    main()
