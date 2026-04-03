"""Tiny training entrypoint for the visual-text projector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from maintenance_triage_copilot.config import load_config
from maintenance_triage_copilot.domain.models import VisualObservation
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Train visual-text projector for maintenance triage"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="JSONL file with observation + target_text",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/projector.pt",
        help="Path to save the trained projector checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    text_encoder = MaintenanceTextEncoder(cfg.text_encoder)
    image_backbone = IJEPAImageAdapter(cfg.image_backbone)
    video_backbone = VJEPAVideoAdapter(cfg.video_backbone)
    projector = VisualTextProjector(
        input_dim=image_backbone.embedding_dim,
        hidden_dim=cfg.adapter.hidden_dim,
        output_dim=cfg.adapter.output_dim,
    )
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)

    rows = []
    with Path(args.data).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    for epoch in range(args.epochs):
        total_loss = 0.0
        for row in rows:
            observation = VisualObservation(**row["observation"])
            if observation.media_type.value == "image":
                visual = image_backbone.encode_observation(observation)
            else:
                visual = video_backbone.encode_observation(observation)
            target = text_encoder.encode_one(row["target_text"])

            optimizer.zero_grad()
            projected = projector(torch.from_numpy(visual).unsqueeze(0))
            loss = 1 - functional.cosine_similarity(
                projected, torch.from_numpy(target).unsqueeze(0)
            ).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(len(rows), 1):.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), output_path)
    print(f"Saved projector checkpoint to {output_path}")


if __name__ == "__main__":
    main()
