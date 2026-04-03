"""Cross-modal adapter."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class VisualTextProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, visual_embedding: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(visual_embedding))
