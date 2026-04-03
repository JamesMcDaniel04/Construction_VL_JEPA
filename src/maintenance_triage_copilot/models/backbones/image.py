"""Image backbone adapter using vendored I-JEPA slices."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from maintenance_triage_copilot.config import ImageBackboneConfig
from maintenance_triage_copilot.domain.models import ReferenceState, VisualObservation
from maintenance_triage_copilot.vendor.meta_ijepa import VisionTransformer


class IJEPAImageAdapter:
    def __init__(self, cfg: ImageBackboneConfig):
        self.cfg = cfg
        self.model = VisionTransformer(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
        )
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self.cfg.embed_dim

    def encode_observation(self, observation: VisualObservation | ReferenceState) -> np.ndarray:
        tensor = observation.load_tensor().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = functional.interpolate(
            tensor,
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            embedding = self.model.pooled(tensor)
        vector = embedding[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-8 else vector / norm
