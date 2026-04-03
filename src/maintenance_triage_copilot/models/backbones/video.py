"""Video backbone adapter using vendored V-JEPA slices."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from maintenance_triage_copilot.config import VideoBackboneConfig
from maintenance_triage_copilot.domain.models import ReferenceState, VisualObservation
from maintenance_triage_copilot.vendor.meta_vjepa import VideoVisionTransformer


class VJEPAVideoAdapter:
    def __init__(self, cfg: VideoBackboneConfig):
        self.cfg = cfg
        self.model = VideoVisionTransformer(
            img_size=cfg.input_size,
            patch_size=cfg.patch_size,
            num_frames=cfg.num_frames,
            tubelet_size=cfg.tubelet_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
        )
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self.cfg.embed_dim

    def sample_frames(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError("Expected video tensor with shape [C, T, H, W]")
        _, frames, _, _ = tensor.shape
        if frames == self.cfg.num_frames:
            return tensor
        indices = torch.linspace(0, max(frames - 1, 0), self.cfg.num_frames).long()
        return tensor[:, indices]

    def encode_observation(self, observation: VisualObservation | ReferenceState) -> np.ndarray:
        tensor = observation.load_tensor().float()
        if tensor.ndim != 4:
            raise ValueError("Video observations must provide [C, T, H, W] tensors")
        tensor = self.sample_frames(tensor).unsqueeze(0)
        batch, channels, frames, _, _ = tensor.shape
        tensor = tensor.view(batch * frames, channels, tensor.shape[-2], tensor.shape[-1])
        tensor = functional.interpolate(
            tensor,
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.view(batch, frames, channels, self.cfg.input_size, self.cfg.input_size)
        tensor = tensor.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            embedding = self.model.pooled(tensor)
        vector = embedding[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-8 else vector / norm
