"""Image backbone adapter for vendored Meta JEPA slices.

Supports two modes:
  1. "vendored" (supported): Uses the vendored Meta I-JEPA VisionTransformer.
  2. "timm" (optional): Uses a timm ViT for local experimentation only.

When a checkpoint_path is set in config, loads saved weights on top of the
pretrained backbone (for fine-tuned models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from maintenance_triage_copilot.config import ImageBackboneConfig
from maintenance_triage_copilot.domain.models import ReferenceState, VisualObservation


class TimmVisionEncoder(nn.Module):
    """Optional ViT backbone via timm for local experimentation."""

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for the optional timm image backbone. "
                "Install with: pip install timm>=0.9.0"
            ) from exc

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # feature extraction mode — no classification head
        )
        self.embed_dim = int(cast(Any, self.model).embed_dim)
        if gradient_checkpointing and hasattr(self.model, "set_grad_checkpointing"):
            cast(Any, self.model).set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return all patch token embeddings [B, N+1, D] (CLS + patches)."""
        return cast(torch.Tensor, cast(Any, self.model).forward_features(x))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token embedding [B, D]."""
        features = self.forward(x)
        # timm ViTs typically put CLS at position 0
        if features.ndim == 3:
            return features[:, 0]
        return features


class IJEPAImageAdapter:
    """Image encoder adapter that wraps either timm or vendored ViT."""

    def __init__(self, cfg: ImageBackboneConfig):
        self.cfg = cfg
        self._use_timm = getattr(cfg, "use_timm", False)
        self.model: Any
        self._embed_dim: int
        self._input_size: int

        if self._use_timm:
            model_name = getattr(cfg, "timm_model_name", "vit_tiny_patch16_224")
            pretrained = getattr(cfg, "timm_pretrained", True)
            self.model = TimmVisionEncoder(
                model_name=model_name,
                pretrained=pretrained,
            )
            self._embed_dim = self.model.embed_dim
            self._input_size = 224  # timm ViT default
        else:
            from maintenance_triage_copilot.vendor.meta_ijepa import VisionTransformer

            self.model = VisionTransformer(
                img_size=cfg.input_size,
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
            )
            self._embed_dim = cfg.embed_dim
            self._input_size = cfg.input_size

        # Load checkpoint if provided
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return int(self._embed_dim)

    def encode_observation(self, observation: VisualObservation | ReferenceState) -> np.ndarray:
        tensor = observation.load_tensor().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = functional.interpolate(
            tensor,
            size=(self._input_size, self._input_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            embedding = cast(torch.Tensor, self.model.pooled(tensor))
        vector = embedding[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-8 else vector / norm

    def encode_raw_image(self, image_bytes: bytes) -> np.ndarray:
        """Encode a raw JPEG/PNG image from bytes."""
        tensor = _decode_image_bytes(image_bytes)
        tensor = functional.interpolate(
            tensor.unsqueeze(0),
            size=(self._input_size, self._input_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            embedding = cast(torch.Tensor, self.model.pooled(tensor))
        vector = embedding[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-8 else vector / norm


def _decode_image_bytes(data: bytes) -> torch.Tensor:
    """Decode JPEG/PNG bytes to a normalized [C, H, W] float tensor."""
    try:
        from torchvision.io import decode_image
        from torchvision.transforms.functional import normalize

        tensor = cast(
            torch.Tensor,
            decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8)),
        )
        tensor = tensor.float() / 255.0
        # ImageNet normalization (matches timm pretrained weights)
        tensor = cast(
            torch.Tensor,
            normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )
        return tensor
    except ImportError:
        # Fallback: PIL + numpy
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(arr.transpose(2, 0, 1))
