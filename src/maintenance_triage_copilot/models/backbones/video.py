"""Video backbone adapter for vendored Meta JEPA slices.

Supports two modes:
  1. "vendored" (supported): Uses the vendored Meta V-JEPA
     VideoVisionTransformer.
  2. "timm" (optional): Uses per-frame timm ViT pooling for local
     experimentation only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional

from maintenance_triage_copilot.config import VideoBackboneConfig
from maintenance_triage_copilot.domain.models import ReferenceState, VisualObservation
from maintenance_triage_copilot.models.backbones.image import TimmVisionEncoder


class VJEPAVideoAdapter:
    """Video encoder adapter that wraps either timm (per-frame) or vendored 3D ViT."""

    def __init__(self, cfg: VideoBackboneConfig):
        self.cfg = cfg
        self._use_timm = getattr(cfg, "use_timm", False)
        self._frame_encoder: TimmVisionEncoder | None
        self.model: Any

        if self._use_timm:
            model_name = getattr(cfg, "timm_model_name", "vit_tiny_patch16_224")
            pretrained = getattr(cfg, "timm_pretrained", True)
            self._frame_encoder = TimmVisionEncoder(
                model_name=model_name,
                pretrained=pretrained,
            )
            self._embed_dim = self._frame_encoder.embed_dim
            self._input_size = 224
            self.model = self._frame_encoder  # for .eval() compatibility
        else:
            from maintenance_triage_copilot.vendor.meta_vjepa import VideoVisionTransformer

            self.model = VideoVisionTransformer(
                img_size=cfg.input_size,
                patch_size=cfg.patch_size,
                num_frames=cfg.num_frames,
                tubelet_size=cfg.tubelet_size,
                embed_dim=cfg.embed_dim,
                depth=cfg.depth,
                num_heads=cfg.num_heads,
            )
            self._frame_encoder = None
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
        return self._embed_dim

    def sample_frames(self, tensor: torch.Tensor) -> torch.Tensor:
        """Uniformly sample num_frames from a [C, T, H, W] tensor."""
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
        tensor = self.sample_frames(tensor)

        if self._frame_encoder is not None:
            # timm mode: encode each frame independently, then pool
            return self._encode_per_frame(tensor)
        else:
            # vendored 3D ViT mode
            return self._encode_3d(tensor)

    def _encode_per_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Encode via per-frame timm ViT + temporal mean pooling."""
        if self._frame_encoder is None:
            raise RuntimeError("Per-frame encoding requires the optional timm video backbone")
        # Reshape to (frames, channels, h, w) for per-frame encoding
        frame_batch = tensor.permute(1, 0, 2, 3)  # [T, C, H, W]
        frame_batch = functional.interpolate(
            frame_batch,
            size=(self._input_size, self._input_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.no_grad():
            # Encode all frames at once as a batch
            frame_embeddings = self._frame_encoder.pooled(frame_batch)  # [T, D]
            # Temporal mean pooling
            embedding = frame_embeddings.mean(dim=0)  # [D]

        vector = embedding.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-8 else vector / norm

    def _encode_3d(self, tensor: torch.Tensor) -> np.ndarray:
        """Encode via vendored 3D VideoVisionTransformer."""
        tensor = tensor.unsqueeze(0)  # [1, C, T, H, W]
        batch, channels, frames, _, _ = tensor.shape
        # Resize spatial dimensions
        tensor = tensor.view(batch * frames, channels, tensor.shape[-2], tensor.shape[-1])
        tensor = functional.interpolate(
            tensor,
            size=(self._input_size, self._input_size),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.view(batch, frames, channels, self._input_size, self._input_size)
        tensor = tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        with torch.no_grad():
            embedding = cast(torch.Tensor, self.model.pooled(tensor))
        vector = embedding[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector if norm < 1e-8 else vector / norm

    def encode_raw_video(self, video_bytes: bytes) -> np.ndarray:
        """Encode raw video bytes (MP4/AVI/etc.) by extracting frames."""
        tensor = _decode_video_bytes(video_bytes, self.cfg.num_frames)
        if self._frame_encoder is not None:
            return self._encode_per_frame(tensor)
        return self._encode_3d(tensor)


def _decode_video_bytes(data: bytes, num_frames: int) -> torch.Tensor:
    """Decode video bytes to a [C, T, H, W] tensor with uniform frame sampling."""
    import tempfile

    try:
        from decord import VideoReader, cpu
    except ImportError:
        try:
            return _decode_video_cv2(data, num_frames)
        except ImportError as exc:
            raise ImportError(
                "Video decoding requires either 'decord' or 'opencv-python'. "
                "Install with: pip install decord  OR  pip install opencv-python"
            ) from exc

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        vr = VideoReader(tmp.name, ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

    frames = frames.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frames = (frames - mean) / std
    # [T, H, W, C] -> [C, T, H, W]
    return torch.from_numpy(frames.transpose(3, 0, 1, 2)).float()


def _decode_video_cv2(data: bytes, num_frames: int) -> torch.Tensor:
    """Fallback video decoding using OpenCV."""
    import tempfile

    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(3, 0, 1, 2)).float()
