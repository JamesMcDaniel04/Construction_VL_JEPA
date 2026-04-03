"""Modality-specific encoders that project into the shared JEPA latent space.

Phase 1 (MVP): SensorPatchEncoder — 1D Temporal Transformer for time-series.
Phase 2: TextEncoder — frozen sentence transformer + projection.
Phase 3: VideoEncoder — ViT-based frame encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Project raw sensor patches into token embeddings.

    Input:  (batch, n_patches, patch_size, n_sensors)
    Output: (batch, n_patches, latent_dim)
    """

    def __init__(self, patch_size: int, n_sensors: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_size * n_sensors, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, P, S) -> flatten patch -> (B, N, P*S) -> project
        b, n, p, s = x.shape
        x = rearrange(x, "b n p s -> b n (p s)")
        return self.norm(self.proj(x))


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, latent_dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2).float() * (-math.log(10000.0) / latent_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SensorPatchEncoder(nn.Module):
    """1D Temporal Transformer encoder for sensor time-series.

    Converts patched sensor windows into latent representations.
    This is the x-encoder (context encoder) in JEPA terminology.

    Input:  (batch, n_patches, patch_size, n_sensors)
    Output: (batch, n_patches, latent_dim)
    """

    def __init__(
        self,
        patch_size: int,
        n_sensors: int,
        latent_dim: int = 256,
        depth: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, n_sensors, latent_dim)
        self.pos_enc = SinusoidalPositionalEncoding(latent_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_patches, patch_size, n_sensors)
            mask: Optional boolean mask (batch, n_patches) — True = keep, False = mask out.

        Returns:
            Latent representations (batch, n_patches + 1, latent_dim) including CLS token.
        """
        tokens = self.patch_embed(x)  # (B, N, D)
        tokens = self.pos_enc(tokens)

        # Prepend CLS token
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)

        # Apply mask if provided (for JEPA masking strategy)
        src_key_padding_mask = None
        if mask is not None:
            # Prepend True for CLS token (always visible)
            cls_mask = torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)
            # TransformerEncoder expects True = ignore, so invert
            src_key_padding_mask = ~torch.cat([cls_mask, mask], dim=1)

        tokens = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        return self.norm(tokens)


class TextEncoder(nn.Module):
    """Frozen sentence transformer with a learned projection layer.

    Phase 2: Encodes maintenance logs / operator notes into the shared latent space.
    Uses a pre-trained sentence transformer (frozen) and projects into latent_dim.
    """

    def __init__(self, latent_dim: int = 256, pretrained_dim: int = 384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(pretrained_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Pre-computed sentence embeddings (batch, pretrained_dim).

        Returns:
            Projected embeddings (batch, 1, latent_dim) — single token per text entry.
        """
        return self.proj(embeddings).unsqueeze(1)


class CrossModalFusion(nn.Module):
    """Cross-attention fusion module for combining modality encodings.

    Phase 2+: Fuses sensor latents with text/video latents via cross-attention.
    """

    def __init__(self, latent_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, sensor_tokens: torch.Tensor, auxiliary_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sensor_tokens: (batch, n_sensor_tokens, latent_dim)
            auxiliary_tokens: (batch, n_aux_tokens, latent_dim) — text/video tokens.

        Returns:
            Fused tokens (batch, n_sensor_tokens, latent_dim).
        """
        # Cross-attend: sensor tokens query into auxiliary tokens
        residual = sensor_tokens
        x = self.norm1(sensor_tokens)
        aux = self.norm1(auxiliary_tokens)
        x, _ = self.cross_attn(x, aux, aux)
        x = x + residual

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x
