# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file vendors a minimal video backbone slice derived from
# facebookresearch/jepa commit 51c59d518fc63c08464af6de585f78ac0c7ed4d5.

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from maintenance_triage_copilot.vendor.meta_ijepa import Block, trunc_normal_


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: int, grid_depth: int) -> np.ndarray:
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)
    h_dim = embed_dim // 4
    w_dim = embed_dim // 4
    d_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(h_dim, grid_h)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_dim, grid_w)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_dim, grid_d)
    return np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        patch_size: int = 8,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.proj(x).flatten(2).transpose(1, 2))


class VideoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        num_frames: int = 8,
        tubelet_size: int = 2,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
        )
        num_patches = (num_frames // tubelet_size) * (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim),
            requires_grad=False,
        )
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            img_size // patch_size,
            num_frames // tubelet_size,
        )
        self.pos_embed.data.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return cast(torch.Tensor, self.norm(x))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(x).mean(dim=1))
