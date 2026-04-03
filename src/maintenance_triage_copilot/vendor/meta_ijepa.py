# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file vendors a minimal image backbone slice derived from
# facebookresearch/ijepa commit 52c1ae95d05f743e000e8f10a1f3a79b10cff048.

from __future__ import annotations

from functools import partial
from typing import cast

import numpy as np
import torch
import torch.nn as nn


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return cast(
        torch.Tensor,
        nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2.0, b=2.0),
    )


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    mesh = np.meshgrid(grid_w, grid_h)
    grid = np.stack(mesh, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return cast(torch.Tensor, self.drop(self.fc2(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return cast(torch.Tensor, self.drop(self.proj(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.proj(x).flatten(2).transpose(1, 2))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim),
            requires_grad=False,
        )
        sincos = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(sincos).float().unsqueeze(0))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads=num_heads, drop=drop_rate) for _ in range(depth)]
        )
        self.norm = partial(nn.LayerNorm, eps=1e-6)(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
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
