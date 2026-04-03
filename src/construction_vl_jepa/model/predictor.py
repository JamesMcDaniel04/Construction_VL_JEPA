"""JEPA Predictor Network.

The predictor takes context encoder output and predicts the target encoder output
in latent space. This is where the "prediction in latent space" happens — the core
of the JEPA architecture.

The prediction error (L2 between predicted and actual target latent) is the
training signal, and at inference time, the anomaly signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from construction_vl_jepa.model.encoders import SinusoidalPositionalEncoding


class JEPAPredictor(nn.Module):
    """Transformer-based predictor that maps context latent -> predicted target latent.

    Input:  Context encoder output (batch, n_context_tokens, latent_dim)
    Output: Predicted target latent  (batch, n_target_tokens, latent_dim)

    The predictor uses learnable query tokens for the target positions,
    cross-attending into the context encoder output.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        n_target_tokens: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_target_tokens = n_target_tokens
        self.latent_dim = latent_dim

        # Learnable query tokens for target positions
        self.target_queries = nn.Parameter(
            torch.randn(1, n_target_tokens, latent_dim) * 0.02
        )
        self.query_pos = SinusoidalPositionalEncoding(latent_dim)

        # Transformer decoder layers (cross-attend into context)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, context_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_latent: Output from context encoder (batch, n_ctx_tokens, latent_dim).

        Returns:
            Predicted target latent (batch, n_target_tokens, latent_dim).
        """
        batch_size = context_latent.size(0)

        # Expand query tokens for batch
        queries = self.target_queries.expand(batch_size, -1, -1)
        queries = self.query_pos(queries)

        # Cross-attend: queries attend to context
        predicted = self.decoder(queries, context_latent)
        return self.norm(predicted)
