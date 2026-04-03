"""Text encoders for grounded retrieval."""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np

from maintenance_triage_copilot.config import TextEncoderConfig


class MaintenanceTextEncoder:
    """Sentence-transformer wrapper with an explicit mock backend for tests."""

    def __init__(self, cfg: TextEncoderConfig):
        self.cfg = cfg
        self._model = None
        if cfg.backend == "sentence-transformer":
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    cfg.model_name,
                    device="cpu",
                    cache_folder=cfg.cache_folder,
                    local_files_only=cfg.local_files_only,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load sentence-transformer model "
                    f"'{cfg.model_name}'. Install the weights or set an explicit "
                    "test-only backend."
                ) from exc
        elif cfg.backend == "mock":
            self._model = None
        else:
            raise ValueError(f"Unsupported text encoder backend: {cfg.backend}")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return cast(np.ndarray, embeddings.astype(np.float32))
        return np.stack([self._hash_embed(text) for text in texts], axis=0)

    def encode_one(self, text: str) -> np.ndarray:
        return cast(np.ndarray, self.encode([text])[0])

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.cfg.embedding_dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(0, len(digest), 4):
                bucket = int.from_bytes(digest[idx : idx + 4], "little") % self.cfg.embedding_dim
                sign = -1.0 if digest[idx] % 2 else 1.0
                vector[bucket] += sign
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector /= norm
        return vector
