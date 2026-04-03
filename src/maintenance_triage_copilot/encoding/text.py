"""Text encoders for grounded retrieval."""

from __future__ import annotations

import hashlib
import math
from typing import cast

import numpy as np

from maintenance_triage_copilot.config import TextEncoderConfig


class MaintenanceTextEncoder:
    """Sentence-transformer wrapper with deterministic hashing fallback."""

    def __init__(self, cfg: TextEncoderConfig):
        self.cfg = cfg
        self._model = None
        if cfg.backend == "sentence-transformer":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    cfg.model_name,
                    device="cpu",
                    local_files_only=True,
                )
            except Exception:
                self._model = None

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
        else:
            vector.fill(1.0 / math.sqrt(self.cfg.embedding_dim))
        return vector
