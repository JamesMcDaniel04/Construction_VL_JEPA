"""Vector retrieval with in-memory fallback and optional Qdrant mirroring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SearchHit:
    item_id: str
    score: float
    payload: dict[str, Any]


class VectorIndex:
    """Stores vectors in memory and optionally mirrors writes to Qdrant."""

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_prefix: str = "maintenance-triage",
    ):
        self.collection_prefix = collection_prefix
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._qdrant = None
        if qdrant_url:
            try:
                from qdrant_client import QdrantClient

                self._qdrant = QdrantClient(url=qdrant_url)
            except Exception:
                self._qdrant = None

    def upsert(
        self,
        namespace: str,
        item_id: str,
        embedding: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        record = {
            "id": item_id,
            "embedding": self._normalize(embedding),
            "payload": payload,
        }
        self._store.setdefault(namespace, [])
        self._store[namespace] = [
            entry for entry in self._store[namespace] if entry["id"] != item_id
        ]
        self._store[namespace].append(record)

        if self._qdrant is not None:
            try:
                from qdrant_client.models import Distance, PointStruct, VectorParams

                collection_name = f"{self.collection_prefix}-{namespace}"
                collections = {c.name for c in self._qdrant.get_collections().collections}
                if collection_name not in collections:
                    self._qdrant.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=len(record["embedding"]),
                            distance=Distance.COSINE,
                        ),
                    )
                self._qdrant.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=item_id,
                            vector=np.asarray(record["embedding"]).tolist(),
                            payload=payload,
                        )
                    ],
                )
            except Exception:
                pass

    def search(
        self,
        namespace: str,
        query: np.ndarray,
        limit: int,
        equipment_family: str | None = None,
    ) -> list[SearchHit]:
        records = self._store.get(namespace, [])
        if not records:
            return []
        query = self._normalize(query)
        hits: list[SearchHit] = []
        for record in records:
            if equipment_family and record["payload"].get("equipment_family") != equipment_family:
                continue
            score = float(np.dot(query, record["embedding"]))
            hits.append(
                SearchHit(
                    item_id=record["id"],
                    score=max(0.0, min(1.0, score)),
                    payload=record["payload"],
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def status(self) -> dict[str, str]:
        return {
            "mode": "qdrant+memory" if self._qdrant is not None else "memory",
            "namespaces": ",".join(sorted(self._store.keys())) or "empty",
        }

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm < 1e-8:
            return array
        return array / norm
