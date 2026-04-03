"""Vector retrieval with in-memory fallback and persistent Qdrant storage.

Writes AND reads from Qdrant when available. Falls back to in-memory
cosine similarity when Qdrant is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from maintenance_triage_copilot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SearchHit:
    item_id: str
    score: float
    payload: dict[str, Any]


class VectorIndex:
    """Stores and searches vectors in Qdrant with in-memory fallback."""

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_prefix: str = "maintenance-triage",
        *,
        required: bool = False,
    ):
        self.collection_prefix = collection_prefix
        self.required = required
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._qdrant = None
        self._qdrant_collections: set[str] = set()
        if qdrant_url:
            try:
                from qdrant_client import QdrantClient

                self._qdrant = QdrantClient(url=qdrant_url)
                # Cache existing collection names
                self._qdrant_collections = {
                    c.name for c in self._qdrant.get_collections().collections
                }
            except Exception as exc:
                if self.required:
                    raise RuntimeError("Qdrant is required but could not be reached") from exc
                log.warning("qdrant_connection_failed error=%s", exc)
                self._qdrant = None

    def _collection_name(self, namespace: str) -> str:
        return f"{self.collection_prefix}-{namespace}"

    def _ensure_collection(self, namespace: str, dim: int) -> None:
        """Create Qdrant collection if it doesn't exist yet."""
        if self._qdrant is None:
            return
        name = self._collection_name(namespace)
        if name in self._qdrant_collections:
            return
        try:
            from qdrant_client.models import Distance, VectorParams

            self._qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self._qdrant_collections.add(name)
        except Exception as exc:
            if self.required:
                raise RuntimeError(
                    f"Failed to create required Qdrant collection: {name}"
                ) from exc
            log.warning("qdrant_collection_create_failed collection=%s error=%s", name, exc)

    def upsert(
        self,
        namespace: str,
        item_id: str,
        embedding: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        normalized = self._normalize(embedding)
        record = {"id": item_id, "embedding": normalized, "payload": payload}

        # Always keep in-memory copy (fast reads, survives Qdrant hiccups)
        self._store.setdefault(namespace, [])
        self._store[namespace] = [
            entry for entry in self._store[namespace] if entry["id"] != item_id
        ]
        self._store[namespace].append(record)

        # Mirror to Qdrant
        if self._qdrant is not None:
            try:
                from qdrant_client.models import PointStruct

                self._ensure_collection(namespace, len(normalized))
                self._qdrant.upsert(
                    collection_name=self._collection_name(namespace),
                    points=[
                        PointStruct(
                            id=item_id,
                            vector=normalized.tolist(),
                            payload=payload,
                        )
                    ],
                )
            except Exception as exc:
                if self.required:
                    raise RuntimeError("Qdrant upsert failed in required mode") from exc
                log.warning("qdrant_upsert_failed namespace=%s error=%s", namespace, exc)

    def search(
        self,
        namespace: str,
        query: np.ndarray,
        limit: int,
        equipment_family: str | None = None,
    ) -> list[SearchHit]:
        query = self._normalize(query)

        # Try Qdrant first for persistent, distributed search
        if self._qdrant is not None:
            name = self._collection_name(namespace)
            if name in self._qdrant_collections:
                try:
                    return self._search_qdrant(name, query, limit, equipment_family)
                except Exception as exc:
                    if self.required:
                        raise RuntimeError("Qdrant search failed in required mode") from exc
                    log.warning("qdrant_search_failed namespace=%s error=%s", namespace, exc)

        # Fallback to in-memory
        if self.required:
            raise RuntimeError("Qdrant is required but search fell back to memory mode")
        return self._search_memory(namespace, query, limit, equipment_family)

    def _search_qdrant(
        self,
        collection_name: str,
        query: np.ndarray,
        limit: int,
        equipment_family: str | None,
    ) -> list[SearchHit]:
        if self._qdrant is None:
            return []

        query_filter = None
        if equipment_family:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="equipment_family",
                        match=MatchValue(value=equipment_family),
                    )
                ]
            )

        qdrant = cast(Any, self._qdrant)
        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query.tolist(),
            limit=limit,
            query_filter=query_filter,
        )
        return [
            SearchHit(
                item_id=str(r.id),
                score=max(0.0, min(1.0, r.score)),
                payload=r.payload or {},
            )
            for r in results
        ]

    def _search_memory(
        self,
        namespace: str,
        query: np.ndarray,
        limit: int,
        equipment_family: str | None,
    ) -> list[SearchHit]:
        records = self._store.get(namespace, [])
        if not records:
            return []
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
        mode = "qdrant+memory" if self._qdrant is not None else "memory"
        namespaces = ",".join(sorted(self._store.keys())) or "empty"
        return {
            "mode": mode,
            "namespaces": namespaces,
            "required": str(self.required).lower(),
        }

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm < 1e-8:
            return array
        return array / norm
