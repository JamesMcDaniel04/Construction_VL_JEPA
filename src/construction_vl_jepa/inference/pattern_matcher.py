"""Pattern matching via vector similarity search.

Stores latent embeddings in Qdrant and queries for similar historical states.
Only labeled failure patterns are returned by default so the system does not
confuse "similar operating regime" with "similar pre-failure trajectory."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from construction_vl_jepa.config import InferenceConfig


@dataclass
class PatternMatch:
    """A matched historical pattern."""

    pattern_id: str
    similarity: float
    machine_id: str
    window_start: str
    window_end: str
    event_type: str | None = None
    lead_time_hours: float | None = None
    metadata: dict | None = None


class PatternMatcher:
    """Vector similarity search over historical latent states.

    Uses Qdrant for storage and retrieval. Falls back to in-memory
    numpy search if Qdrant is unavailable (useful for dev/testing).
    """

    def __init__(self, cfg: InferenceConfig, qdrant_url: str | None = None, collection: str = "latent_states"):
        self.cfg = cfg
        self.collection = collection
        self.client = None
        self._memory_store: list[dict] = []  # fallback in-memory store

        if qdrant_url:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                self.client = QdrantClient(url=qdrant_url)
                # Ensure collection exists
                collections = [c.name for c in self.client.get_collections().collections]
                if collection not in collections:
                    self.client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=256, distance=Distance.COSINE),
                    )
            except Exception:
                self.client = None

    def store_embedding(
        self,
        embedding: np.ndarray,
        machine_id: str,
        window_start: datetime,
        window_end: datetime,
        metadata: dict | None = None,
    ) -> str:
        """Store a latent embedding with metadata.

        Args:
            embedding: Latent vector (latent_dim,).
            machine_id: Machine identifier.
            window_start: Start of the time window.
            window_end: End of the time window.
            metadata: Additional metadata (operating_mode, shift, anomaly_score, etc.)

        Returns:
            ID of the stored point.
        """
        import uuid

        point_id = str(uuid.uuid4())
        payload = {
            "machine_id": machine_id,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            **(metadata or {}),
        }

        if self.client is not None:
            from qdrant_client.models import PointStruct

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                ],
            )
        else:
            self._memory_store.append({
                "id": point_id,
                "vector": embedding.copy(),
                "payload": payload,
            })

        return point_id

    def find_similar(
        self,
        embedding: np.ndarray,
        top_k: int | None = None,
        score_threshold: float | None = None,
        failure_patterns_only: bool = True,
    ) -> list[PatternMatch]:
        """Find historical latent states similar to the given embedding.

        Args:
            embedding: Query vector (latent_dim,).
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.
            failure_patterns_only: Restrict matches to labeled pre-failure patterns.

        Returns:
            List of PatternMatch results, sorted by similarity descending.
        """
        top_k = top_k or self.cfg.pattern_match_top_k
        threshold = score_threshold or self.cfg.pattern_similarity_threshold

        if self.client is not None:
            query_filter = None
            if failure_patterns_only:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="is_failure_pattern",
                            match=MatchValue(value=True),
                        )
                    ]
                )
            results = self.client.search(
                collection_name=self.collection,
                query_vector=embedding.tolist(),
                limit=top_k,
                score_threshold=threshold,
                query_filter=query_filter,
            )
            return [
                PatternMatch(
                    pattern_id=str(r.id),
                    similarity=r.score,
                    machine_id=r.payload.get("machine_id", ""),
                    window_start=r.payload.get("window_start", ""),
                    window_end=r.payload.get("window_end", ""),
                    event_type=r.payload.get("event_type"),
                    lead_time_hours=r.payload.get("lead_time_hours"),
                    metadata=r.payload,
                )
                for r in results
            ]
        else:
            return self._memory_search(
                embedding,
                top_k,
                threshold,
                failure_patterns_only=failure_patterns_only,
            )

    def _memory_search(
        self,
        embedding: np.ndarray,
        top_k: int,
        threshold: float,
        failure_patterns_only: bool,
    ) -> list[PatternMatch]:
        """Fallback in-memory cosine similarity search."""
        if not self._memory_store:
            return []

        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        scored = []
        for item in self._memory_store:
            if failure_patterns_only and not item["payload"].get("is_failure_pattern", False):
                continue
            vec = item["vector"]
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            sim = float(np.dot(query_norm, vec_norm))
            if sim >= threshold:
                scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            PatternMatch(
                pattern_id=item["id"],
                similarity=sim,
                machine_id=item["payload"].get("machine_id", ""),
                window_start=item["payload"].get("window_start", ""),
                window_end=item["payload"].get("window_end", ""),
                event_type=item["payload"].get("event_type"),
                lead_time_hours=item["payload"].get("lead_time_hours"),
                metadata=item["payload"],
            )
            for sim, item in scored[:top_k]
        ]

    def store_failure_pattern(
        self,
        embedding: np.ndarray,
        machine_id: str,
        failure_event: dict,
        lead_time_hours: float,
    ) -> str:
        """Store a known pre-failure latent state as a named pattern.

        This is called during retrospective analysis when a failure is confirmed
        and we want to tag the latent states that preceded it.
        """
        metadata = {
            "event_type": failure_event.get("event_type", "failure"),
            "description": failure_event.get("description", ""),
            "lead_time_hours": lead_time_hours,
            "is_failure_pattern": True,
        }
        window_start = failure_event.get("window_start", datetime.now())
        window_end = failure_event.get("window_end", datetime.now())

        if isinstance(window_start, str):
            window_start = datetime.fromisoformat(window_start)
        if isinstance(window_end, str):
            window_end = datetime.fromisoformat(window_end)

        return self.store_embedding(
            embedding=embedding,
            machine_id=machine_id,
            window_start=window_start,
            window_end=window_end,
            metadata=metadata,
        )
