"""Tests for the pattern matching engine."""

from datetime import datetime

import numpy as np
import pytest

from construction_vl_jepa.config import InferenceConfig
from construction_vl_jepa.inference.pattern_matcher import PatternMatcher


@pytest.fixture
def matcher():
    cfg = InferenceConfig(pattern_match_top_k=5, pattern_similarity_threshold=0.5)
    return PatternMatcher(cfg)


class TestPatternMatcher:
    def test_store_and_retrieve(self, matcher):
        emb = np.random.randn(256).astype(np.float32)
        point_id = matcher.store_embedding(
            embedding=emb,
            machine_id="CNC-042",
            window_start=datetime(2025, 1, 1),
            window_end=datetime(2025, 1, 1, 0, 30),
        )
        assert point_id is not None
        assert len(matcher._memory_store) == 1

    def test_find_similar_returns_matches(self, matcher):
        # Store several embeddings
        base = np.random.randn(256).astype(np.float32)
        for i in range(5):
            noise = np.random.randn(256).astype(np.float32) * 0.1
            matcher.store_embedding(
                embedding=base + noise,
                machine_id=f"machine-{i}",
                window_start=datetime(2025, 1, 1),
                window_end=datetime(2025, 1, 1, 0, 30),
            )

        # Query with similar embedding
        query = base + np.random.randn(256).astype(np.float32) * 0.05
        matches = matcher.find_similar(query)
        assert len(matches) > 0
        assert all(m.similarity >= 0.5 for m in matches)

    def test_dissimilar_not_returned(self, matcher):
        # Store an embedding
        matcher.store_embedding(
            embedding=np.ones(256, dtype=np.float32),
            machine_id="machine-1",
            window_start=datetime(2025, 1, 1),
            window_end=datetime(2025, 1, 1, 0, 30),
        )

        # Query with very different embedding
        query = -np.ones(256, dtype=np.float32)
        matches = matcher.find_similar(query, score_threshold=0.8)
        assert len(matches) == 0

    def test_store_failure_pattern(self, matcher):
        emb = np.random.randn(256).astype(np.float32)
        point_id = matcher.store_failure_pattern(
            embedding=emb,
            machine_id="CNC-042",
            failure_event={
                "event_type": "failure",
                "description": "Bearing failure",
                "window_start": "2025-01-01T00:00:00",
                "window_end": "2025-01-01T00:30:00",
            },
            lead_time_hours=240.0,
        )
        assert point_id is not None
        # Verify it's stored with failure metadata
        stored = matcher._memory_store[-1]
        assert stored["payload"]["is_failure_pattern"] is True
        assert stored["payload"]["lead_time_hours"] == 240.0
