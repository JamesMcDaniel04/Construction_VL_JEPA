from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from maintenance_triage_copilot.api.main import create_app
from maintenance_triage_copilot.domain.models import MediaType, VisualObservation
from maintenance_triage_copilot.models.adapter import VisualTextProjector


def test_visual_observation_rejects_missing_payload() -> None:
    with pytest.raises(ValidationError):
        VisualObservation(
            observation_id="obs-missing",
            media_type=MediaType.image,
        )


def test_visual_observation_rejects_mixed_payload_sources() -> None:
    with pytest.raises(ValidationError):
        VisualObservation(
            observation_id="obs-mixed",
            media_type=MediaType.image,
            tensor_shape=[3, 16, 16],
            tensor_values=[0.1] * (3 * 16 * 16),
            embedding_values=[0.2] * 192,
        )


def test_visual_observation_allows_precomputed_embedding() -> None:
    observation = VisualObservation(
        observation_id="obs-embedding",
        media_type=MediaType.image,
        embedding_values=[0.2] * 192,
    )
    assert observation.has_precomputed_embedding() is True
    assert observation.has_raw_payload() is False
    assert observation.load_embedding().shape == (192,)


def test_create_app_loads_projector_checkpoint(small_config, tmp_path) -> None:
    checkpoint_path = tmp_path / "projector.pt"
    projector = VisualTextProjector(192, 192, 192)
    torch.save(projector.state_dict(), checkpoint_path)
    small_config.adapter.checkpoint_path = str(checkpoint_path)

    app = create_app(small_config)
    response = app.state.service.health()
    projector_status = response["components"]["projector"]
    assert projector_status["checkpoint_loaded"] is True
    assert projector_status["checkpoint_path"] == str(checkpoint_path)


def test_create_app_fails_fast_when_projector_checkpoint_is_missing(small_config, tmp_path) -> None:
    small_config.adapter.checkpoint_path = str(tmp_path / "missing-projector.pt")
    with pytest.raises(FileNotFoundError):
        create_app(small_config)


def test_precomputed_embedding_is_used_directly(triage_service, monkeypatch) -> None:
    original_encode_observation = triage_service.state.image_backbone.encode_observation

    def guard_encode(media: object) -> np.ndarray:
        if isinstance(media, VisualObservation) and media.embedding_values is not None:
            raise AssertionError("Observation embedding should be used directly")
        return original_encode_observation(media)

    monkeypatch.setattr(
        triage_service.state.image_backbone,
        "encode_observation",
        guard_encode,
    )

    observation = VisualObservation(
        observation_id="obs-direct-embedding",
        media_type=MediaType.image,
        embedding_values=[0.25] * 192,
    )
    encoded = triage_service._encode_visual(observation)
    assert encoded.shape == (192,)
