from __future__ import annotations

import numpy as np

from maintenance_triage_copilot.config import ImageBackboneConfig, VideoBackboneConfig
from maintenance_triage_copilot.domain.models import MediaType, VisualObservation
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter


def test_image_adapter_returns_embedding() -> None:
    adapter = IJEPAImageAdapter(
        ImageBackboneConfig(
            input_size=32,
            patch_size=8,
            embed_dim=192,
            depth=2,
            num_heads=3,
        )
    )
    observation = VisualObservation(
        observation_id="obs-image",
        media_type=MediaType.image,
        tensor_shape=[3, 16, 16],
        tensor_values=np.ones((3, 16, 16), dtype=np.float32).reshape(-1).tolist(),
    )
    embedding = adapter.encode_observation(observation)
    assert embedding.shape == (192,)


def test_video_adapter_samples_to_fixed_frames() -> None:
    adapter = VJEPAVideoAdapter(
        VideoBackboneConfig(
            input_size=32,
            patch_size=8,
            num_frames=8,
            tubelet_size=2,
            embed_dim=192,
            depth=2,
            num_heads=3,
        )
    )
    observation = VisualObservation(
        observation_id="obs-video",
        media_type=MediaType.video,
        tensor_shape=[3, 10, 16, 16],
        tensor_values=np.ones((3, 10, 16, 16), dtype=np.float32).reshape(-1).tolist(),
    )
    sampled = adapter.sample_frames(observation.load_tensor())
    assert sampled.shape[1] == 8
    embedding = adapter.encode_observation(observation)
    assert embedding.shape == (192,)
