from __future__ import annotations

import pytest

from maintenance_triage_copilot.config import TextEncoderConfig
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder


def test_mock_backend_is_explicit() -> None:
    encoder = MaintenanceTextEncoder(TextEncoderConfig(backend="mock", embedding_dim=16))
    vector = encoder.encode_one("breaker fault light")
    assert vector.shape == (16,)


def test_sentence_transformer_load_failure_is_explicit() -> None:
    with pytest.raises(RuntimeError):
        MaintenanceTextEncoder(
            TextEncoderConfig(
                backend="sentence-transformer",
                model_name="definitely-not-a-real-model",
                embedding_dim=16,
                local_files_only=True,
            )
        )
